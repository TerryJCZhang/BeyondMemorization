#!/usr/bin/env python3
"""Pipeline for extracting and filtering abstracts.

This module provides functionality to collect paper identifiers, extract abstracts from LaTeX documents,
and filter them using heuristic checks and a language model (LLM) for quality assurance.

- Collect paper IDs from QA datasets with structure: data/<domain>/<year>/<month>/qa_pairs
- Extract abstracts from LaTeX datasets with structure: data/<domain>/<year>/<month>/latex_text
- Filter abstracts using quick heuristics and an LLM to identify low-quality entries
- Save filtered abstracts and generate summary statistics
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import textwrap
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

try:  # Hugging Face datasets are required for reading the arrow artifacts
    from datasets import Dataset, load_from_disk
except ImportError:  # pragma: no cover - handled gracefully at runtime
    Dataset = None  # type: ignore
    load_from_disk = None  # type: ignore

try:  # The OpenAI SDK provides both sync and async OpenRouter clients
    import openai
except ImportError as exc:  # pragma: no cover - surfaced to the caller on first use
    raise RuntimeError(
        "The `openai` package is required for OpenRouter access. "
        "Install it with `pip install openai`."
    ) from exc

LOGGER = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")

DEFAULT_MODEL = "gpt-5-mini"
DEFAULT_FILTER_CONCURRENCY = 100
DEFAULT_MAX_CHARS = 8000
DEFAULT_MAX_LINES = 200
DEFAULT_TIMEOUT = 45
DEFAULT_RETRIES = 3

SYSTEM_PROMPT = (
    "You are a meticulous research editor who inspects scientific abstracts for quality. "
    "You never rewrite, paraphrase, or alter the abstract text. You only output JSON with your decision."
)

CRITICAL_FLAGS = {"empty", "gibberish", "too_long", "full_paper"}


@dataclass
class FilterDecision:
    keep: bool
    flags: List[str]
    source: str
    notes: str = ""
    corrected_abstract: Optional[str] = None


@dataclass
class AggregateStats:
    total: int = 0
    kept: int = 0
    removed_empty: int = 0
    removed_too_long: int = 0
    removed_full_paper: int = 0
    removed_gibberish: int = 0
    removed_latex: int = 0
    removed_other: int = 0
    errors: int = 0
    latex_corrected: int = 0

    def to_dict(self) -> Dict[str, int]:
        return {
            "total": self.total,
            "kept": self.kept,
            "removed_empty": self.removed_empty,
            "removed_too_long": self.removed_too_long,
            "removed_full_paper": self.removed_full_paper,
            "removed_gibberish": self.removed_gibberish,
            "removed_latex": self.removed_latex,
            "removed_other": self.removed_other,
            "errors": self.errors,
            "latex_corrected": self.latex_corrected,
        }

    def log_summary(self) -> None:
        LOGGER.info("Filter summary: kept %d / %d abstracts", self.kept, self.total)
        LOGGER.info(
            "Removed counts - empty:%d too_long:%d full_paper:%d gibberish:%d latex_issue:%d other:%d errors:%d corrections:%d",
            self.removed_empty,
            self.removed_too_long,
            self.removed_full_paper,
            self.removed_gibberish,
            self.removed_latex,
            self.removed_other,
            self.errors,
            self.latex_corrected,
        )


def quick_filters(abstract: str, max_chars: int, max_lines: int) -> Optional[FilterDecision]:
    stripped = abstract.strip()
    if not stripped:
        return FilterDecision(False, ["empty"], "heuristic", "Empty after stripping whitespace")

    if len(stripped) > max_chars:
        return FilterDecision(False, ["too_long"], "heuristic", f"Length {len(stripped)} > {max_chars}")

    line_count = stripped.count("\n") + 1
    if line_count > max_lines:
        return FilterDecision(False, ["full_paper"], "heuristic", f"Line count {line_count} > {max_lines}")

    lowered = stripped.lower()
    if any(marker in lowered for marker in ("\\documentclass", "\\begin{document}", "\\end{document}")):
        return FilterDecision(False, ["full_paper"], "heuristic", "Contains document-level LaTeX markers")

    heavy_macro_count = stripped.count("\\newcommand") + stripped.count("\\def") + stripped.count("\\renewcommand")
    if heavy_macro_count >= 3:
        return FilterDecision(False, ["full_paper"], "heuristic", "Contains many macro definitions")

    return None


def extract_response_text(response: object) -> str:
    text = getattr(response, "output_text", None)
    if text:
        return text

    output = getattr(response, "output", None)
    if output:
        parts: List[str] = []
        for item in output:
            for content in getattr(item, "content", []) or []:
                content_text = getattr(content, "text", None)
                if content_text:
                    parts.append(content_text)
        if parts:
            return "".join(parts)

    raise ValueError("No textual content in response")


def parse_decision(raw_text: str) -> FilterDecision:
    try:
        payload = json.loads(raw_text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"LLM returned non-JSON payload: {exc}: {raw_text[:200]}") from exc

    keep = bool(payload.get("keep"))
    raw_flags = payload.get("flags", [])
    notes = str(payload.get("notes", "")).strip()
    corrected_abstract = payload.get("corrected_abstract")
    if corrected_abstract is not None:
        corrected_abstract = str(corrected_abstract)

    if not isinstance(raw_flags, list):
        raw_flags = []

    normalized_flags: List[str] = []
    for flag in raw_flags:
        if not isinstance(flag, str):
            continue
        flag_lower = flag.strip().lower()
        if flag_lower:
            normalized_flags.append(flag_lower)

    if keep and any(flag in CRITICAL_FLAGS for flag in normalized_flags):
        keep = False

    if corrected_abstract is not None:
        corrected_abstract = corrected_abstract.strip()
        if not corrected_abstract:
            corrected_abstract = None

    return FilterDecision(
        keep=keep,
        flags=normalized_flags,
        source="llm",
        notes=notes,
        corrected_abstract=corrected_abstract,
    )


# ---------------------------------------------------------------------------
# Data classes to keep structured payloads tidy
# ---------------------------------------------------------------------------


@dataclass
class PaperAbstract:
    """Minimal representation of a paper abstract used downstream. Abstract text is kept in raw LaTeX form."""

    paper_id: str
    abstract: str
    title: Optional[str] = None
    category: Optional[str] = None
    paper_link: Optional[str] = None
    citations: Optional[Any] = None
    source_dataset: Optional[str] = None
    domain: Optional[str] = None
    year: Optional[str] = None
    month: Optional[str] = None
    date: Optional[str] = None
    extras: Dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> Dict[str, Any]:
        payload = {
            "paper_id": self.paper_id,
            "abstract": self.abstract,
            "title": self.title,
            "category": self.category,
            "paper_link": self.paper_link,
            "citations": self.citations,
            "source_dataset": self.source_dataset,
            "domain": self.domain,
            "year": self.year,
            "month": self.month,
            "date": self.date,
        }
        payload.update(self.extras)
        return {key: value for key, value in payload.items() if value is not None}

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "PaperAbstract":
        known_keys = {
            "paper_id",
            "abstract",
            "title",
            "category",
            "paper_link",
            "citations",
            "source_dataset",
            "domain",
            "year",
            "month",
            "date",
        }
        data = {key: payload.get(key) for key in known_keys}
        extras = {key: value for key, value in payload.items() if key not in known_keys}
        data["extras"] = extras
        if not data.get("paper_id"):
            raise ValueError("paper_id is required to instantiate PaperAbstract")
        return cls(**data)



# ---------------------------------------------------------------------------
# Core pipeline
# ---------------------------------------------------------------------------


class AbstractPipeline:
    """End-to-end orchestrator for abstract extraction workflows."""

    def __init__(
        self,
        data_dir: str | Path = "data",
        output_dir: str | Path = "output",
        derived_dir: str | Path = "derived",
    ) -> None:
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.derived_dir = Path(derived_dir)
        self.paper_ids_dir = self.derived_dir / "paper_ids"
        self.abstract_dir = self.derived_dir / "abstracts"
        for path in (
            self.paper_ids_dir,
            self.abstract_dir
        ):
            path.mkdir(parents=True, exist_ok=True)

        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError(
                "OpenAI API key missing. Export OPENAI_API_KEY or pass it to AbstractPipeline."
            )

        self.client = openai.AsyncOpenAI(
            api_key=api_key
        )

    # ------------------------------------------------------------------
    # Filesystem helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _write_json(path: Path, payload: Dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)

    @staticmethod
    def _write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row))
                handle.write("\n")

    @staticmethod
    def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
        with path.open("r", encoding="utf-8") as handle:
            return [json.loads(line) for line in handle if line.strip()]

    @staticmethod
    def _write_json_array(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            json.dump(list(rows), handle, indent=2)

    @staticmethod
    def _normalize_month(month: Any) -> Optional[str]:
        if month is None:
            return None
        if isinstance(month, int):
            return f"{month:02d}"
        month_str = str(month).strip()
        if not month_str:
            return None
        if month_str.isdigit():
            return f"{int(month_str):02d}"
        return month_str

    @staticmethod
    def _normalize_date(raw_date: Optional[str], fallback_year: Optional[str], fallback_month: Optional[str]) -> Optional[str]:
        if raw_date:
            if isinstance(raw_date, str):
                normalized = raw_date.strip()
                if normalized:
                    # Common formats: YYYY-MM-DD, YYYY/MM/DD, YYYY.MM.DD
                    match = re.search(r"(\\d{4})[-/\\.](\\d{2})[-/\\.](\\d{2})", normalized)
                    if match:
                        year, month, day = match.groups()
                        return f"{year}-{month}-{day}"
                    match = re.search(r"(\\d{4})[-/\\.](\\d{2})", normalized)
                    if match:
                        year, month = match.groups()
                        return f"{year}-{month}-01"
        if fallback_year and fallback_month:
            month_norm = AbstractPipeline._normalize_month(fallback_month)
            if month_norm:
                return f"{fallback_year}-{month_norm}-01"
        if fallback_year:
            return f"{fallback_year}-01-01"
        return None

    # ------------------------------------------------------------------
    # Stage 1: Collect paper identifiers
    # ------------------------------------------------------------------

    def collect_paper_ids(self, domains: Optional[List[str]] = None) -> Dict[str, Dict[str, Dict[str, Any]]]:
        if load_from_disk is None:
            raise RuntimeError(
                "The `datasets` library is required to read QA datasets. Install it with `pip install datasets`."
            )

        collected: Dict[str, Dict[str, Dict[str, Any]]] = {}

        for domain_dir in sorted(self.data_dir.glob("*/")):
            domain = domain_dir.name.rstrip("/")
            if domains and domain not in domains:
                continue
            collected.setdefault(domain, {})

            for year_dir in sorted(domain_dir.glob("*/")):
                year = year_dir.name.rstrip("/")
                collected[domain].setdefault(year, {})

                for month_dir in sorted(year_dir.glob("*/")):
                    month_raw = month_dir.name.rstrip("/")
                    month = self._normalize_month(month_raw)
                    dataset_path = month_dir / "qa_pairs"
                    if not dataset_path.exists():
                        LOGGER.debug("Skipping %s (qa_pairs missing)", dataset_path)
                        continue

                    dataset = load_from_disk(str(dataset_path))
                    paper_ids = sorted({pid for pid in dataset["paper_id"] if pid})
                    payload = {
                        "domain": domain,
                        "year": year,
                        "month": month,
                        "count": len(paper_ids),
                        "paper_ids": paper_ids,
                        "dataset_path": str(dataset_path),
                    }
                    collected[domain][year][month] = payload

                    output_path = self.paper_ids_dir / domain / year / f"{month}.json"
                    self._write_json(output_path, payload)
                    LOGGER.info(
                        "Collected %d paper ids for %s/%s/%s",
                        len(paper_ids),
                        domain,
                        year,
                        month,
                    )

        return collected

    # ------------------------------------------------------------------
    # Stage 2: Extract abstracts for collected paper ids
    # ------------------------------------------------------------------



    @staticmethod
    def _extract_abstract(full_text: str) -> str:
        if not full_text:
            return ""

        env_pattern = re.compile(
            r"\\begin\{abstract\*?\}(?:\[[^\]]*\])?",
            flags=re.IGNORECASE,
        )
        for match in env_pattern.finditer(full_text):
            start_idx = match.end()
            end_match = re.search(
                r"\\end\{abstract\*?\}",
                full_text[start_idx:],
                flags=re.IGNORECASE,
            )
            if not end_match:
                continue
            end_idx = start_idx + end_match.start()
            content = full_text[start_idx:end_idx]
            cleaned = AbstractPipeline._normalize_abstract(content)
            if cleaned:
                return cleaned
            after_idx = start_idx + end_match.end()
            trailing = AbstractPipeline._capture_following_paragraph(full_text[after_idx:])
            if trailing:
                return AbstractPipeline._normalize_abstract(trailing)

        cmd_pattern = re.compile(
            r"\\abstract\*?(?:\[[^\]]*\])?",
            flags=re.IGNORECASE,
        )
        for match in cmd_pattern.finditer(full_text):
            idx = AbstractPipeline._skip_whitespace(full_text, match.end())
            if idx < len(full_text) and full_text[idx] == "{":
                body, end_idx = AbstractPipeline._consume_braced_block(full_text, idx)
                cleaned = AbstractPipeline._normalize_abstract(body)
                if cleaned:
                    return cleaned
                trailing = AbstractPipeline._capture_following_paragraph(full_text[end_idx:])
                if trailing:
                    return AbstractPipeline._normalize_abstract(trailing)
            else:
                trailing = AbstractPipeline._capture_following_paragraph(full_text[idx:])
                if trailing:
                    return AbstractPipeline._normalize_abstract(trailing)

        for env in ("quotation", "quote"):
            pattern = re.compile(
                rf"\\begin\{{{env}\}}(.*?)\\end\{{{env}\}}",
                flags=re.IGNORECASE | re.DOTALL,
            )
            for block_match in pattern.finditer(full_text):
                body = block_match.group(1)
                if "abstract" not in body.lower():
                    continue
                label_match = re.search(
                    r"(\\textbf\{Abstract\.?\}|\{\\bf\s*Abstract\.?\}|\\bf\s*Abstract\.?)\s*(.*)",
                    body,
                    flags=re.IGNORECASE | re.DOTALL,
                )
                if not label_match:
                    continue
                abstract_text = label_match.group(2)
                split_pattern = re.compile(
                    r"\\textbf\{(?:Keywords?|Key\s*words?)\}",
                    flags=re.IGNORECASE | re.DOTALL,
                )
                abstract_text = split_pattern.split(abstract_text, maxsplit=1)[0]
                cleaned = AbstractPipeline._normalize_abstract(abstract_text)
                if cleaned:
                    return cleaned

        label_pattern = re.compile(
            r"(?:\\noindent\s*)?(\\textbf\{Abstract\.?\}|\{\\bf\s*Abstract\.?\}|\\bf\s*Abstract\.?)",
            flags=re.IGNORECASE,
        )
        label_match = label_pattern.search(full_text)
        if label_match:
            following = AbstractPipeline._capture_following_paragraph(full_text[label_match.end():])
            if following:
                return AbstractPipeline._normalize_abstract(following)

        return ""


    @staticmethod
    def _normalize_abstract(text: str) -> str:
        """Preserve LaTeX content while taming excessive blank space."""

        stripped = text.strip()
        stripped = stripped.replace("\r\n", "\n").replace("\r", "\n")
        stripped = re.sub(r"\n{3,}", "\n\n", stripped)
        return stripped

    @staticmethod
    def _skip_whitespace(text: str, index: int) -> int:
        while index < len(text) and text[index].isspace():
            index += 1
        return index

    @staticmethod
    def _consume_braced_block(text: str, start_index: int) -> Tuple[str, int]:
        if start_index >= len(text) or text[start_index] != "{":
            return "", start_index

        depth = 0
        index = start_index
        while index < len(text):
            char = text[index]
            if char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    return text[start_index + 1 : index], index + 1
            elif char == "\\" and index + 1 < len(text):
                index += 1
            index += 1
        return text[start_index + 1 : index], index

    @staticmethod
    def _capture_following_paragraph(text: str) -> str:
        if not text:
            return ""

        work = text.lstrip()
        if not work:
            return ""

        command_prefix = re.compile(
            r"^(\\(?:noindent|bigskip|medskip|smallskip|vskip)[^\\n]*|\\maketitle|\\par)\\s*",
            flags=re.IGNORECASE,
        )
        while True:
            match = command_prefix.match(work)
            if not match:
                break
            work = work[match.end():].lstrip()
            if not work:
                return ""

        stop_pattern = re.compile(
            r"\\(section|subsection|subsubsection|paragraph|keywords?|begin|tableofcontents)\\b",
            flags=re.IGNORECASE,
        )
        stop_match = stop_pattern.search(work)
        end_index = stop_match.start() if stop_match else len(work)
        snippet = work[:end_index]

        split_pattern = re.compile(
            r"\\(?:bigskip|medskip|smallskip|keywords?)\\b",
            flags=re.IGNORECASE,
        )
        snippet = split_pattern.split(snippet, maxsplit=1)[0]

        return snippet.strip()

    def build_abstract_datasets(
        self,
        paper_index: Dict[str, Dict[str, Dict[str, Any]]],
    ) -> Dict[str, Dict[str, Dict[str, List[PaperAbstract]]]]:
        if load_from_disk is None:
            raise RuntimeError(
                "The `datasets` library is required to read LaTeX datasets. Install it with `pip install datasets`."
            )

        output: Dict[str, Dict[str, Dict[str, List[PaperAbstract]]]] = {}

        for domain, years in paper_index.items():
            output.setdefault(domain, {})
            domain_root = self.output_dir / domain

            for year, months in years.items():
                output[domain].setdefault(year, {})

                for month, meta in months.items():
                    target_ids = set(meta.get("paper_ids", []))
                    if not target_ids:
                        LOGGER.warning("No paper ids for %s/%s/%s", domain, year, month)
                        continue

                    month_records: Dict[str, PaperAbstract] = {}
                    normalized_month = self._normalize_month(month)
                    latex_dirs: List[Path] = []
                    dataset_root = meta.get("dataset_path")
                    if dataset_root:
                        candidate = Path(dataset_root).parent / "latex_text"
                        if candidate.exists():
                            latex_dirs.append(candidate)
                    if not latex_dirs:
                        latex_dirs = list(domain_root.glob(f"**/{year}/{normalized_month}/latex_text"))
                    if not latex_dirs:
                        LOGGER.warning(
                            "No latex datasets found for %s/%s/%s", domain, year, month
                        )

                    for latex_dir in latex_dirs:
                        try:
                            dataset = load_from_disk(str(latex_dir))
                        except FileNotFoundError:
                            LOGGER.error("Latex dataset missing: %s", latex_dir)
                            continue

                        for row in dataset:
                            paper_id = row.get("id")
                            if paper_id not in target_ids:
                                continue

                            if paper_id in month_records:
                                continue

                            abstract = self._extract_abstract(row.get("full_text", ""))
                            date = self._normalize_date(
                                row.get("published")
                                or row.get("update_date")
                                or row.get("date")
                                or row.get("metadata_date"),
                                year,
                                normalized_month,
                            )
                            month_records[paper_id] = PaperAbstract(
                                paper_id=paper_id,
                                abstract=abstract,
                                title=row.get("title"),
                                category=row.get("category"),
                                paper_link=row.get("paper_link"),
                                citations=row.get("citations"),
                                source_dataset=str(latex_dir),
                                domain=domain,
                                year=year,
                                month=normalized_month,
                                date=date,
                            )

                    output[domain][year][normalized_month] = list(month_records.values())
                    jsonl_path = self.abstract_dir / domain / year / f"{normalized_month}.jsonl"
                    self._write_jsonl(jsonl_path, (record.to_json() for record in month_records.values()))
                    LOGGER.info(
                        "Saved %d abstracts for %s/%s/%s",
                        len(month_records),
                        domain,
                        year,
                        normalized_month,
                    )

        return output

    def load_abstract_index(
        self,
        domains: Optional[List[str]] = None,
    ) -> Dict[str, Dict[str, Dict[str, List[PaperAbstract]]]]:
        """Reconstruct the cached abstract index from JSONL files on disk."""

        index: Dict[str, Dict[str, Dict[str, List[PaperAbstract]]]] = {}

        for domain_dir in sorted(self.abstract_dir.glob("*/")):
            domain = domain_dir.name.rstrip("/")
            if domains and domain not in domains:
                continue

            index.setdefault(domain, {})

            for year_dir in sorted(domain_dir.glob("*/")):
                year = year_dir.name.rstrip("/")
                index[domain].setdefault(year, {})

                for month_file in sorted(year_dir.glob("*.jsonl")):
                    month = month_file.stem
                    records = self._read_jsonl(month_file)
                    index[domain][year][month] = [
                        PaperAbstract.from_dict(record) for record in records
                    ]

        return index

    def flatten_abstract_index(
        self,
        abstract_index: Dict[str, Dict[str, Dict[str, List[PaperAbstract]]]],
    ) -> List[Dict[str, Any]]:
        flattened: List[Dict[str, Any]] = []
        for domain, years in abstract_index.items():
            for year, months in years.items():
                for month, records in months.items():
                    for record in records:
                        payload = record.to_json()
                        payload["domain"] = payload.get("domain") or domain
                        payload["year"] = payload.get("year") or year
                        payload["month"] = self._normalize_month(payload.get("month") or month)
                        if payload["month"] is None:
                            payload["month"] = self._normalize_month(month)
                        payload["date"] = self._normalize_date(
                            payload.get("date"),
                            payload.get("year"),
                            payload.get("month"),
                        )
                        flattened.append(payload)
        return flattened

    async def filter_records(
        self,
        records: Sequence[Dict[str, Any]],
        *,
        concurrency: int = DEFAULT_FILTER_CONCURRENCY,
        max_chars: int = DEFAULT_MAX_CHARS,
        max_lines: int = DEFAULT_MAX_LINES,
        timeout: int = DEFAULT_TIMEOUT,
        retries: int = DEFAULT_RETRIES,
        model: str = DEFAULT_MODEL,
        use_llm: bool = True,
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], AggregateStats]:
        if not records:
            return [], [], AggregateStats()

        semaphore = asyncio.Semaphore(concurrency)
        decisions: List[Optional[FilterDecision]] = [None] * len(records)
        llm_tasks: List[asyncio.Task[Tuple[int, FilterDecision]]] = []

        for idx, record in enumerate(records):
            abstract = record.get("abstract", "")
            quick_decision = quick_filters(abstract, max_chars, max_lines)
            if quick_decision:
                decisions[idx] = quick_decision
            elif use_llm:
                llm_tasks.append(
                    asyncio.create_task(
                        self._evaluate_with_llm(
                            idx,
                            record,
                            semaphore,
                            timeout=timeout,
                            retries=retries,
                            model=model,
                        )
                    )
                )
            else:
                decisions[idx] = FilterDecision(True, [], "heuristic")

        if llm_tasks:
            results = await asyncio.gather(*llm_tasks)
            for idx, decision in results:
                decisions[idx] = decision

        stats = AggregateStats()
        kept_records: List[Dict[str, Any]] = []
        rejected_records: List[Dict[str, Any]] = []

        for record, decision in zip(records, decisions):
            if decision is None:
                decision = FilterDecision(False, ["other"], "internal", "Missing decision")

            stats.total += 1

            record_flags = list(decision.flags)
            record_notes = decision.notes

            if decision.keep:
                normalized_flags = {flag.lower() for flag in record_flags}
                if "latex_issue" in normalized_flags and not decision.corrected_abstract:
                    decision.keep = False
                elif decision.corrected_abstract:
                    if normalized_flags - {"latex_issue"}:
                        decision.keep = False
                    else:
                        record = dict(record)
                        record["abstract"] = decision.corrected_abstract
                        stats.latex_corrected += 1

                if decision.keep:
                    stats.kept += 1
                    kept_record = dict(record)
                    if record_flags:
                        kept_record["filter_flags"] = record_flags
                    if record_notes:
                        kept_record["filter_notes"] = record_notes
                    kept_records.append(kept_record)
                    continue

            if "empty" in record_flags:
                stats.removed_empty += 1
            elif "too_long" in record_flags:
                stats.removed_too_long += 1
            elif "full_paper" in record_flags:
                stats.removed_full_paper += 1
            elif "gibberish" in record_flags:
                stats.removed_gibberish += 1
            elif "latex_issue" in record_flags:
                stats.removed_latex += 1
            else:
                stats.removed_other += 1

            if decision.source in {"llm", "internal"} and "other" in record_flags:
                stats.errors += 1

            rejected_record = dict(record)
            rejected_record["filter_flags"] = record_flags
            rejected_record["filter_notes"] = record_notes
            rejected_record["filter_source"] = decision.source
            rejected_records.append(rejected_record)

        return kept_records, rejected_records, stats

    async def _evaluate_with_llm(
        self,
        idx: int,
        record: Dict[str, Any],
        semaphore: asyncio.Semaphore,
        *,
        timeout: int,
        retries: int,
        model: str,
    ) -> Tuple[int, FilterDecision]:
        abstract = record.get("abstract", "")
        title = record.get("title", "")
        paper_id = record.get("paper_id", "")
        domain = record.get("domain", "")
        year = record.get("year", "")
        month = record.get("month", "")

        user_prompt = textwrap.dedent(
            f"""
            Evaluate whether the following scientific abstract should be kept. Requirements:
            - If the abstract text is empty or whitespace, treat it as invalid.
            - If the text is incoherent, mostly punctuation, latex boilerplate, or metadata rather than a concise abstract, mark it as gibberish.
            - If the text has LaTeX syntax issues (unbalanced braces, stray percent signs that comment out content, unmatched $ delimiters, etc.), flag a LaTeX issue.
            - If LaTeX issues are the only serious problem and can be resolved without changing any words, punctuation, or formulas, correct the LaTeX while keeping the wording identical and provide the corrected text.
            - Do not paraphrase, rewrite, or alter the wording of the abstract. Only fix minimal LaTeX syntax if requested above.
            - Return strict JSON with keys: "keep" (boolean), "flags" (array of lower-case strings), "notes" (short plain text <= 25 words), "corrected_abstract" (string, required when you fix LaTeX).
            - Allowed flag values: empty, gibberish, latex_issue, too_long, full_paper, other.
            - If any serious issue other than latex issues is present, set keep to false.

            Context (do not rewrite based on this):
            - domain: {domain}
            - year: {year}
            - month: {month}
            - paper_id: {paper_id}
            - title: {title}

            Abstract:
            <<<ABSTRACT>>>
            {abstract}
            <<<END>>>
            """
        ).strip()

        attempt = 0
        last_error: Optional[Exception] = None
        while attempt < retries:
            attempt += 1
            try:
                async with semaphore:
                    response = await asyncio.wait_for(
                        self.client.responses.create(
                            model=model,
                            input=f"{SYSTEM_PROMPT}\n\n{user_prompt}",
                            max_output_tokens=600,
                            text={"verbosity": "low", "format": {"type": "json_object"}},
                        ),
                        timeout=timeout,
                    )
                raw_text = extract_response_text(response)
                decision = parse_decision(raw_text)
                return idx, decision
            except Exception as exc:
                last_error = exc
                LOGGER.warning(
                    "LLM evaluation failed for record %s (attempt %d/%d): %s",
                    paper_id or idx,
                    attempt,
                    retries,
                    exc,
                )
                if attempt < retries:
                    await asyncio.sleep(2 ** (attempt - 1))

        message = f"LLM error: {last_error}" if last_error else "Exceeded retries"
        return idx, FilterDecision(False, ["other"], "llm", message)
    

# ------------------------------------------------------------------


def build_pipeline(**kwargs: Any) -> AbstractPipeline:
    """Factory helper that mirrors the AbstractPipeline constructor."""

    return AbstractPipeline(**kwargs)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Paper abstract extraction and filtering pipeline.")
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data",
        help="Directory containing QA dataset for paper ids.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="output",
        help="Directory containing LaTeX dataset for abstracts.",
    )
    parser.add_argument(
        "--derived-dir",
        type=str,
        default="derived",
        help="Directory to store intermediate abstract results.",
    )
    parser.add_argument(
        "--domains",
        nargs="*",
        default=None,
        help="Specific domains to process (e.g., 'cs', 'math'). Process all if omitted.",
    )
    parser.add_argument(
        "--flat-output",
        type=str,
        help="Optional path for the flattened abstracts JSON (defaults to <derived_dir>/abstracts/abstracts_raw.json).",
    )
    parser.add_argument(
        "--filtered-output",
        type=str,
        help="Optional path for the filtered abstracts JSON (defaults to <derived_dir>/abstracts/abstracts_filtered.json).",
    )
    parser.add_argument(
        "--rejected-output",
        type=str,
        help="Optional path for rejected abstracts JSON (defaults to <derived_dir>/abstracts/abstracts_rejected.json).",
    )
    parser.add_argument(
        "--skip-filter",
        action="store_true",
        help="Skip the filtering stage and only emit the raw flattened abstracts.",
    )
    parser.add_argument(
        "--no-llm",
        action="store_true",
        help="Use heuristic filters only (no LLM calls). Ignored if --skip-filter is provided.",
    )
    parser.add_argument(
        "--filter-concurrency",
        type=int,
        default=DEFAULT_FILTER_CONCURRENCY,
        help="Maximum concurrent LLM moderation calls.",
    )
    parser.add_argument(
        "--filter-max-chars",
        type=int,
        default=DEFAULT_MAX_CHARS,
        help="Heuristic filter: maximum characters before rejecting as too long.",
    )
    parser.add_argument(
        "--filter-max-lines",
        type=int,
        default=DEFAULT_MAX_LINES,
        help="Heuristic filter: maximum lines before rejecting as full paper.",
    )
    parser.add_argument(
        "--filter-timeout",
        type=int,
        default=DEFAULT_TIMEOUT,
        help="Timeout (seconds) per LLM moderation call.",
    )
    parser.add_argument(
        "--filter-retries",
        type=int,
        default=DEFAULT_RETRIES,
        help="Retries per LLM moderation call.",
    )
    parser.add_argument(
        "--filter-model",
        type=str,
        default=DEFAULT_MODEL,
        help="Model name for the moderation LLM (OpenRouter format).",
    )

    args = parser.parse_args()

    pipeline = build_pipeline(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        derived_dir=args.derived_dir,
    )

    # Step 1: Collect paper IDs
    paper_index = pipeline.collect_paper_ids(domains=args.domains)

    # Step 2: Extract abstracts
    abstract_index = pipeline.build_abstract_datasets(paper_index)
    flattened_records = pipeline.flatten_abstract_index(abstract_index)

    raw_output_path = Path(args.flat_output) if args.flat_output else pipeline.abstract_dir / "abstracts_raw.json"
    pipeline._write_json_array(raw_output_path, flattened_records)
    LOGGER.info("Wrote %d raw abstracts to %s", len(flattened_records), raw_output_path)

    if args.skip_filter:
        filtered_records = flattened_records
        rejected_records: List[Dict[str, Any]] = []
        LOGGER.info("Filtering skipped (--skip-filter); raw abstracts copied to filtered output.")
    else:
        filtered_records, rejected_records, stats = asyncio.run(
            pipeline.filter_records(
                flattened_records,
                concurrency=args.filter_concurrency,
                max_chars=args.filter_max_chars,
                max_lines=args.filter_max_lines,
                timeout=args.filter_timeout,
                retries=args.filter_retries,
                model=args.filter_model,
                use_llm=not args.no_llm,
            )
        )
        stats.log_summary()

    filtered_output_path = Path(args.filtered_output) if args.filtered_output else pipeline.abstract_dir / "abstracts_filtered.json"
    pipeline._write_json_array(filtered_output_path, filtered_records)
    LOGGER.info("Wrote %d filtered abstracts to %s", len(filtered_records), filtered_output_path)

    rejected_output_path = Path(args.rejected_output) if args.rejected_output else pipeline.abstract_dir / "abstracts_rejected.json"
    if rejected_records or args.rejected_output:
        pipeline._write_json_array(rejected_output_path, rejected_records)
        LOGGER.info("Wrote %d rejected abstracts to %s", len(rejected_records), rejected_output_path)
