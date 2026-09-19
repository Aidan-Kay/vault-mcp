"""Environment-derived settings, resolved once at import.

Every knob the container has is here. Nothing else reads os.environ.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def _csv(name: str, default: str) -> tuple[str, ...]:
    raw = os.environ.get(name, default)
    return tuple(part.strip() for part in raw.split(",") if part.strip())


def _int(name: str, default: int) -> int:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError as exc:
        raise RuntimeError(f"{name} must be an integer, got {raw!r}") from exc


def _bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name, "").strip().lower()
    if not raw:
        return default
    if raw in {"1", "true", "yes", "on"}:
        return True
    if raw in {"0", "false", "no", "off"}:
        return False
    raise RuntimeError(f"{name} must be a boolean, got {raw!r}")


def _float(name: str, default: float) -> float:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError as exc:
        raise RuntimeError(f"{name} must be a number, got {raw!r}") from exc


@dataclass(frozen=True, slots=True)
class Settings:
    vault_path: Path
    api_key: str
    allowed_hosts: tuple[str, ...]
    ollama_url: str
    embed_model: str
    search_exclude_dirs: frozenset[str]
    index_exclude_dirs: tuple[str, ...]
    doc_suffixes: frozenset[str]
    doc_files_dir: str
    doc_ocr: bool
    chunk_target_tokens: int
    chunk_overlap_tokens: int
    chunk_min_tokens: int
    search_default_k: int
    embed_batch_size: int
    embed_dims: int
    watch_debounce_seconds: float
    index_reconcile_seconds: float
    host: str
    port: int


def load() -> Settings:
    api_key = os.environ.get("VAULT_MCP_API_KEY", "").strip()
    if not api_key:
        # Fail closed. An empty key must never be read as "auth disabled" for a
        # service that serves finances, insurance and addresses as plain text.
        raise RuntimeError("VAULT_MCP_API_KEY is unset - refusing to start")

    vault_path = Path(os.environ.get("VAULT_PATH", "/vault")).resolve()
    if not vault_path.is_dir():
        raise RuntimeError(f"VAULT_PATH {vault_path} is not a directory")

    ollama_url = os.environ.get("OLLAMA_URL", "http://ollama:11434/v1").rstrip("/")

    return Settings(
        vault_path=vault_path,
        api_key=api_key,
        allowed_hosts=_csv("MCP_ALLOWED_HOSTS", "vault-mcp:8080,127.0.0.1:8090"),
        ollama_url=ollama_url,
        embed_model=os.environ.get("EMBED_MODEL", "nomic-embed-text"),
        # Bare folder *names*, matched against every part of a path, so
        # "Workflows" drops the folder wherever it appears.
        search_exclude_dirs=frozenset(
            _csv("SEARCH_EXCLUDE_DIRS", "Workflows,Reports,.obsidian")
        ),
        # Root-relative path *prefixes* - "Approvals" alone matches nothing.
        # The shapes differ because the questions do, so the two lists are not
        # interchangeable despite the parallel names.
        #
        # Kept apart from SEARCH_EXCLUDE_DIRS on purpose. That one drops
        # Workflows/ and Reports/ wholesale from *search*; index.md must not,
        # because curated notes live inside both. These are the generated series
        # only, and the list mirrors the "Excluded folders" table in
        # Meta/Conventions.md.
        index_exclude_dirs=_csv(
            "INDEX_EXCLUDE_DIRS",
            "Workflows/Approvals,"
            "Workflows/Email Triage/Logs,"
            "Reports/Daily Coffee Read,"
            "Reports/Plex Music Recommendations,"
            "Reports/Vault Maintenance,"
            "Reports/Monthly Events Discovery",
        ),
        # Binary document types the vault will carry. One list, read by the
        # read path, the write path, the move path and the index walk, so a
        # suffix cannot be uploadable and unreadable at the same time. Every
        # entry needs an extractor in documents.py, which is why adding .docx
        # is a code change and not a config change.
        doc_suffixes=frozenset(
            part if part.startswith(".") else f".{part}"
            for part in (p.lower() for p in _csv("DOC_SUFFIXES", ".pdf"))
        ),
        # The folder name a document upload must land directly inside. This is
        # the containment control: a credential that can carry bytes cannot put
        # them anywhere a note lives. Configurable because it is a vault
        # convention rather than a law, but changing it changes the vault.
        doc_files_dir=os.environ.get("DOC_FILES_DIR", "Files").strip("/ ") or "Files",
        # OCR a document that has no text layer. Requires a Tesseract binary and
        # its tessdata on the host; documents.py reports it unavailable rather
        # than failing when there is none. See the OCR note in the README.
        doc_ocr=_bool("DOC_OCR", True),
        chunk_target_tokens=_int("CHUNK_TARGET_TOKENS", 400),
        chunk_overlap_tokens=_int("CHUNK_OVERLAP_TOKENS", 60),
        chunk_min_tokens=_int("CHUNK_MIN_TOKENS", 120),
        search_default_k=_int("SEARCH_DEFAULT_K", 6),
        embed_batch_size=_int("EMBED_BATCH_SIZE", 64),
        embed_dims=_int("EMBED_DIMS", 768),
        watch_debounce_seconds=_float("WATCH_DEBOUNCE_SECONDS", 2.0),
        # index.md is otherwise only ever updated one note at a time, so an
        # event that never arrives costs a restart to notice rather than one
        # pass. watcher.py handles the gap that has actually bitten us; this is
        # for the ones that have not yet. 0 disables the pass.
        index_reconcile_seconds=_float("INDEX_RECONCILE_SECONDS", 900.0),
        host=os.environ.get("BIND_HOST", "0.0.0.0"),
        port=_int("BIND_PORT", 8080),
    )


settings = load()
