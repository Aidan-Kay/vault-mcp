# ─── Build Stage ───────────────────────────────────────────────────────────────
FROM python:3.13-slim AS builder

# git + ca-certificates are required to fetch the repository over HTTPS
RUN apt-get update && \
    apt-get install -y --no-install-recommends git ca-certificates && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /src

# Clone the repository (shallow clone — no history needed). Docker caches this
# layer on the URL alone, so a new commit on main does NOT invalidate it:
# rebuild with `docker compose build --no-cache vault-mcp` to pick one up.
RUN git clone --depth 1 https://github.com/Aidan-Kay/vault-mcp.git .

# ─── Runtime Stage ────────────────────────────────────────────────────────────
FROM python:3.13-slim AS runtime

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# git is left behind in the builder — the runtime image never needs it.
COPY --from=builder /src/requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# OCR for documents that arrive as scans, which is what utility providers email.
# Tesseract is not a Python dependency and does not arrive with the wheel: it is
# a binary PyMuPDF shells out to, plus its language data. Without it a scanned
# bill files correctly and extracts to nothing, which src/documents.py reports
# as `needs_ocr` rather than as success - so the image is usable without this
# layer, it just cannot read scans. Set DOC_OCR=false to stop it trying.
RUN apt-get update && \
    apt-get install -y --no-install-recommends tesseract-ocr tesseract-ocr-eng && \
    rm -rf /var/lib/apt/lists/*
# TESSDATA_PREFIX is deliberately not set. PyMuPDF returns that variable
# verbatim when it is set, without checking the directory exists, so a path
# hardcoded here that a future base image moves would look configured and fail
# at OCR time. Unset, it asks `tesseract --list-langs` where its own data is,
# which is authoritative and survives the upgrade.

COPY --from=builder /src/src/ ./src/

# AGPL-3.0: the licence travels with the binary, so a running container can answer
# what terms it is under without reference to the repo it was built from.
COPY --from=builder /src/LICENSE ./

# The :ro mount that used to be the primary control is gone - this container
# writes now. Path containment in safe_resolve() is the control; running as a
# non-root uid is what is left of defence in depth. uid 1000 also matches the
# vault's file ownership, so written notes keep the ownership Samba expects.
RUN useradd --uid 1000 --create-home --shell /usr/sbin/nologin appuser
USER appuser

EXPOSE 8080

CMD ["python", "-m", "src.server"]
