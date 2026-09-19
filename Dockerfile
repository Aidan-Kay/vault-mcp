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

# The runtime stage carries no git metadata, so the commit is written down here or
# nowhere. /readyz reports it: AGPL section 13 asks a running service to offer its
# source, and a repository URL alone says where the project lives rather than
# which of its states is answering.
RUN git rev-parse HEAD > REVISION

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
COPY --from=builder /src/REVISION ./

# AGPL-3.0: the licence travels with the binary, so a running container can answer
# what terms it is under without reference to the repo it was built from.
COPY --from=builder /src/LICENSE ./

# The :ro mount that used to be the primary control is gone - this container
# writes now. Path containment in safe_resolve() is the control; running as a
# non-root uid is what is left of defence in depth. uid 1000 also matches the
# vault's file ownership, so written notes keep the ownership Samba expects.
RUN useradd --uid 1000 --create-home --shell /usr/sbin/nologin appuser

# The chunk-and-vector cache, so a restart is not a re-embed of the whole vault.
# Named explicitly rather than left to the XDG default, because HOME is not
# reliably set for a USER in a container and an unwritable default would be a
# silent loss of the whole feature. Mount a volume here to keep the cache across
# `up --force-recreate`; without one it survives a restart and no more. It holds
# the vault's text outside the vault, hence 0700 and INDEX_CACHE_PATH= to disable.
ENV HOME=/home/appuser \
    INDEX_CACHE_PATH=/cache/index.npz
RUN mkdir -p /cache && chown appuser:appuser /cache && chmod 700 /cache

USER appuser

EXPOSE 8080

# Liveness, not readiness. /healthz depends on nothing but the port being bound,
# because an unhealthy container is one something restarts - and restarting to
# recover a broken subsystem takes out every working one with it. Whether search
# is usable is /readyz, which is reported and never acted on automatically.
#
# python rather than curl: the slim image has no curl, and adding one to ask a
# question the interpreter can already ask is 4 MB for nothing. BIND_PORT is read
# so a moved port does not leave a healthcheck quietly failing against 8080.
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD ["python", "-c", "import os,urllib.request; urllib.request.urlopen('http://127.0.0.1:'+os.environ.get('BIND_PORT','8080')+'/healthz', timeout=4).read()"]

CMD ["python", "-m", "src.server"]
