FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    TZ=Asia/Shanghai

WORKDIR /app
RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates fonts-noto-cjk tzdata \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml /tmp/lysbot-pyproject.toml
RUN --mount=type=cache,target=/root/.cache/pip \
    python -c "import pathlib, subprocess, sys, tomllib; deps = tomllib.loads(pathlib.Path('/tmp/lysbot-pyproject.toml').read_text())['project']['dependencies']; subprocess.check_call([sys.executable, '-m', 'pip', 'install', *deps])"

COPY pyproject.toml README.md LICENSE NOTICE AGENTS.md QUICKSTART.md CONTRIBUTING.md ./
COPY src ./src
COPY lysbot ./lysbot
COPY personas ./personas
COPY knowledge/project.md ./knowledge/project.md
COPY scripts ./scripts
COPY docs ./docs
COPY extensions/.gitkeep ./extensions/.gitkeep
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --no-deps . && pip check

COPY bot.py ./
CMD ["python", "bot.py"]
