FROM python:3.12-slim-trixie

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    MPLBACKEND=Agg \
    MPLCONFIGDIR=/tmp/syscall-hids-mpl

RUN apt-get update \
    && apt-get install -y --no-install-recommends strace \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --uid 10001 --create-home --shell /usr/sbin/nologin hids \
    && install -d -o hids -g hids /work

WORKDIR /opt/syscall-hids
COPY pyproject.toml requirements.txt README.md main.py ./
COPY core ./core
COPY visualizers ./visualizers
COPY scripts ./scripts
RUN python -m pip install .

WORKDIR /work
USER hids
ENTRYPOINT ["syscall-hids"]
CMD ["--help"]
