# manforge — reproducible gfortran + Python 3.12 build environment
#
# Provides gfortran, f2py (via numpy), and all Python dependencies.
# Used to run Fortran-vs-Python equivalence benchmarks reproducibly.
#
# Build:
#   docker build -t manforge-fortran .
#
# Run tests inside container:
#   docker run --rm -v $(pwd):/workspace -w /workspace manforge-fortran \
#       bash -c "make fortran-build-umat && make test-benchmarks-fortran"

FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
        gfortran \
        make \
        liblapack-dev \
        libblas-dev \
    && rm -rf /var/lib/apt/lists/*

# The venv lives outside /workspace because that path is bind-mounted at run
# time, which would otherwise hide it behind the host's own .venv.
ENV UV_PROJECT_ENVIRONMENT=/opt/venv
ENV PATH=/opt/venv/bin:$PATH

WORKDIR /build
# README.md is required here: pyproject.toml declares it as the project readme,
# so the manforge build in `uv sync` fails without it.
COPY pyproject.toml uv.lock README.md ./
RUN pip install --no-cache-dir uv \
 && uv sync --extra dev --extra fortran

WORKDIR /workspace

COPY . .

# Default: compile UMAT and run Fortran equivalence benchmarks
CMD ["bash", "-c", "make fortran-build-umat && make test-benchmarks-fortran"]
