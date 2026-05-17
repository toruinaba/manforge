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
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

COPY pyproject.toml uv.lock ./
RUN pip install --no-cache-dir uv \
 && uv sync --extra dev --extra fortran

COPY . .

# Default: compile UMAT and run Fortran equivalence benchmarks
CMD ["bash", "-c", "make fortran-build-umat && make test-benchmarks-fortran"]
