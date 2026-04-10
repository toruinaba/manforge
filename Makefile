# manforge Makefile
# Provides shortcuts for Fortran compilation and test execution.

.PHONY: fortran-build fortran-test test docker-build docker-test clean

# ---------------------------------------------------------------------------
# Fortran build (host)
# ---------------------------------------------------------------------------

## Compile fortran/test_basic.f90 into a Python extension via f2py
fortran-build:
	cd fortran && uv run python -m numpy.f2py -c test_basic.f90 -m manforge_test_basic

# ---------------------------------------------------------------------------
# Test targets
# ---------------------------------------------------------------------------

## Run Fortran integration tests only
fortran-test:
	uv run pytest tests/test_fortran_basic.py -v

## Run full test suite (excluding slow fitting tests)
test:
	uv run pytest tests/ --ignore=tests/test_fitting.py -v

## Run complete test suite (slow — includes fitting tests)
test-all:
	uv run pytest tests/ -v

# ---------------------------------------------------------------------------
# Docker targets (requires Docker installed)
# ---------------------------------------------------------------------------

## Build Docker image with gfortran + Python 3.12
docker-build:
	docker build -t manforge-fortran .

## Run Fortran build and tests inside Docker container
docker-test:
	docker run --rm -v $$(pwd):/workspace -w /workspace manforge-fortran \
		bash -c "make fortran-build && make fortran-test"

# ---------------------------------------------------------------------------
# Cleanup
# ---------------------------------------------------------------------------

## Remove f2py compiled artifacts in fortran/
clean:
	rm -f fortran/*.so fortran/*.mod fortran/*.o fortran/*module.c fortran/*-f2pywrappers*.f90
