# manforge Makefile
# Provides shortcuts for Fortran compilation and test execution.

.PHONY: fortran-build fortran-build-umat fortran-test fortran-test-umat test test-unit test-integration test-slow test-fortran test-all docker-build docker-test clean

# ---------------------------------------------------------------------------
# Fortran build (host)
# ---------------------------------------------------------------------------

## Compile fortran/test_basic.f90 into a Python extension via f2py
fortran-build:
	cd fortran && uv run python -m numpy.f2py -c test_basic.f90 -m manforge_test_basic

## Compile UMAT sources (abaqus_stubs + j2_isotropic_3d) into a Python extension via f2py
fortran-build-umat:
	cd fortran && uv run python -m numpy.f2py -c abaqus_stubs.f90 j2_isotropic_3d.f90 -m j2_isotropic_3d

# ---------------------------------------------------------------------------
# Test targets
# ---------------------------------------------------------------------------

## Run Fortran basic integration tests only
fortran-test:
	uv run pytest tests/test_fortran_basic.py -v

## Run Fortran UMAT cross-validation tests only
fortran-test-umat:
	uv run pytest tests/test_fortran_umat.py -v

## Run fast tests: unit + integration, excluding slow and fortran (default)
test:
	uv run pytest tests/unit tests/integration -m "not slow" -v

## Run unit tests only (fastest)
test-unit:
	uv run pytest tests/unit -v

## Run integration tests only (no slow)
test-integration:
	uv run pytest tests/integration -m "not slow" -v

## Run slow tests only (fitting, long FD tangent checks)
test-slow:
	uv run pytest tests/slow tests/integration tests/unit -m "slow" -v

## Run Fortran-dependent tests (requires compiled .so)
test-fortran:
	uv run pytest tests/fortran -v

## Run complete test suite (includes slow and fortran if modules present)
test-all:
	uv run pytest tests -v

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
