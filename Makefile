# manforge Makefile
# Provides shortcuts for Fortran compilation and test execution.

.PHONY: fortran-build fortran-build-umat test test-unit test-integration test-e2e test-e2e-slow test-benchmarks test-benchmarks-fortran test-all docker-build docker-test clean

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

## Run fast tests: unit + integration, excluding slow and fortran (default)
test:
	uv run pytest tests/unit tests/integration -m "not slow and not fortran" -v

## Run unit tests only (fastest)
test-unit:
	uv run pytest tests/unit -m "not slow and not fortran" -v

## Run integration tests excluding slow
test-integration:
	uv run pytest tests/integration -m "not slow and not fortran" -v

## Run e2e tests (CLI subprocess + fitting smoke)
test-e2e:
	uv run pytest tests/e2e -m "not slow and not fortran" -v

## Run slow e2e tests (fitting pipeline etc.)
test-e2e-slow:
	uv run pytest tests/e2e -m "slow" -v

## Run benchmark tests (Path A: analytical vs numerical; Fortran parts skipped)
test-benchmarks:
	uv run pytest tests/benchmarks -m "not fortran" -v

## Run Fortran benchmark tests (Path B: Python NR vs Fortran UMAT; requires compiled .so)
test-benchmarks-fortran:
	uv run pytest tests/benchmarks -m "fortran" -v

## Run complete test suite (includes slow and fortran if modules present)
test-all:
	uv run pytest tests -v

# ---------------------------------------------------------------------------
# Docker targets (requires Docker installed)
# ---------------------------------------------------------------------------

## Build Docker image with gfortran + Python 3.12
docker-build:
	docker build -t manforge-fortran .

## Run Fortran build and Fortran benchmark tests inside Docker container
docker-test:
	docker run --rm -v $$(pwd):/workspace -w /workspace manforge-fortran \
		bash -c "make fortran-build-umat && make test-benchmarks-fortran"

# ---------------------------------------------------------------------------
# Cleanup
# ---------------------------------------------------------------------------

## Remove f2py compiled artifacts in fortran/
clean:
	rm -f fortran/*.so fortran/*.mod fortran/*.o fortran/*module.c fortran/*-f2pywrappers*.f90
