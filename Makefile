# manforge Makefile
# Provides shortcuts for Fortran compilation and test execution.

.PHONY: fortran-build fortran-build-umat fortran-build-yu fortran-build-yu-ps fortran-build-yu-proj-ps test test-unit test-integration test-e2e test-e2e-slow test-slow test-benchmarks test-benchmarks-fortran test-all docker-build docker-test docker-test-yu clean
# Archived targets (fixed-form .for files moved to archives/fortran_fixed_form/):
#   fortran-build-yu-fixed  -- cd fortran && uv run python -m numpy.f2py -c abaqus_stubs.f90 yu_kinematic_3d_fixed.for -m yu_kinematic_3d_fixed -llapack -lblas

# ---------------------------------------------------------------------------
# Fortran build (host)
# ---------------------------------------------------------------------------

# PY selects the interpreter that drives f2py.  Inside the Docker image the venv
# is read-only and lives outside the bind-mounted tree, so `uv run` would try to
# re-sync it and fail; override with `make PY=python fortran-build-yu` there.
PY ?= uv run python

## Compile fortran/test_basic.f90 into a Python extension via f2py
fortran-build:
	cd fortran && $(PY) -m numpy.f2py -c test_basic.f90 -m manforge_test_basic

## Compile UMAT sources (abaqus_stubs + j2_isotropic_3d) into a Python extension via f2py
fortran-build-umat:
	cd fortran && $(PY) -m numpy.f2py -c abaqus_stubs.f90 j2_isotropic_3d.f90 -m j2_isotropic_3d

## Compile YU Kinematic UMAT (abaqus_stubs + yu_kinematic_3d) into a Python extension via f2py
fortran-build-yu:
	cd fortran && $(PY) -m numpy.f2py -c abaqus_stubs.f90 yu_kinematic_3d.f90 -m yu_kinematic_3d -llapack -lblas

## Compile YU Kinematic plane-stress UMAT (abaqus_stubs + yu_kinematic_ps) via f2py
fortran-build-yu-ps:
	cd fortran && $(PY) -m numpy.f2py -c abaqus_stubs.f90 yu_kinematic_ps.f90 -m yu_kinematic_ps -llapack -lblas

## Compile YU Kinematic plane-stress UMAT with the projected stagnation update
fortran-build-yu-proj-ps:
	cd fortran && $(PY) -m numpy.f2py -c abaqus_stubs.f90 yu_kinematic_proj_ps.f90 -m yu_kinematic_proj_ps -llapack -lblas

## (archived) fortran-build-yu-fixed and fortran-build-yu-abaqus moved to archives/fortran_fixed_form/

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

## Run all slow-marked tests across unit + integration + e2e
test-slow:
	uv run pytest tests/unit tests/integration tests/e2e -m "slow and not fortran" -v

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

# Runs as the invoking user so build artifacts in the bind mount stay writable
# on the host; PY=python bypasses `uv run` because /opt/venv is not writable.
DOCKER_RUN = docker run --rm --user $$(id -u):$$(id -g) \
	-v $$(pwd):/workspace -w /workspace -e HOME=/tmp \
	-e PYTHONPATH=/workspace/src:/workspace/fortran manforge-fortran

## Run Fortran build and Fortran benchmark tests inside Docker container
docker-test:
	$(DOCKER_RUN) bash -c "make PY=python fortran-build-umat && \
		python -m pytest tests/benchmarks -m fortran -v"

## Build both YU Fortran modules and run the YU Fortran benchmarks in Docker
docker-test-yu:
	$(DOCKER_RUN) bash -c "make PY=python fortran-build-yu && \
		make PY=python fortran-build-yu-ps && \
		make PY=python fortran-build-yu-proj-ps && \
		python -m pytest tests/benchmarks/yu_kinematic -m fortran -v"

# ---------------------------------------------------------------------------
# Cleanup
# ---------------------------------------------------------------------------

## Remove f2py compiled artifacts in fortran/
clean:
	rm -f fortran/*.so fortran/*.mod fortran/*.o fortran/*module.c fortran/*-f2pywrappers*.f90
