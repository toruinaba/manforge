# returnmap Makefile
# Provides shortcuts for Fortran compilation and test execution.

.PHONY: fortran-build fortran-build-umat fortran-build-yu fortran-build-yu-ps fortran-build-yu-3d-bench docker-bench-yu-3d test test-unit test-integration test-e2e test-e2e-slow test-slow test-models test-models-fortran test-all docker-build docker-test docker-test-yu docs docs-clean clean
# Archived targets (fixed-form .for files moved to archives/fortran_fixed_form/):
#   fortran-build-yu-fixed  -- uv run returnmap build --abaqus-stubs archives/fortran_fixed_form/yu_kinematic_3d_fixed.for --name yu_kinematic_3d_fixed --link lapack --link blas

# ---------------------------------------------------------------------------
# Fortran build (host)
# ---------------------------------------------------------------------------

# PY selects the interpreter that drives the build.  Inside the Docker image the venv
# is read-only and lives outside the bind-mounted tree, so `uv run` would try to
# re-sync it and fail; override with `make PY=python fortran-build-yu` there.
PY ?= uv run python

# Each model family keeps its UMAT sources next to the model.  The compiled
# modules are all built into fortran/, which the tests put on sys.path.
# `returnmap build --abaqus-stubs` compiles in the abaqus_stubs.f90 the
# package ships (src/returnmap/fortran/).
MODELS = src/returnmap/models
BUILD = $(PY) -m returnmap build

## Compile fortran/test_basic.f90 into a Python extension via f2py
fortran-build:
	$(BUILD) fortran/test_basic.f90 --name returnmap_test_basic

## Compile UMAT sources (abaqus_stubs + j2_isotropic_3d) into a Python extension via f2py
fortran-build-umat:
	$(BUILD) --abaqus-stubs $(MODELS)/j2_isotropic/fortran/j2_isotropic_3d.f90 --name j2_isotropic_3d

## Compile YU Kinematic UMAT (abaqus_stubs + yu_kinematic_3d) into a Python extension via f2py
fortran-build-yu:
	$(BUILD) --abaqus-stubs $(MODELS)/yu_kinematic/fortran/yu_kinematic_3d.f90 --name yu_kinematic_3d --link lapack --link blas

## Compile YU Kinematic plane-stress UMAT (abaqus_stubs + yu_kinematic_ps) via f2py
fortran-build-yu-ps:
	$(BUILD) --abaqus-stubs $(MODELS)/yu_kinematic/fortran/yu_kinematic_ps.f90 --name yu_kinematic_ps --link lapack --link blas

## (archived) fortran-build-yu-fixed and fortran-build-yu-abaqus moved to archives/fortran_fixed_form/

# ---------------------------------------------------------------------------
# Test targets
# ---------------------------------------------------------------------------

## Run fast tests: framework unit + integration and every model family, excluding slow and fortran (default)
test:
	uv run pytest tests/unit tests/integration $(MODELS) -m "not slow and not fortran" -v

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

## Run all slow-marked tests (framework and models)
test-slow:
	uv run pytest -m "slow and not fortran" -v

## Run the tests of the model families (slow included; Fortran parts skipped).
## One family: uv run pytest $(MODELS)/yu_kinematic -m "not fortran"
test-models:
	uv run pytest $(MODELS) -m "not fortran" -v

## Run the Python-vs-UMAT tests of the model families (requires compiled .so)
test-models-fortran:
	uv run pytest $(MODELS) -m "fortran" -v

## Run complete test suite (includes slow and fortran if modules present)
test-all:
	uv run pytest -v

# ---------------------------------------------------------------------------
# Docker targets (requires Docker installed)
# ---------------------------------------------------------------------------

## Build Docker image with gfortran + Python 3.12
docker-build:
	docker build -t returnmap-fortran .

# Runs as the invoking user so build artifacts in the bind mount stay writable
# on the host; PY=python bypasses `uv run` because /opt/venv is not writable.
DOCKER_RUN = docker run --rm --user $$(id -u):$$(id -g) \
	-v $$(pwd):/workspace -w /workspace -e HOME=/tmp \
	-e PYTHONPATH=/workspace/src:/workspace/fortran returnmap-fortran

## Build the J2 UMAT and run the Python-vs-UMAT tests inside Docker
docker-test:
	$(DOCKER_RUN) bash -c "make PY=python fortran-build-umat && \
		python -m pytest $(MODELS) -m fortran -v"

## Build both YU Fortran modules and run the YU Python-vs-UMAT tests in Docker
docker-test-yu:
	$(DOCKER_RUN) bash -c "make PY=python fortran-build-yu && \
		make PY=python fortran-build-yu-ps && \
		python -m pytest $(MODELS)/yu_kinematic -m fortran -v"

# ---------------------------------------------------------------------------
# UMAT timing harness (benchmarks/umat_timing, kept out of fortran/)
# ---------------------------------------------------------------------------

# The harness lives outside fortran/ so that directory stays purely shippable
# UMAT sources.  Each build pairs one implementation file with its wrapper file;
# they must stay separate compilation units, otherwise the optimiser sees the
# callee bodies and hoists the repeat loops away.
BENCH_DIR = benchmarks/umat_timing

## Measure absolute us/call for the 3D UMAT inside Docker
docker-bench-yu-3d:
	$(DOCKER_RUN) bash -c "cd $(BENCH_DIR) && \
		python -m numpy.f2py -c \$$(python -m returnmap stubs) \
			../../$(MODELS)/yu_kinematic/fortran/yu_kinematic_3d.f90 yu_3d_bench.f90 \
			-m yu_3d_bench -llapack -lblas && \
		PYTHONPATH=/workspace/src:\$$PWD python bench_3d.py"

# ---------------------------------------------------------------------------
# Documentation
# ---------------------------------------------------------------------------

## Build the HTML documentation into docs/_build/html (needs: uv sync --extra docs).
## Warnings are errors: a dead cross-reference or an undocumented public name fails the build.
docs:
	uv run sphinx-build -b html -W --keep-going -n docs docs/_build/html

## Remove the built documentation
docs-clean:
	rm -rf docs/_build

# ---------------------------------------------------------------------------
# Cleanup
# ---------------------------------------------------------------------------

## Remove f2py compiled artifacts in fortran/
clean:
	rm -f fortran/*.so fortran/*.mod fortran/*.o fortran/*module.c fortran/*-f2pywrappers*.f90
	rm -f $(BENCH_DIR)/*.so $(BENCH_DIR)/*.mod $(BENCH_DIR)/*.o $(BENCH_DIR)/*module.c $(BENCH_DIR)/*-f2pywrappers*.f
