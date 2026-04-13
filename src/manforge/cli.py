"""manforge CLI — Fortran build toolkit for Python users.

Usage:
  manforge build <files...> --name <module_name> [--output-dir <dir>] [--verbose]
  manforge clean [--dir <dir>] [--dry-run]
  manforge list  [--dir <dir>]
"""

import argparse
import glob
import platform
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------


def _find_project_root() -> Path:
    """Walk up from cwd looking for pyproject.toml. Fallback to cwd."""
    current = Path.cwd()
    for parent in [current, *current.parents]:
        if (parent / "pyproject.toml").exists():
            return parent
    return current


def _default_output_dir() -> Path:
    return _find_project_root() / "fortran"


def _check_gfortran() -> None:
    if shutil.which("gfortran") is not None:
        return

    system = platform.system()
    if system == "Darwin":
        hint = "  macOS:   brew install gcc"
    elif system == "Linux":
        hint = "  Ubuntu:  sudo apt-get install gfortran\n  Fedora:  sudo dnf install gcc-gfortran"
    else:
        hint = "  See https://gcc.gnu.org/wiki/GFortranBinaries"

    print("Error: gfortran not found.")
    print("Install it with:")
    print(hint)
    print()
    print("Alternatively, use Docker (requires Docker installed):")
    print("  make docker-build && make docker-test")
    sys.exit(1)


def _check_f2py() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "numpy.f2py", "--version"],
        capture_output=True,
    )
    if result.returncode != 0:
        print("Error: numpy.f2py is not available.")
        print("Install numpy:")
        print("  uv sync --extra fortran")
        print("  # or: pip install numpy meson ninja")
        sys.exit(1)


def _check_meson() -> None:
    if shutil.which("meson") is not None:
        return
    print("Error: meson not found.")
    print("f2py requires meson as its build backend on Python 3.12+.")
    print("Install with:")
    print("  uv sync --extra fortran")
    print("  # or: pip install meson ninja")
    sys.exit(1)


def _human_size(nbytes: int) -> str:
    for unit in ("B", "KB", "MB"):
        if nbytes < 1024:
            return f"{nbytes:.0f} {unit}"
        nbytes //= 1024
    return f"{nbytes:.0f} GB"


def _extract_module_name(so_path: Path) -> str:
    """Extract the Python module name from a .so/.pyd filename.

    e.g. 'j2_isotropic_3d.cpython-312-x86_64-linux-gnu.so' → 'j2_isotropic_3d'
    """
    return so_path.name.split(".")[0]


# ---------------------------------------------------------------------------
# Subcommand implementations
# ---------------------------------------------------------------------------


def cmd_build(args: argparse.Namespace) -> None:
    _check_gfortran()
    _check_f2py()
    _check_meson()

    # Validate source files
    sources: list[Path] = []
    for f in args.files:
        p = Path(f).resolve()
        if not p.exists():
            print(f"Error: file not found: {f}")
            sys.exit(1)
        if p.suffix not in (".f90", ".f", ".F90", ".F"):
            print(f"Error: not a Fortran source file: {f}")
            print("Expected extensions: .f90, .f, .F90, .F")
            sys.exit(1)
        sources.append(p)

    output_dir = Path(args.output_dir).resolve() if args.output_dir else _default_output_dir()
    output_dir.mkdir(parents=True, exist_ok=True)

    module_name: str = args.name
    source_names = ", ".join(s.name for s in sources)
    print(f"Building module '{module_name}' from: {source_names}")
    print(f"Output directory: {output_dir}")
    print()

    cmd = [sys.executable, "-m", "numpy.f2py", "-c"] + [str(s) for s in sources] + ["-m", module_name]

    if args.verbose:
        result = subprocess.run(cmd, cwd=output_dir)
    else:
        result = subprocess.run(cmd, cwd=output_dir, capture_output=True, text=True)

    if result.returncode != 0:
        print("Build failed.")
        if not args.verbose:
            print()
            print("--- f2py output ---")
            if result.stdout:
                print(result.stdout)
            if result.stderr:
                print(result.stderr)
            print("-------------------")
            print()
            print("Tip: run with --verbose for full compiler output.")
        sys.exit(1)

    # Locate the produced .so / .pyd
    patterns = [f"{module_name}.cpython-*.so", f"{module_name}.*.pyd", f"{module_name}.so"]
    built_files = []
    for pat in patterns:
        built_files.extend(output_dir.glob(pat))

    print(f"Build succeeded.")
    if built_files:
        so = built_files[0]
        print(f"Module file: {so}")

    print()
    print("To use in Python:")
    print(f"  import sys")
    print(f"  sys.path.insert(0, '{output_dir}')")
    print(f"  import {module_name}")
    print()
    print("Or with FortranUMAT:")
    print(f"  from manforge.verification import FortranUMAT")
    print(f"  import sys; sys.path.insert(0, '{output_dir}')")
    print(f"  umat = FortranUMAT('{module_name}')")


def cmd_clean(args: argparse.Namespace) -> None:
    target_dir = Path(args.dir).resolve() if args.dir else _default_output_dir()

    if not target_dir.exists():
        print(f"Nothing to clean (directory not found: {target_dir})")
        return

    patterns = ["*.so", "*.pyd", "*.mod", "*.o", "*module.c", "*-f2pywrappers*.f90"]
    matches: list[Path] = []
    for pat in patterns:
        matches.extend(target_dir.glob(pat))

    if not matches:
        print("Nothing to clean.")
        return

    dry_run: bool = args.dry_run
    if dry_run:
        print(f"Dry run — would remove {len(matches)} file(s):")
    else:
        print(f"Removing {len(matches)} file(s):")

    for f in sorted(matches):
        print(f"  {f}")
        if not dry_run:
            f.unlink()

    if not dry_run:
        print("Done.")


def cmd_list(args: argparse.Namespace) -> None:
    target_dir = Path(args.dir).resolve() if args.dir else _default_output_dir()

    if not target_dir.exists():
        print(f"No compiled modules found (directory not found: {target_dir})")
        return

    so_files: list[Path] = []
    so_files.extend(target_dir.glob("*.so"))
    so_files.extend(target_dir.glob("*.pyd"))
    so_files = sorted(so_files)

    if not so_files:
        print("No compiled modules found.")
        print(f"Run 'manforge build' to compile a Fortran module.")
        return

    name_width = max(len(_extract_module_name(f)) for f in so_files)
    name_width = max(name_width, 6)

    header = f"{'Module':<{name_width}}  {'Size':>8}  {'Modified':<20}  File"
    print(header)
    print("-" * len(header))
    for f in so_files:
        stat = f.stat()
        mtime = datetime.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%d %H:%M")
        size = _human_size(stat.st_size)
        module = _extract_module_name(f)
        print(f"{module:<{name_width}}  {size:>8}  {mtime:<20}  {f.name}")


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="manforge",
        description="manforge — Fortran UMAT build toolkit for Python users",
    )
    subparsers = parser.add_subparsers(dest="command", metavar="<command>")
    subparsers.required = True

    # --- build ---
    build_parser = subparsers.add_parser(
        "build",
        help="Compile Fortran source files into a Python extension module",
        description=(
            "Compile one or more Fortran source files into a Python extension module "
            "using f2py. The compiled .so file is placed in the fortran/ directory by default."
        ),
    )
    build_parser.add_argument(
        "files",
        nargs="+",
        metavar="<file.f90>",
        help="Fortran source file(s) to compile (.f90 or .f)",
    )
    build_parser.add_argument(
        "--name", "-n",
        required=True,
        metavar="<module_name>",
        help="Python module name for the compiled extension",
    )
    build_parser.add_argument(
        "--output-dir", "-o",
        default=None,
        metavar="<dir>",
        help="Output directory for the compiled .so file (default: fortran/)",
    )
    build_parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Show full f2py / compiler output",
    )
    build_parser.set_defaults(func=cmd_build)

    # --- clean ---
    clean_parser = subparsers.add_parser(
        "clean",
        help="Remove compiled Fortran artifacts",
        description="Remove f2py build artifacts (*.so, *.mod, *.o, etc.) from the fortran/ directory.",
    )
    clean_parser.add_argument(
        "--dir", "-d",
        default=None,
        metavar="<dir>",
        help="Directory to clean (default: fortran/)",
    )
    clean_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be removed without actually deleting",
    )
    clean_parser.set_defaults(func=cmd_clean)

    # --- list ---
    list_parser = subparsers.add_parser(
        "list",
        help="List compiled Fortran extension modules",
        description="Show compiled Python extension modules (.so files) in the fortran/ directory.",
    )
    list_parser.add_argument(
        "--dir", "-d",
        default=None,
        metavar="<dir>",
        help="Directory to scan (default: fortran/)",
    )
    list_parser.set_defaults(func=cmd_list)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
