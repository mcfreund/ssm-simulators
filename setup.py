from setuptools import setup, Extension, find_packages
import os
import subprocess
import sys
import numpy

# ---------------------------------------------------------------------------
# Optional OpenMP + GSL support for multithreaded simulators (cssm.pyx).
#
# Both are OPTIONAL: the extension builds and works single-threaded without
# them (gsl_rng.h provides no-op stubs under !HAVE_GSL, and the parallel code
# paths are only entered when n_threads > 1 *and* GSL is available). We keep
# the flag set minimal on purpose — just -fopenmp + GSL, NOT -O3/-ffast-math/
# -flto — so the sequential (n_threads=1) code path stays byte-identical to the
# unoptimized baseline build (parity is a design goal; see plan).
#
# Force on/off via SSMS_FORCE_OPENMP / SSMS_FORCE_GSL (0/1).
# ---------------------------------------------------------------------------


def gsl_flags():
    """Return dict of GSL compile/link flags via gsl-config, or None if absent."""
    force = os.environ.get("SSMS_FORCE_GSL", "").lower()
    if force in ("0", "false"):
        print("GSL disabled via SSMS_FORCE_GSL=0")
        return None
    try:
        cflags = subprocess.check_output(
            ["gsl-config", "--cflags"], text=True, stderr=subprocess.DEVNULL
        ).split()
        libs = subprocess.check_output(
            ["gsl-config", "--libs"], text=True, stderr=subprocess.DEVNULL
        ).split()
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("GSL not available (gsl-config not found) — building without GSL "
              "(parallel RNG unavailable; on Oscar: module load gsl/2.8-cpuv)")
        return None
    include_dirs = [f[2:] for f in cflags if f.startswith("-I")]
    compile_args = [f for f in cflags if not f.startswith("-I")]
    # Bake the GSL lib dir(s) into the .so via RPATH so libgsl is found at
    # runtime WITHOUT needing `module load gsl` in every context that imports
    # cssm (widget, empirical scripts, SLURM jobs). The spack module path is
    # stable on Oscar. Without this, importing cssm raises a libgsl.so ImportError.
    rpath_args = [f"-Wl,-rpath,{f[2:]}" for f in libs if f.startswith("-L")]
    print(f"GSL available: {' '.join(libs)}")
    return {
        "include_dirs": include_dirs,
        "extra_compile_args": compile_args,
        "extra_link_args": list(libs) + rpath_args,
        "define_macros": [("HAVE_GSL", "1")],
    }


def openmp_flags():
    """Return dict of OpenMP compile/link flags, or None if unavailable."""
    force = os.environ.get("SSMS_FORCE_OPENMP", "").lower()
    if force in ("0", "false"):
        print("OpenMP disabled via SSMS_FORCE_OPENMP=0")
        return None
    if sys.platform == "darwin":
        try:
            prefix = subprocess.check_output(
                ["brew", "--prefix", "libomp"], text=True, stderr=subprocess.DEVNULL
            ).strip()
        except (subprocess.CalledProcessError, FileNotFoundError):
            prefix = "/opt/homebrew/opt/libomp"
        return {
            "include_dirs": [f"{prefix}/include"],
            "extra_compile_args": ["-Xpreprocessor", "-fopenmp", f"-I{prefix}/include"],
            "extra_link_args": [f"-L{prefix}/lib", "-lomp"],
            "define_macros": [("HAVE_OPENMP", "1")],
        }
    if sys.platform == "win32":
        return {
            "include_dirs": [],
            "extra_compile_args": ["/openmp"],
            "extra_link_args": [],
            "define_macros": [("HAVE_OPENMP", "1")],
        }
    # Linux / GCC
    return {
        "include_dirs": [],
        "extra_compile_args": ["-fopenmp"],
        "extra_link_args": ["-fopenmp"],
        "define_macros": [("HAVE_OPENMP", "1")],
    }


def cssm_extension():
    include_dirs = [numpy.get_include(), "src"]
    compile_args, link_args, define_macros = [], [], []
    for flags in (openmp_flags(), gsl_flags()):
        if flags is None:
            continue
        include_dirs.extend(flags["include_dirs"])
        compile_args.extend(flags["extra_compile_args"])
        link_args.extend(flags["extra_link_args"])
        define_macros.extend(flags["define_macros"])
    return Extension(
        "cssm",
        ["src/cssm.pyx"],
        language="c++",
        include_dirs=include_dirs,
        extra_compile_args=compile_args,
        extra_link_args=link_args,
        define_macros=define_macros,
    )


# Try to build with Cython if available
try:
    from Cython.Build import cythonize

    ext_modules = cythonize(
        [cssm_extension()],
        compiler_directives={"language_level": "3"},
    )
except ImportError:
    ext_modules = [cssm_extension()]

# Use find_packages to automatically discover all packages
packages = find_packages(include=["ssms", "ssms.*"])

setup(
    name="ssm-simulators",
    version="0.10.2",
    packages=packages,
    package_data={
        "ssms": ["**/*.py", "**/*.pyx", "**/*.pxd", "**/*.so", "**/*.pyd"],
    },
    include_package_data=True,
    include_dirs=[numpy.get_include()],
    ext_modules=ext_modules,
    install_requires=[
        "numpy",
        "pandas",
        "scipy",
        "matplotlib",
        "tqdm",
        "pyyaml",
        "typer",
    ],
)
