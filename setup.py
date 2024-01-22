import glob
import os
import subprocess
import sys
import warnings

import torch
from packaging.version import Version, parse
from setuptools import find_packages, setup
from torch.utils.cpp_extension import (
    CUDA_HOME,
    BuildExtension,
    CppExtension,
    CUDAExtension,
    load,
)

PYTORCH_HOME = (
    os.path.abspath(os.environ["PYTORCH_HOME"])
    if "PYTORCH_HOME" in os.environ
    else None
)

# ninja build does not work unless include_dirs are abs path
this_dir = os.path.dirname(os.path.abspath(__file__))


def get_cuda_bare_metal_version(cuda_dir):
    raw_output = subprocess.check_output(
        [cuda_dir + "/bin/nvcc", "-V"], universal_newlines=True
    )
    output = raw_output.split()
    release_idx = output.index("release") + 1
    bare_metal_version = parse(output[release_idx].split(",")[0])

    return raw_output, bare_metal_version


def check_cuda_torch_binary_vs_bare_metal(cuda_dir):
    raw_output, bare_metal_version = get_cuda_bare_metal_version(cuda_dir)
    torch_binary_version = parse(torch.version.cuda)

    print("\nCompiling cuda extensions with")
    print(raw_output + "from " + cuda_dir + "/bin\n")

    if bare_metal_version != torch_binary_version:
        raise RuntimeError(
            "Cuda extensions are being compiled with a version of Cuda that does "
            "not match the version used to compile Pytorch binaries.  "
            "Pytorch binaries were compiled with Cuda {}.\n".format(torch.version.cuda)
            + "In some cases, a minor-version mismatch will not cause later errors:  "
            "https://github.com/NVIDIA/apex/pull/323#discussion_r287021798.  "
            "You can try commenting out this check (at your own risk)."
        )


def raise_if_cuda_home_none(global_option: str) -> None:
    if CUDA_HOME is not None:
        return
    raise RuntimeError(
        f"{global_option} was requested, but nvcc was not found.  Are you sure your environment has nvcc available?  "
        "If you're installing within a container from https://hub.docker.com/r/pytorch/pytorch, "
        "only images whose names contain 'devel' will provide nvcc."
    )


def check_cudnn_version_and_warn(
    global_option: str, required_cudnn_version: int
) -> bool:
    cudnn_available = torch.backends.cudnn.is_available()
    cudnn_version = torch.backends.cudnn.version() if cudnn_available else None
    if not (cudnn_available and (cudnn_version >= required_cudnn_version)):
        warnings.warn(
            f"Skip `{global_option}` as it requires cuDNN {required_cudnn_version} or later, "
            f"but {'cuDNN is not available' if not cudnn_available else cudnn_version}"
        )
        return False
    return True


if not torch.cuda.is_available():
    # https://github.com/NVIDIA/apex/issues/486
    # Extension builds after https://github.com/pytorch/pytorch/pull/23408 attempt to query torch.cuda.get_device_capability(),
    # which will fail if you are compiling in an environment without visible GPUs (e.g. during an nvidia-docker build command).
    print(
        "\nWarning: Torch did not find available GPUs on this system.\n",
        "If your intention is to cross-compile, this is not an error.\n"
        "By default, Apex will cross-compile for Pascal (compute capabilities 6.0, 6.1, 6.2),\n"
        "Volta (compute capability 7.0), Turing (compute capability 7.5),\n"
        "and, if the CUDA version is >= 11.0, Ampere (compute capability 8.0).\n"
        "If you wish to cross-compile for a single specific architecture,\n"
        'export TORCH_CUDA_ARCH_LIST="compute capability" before running setup.py.\n',
    )
    if os.environ.get("TORCH_CUDA_ARCH_LIST", None) is None and CUDA_HOME is not None:
        _, bare_metal_version = get_cuda_bare_metal_version(CUDA_HOME)
        if bare_metal_version >= Version("11.8"):
            os.environ["TORCH_CUDA_ARCH_LIST"] = "6.0;6.1;6.2;7.0;7.5;8.0;8.6;9.0"
        elif bare_metal_version >= Version("11.1"):
            os.environ["TORCH_CUDA_ARCH_LIST"] = "6.0;6.1;6.2;7.0;7.5;8.0;8.6"
        elif bare_metal_version == Version("11.0"):
            os.environ["TORCH_CUDA_ARCH_LIST"] = "6.0;6.1;6.2;7.0;7.5;8.0"
        else:
            os.environ["TORCH_CUDA_ARCH_LIST"] = "6.0;6.1;6.2;7.0;7.5"

print("\n\ntorch.__version__  = {}\n\n".format(torch.__version__))
TORCH_MAJOR = int(torch.__version__.split(".")[0])
TORCH_MINOR = int(torch.__version__.split(".")[1])

if TORCH_MAJOR == 0 and TORCH_MINOR < 4:
    raise RuntimeError(
        "Apex requires Pytorch 0.4 or newer.\nThe latest stable release can be obtained from https://pytorch.org/"
    )

cmdclass = {}
ext_modules = []

extras = {}

if "--cpp_ext" in sys.argv or "--cuda_ext" in sys.argv:
    if TORCH_MAJOR == 0:
        raise RuntimeError(
            "--cpp_ext requires Pytorch 1.0 or later, "
            "found torch.__version__ = {}".format(torch.__version__)
        )

if "--cpp_ext" in sys.argv:
    sys.argv.remove("--cpp_ext")
    ext_modules.append(CppExtension("apex_C", ["csrc/flatten_unflatten.cpp"]))


# Set up macros for forward/backward compatibility hack around
# https://github.com/pytorch/pytorch/commit/4404762d7dd955383acee92e6f06b48144a0742e
# and
# https://github.com/NVIDIA/apex/issues/456
# https://github.com/pytorch/pytorch/commit/eb7b39e02f7d75c26d8a795ea8c7fd911334da7e#diff-4632522f237f1e4e728cb824300403ac
version_ge_1_1 = []
if (TORCH_MAJOR > 1) or (TORCH_MAJOR == 1 and TORCH_MINOR > 0):
    version_ge_1_1 = ["-DVERSION_GE_1_1"]
version_ge_1_3 = []
if (TORCH_MAJOR > 1) or (TORCH_MAJOR == 1 and TORCH_MINOR > 2):
    version_ge_1_3 = ["-DVERSION_GE_1_3"]
version_ge_1_5 = []
if (TORCH_MAJOR > 1) or (TORCH_MAJOR == 1 and TORCH_MINOR > 4):
    version_ge_1_5 = ["-DVERSION_GE_1_5"]
version_dependent_macros = version_ge_1_1 + version_ge_1_3 + version_ge_1_5

_, bare_metal_version = get_cuda_bare_metal_version(CUDA_HOME)

if "--cuda_ext" in sys.argv:
    sys.argv.remove("--cuda_ext")
    raise_if_cuda_home_none("--cuda_ext")
    check_cuda_torch_binary_vs_bare_metal(CUDA_HOME)

    ext_modules.append(
        CUDAExtension(
            name="amp_C",
            sources=[
                "csrc/amp_C_frontend.cpp",
                "csrc/multi_tensor_adam.cu",
            ],
            extra_compile_args={
                "cxx": ["-O3"] + version_dependent_macros,
                "nvcc": [
                    "-lineinfo",
                    "-O3",
                    # '--resource-usage',
                    "--use_fast_math",
                ]
                + version_dependent_macros,
            },
        )
    )

    if bare_metal_version >= Version("11.0"):
        cc_flag = []
        cc_flag.append("-gencode")
        cc_flag.append("arch=compute_70,code=sm_70")
        cc_flag.append("-gencode")
        cc_flag.append("arch=compute_80,code=sm_80")
        if bare_metal_version >= Version("11.1"):
            cc_flag.append("-gencode")
            cc_flag.append("arch=compute_86,code=sm_86")
        if bare_metal_version >= Version("11.8"):
            cc_flag.append("-gencode")
            cc_flag.append("arch=compute_90,code=sm_90")

        ext_modules.append(
            CUDAExtension(
                name="fused_weight_gradient_mlp_cuda",
                include_dirs=[os.path.join(this_dir, "csrc")],
                sources=[
                    "csrc/megatron/fused_weight_gradient_dense.cpp",
                    "csrc/megatron/fused_weight_gradient_dense_cuda.cu",
                    "csrc/megatron/fused_weight_gradient_dense_16bit_prec_cuda.cu",
                ],
                extra_compile_args={
                    "cxx": ["-O3"] + version_dependent_macros,
                    "nvcc": [
                        "-O3",
                        "-U__CUDA_NO_HALF_OPERATORS__",
                        "-U__CUDA_NO_HALF_CONVERSIONS__",
                        "--expt-relaxed-constexpr",
                        "--expt-extended-lambda",
                        "--use_fast_math",
                    ]
                    + version_dependent_macros
                    + cc_flag,
                },
            )
        )

if PYTORCH_HOME is not None and os.path.exists(PYTORCH_HOME):
    nvfuser_is_refactored = "nvfuser" in (
        os.path.join(d)
        for d in os.listdir(os.path.join(PYTORCH_HOME, "third_party"))
        if os.path.isdir(os.path.join(os.path.join(PYTORCH_HOME, "third_party"), d))
    )
    import nvfuser  # NOQA

    print(PYTORCH_HOME)
    include_dirs = [PYTORCH_HOME]
    library_dirs = []
    extra_link_args = []
    if nvfuser_is_refactored:
        include_dirs.append(os.path.join(PYTORCH_HOME, "third_party/nvfuser/csrc"))
        library_dirs = nvfuser.__path__
        extra_link_args.append("-lnvfuser_codegen")
    ext_modules.append(
        CUDAExtension(
            name="instance_norm_nvfuser_cuda",
            sources=[
                "csrc/instance_norm_nvfuser.cpp",
                "csrc/instance_norm_nvfuser_kernel.cu",
            ],
            include_dirs=include_dirs,
            library_dirs=library_dirs,
            extra_link_args=extra_link_args,
            extra_compile_args={
                "cxx": ["-O3"] + version_dependent_macros,
                "nvcc": ["-O3"]
                + version_dependent_macros
                + [f"-DNVFUSER_THIRDPARTY={int(nvfuser_is_refactored)}"],
            },
        )
    )

# Check, if ATen/CUDAGeneratorImpl.h is found, otherwise use ATen/cuda/CUDAGeneratorImpl.h
# See https://github.com/pytorch/pytorch/pull/70650
generator_flag = []
torch_dir = torch.__path__[0]
if os.path.exists(os.path.join(torch_dir, "include", "ATen", "CUDAGeneratorImpl.h")):
    generator_flag = ["-DOLD_GENERATOR_PATH"]

setup(
    name="apex",
    version="0.1",
    packages=find_packages(
        exclude=(
            "build",
            "csrc",
            "include",
            "tests",
            "dist",
            "docs",
            "tests",
            "examples",
            "apex.egg-info",
        )
    ),
    install_requires=["packaging>20.6"],
    description="PyTorch Extensions written by NVIDIA",
    ext_modules=ext_modules,
    cmdclass={"build_ext": BuildExtension} if ext_modules else {},
    extras_require=extras,
)
