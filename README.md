# High-Leverage Computing Library

## What it does
It mimics the api style of famous Eigen library , and computes using multiple backend , such as Kompute and AdaptiveCPP

## Is it necessary
Yes it does , the community lack of this type of library using gpu backend and also compatible with cpu backend .
Even those who could do all above , they usually require custom toolchain , unlike sycl or Kompute , they use regular toolchain .

## Why not just use AdaptiveCPP
Because AdaptiveCPP uses clspv toolchain , which is not easy to use and embed.
Also I was in charge of AdaptiveCPP pack in xmake and I temporarily have no time to finish clspv toolchain integration .
If you wish to fully use AdaptiveCPP , it's welcome to pr to xmake-repo and here!

## How can we sure about that your project is not a toy
For one who is very serious about correctness , I build the whole test pipeline , and support coverage computing .

## Requirements
* C++20 (concepts / span ; `-UNDEBUG`-independent test asserts)
* Optional backends: AdaptiveCpp (SYCL 2020, `--gpu=y`) and Kompute v0.8.0 (Vulkan compute, `--kompute=y`)
* GPU kernel backend (`--gpu=y`): `--gpu_backend=omp|opencl|cuda` (default `omp` = OpenMP host, runs without a GPU).
  CUDA needs an NVIDIA driver + CUDA toolkit on the build machine, plus `--cuda_arch=sm_XX` (GTX 16xx `sm_75`, RTX 30xx `sm_86`, RTX 40xx `sm_89`).
  Example: `xmake f --gpu=y --gpu_backend=cuda --cuda_arch=sm_86 && xmake -r run_hlcl_gpu`

## Architecture (since v0.0.2)
Vector / Matrix / Quaternion are single primary templates parameterized on
`Backend { CPU, GPU, Kompute }`. All backend differences live in
`BackendTraits<B>` (storage + kernel dispatch, `include/hlcl/backend_traits.hpp`,
`gpu_impl.hpp`, `kompute_impl.hpp`) — one implementation, three backends, no
per-backend class copies. Elementwise ops dispatch to SYCL / Vulkan kernels
with CPU-bit-identical numerics (true division, no reciprocal multiply);
reductions stay on the host (all storages are host-readable).
