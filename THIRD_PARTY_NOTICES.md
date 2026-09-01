# Third-party notices

This fork contains a small adapter and a set of CUDA/C++ source overrides for
the EXL3 kernels shipped by [ExLlamaV3](https://github.com/turboderp-org/exllamav3).
The upstream project is copyright Turboderp and is distributed under the MIT
License. The applicable license text is included at
[`third_party/exllamav3_ext_shared_had/LICENSE-MIT`](third_party/exllamav3_ext_shared_had/LICENSE-MIT).

The files under `third_party/exllamav3_ext_shared_had/` are a modified snapshot
of the native extension sources shipped by `exllamav3==0.0.43`; they are not a
copy of the full Python package. The changes add the raw grouped EXL3 and GUAD
entry points used by the vLLM adapter and keep the SM80 code path compatible
with the target GPU.

The vLLM portions of this repository remain Apache-2.0, as described in
[`LICENSE`](LICENSE). Model checkpoints, Marlin sidecars, DFlash2 weights,
and the installed ExLlamaV3 package are not redistributed by this repository;
review their respective upstream licenses before deployment.
