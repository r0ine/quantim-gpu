# quantim-gpu

CUDA backend for the Quantim state-vector simulator. This is a research project. See `docs/gpu-plan.md` for the design, `docs/yurutucu.md` for the current runner, and `docs/olcum.md` for CPU/GPU timing. The CPU implementation and OpenQASM parser come from the [Quantim](https://github.com/r0ine/quantim) repository under `extern/quantim` (a submodule pinned to a commit). That repository is currently private, so the build only works for people with access to it; until it is published, treat this repository as a read-only reference for the design and the measurements.

Target hardware: RTX 4060 Laptop 8 GB, WSL2 Ubuntu 24.04 on Windows 11, and CUDA toolkit 12.9.

## Usage

The GPU path has its own CLI; `--device` was not added to Quantim's `quantim run` command.

```
quantim-gpu run devre.qasm [--shots N] [--seed S] [--json] [--no-fuse] [--device cuda]
quantim-gpu run - [...]        # circuit from standard input
```

The histogram and JSON formats match `quantim run`. Device information goes to standard error and the `"device"` JSON field. `--device` accepts only `cuda`. The GPU path has no gate fusion or qubit remapping; `--no-fuse` is accepted but has no effect. More than 24 qubits requires `QUANTIM_GPU_LARGE=1`.

## Status

| Stage | Work | Status |
| --- | --- | --- |
| M0 | Device inventory, bandwidth and temperature measurements | Complete |
| M2 | 2x2, diagonal, controlled X, swap/iSWAP kernels | Complete; amplitudes match CPU Scalar bit for bit at 1–24 and 26 qubits |
| M3 | 4x4, intermediate measurement, terminal sampling | Complete; same sample sequence as CPU with the same seed |
| M4 | `quantim-gpu run` and QASM runner | Complete; same histogram as `QUANTIM_SIMD=scalar quantim run --no-fuse` |
| M5 | Alternating CPU/GPU timing | Measured at 20–26 qubits; 28 qubits not measured (`docs/olcum.md`) |

The M1 CMake option and `quantim run --device` integration were not implemented. The GPU path remains a separate binary in this repository.

## Scope

- Pure state vectors only, using `double`, up to 28 qubits (4 GiB).
- CPU `Scalar` is the correctness reference. No speed claim precedes measurement.
- Noise, `--certified`, and `trace` are unavailable on GPU.
- AI model acceleration and multiple GPUs are out of scope.

## Implementation

CUDA C++ (host C++20, `.cu` kernels, `sm_89`), CMake, and CTest. Reference and benchmark scripts use Python. CUDA code is compiled with `--fmad=false -ftz=false -prec-div=true -prec-sqrt=true`.

```
src/        GPU backend (state vector, kernels, sampling, CLI)
tests/      comparison tests against CPU Scalar
bench/      benchmark scripts
tools/      device probe, temperature guard, random QASM generator
docs/       design and findings
extern/     quantim (submodule)
```

## License

Licensed under PolyForm Small Business 1.0.0 (see LICENSE). This is source-available, not OSI open source: free for businesses with fewer than 100 people and under USD 1M annual revenue; larger companies need separate permission from the author.
