# Quantim CUDA state-vector plan

This is a design document; it contains no CUDA code, build, or GPU measurement. `PROVEN` applies only to explicit arithmetic or structure visible in the source. `TESTED` refers to historical test records in the repository. Machine-specific figures were not measured in this session. Speed and tolerance estimates below are marked `ASSUMPTION` or `TO MEASURE`.

## 1. Goal and scope

The first CUDA backend runs the pure state-vector simulation in `cpp/quantum` on an RTX 4060 Laptop GPU under WSL2 Ubuntu 24.04. The OpenQASM parser's bit order stays the same: qubit `q` is bit `q` of the basis-state index. Stabilizer, `CertifiedState`, Python's `fracvm`, cryptography research, AI/LLM workloads, and multiple GPUs are out of scope. GPU correctness is measured against the CPU `Scalar` path of `StateVector`; no certified interval is claimed. There is no CPU-relative speed claim before measurement.

Environment input: 8 GB VRAM, about 7 GB available, Windows driver 616.92, and only the CUDA toolkit inside WSL2. The display uses the same GPU. [NVIDIA's WSL guide](https://docs.nvidia.com/cuda/wsl-user-guide/index.html) says not to install a separate Linux display driver inside WSL. Memory clock, sustained bandwidth, and thermal behavior are `TO MEASURE`.

## 2. Memory budget

`PROVEN`: A single vector occupies `2^n × sizeof(complex<T>)`. For this calculation, `complex<double>` = 16 bytes and `complex<float>` = 8 bytes; the GPU representation is fixed to that layout. Units are binary: 1 GiB = `2^30` bytes. This budget treats 7 GiB as `7 × 2^30` bytes; the user's “about 7 GB” input must be checked with `cudaMemGetInfo` before allocation.

| Qubits `n` | `complex<double>` | `complex<float>` | One vector within 7 GiB |
| ---: | ---: | ---: | --- |
| 20 | 16 MiB | 8 MiB | Both fit |
| 22 | 64 MiB | 32 MiB | Both fit |
| 24 | 256 MiB | 128 MiB | Both fit |
| 26 | 1 GiB | 512 MiB | Both fit |
| 27 | 2 GiB | 1 GiB | Both fit |
| 28 | 4 GiB | 2 GiB | Both fit |
| 29 | 8 GiB | 4 GiB | Only `float` fits |
| 30 | 16 GiB | 8 GiB | Neither fits |

The theoretical single-vector limit is therefore 28 qubits for `double` and 29 for `float`. The first version uses `double` and at most 28 qubits, matching `StateVector::kMaxQubits = 28`. `float` needs a separate correctness study; 29-qubit support is not an acceptance criterion here. At 28 qubits, the `double` vector occupies 4 GiB; a second full vector would take the total to 8 GiB and exceed the budget. The first design therefore updates in place. A full-size probability array would add `2^28 × 8 = 2 GiB` at 28 qubits, so sampling uses block sums. Remaining room for phase tables, measurement workspace, the CUDA context, and the display is checked dynamically. `ASSUMPTION`: initial safety margin of 1 GiB; M0 measures whether it is sufficient. It does not replace the fixed vector limit.

## 3. Roofline estimate and fusion decision

[NVIDIA's laptop comparison table](https://www.nvidia.com/en-us/geforce/laptops/compare/) lists a 128-bit memory bus, 3072 CUDA cores, and a 1470–2370 MHz boost range for the 4060 Laptop. It does not list the memory data rate. `ASSUMPTION`: 16 Gb/s/pin gives theoretical bandwidth of `128 × 16 / 8 = 256 GB/s` (decimal). Device memory speed and sustained bandwidth still need measurement. An in-place pass that reads and writes every amplitude has minimum traffic of `32 × 2^n` bytes for `double` and `16 × 2^n` for `float`. The times below are **optimistic lower bounds** from `traffic / 256e9`, not runtime predictions or benchmark results.

| Qubits | `double` read+write | `double` lower bound | `float` lower bound |
| ---: | ---: | ---: | ---: |
| 20 | 32 MiB | 0,131 ms | 0,066 ms |
| 24 | 512 MiB | 2,097 ms | 1,049 ms |
| 26 | 2 GiB | 8,389 ms | 4,194 ms |
| 28 | 8 GiB | 33,554 ms | 16,777 ms |
| 29 | `double` does not fit | — | 33,554 ms |

The machine input gives an FP64 rate of 1/64 of FP32. `ASSUMPTION`: if the maximum 2,37 GHz boost were sustained, the FP64 ceiling would be `3072 × 2 × 2,37e9 / 64 = 227,52 GFLOP/s`, with a bandwidth/compute balance of `227,52 / 256 = 0,889 FLOP/byte`. This is neither the actual operating clock nor measured FP64 throughput. A simple dense `2^k × 2^k` complex-matrix model counts roughly `8 × 2^k` FLOP per output amplitude: 16 FLOP/32 bytes = 0,5 FLOP/byte for `k=1`, 1 for `k=2`, and 2 for `k=3`. In this model, dense single-qubit and diagonal gates may be bandwidth-bound, dense 4x4 is near the boundary, and dense 8x8 is compute-bound. It does not follow that all GPU gates are bandwidth-bound. At 28 qubits, the same optimistic FP64 ceiling gives compute lower bounds of 18,88/37,75/75,51 ms for dense 2x2/4x4/8x8 respectively; compare each with its memory bound and take the larger. Controls, irregular access, cache behavior, and register spills increase runtime.

The memory lower bound for `g` separate full passes is `g × Tmem`. Fusion into one pass retains only `Tmem` of that memory cost but does more FP64 work. For example, at 28 `double` qubits, two separate full passes have a 67,108 ms memory floor, while one dense 4x4 pass has a roofline floor of at least 37,75 ms. Dense three-qubit fusion is not enabled automatically: the 8x8 compute lower bound is 75,51 ms, and CPU phase 13 found it did not help every circuit family (`docs/perf-plan.md`). No fixed number of gates is assumed as the GPU fusion threshold. Device times for 2x2, 4x4, 8x8, and successive simple gates in the same block will determine the pass/compute tradeoff. Launch and copy costs also matter for small circuits.

## 4. Kernel design and existing planner

The first `double` representation uses adjacent `re, im` pairs and the same index order as `std::complex<double>`. Direct transfer of the C++ object representation into CUDA memory is not assumed; an explicit `double2` / two-`double` layout and conversion must be verified. Each operation owns disjoint amplitude pairs or groups of four; two blocks never write the same address. Initial geometry is an `ASSUMPTION`: 128–256 threads/block and a grid-stride loop, with each thread handling one or a few independent pairs/groups. Occupancy, register count, and device measurements determine the final block size.

The single-qubit 2x2 kernel reads both inputs of each pair with a zero `target` bit and writes both outputs in place. For low bits such as qubit 0/1, bit insertion arranges neighboring thread addresses for coalesced reads and writes. For high bits, threads process consecutive addresses within each half-block; the long stride stays between the two addresses of each pair. The coalesced-access principle in the [CUDA memory guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html#coalesced-access-to-global-memory) motivates separate profiling for low- and high-bit layouts. Permutations such as `X`, `CNOT`, and `SWAP`, and pure phases, keep specialized kernels; forcing them through a dense matrix adds unnecessary FP64 work.

The two-qubit 4x4 kernel reads four amplitudes once and writes four outputs from the thread/block that owns the group. Diagonal gates such as `CZ`, `CP`, and `RZ` scale only relevant indices. `PhaseTable` could become a small device table, but the CPU `gather_` byte tables are not copied as-is. Controlled gates apply the control mask to GPU indices: unmatched groups stay unchanged, matched groups are updated. Empty warp/read cost and branching with sparse control bits need separate measurement. Compressing active indices is considered only afterward.

The general three-qubit 8x8 pass remains experimental, bounded by `k <= 3`. Eight complex inputs, eight outputs, and 64 coefficients per thread can pressure registers. Coefficients may live in constant memory/cache or be loaded once per block; a `ptxas` register and spill report is an exit criterion. For low bits, coalesced global reads can stage a tile of amplitudes in shared memory for several gates; inter-block dependencies require separate launches. High bits use split two/four/eight-element accesses. Full-vector bit permutation is considered only if it saves multiple later passes. GPU shared-memory size, bank conflicts, and feasible tile size are `TO MEASURE`.

`qasm::Circuit`, `Instruction`, CPU matrix construction, `Pass` kinds (`Single`, `Pair`, `Exchange`, `Diagonal`, `Table`), and the ordering/commutation rules of `FusionBuffer` offer reusable semantics. The current `FusionBuffer` and `PassQueue` hold `StateVector&`; `Target`, `Subspace`, `parallelFor`, the AVX2/FMA path in `simd.cpp`, and CPU `PhaseTable::gather_` execution cannot be moved to CUDA. Extract a CPU-side pass description for both runners while preserving the CPU planner's gate order in the first delivery. The logical-to-physical mapping in `planRemaps` and `restoreLayout` is useful, but the CPU 14-qubit chunk size, `kMinRemapPasses = 2` threshold, and CPU bit-swap kernel are not GPU tuning parameters. GPU remapping stays off until initial measurement. Restore layout when exporting the final state vector; terminal sampling may use the physical bit map without a full restore. `NormCheck::EveryGate` needs a separate reduction after every GPU gate. Initial speed measurements use the current CLI default, `AtMeasure`; strict norm mode is checked separately.

## 5. GPU measurement and sampling

For terminal measurement, the vector stays on GPU. Each block sums `|a_i|²` in a fixed tree order, followed by a prefix scan of block totals; samples generated from the seed are located in that cumulative mass. Neither the full vector nor a full-size probability array is moved to host memory for `shots`. Output contains only measured basis-state indices/histogram. Ordered samples and within-block search for multiple shots are compared with `sampleTerminalMeasurements`; prefix-scan memory and launch costs are measured separately.

Keep `std::mt19937_64` and the current `std::generate_canonical<double, 53>` call order on the host, sending thresholds to GPU. A fixed block size and reduction tree, no atomic probability sums, and a fixed boundary-comparison rule aim for reproducibility with the same binary/device/seed. The CPU histogram is **not guaranteed** to match: different summation or amplitude rounding can change a shot near a threshold. For intermediate measurement, first reduce the chosen qubit's probability, draw once from the host RNG, zero half the vector on GPU, normalize the remainder, and continue the gates. Test this separately from the terminal sampling shortcut.

## 6. Numerical contract

The reference is CPU `QUANTIM_SIMD=scalar`; CPU `Fma` may be faster but is not the reference. The first CUDA correctness mode uses `double`, round-to-nearest, `--fmad=false`, `-ftz=false`, `-prec-div=true`, `-prec-sqrt=true`, and no `--use_fast_math`. These keep the differences described in [NVIDIA's floating-point guide](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/mathematical-functions.html) visible. A performance mode with `--fmad=true` is tested separately. Summation order inside kernels will be documented; both modes must meet the same tolerance contract. CPU-built angle/matrix coefficients are copied to the device with the same `double` bits.

`HYPOTHESIS / initial acceptance threshold`: For a `double` circuit of at most 200 logical gates with no intermediate measurement, each complex amplitude must satisfy `|a_gpu - a_scalar| <= 5e-13 × max(1, D) + 1e-10 × |a_scalar|`, where `D` is the number of applied logical gates. Also require `|Σ|a_gpu|² - 1| <= 1e-9` (current `StateVector::kNormTolerance`) and absolute difference `<= 2e-9` for a single measurement probability. These are not measured error bounds or mathematical certificates. Review them against reference circuits in M2–M4 and explicitly adjust for depth and precision if needed. `float` is not accepted under these thresholds. Fusion changes matrix multiplication order, FMA changes rounding, GPU reduction changes the sum tree, and trigonometric libraries can change coefficients. Even `--fmad=false` does not promise bit identity between CPU and GPU. Reproducibility for the same GPU binary, launch layout, and seed is a separate requirement; NaN and overflow inputs are handled separately.

## 7. Build and CLI integration

Implementation diverged from the plan: the GPU path was not added as a `--device` flag to Quantim, but remains a separate binary in this repository. Quantim stays pinned as a submodule, without a CUDA dependency in the CPU library. This CMake project always requires CUDA (`find_package(CUDAToolkit REQUIRED)`, architecture 89); there is no `QUANTIM_CUDA` option.

```
quantim-gpu run <dosya.qasm | -> [--shots N] [--seed S] [--json] [--no-fuse] [--device cuda]
```

QASM uses Quantim's parser (`qasm::parseFile`/`parseString`). Histogram and JSON formats match `quantim run`; device name and free VRAM go to standard error, with a `"device"` JSON field. `--device` accepts only `cuda`; other values produce an error directing CPU users to `quantim`. There is no silent CPU fallback: missing device, insufficient VRAM (`cudaMemGetInfo` plus margin before allocation), or CUDA errors return a Turkish error and a nonzero exit code. Qubit count is 1–28; above 24 requires `QUANTIM_GPU_LARGE=1`.

The GPU runner follows the unfused CPU path: each gate goes to the kernel the CPU would select, with a matrix built in the same way. There is no fusion or remapping; `--no-fuse` is accepted but has no effect (see `docs/yurutucu.md`). The histogram matches `QUANTIM_SIMD=scalar quantim run --no-fuse`; kernel tests found bit-for-bit amplitude identity with CPU Scalar (`TESTED`, M2–M4).

`--certified`, `--noise`, `trace`, and `info` are unavailable in the GPU CLI and fail as unknown options. The error bound and 24-qubit limit of `CertifiedState` do not carry over to the GPU `double` path. Noise trajectories rerun the vector for each shot and the RNG/Kraus step at every noise point, so they need a separate design and cost measurement; they were not part of M5. The C ABI (`quantim.h`) and Python bindings did not change. `quantim run --device cuda` integration can be reconsidered after Quantim exposes an interface separating planning from execution (proposal in `docs/yurutucu.md`).

## 8. Test plan

After `M0`, if a GPU is available, compare kernel unit tests with CPU `Scalar` from the same initial vector. Cover at least 1, 2, 3, 5, 10, 14, 20 and, resources permitting, 24/28 qubits; low, middle, and highest target bits; and distant, nonadjacent pairs. Cover all single/two/three-qubit families in `gates.hpp`, every QASM parser gate, phase tables, controlled/multicontrolled X and phase, permutations, final `restoreLayout`, `--no-fuse`, and successive gates on the same physical bit. Keep 28-qubit circuits short and behind a separate resource gate. Compare full amplitudes for small cases; selected indices, norm, and distribution for large ones. Selected indices alone do not prove correctness of the full vector; require full comparison at 20–24 qubits when memory permits.

Generate fixed-seed random circuits with varying depth, measurement position, and control density. Check amplitude/probability tolerance against CPU, repeatability within GPU, and classical bit order. Also test terminal shots, intermediate measurement, `barrier`, `swap`, `id`, zero shots, invalid device, and insufficient VRAM. The `CTest` GPU smoke test reports one explicit `SKIP` if the CUDA binary exists but no suitable device is present; if the GPU binary is disabled, the test is not registered. A CUDA error with a GPU present is a failure, not a `SKIP`. Keep CPU and Python test gates. No tests were run while this document was prepared.

## 9. Benchmark plan

Try the `bench/circuits.py` families `layered`, `qft`, `ghz`, and `random` at 20, 22, 24, 26, and 28 qubits. Mark rows blocked by VRAM or temperature as `not run`. In the same WSL environment, compare the CPU `Scalar` reference and production CPU path with 1 and 4 threads against GPU `double` correctness and performance modes. Old VirtualBox times are not a direct speed-ratio baseline. Include `Qulacs` / `Aer` only **if** GPU-enabled builds actually exist on this machine; their GPU versions have not been verified here. CPU Aer/Qiskit being present in the guest's `~/qenv` does not count as a GPU comparison.

Record circuit-generation seed, `shots`, fusion, norm mode, device, CUDA/toolkit version, and driver version for each measurement. Measure GPU kernel time with CUDA events, and allocation + H2D + planning + execution + sampling + D2H total with a host clock. Effective bandwidth (`bytes read+written / kernel time`), device memory clock, and temperature are `TO MEASURE`. Run each A/B pair in order for at least five rounds; use the alternating A/B median from `HAFIZA.md` and show min–max. Keep warmup and cooldown consistent, then check drift with a second sequence in reverse order. Record temperature, clock, power, and VRAM each round. No ratio comes from a single run.

## 10. Operational safety

[Microsoft's WDDM TDR documentation](https://learn.microsoft.com/en-us/windows-hardware/drivers/display/timeout-detection-and-recovery) gives a default timeout of two seconds. Avoid a single long launch on the display GPU used through WSL: divide grid-stride work and limit gate/fusion groups. `ASSUMPTION / initial operational target`: each launch below 100 ms; M0 may tighten this. Do not change TDR registry settings. Check large `cudaMalloc` allocations and `cudaMemGetInfo` before every run; show allocated space and extra temporary memory, and stop before execution if VRAM is insufficient.

Scale through 20 → 22 → 24 → 26 → 28 qubits; do not advance until correctness, launch duration, free VRAM, and temperature have been checked at each step. Stop measurements **above** 85 °C and do not begin another round near the limit. A separate stop script terminates only the benchmark process, records the CUDA error, and releases resources; it does not shut down the Windows driver or other GPU jobs. If WSL `nvidia-smi` telemetry is incomplete, do not run long 26–28-qubit jobs without monitoring via Windows sensors. Do not retry automatically after display unresponsiveness, `cudaErrorLaunchTimeout`, memory overflow, or overheating.

## 11. Session milestones

| Stage | Session output and exit criterion | Difficulty |
| --- | --- | --- |
| M0 | WSL toolkit/device inventory; `cudaMemGetInfo`, memory speed, temperature telemetry, copy bandwidth; estimated safety gate for 20–28 qubits. Figures come with raw records. | Routine, sub-agent |
| M1 | CMake option and CLI device selection; smoke tests for no GPU/disabled build; CPU default and C ABI unchanged. | Routine, sub-agent |
| M2 | `double` single-qubit, diagonal, controlled/permutation kernels; full comparison with CPU `Scalar` at 1–24 qubits and fixed-seed repeat test. | Hard, Opus |
| M3 | 4x4 and terminal sampling; separate validation for GPU block reduction, shots, and intermediate measurement; histogram without a full-vector host copy. | Hard, Opus |
| M4 | Connect CPU planner's pass description to GPU runner; fused/plain results, virtual `swap`, `restoreLayout`, and four QASM circuit families pass. Physical remapping and dense `k=3` await separate measurement decisions. | Hard, Opus |
| M5 | Alternating A/B timings, roofline comparison, 20–28-qubit limits, thermal log, and fusion on/off decision; leave unmeasured rows empty. | Routine measurement, sub-agent; Opus interprets results |

Each stage closes with its tests and a short findings record; success at one stage does not establish the next stage's speed claim. `HAFIZA.md` requires separate review of critical kernel and planner changes. This document does not delegate the work to an agent.

## 12. Risks and open questions

- **VRAM:** 7 GiB is approximate. The display, WSL context, and other processes may prevent a 28-qubit `double` allocation. M0 data establishes actual free space and the safety margin.
- **FP64:** The 1/64 ratio may bring dense 4x4/8x8 and long fusion to the compute ceiling early. M5 measurements determine which gate families run faster on GPU than CPU.
- **Access pattern:** High-bit strides, low-bit shuffles, and control masks affect effective bandwidth differently. Coalescing and shared-memory decisions need profiler data.
- **Numerical divergence:** Bit identity between CPU `Scalar` and GPU is not assured. Tests must isolate coefficient generation, FMA, fusion, and reduction effects. The stated tolerances are provisional acceptance thresholds.
- **Planner:** The CPU's 14-qubit chunk and remapping threshold may give a wrong GPU cost model. Keep pass semantics while choosing device-specific costs.
- **Sampling:** The terminal measurement shortcut may help with many shots; intermediate measurement and future noise cannot use it. Per-shot GPU trajectory allocation/startup cost is unknown.
- **WSL/display GPU:** TDR, heat, and power limits can distort results or interrupt a session. Interpret all performance results alongside temperature and clock logs.

References: source files `cpp/quantum/src/{kernels.hpp,simd.cpp,quantum.cpp,qasm.cpp}` and `cpp/quantum/include/quantim/*.hpp`; historical performance records in `docs/perf-plan.md` and `HAFIZA.md`. Web links are primary sources for hardware/tool behavior, not substitutes for machine-specific speed measurements.
