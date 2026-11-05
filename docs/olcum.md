# M5: CPU and GPU timing

All figures here come from `bench/ab_runner.py` logs (`/root/build/ab_20_22.jsonl`, `ab_24.jsonl`, `ab_26.jsonl` in WSL) and the `gpu_guard` logs from the same runs (`ab2022.guard.log`, `ab24.guard.log`, `ab26.guard.log`). Medians and ratios were recomputed from raw rounds and match the script's tables (`ab2022.out`, `ab24.out`, `ab26.out`). Nothing absent from the logs is presented as a result; interpretations and assumptions are marked separately.

## What was measured

Timing is the externally measured wall time of the `quantim run` and `quantim-gpu run` processes (`time.perf_counter`, from start to end of `subprocess.run`). It includes process startup, QASM parsing, allocation, circuit execution, and sampling for one shot. On GPU, it also includes CUDA context creation and `cudaMalloc`. Kernel time was not measured separately; these tables do not show kernel speed.

| Condition | Value |
| --- | --- |
| Hardware | RTX 4060 Laptop 8 GB, Windows 11 + WSL2 Ubuntu 24.04 (`.wslconfig`: 7 GB RAM, 6 processors) |
| Software | CUDA toolkit 12.9 (nvcc 12.9.86), Windows driver 616.92 |
| CPU binary | `/root/build/cpu/quantim`, CLI defaults: fusion and remapping enabled, default SIMD path |
| GPU binary | `/root/build/qgpu/quantim-gpu`, no fusion or remapping, `double` |
| CPU threads | `--threads 1` and `--threads 4` |
| Circuits | `extern/quantim/cpp/quantum/bench/circuits.py` `SPEED_FAMILIES`: layered, qft, ghz, random |
| Shots, seed | 1 shot, seed 7 |
| Round order | `cpu1, gpu, gpu, cpu4` each round; reversed on odd-numbered rounds |
| Warmup | 1 round (excluded from medians) |

GPU ran twice per round, so it has twice as many samples as either CPU setting.

| Run | Date (UTC+3) | Measured rounds | Cooldown between rounds | Samples (CPU1 / CPU4 / GPU) |
| --- | --- | ---: | ---: | --- |
| 20 and 22 qubits | 2026-11-02 21:21–21:35 | 5 | 20 s | 5 / 5 / 10 |
| 24 qubits | 2026-11-02 21:36–21:48 | 5 | 30 s | 5 / 5 / 10 |
| 26 qubits | 2026-11-03 17:01–17:12 | 3 | 30 s | 3 / 3 / 6 |

The 26-qubit run used three rounds rather than the planned five (`"rounds": 3`). A median of three samples is less stable than a median of five; treat these ratios as preliminary. GPU runs above 24 qubits require `QUANTIM_GPU_LARGE=1`. The successful 26-qubit run implies it was set, but the setup log does not record it.

## Results

Times are median seconds, with min–max in parentheses. Ratio = CPU median / GPU median; above 1 means GPU took less time.

### 20 qubits

| Family | Gates | CPU 1 | CPU 4 | GPU | CPU1/GPU | CPU4/GPU |
| --- | ---: | --- | --- | --- | ---: | ---: |
| layered | 295 | 0,0693 (0,0610–0,0723) | 0,0237 (0,0219–0,0279) | 0,2327 (0,1852–0,5685) | 0,30 | 0,10 |
| qft | 212 | 0,0511 (0,0457–0,0549) | 0,0248 (0,0197–0,0359) | 0,2360 (0,1867–0,3415) | 0,22 | 0,11 |
| ghz | 60 | 0,0335 (0,0308–0,0399) | 0,0138 (0,0133–0,0164) | 0,2202 (0,1798–0,2650) | 0,15 | 0,06 |
| random | 400 | 0,1333 (0,1271–0,1352) | 0,0412 (0,0381–0,0534) | 0,4727 (0,2335–0,7098) | 0,28 | 0,09 |

### 22 qubits

| Family | Gates | CPU 1 | CPU 4 | GPU | CPU1/GPU | CPU4/GPU |
| --- | ---: | --- | --- | --- | ---: | ---: |
| layered | 325 | 0,3714 (0,3542–0,4037) | 0,2034 (0,1979–0,2151) | 0,4105 (0,3498–0,4380) | 0,90 | 0,50 |
| qft | 255 | 0,1928 (0,1869–0,1991) | 0,0884 (0,0836–0,1090) | 0,3150 (0,2283–0,4344) | 0,61 | 0,28 |
| ghz | 66 | 0,1843 (0,1755–0,2000) | 0,0901 (0,0829–0,1054) | 0,3135 (0,2241–0,8595) | 0,59 | 0,29 |
| random | 440 | 0,5711 (0,5667–0,5910) | 0,2096 (0,2071–0,2186) | 0,5995 (0,4732–1,2951) | 0,95 | 0,35 |

### 24 qubits

| Family | Gates | CPU 1 | CPU 4 | GPU | CPU1/GPU | CPU4/GPU |
| --- | ---: | --- | --- | --- | ---: | ---: |
| layered | 355 | 1,7570 (1,7253–1,7648) | 1,1446 (1,0942–1,1579) | 1,1065 (0,9348–1,5822) | 1,59 | 1,03 |
| qft | 302 | 0,9053 (0,8709–0,9903) | 0,4036 (0,3723–0,4343) | 0,5467 (0,4381–0,6374) | 1,66 | 0,74 |
| ghz | 72 | 0,8889 (0,7893–0,9167) | 0,4131 (0,3626–0,4511) | 0,3714 (0,2978–0,4371) | 2,39 | 1,11 |
| random | 480 | 2,8330 (2,7358–2,9089) | 1,1006 (1,0144–1,6254) | 1,1366 (0,9106–1,2230) | 2,49 | 0,97 |

### 26 qubits (3 rounds)

| Family | Gates | CPU 1 | CPU 4 | GPU | CPU1/GPU | CPU4/GPU |
| --- | ---: | --- | --- | --- | ---: | ---: |
| layered | 385 | 8,6822 (8,6555–9,0580) | 6,0691 (5,3246–6,1931) | 3,7403 (3,5535–3,9584) | 2,32 | 1,62 |
| qft | 353 | 5,0705 (4,8972–5,4100) | 2,2433 (2,1787–2,4311) | 1,5363 (1,4091–1,5965) | 3,30 | 1,46 |
| ghz | 78 | 4,5305 (4,3630–4,5754) | 1,8253 (1,8160–2,0077) | 0,8328 (0,7425–1,0431) | 5,44 | 2,19 |
| random | 520 | 13,3132 (13,1697–14,1550) | 5,2918 (4,8470–5,9128) | 4,0891 (3,9216–4,4001) | 3,26 | 1,29 |

Other agent processes were running during the 26-qubit run (session note; the logs do not quantify this load). CPU times may have been affected, inflating the 26-qubit ratios in GPU's favor. The size of that effect is unknown.

## Thermal and memory logs

`gpu_guard` reads `nvidia-smi` every 2 seconds. VRAM is the total used by all GPU processes; other applications occupied 982–1149 MiB before the runs.

| Run | Peak temperature | Peak VRAM | Peak power | Samples |
| --- | ---: | ---: | ---: | ---: |
| 20 and 22 qubits | 67 °C | 1372 MiB | 29,96 W | 409 |
| 24 qubits | 67 °C | 1891 MiB | 46,26 W | 335 |
| 26 qubits | 76 °C | 2653 MiB | 55,42 W | 299 |

No run triggered the guard's 85 °C stop or reached `ab_runner`'s 80 °C limit; the logs contain no errors. Before/after GPU readings for 26 qubits were 60–75 °C.

## Interpretation

- At 20 and 22 qubits, GPU was slower than both CPU settings for every family. At 20 qubits, GPU medians cluster around 0,22–0,24 s regardless of circuit (except random at 0,47 s); 60-gate ghz and 295-gate layered take nearly the same time. Attributing this fixed cost to CUDA context and driver startup is an **unmeasured assumption**: no breakdown of context, `cudaMalloc`, kernels, or sampling was recorded.
- At 24 qubits, GPU took less time than single-threaded CPU (1,59–2,49×). Against four threads, it was narrowly ahead for layered and ghz, and behind or roughly equal for qft and random (0,74–1,11).
- At 26 qubits, GPU took less time than both CPU settings across all four families: CPU1/GPU 2,32–5,44, CPU4/GPU 1,29–2,19. These rows have three rounds and background load.
- GPU min–max ranges are markedly wider than CPU ranges (for example, 22-qubit random at 0,47–1,30 s). The cause was not isolated.
- CPU uses fusion and remapping; GPU makes one pass per gate. These ratios compare those two paths, not the same algorithm on two devices. The effect of GPU fusion was not measured.

## Relation to the roofline model

Process-level A/B timing says nothing about memory bandwidth. The only bandwidth evidence comes from separate, single-run kernel timings (not A/B):

- M0 (`gpu_probe`): theoretical 256 GB/s (8001 MHz, 128 bit), triad 200 and 237–238 GB/s, copy ≈199 GB/s.
- M2: full 2x2 pass at 24 qubits, 3,1–3,2 ms; `32 × 2^24` bytes of traffic imply 165–172 GB/s effective.
- `launch_time` (after M3): 24-qubit 2x2, 2,997 ms (≈179 GB/s); device norm, 1,583 ms (read only, ≈170 GB/s); 4x4, 6,194 ms.

The 24-qubit 2x2 pass thus reaches about 65–70% of theoretical bandwidth. The 4x4 pass taking twice as long is consistent with the plan's estimate that 4x4 sits near the FP64 limit. This is insufficient to call the entire GPU path bandwidth-bound: kernel time and traffic were not measured for each gate family. A simple “gate count × full-pass lower bound” model also fails: at 26 qubits, qft gives 353 × 8,39 ms ≈ 2,96 s, while the measured process time was 1,54 s. Diagonal and controlled gates do not touch the whole vector; traffic per gate remains TO MEASURE.

## Not measured

- **28 qubits:** Not run. At measurement time, free RAM on Windows was insufficient. CPU would have needed to hold a 4 GiB state vector inside WSL's 7 GB limit, so the run was not started. A prior GPU 28-qubit allocation and short correctness check (`gpu_large_check`, outside A/B, without CPU comparison) passed; no timing data exists.
- Within-process timing breakdown (CUDA context, allocation, kernels, sampling) or CUDA-event kernel time.
- Multiple-shot (`shots` > 1) or intermediate-measurement circuits; every run used one shot and terminal measurement.
- Timing against the CPU `QUANTIM_SIMD=scalar` reference path or CPU `--no-fuse`.
- GPU `--fmad=true` performance mode (not compiled).
- Qulacs and Aer GPU builds are not installed on this machine; no comparison was made.
- GPU fusion or remapping; there is no evidence for an on/off fusion decision.
