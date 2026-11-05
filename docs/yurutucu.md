# GPU runner (M4) and proposed shared pass interface

## Current state

`quantim-gpu run dosya.qasm [--shots N] [--seed S] [--json] [--no-fuse] [--device cuda]`
uses Quantim's QASM parser (`extern/quantim`, `qasm::parseFile`) and runs the circuit on GPU one gate at a time. Histogram and JSON formats match `quantim run`; the device line goes to standard error and the `"device"` field goes into JSON.

`src/runner.cpp` follows the CPU path without fusion: each gate uses the kernel selected by `gates.cpp` (exchange, diagonal, 2x2, swap) with a matrix built the same way. A matrix with zero off-diagonal entries uses the diagonal kernel, including `rx(0)`. Terminal measurements use the sampling shortcut. Intermediate measurements rerun the circuit for each shot. The host advances the RNG in the same order as the CPU. The kernels reproduce CPU Scalar arithmetic term by term, so the histogram is **identical** to `QUANTIM_SIMD=scalar quantim run --no-fuse` (TESTED: 914 checks in `gpu_runner_tests`; qft/ghz/layered/random at 5–20 qubits through the CLI; five circuits and five samples from `random_qasm.py`, with exact agreement across 22 circuits). Results also agree at the distribution level with the default CPU path (fusion + Fma + remapping): `bench/compare_cpu_gpu.py` stayed within TVD and χ² limits on 12 circuits.

## Why the first version has no fusion

- The CPU's `FusionBuffer`, `PassQueue`, and `planRemaps` live in the anonymous namespace of `qasm.cpp` and depend directly on `StateVector&` and CPU kernels. They cannot be called without changing the submodule. Copying them would maintain roughly 790 lines, from `PhaseTable` through `FusionBuffer` (qasm.cpp 1189–1980), in two places.
- The benefit of GPU fusion has not been measured. At 24 qubits, a pass per gate takes about 3 ms for 2x2 and 6 ms for 4x4 (FP64 limited). The CPU rationale of reducing memory passes must be weighed against the doubled 4x4 compute cost. M5 measurement should decide.
- The unfused path allows bit-for-bit comparison with CPU; it establishes the correctness baseline.

`--no-fuse` is accepted but has no effect. There is no remapping.

## Proposal: expose the Quantim planner's passes

To add GPU fusion, separate planner output from execution. The proposed small interface belongs upstream in Quantim, not in the submodule copy:

```cpp
namespace quantim::quantum::qasm {
struct PlannedPass {            // Same fields as the current Pass, without StateVector
    enum class Kind { Single, Pair, Exchange, Diagonal, Table, BitSwaps, Measure } kind;
    std::size_t first, second, controls, clbit;
    detail::Matrix2 matrix2; detail::Matrix4 matrix4;
    std::complex<double> zero, one;
    std::shared_ptr<const PhaseTable> table;   // Diagonal group: qubits and phase array
    std::vector<std::pair<std::size_t, std::size_t>> swaps;
};
class PassSink {                // Implemented by CPU PassQueue and the GPU runner
public:
    virtual void apply(const PlannedPass&) = 0;
    virtual std::size_t measure(std::size_t physicalQubit, std::mt19937_64&) = 0;
};
void plan(const Circuit&, PassSink&, const PlanOptions&);   // FusionBuffer and remapping decisions
}
```

Fusion and remapping decisions (commutation, virtual swaps, `restoreLayout`) would stay in one place. CPU and GPU would receive the same pass sequence, leaving kernel behavior as the only source of result differences. Chunk size (`chunkQubits`, 14) and the remapping threshold are CPU-specific, so they would move into `PlanOptions`; GPU values would come from M5 measurements. The `PhaseTable` `gather_` byte tables belong to CPU execution; the interface would carry only qubits and phases.

Quantim's own tests (`quantum_qasm_tests`, `kernels_test`) would need to preserve bit-for-bit results before connecting the GPU path. This change requires user approval and a separate session.

## Limits

- Noise (`--noise`), `--certified`, `trace`, and `info` are unavailable on GPU.
- More than 24 qubits requires `QUANTIM_GPU_LARGE=1` (increase gradually while monitoring temperature).
- A circuit with intermediate measurements reruns the full circuit on GPU for every shot.
