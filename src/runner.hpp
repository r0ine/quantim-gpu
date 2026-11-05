// SPDX-License-Identifier: PolyForm-Small-Business-1.0.0
#ifndef QUANTIM_GPU_RUNNER_HPP
#define QUANTIM_GPU_RUNNER_HPP

#include <cstddef>
#include <cstdint>

#include "device_state.hpp"
#include "quantim/qasm.hpp"

// Runs a parsed Quantim circuit on the GPU the way the CPU's unfused path does:
// each gate goes to the kernel and matrix gates.cpp would pick, measurements
// and terminal sampling follow qasm.cpp's runShots with one seeded generator.
// With CPU Scalar kernels the amplitudes, and so the histograms, are the same
// as `QUANTIM_SIMD=scalar quantim run --no-fuse`. There is no fusion or qubit
// remapping yet.
namespace quantim::gpu {

void applyGate(DeviceState& state, const quantum::qasm::Instruction& instruction);

quantum::qasm::RunResult runShots(const quantum::qasm::Circuit& circuit, std::size_t shots,
                                  std::uint64_t seed);

}  // namespace quantim::gpu

#endif
