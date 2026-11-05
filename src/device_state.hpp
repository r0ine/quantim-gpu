// SPDX-License-Identifier: PolyForm-Small-Business-1.0.0
#ifndef QUANTIM_GPU_DEVICE_STATE_HPP
#define QUANTIM_GPU_DEVICE_STATE_HPP

#include <array>
#include <complex>
#include <cstddef>
#include <random>
#include <span>
#include <string>
#include <vector>

// State vector in device memory, same layout and bit order as the CPU
// StateVector: amplitude k is interleaved re, im at doubles 2k and 2k + 1, and
// qubit q is bit q of k. The kernels repeat the arithmetic of the CPU Scalar
// level term by term; built with --fmad=false they round the same way.
//
// Every kernel launch covers at most kItemsPerLaunch pairs, so one launch
// stays far below the display driver's timeout even at 28 qubits. Errors are
// std::runtime_error with a Turkish message; nothing falls back to the CPU.
namespace quantim::gpu {

using Matrix2 = std::array<std::complex<double>, 4>;  // row-major, {|0>, |1>}
// Row-major, local index 2 * bit(first) + bit(second), as in the CPU kernels.
using Matrix4 = std::array<std::complex<double>, 16>;

// Probability mass of the whole state and of the part with one qubit set.
struct Weights {
    double total = 0.0;
    double one = 0.0;
};

struct DeviceInfo {
    std::string name;
    int major = 0;
    int minor = 0;
    std::size_t freeBytes = 0;
    std::size_t totalBytes = 0;
};

// Throws when no CUDA device is visible.
DeviceInfo deviceInfo();

class DeviceState {
public:
    static constexpr std::size_t kMaxQubits = 28;
    // Above this a run needs QUANTIM_GPU_LARGE=1: larger states are only run
    // step by step, with the temperature watched.
    static constexpr std::size_t kUnguardedQubits = 24;
    static constexpr std::size_t kReserveBytes = std::size_t{512} << 20;
    static constexpr std::size_t kItemsPerLaunch = std::size_t{1} << 22;
    // Norm partial sums cover this many amplitudes each, summed in a fixed
    // tree, so normSquared() does not depend on launch layout.
    static constexpr std::size_t kSumSpan = std::size_t{1} << 14;

    explicit DeviceState(std::size_t qubits);  // |0...0>
    ~DeviceState();
    DeviceState(const DeviceState&) = delete;
    DeviceState& operator=(const DeviceState&) = delete;
    DeviceState(DeviceState&& other) noexcept;
    DeviceState& operator=(DeviceState&& other) noexcept;

    [[nodiscard]] std::size_t qubitCount() const noexcept { return qubits_; }
    [[nodiscard]] std::size_t dimension() const noexcept { return std::size_t{1} << qubits_; }

    void reset();
    void upload(std::span<const std::complex<double>> amplitudes);
    [[nodiscard]] std::vector<std::complex<double>> download() const;
    [[nodiscard]] std::complex<double> amplitude(std::size_t index) const;
    // Sum of |a|^2 reduced on the device; only the partial sums cross the bus.
    [[nodiscard]] double normSquared() const;

    // A 2x2 on `target` where every bit of `controls` is set.
    void applyMatrix(std::size_t target, std::size_t controls, const Matrix2& matrix);
    // Scales the target's |0> and |1> amplitudes; a phase equal to 1 is skipped.
    void applyDiagonal(std::size_t target, std::size_t controls, std::complex<double> zero,
                       std::complex<double> one);
    // X on `target`, controlled.
    void applyExchange(std::size_t target, std::size_t controls);
    // Swaps |first=1, second=0> with |first=0, second=1>, both scaled by
    // `phase` unless it is 1 (i for iSWAP).
    void applySwap(std::size_t first, std::size_t second, std::size_t controls,
                   std::complex<double> phase);
    void applyMatrix4(std::size_t first, std::size_t second, std::size_t controls,
                      const Matrix4& matrix);

    // Measurement in the CPU's order: probabilities are summed serially in
    // blocks of kSampleBlock amplitudes, blocks are added on the host and the
    // generator is drawn on the host, so with the same amplitudes and seed the
    // outcome matches the CPU run.
    [[nodiscard]] Weights weigh(std::size_t qubit) const;
    void collapse(std::size_t qubit, bool outcome, double scale);
    // One measurement of `qubit` with the draw taken from `rng`; returns 0 or 1.
    std::size_t measure(std::size_t qubit, std::mt19937_64& rng);
    // Basis indices of `shots` samples of the whole register, the same sequence
    // the CPU's terminal sampling produces. The state is left unchanged.
    [[nodiscard]] std::vector<std::size_t> sample(std::size_t shots, std::mt19937_64& rng) const;

    static constexpr std::size_t kSampleBlock = std::size_t{1} << 14;
    static constexpr double kNormTolerance = 1e-9;

    void synchronize() const;

private:
    void checkQubit(std::size_t qubit, std::size_t controls, const char* what) const;

    [[nodiscard]] std::size_t sumBlocks() const noexcept;

    std::size_t qubits_ = 0;
    double* data_ = nullptr;
    double* partials_ = nullptr;
};

}  // namespace quantim::gpu

#endif
