// SPDX-License-Identifier: PolyForm-Small-Business-1.0.0
#include "device_state.hpp"
#include "kernels.hpp"

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <iomanip>
#include <iostream>
#include <numbers>
#include <random>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

using quantim::gpu::DeviceState;
using quantim::quantum::NormCheck;
using quantim::quantum::StateVector;
using Complex = std::complex<double>;
using Amplitudes = std::vector<Complex>;
namespace detail = quantim::quantum::detail;

int checks = 0;
int failures = 0;

void check(bool condition, const std::string& message) {
    ++checks;
    if (!condition) {
        ++failures;
        std::cerr << "BAŞARISIZ: " << message << '\n';
    }
}

void checkState(const Amplitudes& gpu, const Amplitudes& cpu, const std::string& label) {
    const bool sameSize = gpu.size() == cpu.size();
    const bool sameBits = sameSize &&
                          std::memcmp(gpu.data(), cpu.data(), gpu.size() * sizeof(Complex)) == 0;
    if (!sameBits && sameSize) {
        double largest = 0.0;
        std::size_t first = gpu.size();
        for (std::size_t index = 0; index < gpu.size(); ++index) {
            if (std::memcmp(&gpu[index], &cpu[index], sizeof(Complex)) != 0 && first == gpu.size()) {
                first = index;
            }
            largest = std::max(largest, std::abs(gpu[index] - cpu[index]));
        }
        std::cerr << std::setprecision(17) << label << ": ilk fark=" << first
                  << ", en büyük mutlak fark=" << largest << '\n';
    }
    check(sameBits, label + (sameSize ? " bit farkı" : " boyut farkı"));
}

template <typename Work>
void expectInvalid(const std::string& label, Work work) {
    try {
        work();
        check(false, label + " kabul edildi");
    } catch (const std::invalid_argument&) {
        check(true, label);
    } catch (const std::exception& error) {
        check(false, label + ": " + error.what());
    }
}

std::size_t bit(std::size_t qubit) { return std::size_t{1} << qubit; }

double angle(std::mt19937_64& rng) {
    return std::uniform_real_distribution<double>(-std::numbers::pi, std::numbers::pi)(rng);
}

detail::Matrix2 unitary(std::mt19937_64& rng) {
    const double theta = std::uniform_real_distribution<double>(0.0, std::numbers::pi)(rng);
    const double phi = angle(rng);
    const double lambda = angle(rng);
    const double cosine = std::cos(theta / 2);
    const double sine = std::sin(theta / 2);
    return {Complex{cosine, 0.0}, -std::polar(sine, lambda),
            std::polar(sine, phi), std::polar(cosine, phi + lambda)};
}

void randomize(StateVector& state, std::mt19937_64& rng) {
    std::normal_distribution<double> normal;
    double norm = 0.0;
    for (Complex& amplitude : state.mutableAmplitudes()) {
        amplitude = {normal(rng), normal(rng)};
        norm += std::norm(amplitude);
    }
    const double scale = 1.0 / std::sqrt(norm);
    for (Complex& amplitude : state.mutableAmplitudes()) amplitude *= scale;
}

enum class Family { Matrix, Diagonal, Exchange, Swap };

struct Operation {
    Family family = Family::Matrix;
    std::size_t target = 0;
    std::size_t second = 0;
    std::size_t controls = 0;
    detail::Matrix2 matrix{};
    Complex zero{1.0, 0.0};
    Complex one{1.0, 0.0};
};

void apply(DeviceState& gpu, const Operation& gate) {
    switch (gate.family) {
    case Family::Matrix:
        gpu.applyMatrix(gate.target, gate.controls, gate.matrix);
        break;
    case Family::Diagonal:
        gpu.applyDiagonal(gate.target, gate.controls, gate.zero, gate.one);
        break;
    case Family::Exchange:
        gpu.applyExchange(gate.target, gate.controls);
        break;
    case Family::Swap:
        gpu.applySwap(gate.target, gate.second, gate.controls, gate.one);
        break;
    }
}

void apply(DeviceState& gpu, StateVector& cpu, const Operation& gate) {
    apply(gpu, gate);
    switch (gate.family) {
    case Family::Matrix:
        detail::applyMatrix(detail::Target(cpu), gate.target, gate.controls, gate.matrix);
        break;
    case Family::Diagonal:
        detail::applyDiagonal(detail::Target(cpu), gate.target, gate.controls, gate.zero, gate.one);
        break;
    case Family::Exchange:
        detail::applyExchange(detail::Target(cpu), gate.target, gate.controls);
        break;
    case Family::Swap:
        detail::applySwap(detail::Target(cpu), gate.target, gate.second, gate.controls, gate.one);
        break;
    }
}

std::vector<std::size_t> controlsFor(std::size_t qubits, std::size_t target) {
    std::vector<std::size_t> others;
    for (std::size_t qubit = 0; qubit < qubits; ++qubit) {
        if (qubit != target) others.push_back(qubit);
    }
    std::vector<std::size_t> masks{0};
    if (others.empty()) return masks;
    masks.push_back(bit(others.front()));
    if (others.back() != others.front()) masks.push_back(bit(others.back()));
    if (qubits >= 4) masks.push_back(bit(others.front()) | bit(others.back()));
    return masks;
}

std::vector<Operation> sequence(std::size_t qubits, std::mt19937_64& rng) {
    std::vector<Operation> gates;
    gates.reserve(200);
    for (std::size_t step = 0; step < 200; ++step) {
        Operation gate;
        gate.family = static_cast<Family>(rng() % 4);
        gate.target = rng() % qubits;
        if (gate.family == Family::Swap) {
            gate.second = rng() % (qubits - 1);
            if (gate.second >= gate.target) ++gate.second;
        }
        std::vector<std::size_t> available;
        for (std::size_t qubit = 0; qubit < qubits; ++qubit) {
            if (qubit != gate.target &&
                (gate.family != Family::Swap || qubit != gate.second)) {
                available.push_back(qubit);
            }
        }
        std::shuffle(available.begin(), available.end(), rng);
        const std::size_t controlCount = std::min<std::size_t>(rng() % 3, available.size());
        for (std::size_t index = 0; index < controlCount; ++index) {
            gate.controls |= bit(available[index]);
        }
        if (gate.family == Family::Matrix) gate.matrix = unitary(rng);
        if (gate.family == Family::Diagonal) {
            if ((rng() & 1u) != 0) gate.zero = std::polar(1.0, angle(rng));
            gate.one = std::polar(1.0, angle(rng));
        }
        if (gate.family == Family::Swap) {
            gate.one = (rng() & 1u) != 0 ? Complex{0.0, 1.0} : Complex{1.0, 0.0};
        }
        gates.push_back(gate);
    }
    return gates;
}

void runSequence(std::size_t qubits, DeviceState& gpu, StateVector& cpu,
                 std::mt19937_64& rng) {
    const std::vector<Operation> gates = sequence(qubits, rng);
    randomize(cpu, rng);
    const Amplitudes initial = cpu.amplitudes();
    gpu.upload(initial);
    for (const Operation& gate : gates) apply(gpu, cpu, gate);
    const Amplitudes firstRun = gpu.download();
    const std::string label = "dizi n=" + std::to_string(qubits);
    checkState(firstRun, cpu.amplitudes(), label);
    double norm = 0.0;
    for (const Complex amplitude : firstRun) norm += std::norm(amplitude);
    check(std::abs(norm - 1.0) <= 1e-9, label + " normu");
    if (qubits == 5) {
        gpu.upload(initial);
        for (const Operation& gate : gates) apply(gpu, gate);
        checkState(gpu.download(), firstRun, label + " tekrar");
    }
}

void checkErrors(DeviceState& gpu, std::size_t qubits) {
    const detail::Matrix2 identity{1.0, 0.0, 0.0, 1.0};
    if (qubits == 1) {
        expectInvalid("matris hedef sınırı", [&] { gpu.applyMatrix(1, 0, identity); });
        expectInvalid("köşegen hedef sınırı", [&] { gpu.applyDiagonal(1, 0, 1.0, 1.0); });
        expectInvalid("değiştirme hedef sınırı", [&] { gpu.applyExchange(1, 0); });
        expectInvalid("takas ilk hedef sınırı", [&] { gpu.applySwap(1, 0, 0, 1.0); });
        expectInvalid("takas ikinci hedef sınırı", [&] { gpu.applySwap(0, 1, 0, 1.0); });
        expectInvalid("matris hedef kontrolü", [&] { gpu.applyMatrix(0, bit(0), identity); });
        expectInvalid("köşegen hedef kontrolü", [&] {
            gpu.applyDiagonal(0, bit(0), 1.0, 1.0);
        });
        expectInvalid("değiştirme hedef kontrolü", [&] { gpu.applyExchange(0, bit(0)); });
        expectInvalid("aynı kübit takası", [&] { gpu.applySwap(0, 0, 0, 1.0); });
        expectInvalid("yanlış yükleme boyu", [&] { gpu.upload(std::span<const Complex>{}); });
    }
    if (qubits == 2) {
        expectInvalid("takas ilk hedef kontrolü", [&] { gpu.applySwap(0, 1, bit(0), 1.0); });
        expectInvalid("takas ikinci hedef kontrolü", [&] { gpu.applySwap(0, 1, bit(1), 1.0); });
    }
}

void runCases(std::size_t qubits, DeviceState& gpu, StateVector& cpu,
              std::mt19937_64& rng) {
    const auto runCase = [&](const std::string& label, const Operation& gate) {
        randomize(cpu, rng);
        gpu.upload(cpu.amplitudes());
        apply(gpu, cpu, gate);
        checkState(gpu.download(), cpu.amplitudes(),
                   "n=" + std::to_string(qubits) + " " + label);
    };

    if (qubits == 24) {
        // Each target has 2^23 pairs; the kItemsPerLaunch=2^22 limit splits the launch.
        for (const std::size_t target : {std::size_t{0}, std::size_t{23}}) {
            Operation gate;
            gate.target = target;
            gate.matrix = unitary(rng);
            runCase("parçalı matris hedef=" + std::to_string(target), gate);
        }
        return;
    }

    std::vector<std::size_t> targets{0, 1, qubits / 2, qubits - 1};
    targets.erase(std::remove_if(targets.begin(), targets.end(),
                                 [qubits](std::size_t target) { return target >= qubits; }),
                  targets.end());
    std::sort(targets.begin(), targets.end());
    targets.erase(std::unique(targets.begin(), targets.end()), targets.end());

    for (const std::size_t target : targets) {
        for (const std::size_t controls : controlsFor(qubits, target)) {
            const std::string label = "hedef=" + std::to_string(target) +
                                      " kontrol=" + std::to_string(controls);
            Operation gate;
            gate.target = target;
            gate.controls = controls;
            gate.matrix = unitary(rng);
            runCase("matris " + label, gate);

            gate.family = Family::Diagonal;
            gate.zero = std::polar(1.0, angle(rng));
            gate.one = std::polar(1.0, angle(rng));
            runCase("köşegen iki faz " + label, gate);
            gate.zero = 1.0;
            runCase("köşegen tek faz " + label, gate);
            gate.one = 1.0;
            runCase("köşegen birim " + label, gate);

            gate.family = Family::Exchange;
            runCase("değiştirme " + label, gate);
        }
    }

    std::vector<std::pair<std::size_t, std::size_t>> pairs;
    const auto addPair = [&](std::size_t first, std::size_t second) {
        const std::pair<std::size_t, std::size_t> pair{first, second};
        if (first < qubits && second < qubits && first != second &&
            std::find(pairs.begin(), pairs.end(), pair) == pairs.end()) {
            pairs.push_back(pair);
        }
    };
    addPair(0, 1);
    addPair(1, 0);
    addPair(0, qubits - 1);
    addPair(qubits / 2, qubits / 2 + 1);
    for (const auto& [first, second] : pairs) {
        std::vector<std::size_t> masks{0};
        for (std::size_t qubit = 0; qubit < qubits; ++qubit) {
            if (qubit != first && qubit != second) {
                masks.push_back(bit(qubit));
                break;
            }
        }
        for (const std::size_t controls : masks) {
            for (const Complex phase : {Complex{1.0, 0.0}, Complex{0.0, 1.0}}) {
                Operation gate;
                gate.family = Family::Swap;
                gate.target = first;
                gate.second = second;
                gate.controls = controls;
                gate.one = phase;
                runCase("takas " + std::to_string(first) + "," + std::to_string(second) +
                            " kontrol=" + std::to_string(controls) +
                            (phase == Complex{1.0, 0.0} ? " faz=1" : " faz=i"),
                        gate);
            }
        }
    }
}

}  // namespace

int main() {
    quantim::gpu::DeviceInfo info;
    try {
        info = quantim::gpu::deviceInfo();
    } catch (const std::exception& error) {
        std::cout << "SKIP: " << error.what() << '\n';
        return 77;
    }
    std::cout << info.name << " CC " << info.major << '.' << info.minor
              << " VRAM " << info.freeBytes << '/' << info.totalBytes << " bayt\n";

    try {
        detail::simd::use(detail::simd::Level::Scalar);
        if (::unsetenv("QUANTIM_GPU_LARGE") != 0) {
            throw std::runtime_error("QUANTIM_GPU_LARGE kaldırılamadı");
        }
        // The constructor rejects these values before cudaMemGetInfo and cudaMalloc.
        expectInvalid("sıfır kübit", [] { DeviceState state(0); });
        expectInvalid("üst sınır dışı", [] { DeviceState state(29); });
        expectInvalid("korumasız 25 kübit", [] { DeviceState state(25); });

        std::mt19937_64 rng(20261102);
        for (const std::size_t qubits : {1u, 2u, 3u, 5u, 10u, 14u, 20u, 24u}) {
            DeviceState gpu(qubits);
            StateVector cpu(qubits, 0, NormCheck::AtMeasure);
            checkErrors(gpu, qubits);
            runCases(qubits, gpu, cpu, rng);
            if (qubits == 5 || qubits == 14 || qubits == 20) {
                runSequence(qubits, gpu, cpu, rng);
            }
        }
    } catch (const std::exception& error) {
        ++failures;
        std::cerr << "BAŞARISIZ: " << error.what() << '\n';
    }

    if (failures != 0) {
        std::cerr << failures << " hata, " << checks << " kontrol\n";
        return 1;
    }
    std::cout << "GPU durum testleri: " << checks << " kontrol geçti\n";
    return 0;
}
