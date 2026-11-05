// SPDX-License-Identifier: PolyForm-Small-Business-1.0.0
// Manual check for 25-28 qubits, run one size at a time under the temperature
// guard (not part of CTest):
//   large_check <qubits> alloc   allocation and one pass on the lowest and highest qubit
//   large_check <qubits> full    H on every qubit, 40 random operations, norm, repeat;
//                                with qubits <= 26 also the full CPU Scalar comparison
#include "device_state.hpp"
#include "kernels.hpp"

#include <algorithm>
#include <bit>
#include <chrono>
#include <cmath>
#include <complex>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <numbers>
#include <random>
#include <string>
#include <vector>

namespace {

using quantim::gpu::DeviceState;
using Complex = std::complex<double>;
namespace detail = quantim::quantum::detail;

int failures = 0;
int checks = 0;
double slowestLaunch = 0.0;

void check(bool condition, const std::string& message) {
    ++checks;
    if (!condition) {
        ++failures;
        std::cerr << "BAŞARISIZ: " << message << '\n';
    }
}

struct Operation {
    int family = 0;  // 0 matrix, 1 diagonal, 2 exchange, 3 swap
    std::size_t target = 0;
    std::size_t second = 0;
    std::size_t controls = 0;
    detail::Matrix2 matrix{};
    Complex zero{1.0, 0.0};
    Complex one{1.0, 0.0};
};

std::vector<Operation> operations(std::size_t qubits, std::uint64_t seed) {
    std::mt19937_64 rng(seed);
    std::uniform_real_distribution<double> angle(-std::numbers::pi, std::numbers::pi);
    std::vector<Operation> result;
    for (int step = 0; step < 40; ++step) {
        Operation gate;
        gate.family = static_cast<int>(rng() % 4);
        gate.target = rng() % qubits;
        gate.second = (gate.target + 1 + rng() % (qubits - 1)) % qubits;
        const std::size_t control = rng() % qubits;
        if (rng() % 2 == 0 && control != gate.target && control != gate.second) {
            gate.controls = std::size_t{1} << control;
        }
        const double theta = angle(rng) / 2;
        const double phi = angle(rng);
        const double lambda = angle(rng);
        gate.matrix = {Complex{std::cos(theta), 0.0}, -std::polar(std::sin(theta), lambda),
                       std::polar(std::sin(theta), phi), std::polar(std::cos(theta), phi + lambda)};
        gate.zero = std::polar(1.0, angle(rng));
        gate.one = gate.family == 3 ? Complex{0.0, 1.0} : std::polar(1.0, angle(rng));
        result.push_back(gate);
    }
    return result;
}

std::size_t launchesOf(const DeviceState& state, const Operation& gate) {
    const std::size_t skipped = (gate.family == 3 ? 2 : 1) + std::popcount(gate.controls);
    const std::size_t items = state.dimension() >> skipped;
    return std::max<std::size_t>(1, (items + DeviceState::kItemsPerLaunch - 1) /
                                        DeviceState::kItemsPerLaunch);
}

void apply(DeviceState& state, const Operation& gate) {
    const auto begin = std::chrono::steady_clock::now();
    switch (gate.family) {
    case 0: state.applyMatrix(gate.target, gate.controls, gate.matrix); break;
    case 1: state.applyDiagonal(gate.target, gate.controls, gate.zero, gate.one); break;
    case 2: state.applyExchange(gate.target, gate.controls); break;
    default: state.applySwap(gate.target, gate.second, gate.controls, gate.one); break;
    }
    state.synchronize();
    const std::chrono::duration<double, std::milli> elapsed =
        std::chrono::steady_clock::now() - begin;
    slowestLaunch = std::max(slowestLaunch,
                             elapsed.count() / static_cast<double>(launchesOf(state, gate)));
}

void apply(quantim::quantum::StateVector& state, const Operation& gate) {
    const detail::Target target(state);
    switch (gate.family) {
    case 0: detail::applyMatrix(target, gate.target, gate.controls, gate.matrix); break;
    case 1: detail::applyDiagonal(target, gate.target, gate.controls, gate.zero, gate.one); break;
    case 2: detail::applyExchange(target, gate.target, gate.controls); break;
    default: detail::applySwap(target, gate.target, gate.second, gate.controls, gate.one); break;
    }
}

Operation hadamard(std::size_t qubit) {
    const double s = 1.0 / std::sqrt(2.0);
    Operation gate;
    gate.target = qubit;
    gate.matrix = {s, s, s, -s};
    return gate;
}

std::vector<std::size_t> sampleIndices(std::size_t dimension) {
    std::mt19937_64 rng(20261102);
    std::vector<std::size_t> indices{0, dimension - 1};
    for (int index = 0; index < 1022; ++index) indices.push_back(rng() % dimension);
    return indices;
}

void allocation(std::size_t qubits) {
    DeviceState state(qubits);
    const auto info = quantim::gpu::deviceInfo();
    std::cout << "ayrıldı: " << (sizeof(Complex) << qubits) << " bayt, boş VRAM "
              << info.freeBytes << " bayt\n";
    check(info.freeBytes >= (std::size_t{1} << 30), "ayırma sonrası boş VRAM 1 GiB altında");
    apply(state, hadamard(0));
    apply(state, hadamard(qubits - 1));
    const double norm = state.normSquared();
    check(std::abs(norm - 1.0) <= 1e-9, "iki geçiş sonrası norm");
    std::cout << "norm " << std::setprecision(17) << norm << '\n';
}

void full(std::size_t qubits) {
    const std::vector<Operation> gates = operations(qubits, 20261102 + qubits);
    const std::vector<std::size_t> indices = sampleIndices(std::size_t{1} << qubits);
    DeviceState state(qubits);

    // H on every qubit gives the same amplitude everywhere, bit for bit.
    for (std::size_t qubit = 0; qubit < qubits; ++qubit) apply(state, hadamard(qubit));
    const Complex uniform = state.amplitude(0);
    bool same = true;
    for (const std::size_t index : indices) same = same && state.amplitude(index) == uniform;
    check(same, "tüm H sonrası seçilmiş genlikler eşit");
    check(std::abs(std::norm(uniform) * std::ldexp(1.0, static_cast<int>(qubits)) - 1.0) <= 1e-12,
          "tüm H sonrası genlik 2^(-n/2)");

    for (const Operation& gate : gates) apply(state, gate);
    const double norm = state.normSquared();
    check(std::abs(norm - 1.0) <= 1e-9, "dizi sonrası norm");
    std::vector<Complex> first;
    for (const std::size_t index : indices) first.push_back(state.amplitude(index));

    state.reset();
    for (std::size_t qubit = 0; qubit < qubits; ++qubit) apply(state, hadamard(qubit));
    for (const Operation& gate : gates) apply(state, gate);
    bool repeated = true;
    for (std::size_t slot = 0; slot < indices.size(); ++slot) {
        const Complex value = state.amplitude(indices[slot]);
        repeated = repeated && std::memcmp(&value, &first[slot], sizeof(value)) == 0;
    }
    check(repeated, "tekrar koşusu seçilmiş genliklerde bit bit aynı");
    check(state.normSquared() == norm, "tekrar koşusu normu bit bit aynı");
    std::cout << "norm " << std::setprecision(17) << norm << '\n';

    if (qubits <= 26) {
        detail::simd::use(detail::simd::Level::Scalar);
        quantim::quantum::StateVector cpu(qubits, 0, quantim::quantum::NormCheck::AtMeasure);
        for (std::size_t qubit = 0; qubit < qubits; ++qubit) apply(cpu, hadamard(qubit));
        for (const Operation& gate : gates) apply(cpu, gate);
        const std::vector<Complex> gpu = state.download();
        check(std::memcmp(gpu.data(), cpu.amplitudes().data(), gpu.size() * sizeof(Complex)) == 0,
              "CPU Scalar ile tam vektör bit bit aynı");
    }
}

}  // namespace

int main(int argc, char** argv) {
    if (argc != 3) {
        std::cerr << "Kullanım: large_check <kübit> alloc|full\n";
        return 2;
    }
    const std::size_t qubits = std::stoul(argv[1]);
    const std::string mode = argv[2];
    try {
        const auto info = quantim::gpu::deviceInfo();
        std::cout << info.name << ", boş VRAM " << info.freeBytes << " bayt\n";
        if (mode == "alloc") {
            allocation(qubits);
        } else if (mode == "full") {
            full(qubits);
        } else {
            std::cerr << "bilinmeyen kip: " << mode << '\n';
            return 2;
        }
    } catch (const std::exception& error) {
        std::cerr << "HATA: " << error.what() << '\n';
        return 1;
    }
    std::cout << "en yavaş launch " << std::setprecision(4) << slowestLaunch << " ms, " << checks
              << " kontrol, " << failures << " hata\n";
    return failures == 0 ? 0 : 1;
}
