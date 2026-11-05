// SPDX-License-Identifier: PolyForm-Small-Business-1.0.0
#include "runner.hpp"

#include <cmath>
#include <complex>
#include <map>
#include <numbers>
#include <random>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace quantim::gpu {

namespace {

using quantum::qasm::Circuit;
using quantum::qasm::Instruction;
using quantum::qasm::RunResult;
using Complex = std::complex<double>;

std::size_t bitMask(std::size_t qubit) { return std::size_t{1} << qubit; }

// The matrices below are built exactly as gates.cpp builds them, term by term,
// so the coefficients carry the same bits.
Matrix2 rxMatrix(double theta) {
    const double half = theta / 2.0;
    const double cosine = std::cos(half);
    const double sine = std::sin(half);
    return {Complex{cosine, 0.0}, Complex{0.0, -sine}, Complex{0.0, -sine}, Complex{cosine, 0.0}};
}

Matrix2 ryMatrix(double theta) {
    const double half = theta / 2.0;
    const double cosine = std::cos(half);
    const double sine = std::sin(half);
    return {Complex{cosine, 0.0}, Complex{-sine, 0.0}, Complex{sine, 0.0}, Complex{cosine, 0.0}};
}

Matrix2 rzMatrix(double theta) {
    const double half = theta / 2.0;
    return {std::polar(1.0, -half), Complex{}, Complex{}, std::polar(1.0, half)};
}

Matrix2 phaseMatrix(double theta) {
    return {Complex{1.0, 0.0}, Complex{}, Complex{}, std::polar(1.0, theta)};
}

Matrix2 u3Matrix(double theta, double phi, double lambda) {
    const double half = theta / 2.0;
    const double cosine = std::cos(half);
    const double sine = std::sin(half);
    return {Complex{cosine, 0.0}, -std::polar(sine, lambda), std::polar(sine, phi),
            std::polar(cosine, phi + lambda)};
}

// gates.cpp sends a matrix with zero off-diagonal to the diagonal kernel.
void applyMatrix1(DeviceState& state, std::size_t target, std::size_t controls,
                  const Matrix2& matrix) {
    if (matrix[1] == Complex{} && matrix[2] == Complex{}) {
        state.applyDiagonal(target, controls, matrix[0], matrix[3]);
    } else {
        state.applyMatrix(target, controls, matrix);
    }
}

std::size_t controlMask(const std::vector<std::size_t>& qubits, std::size_t count) {
    std::size_t mask = 0;
    for (std::size_t position = 0; position < count; ++position) {
        if ((mask & bitMask(qubits[position])) != 0) {
            throw std::invalid_argument("kontrol kübiti " + std::to_string(qubits[position]) +
                                        " birden fazla kez verildi");
        }
        mask |= bitMask(qubits[position]);
    }
    return mask;
}

const Matrix2& matrixY() {
    static const Matrix2 matrix{Complex{0.0, 0.0}, Complex{0.0, -1.0}, Complex{0.0, 1.0},
                                Complex{0.0, 0.0}};
    return matrix;
}

const Matrix2& matrixH() {
    static const double scale = 1.0 / std::numbers::sqrt2;
    static const Matrix2 matrix{Complex{scale, 0.0}, Complex{scale, 0.0}, Complex{scale, 0.0},
                                Complex{-scale, 0.0}};
    return matrix;
}

bool hasTerminalMeasurements(const Circuit& circuit) {
    bool seenMeasure = false;
    for (const Instruction& instruction : circuit.instructions) {
        if (instruction.kind == Instruction::Kind::Measure) {
            seenMeasure = true;
        } else if (instruction.kind == Instruction::Kind::Gate && seenMeasure) {
            return false;
        }
    }
    return true;
}

std::string bitString(const std::vector<int>& bits) {
    std::string key;
    key.reserve(bits.size());
    for (std::size_t index = bits.size(); index-- > 0;) key.push_back(bits[index] != 0 ? '1' : '0');
    return key;
}

void validate(const Circuit& circuit) {
    if (circuit.qubitCount == 0 || circuit.qubitCount > DeviceState::kMaxQubits) {
        throw std::invalid_argument("devre " + std::to_string(circuit.qubitCount) +
                                    " kübit içeriyor; 1 ile " +
                                    std::to_string(DeviceState::kMaxQubits) + " arasında olmalı");
    }
    for (const Instruction& instruction : circuit.instructions) {
        for (const std::size_t qubit : instruction.qubits) {
            if (qubit >= circuit.qubitCount) {
                throw std::invalid_argument("kapı indeksi " + std::to_string(qubit) +
                                            " aralık dışı");
            }
        }
    }
}

void checkNorm(const DeviceState& state) {
    const double total = state.normSquared();
    if (!std::isfinite(total) || std::abs(total - 1.0) > DeviceState::kNormTolerance) {
        throw std::runtime_error("durum vektörünün normu bozuldu: " + std::to_string(total));
    }
}

}  // namespace

void applyGate(DeviceState& state, const Instruction& instruction) {
    const std::string& name = instruction.name;
    const std::vector<std::size_t>& q = instruction.qubits;
    const std::vector<double>& p = instruction.params;
    if (name == "id") return;
    if (name == "x") return state.applyExchange(q[0], 0);
    if (name == "y") return applyMatrix1(state, q[0], 0, matrixY());
    if (name == "z") return state.applyDiagonal(q[0], 0, {1.0, 0.0}, {-1.0, 0.0});
    if (name == "h") return state.applyMatrix(q[0], 0, matrixH());
    if (name == "s") return applyMatrix1(state, q[0], 0, {1.0, Complex{}, Complex{}, Complex{0.0, 1.0}});
    if (name == "sdg") return applyMatrix1(state, q[0], 0, {1.0, Complex{}, Complex{}, Complex{0.0, -1.0}});
    if (name == "t") {
        return applyMatrix1(state, q[0], 0,
                            {1.0, Complex{}, Complex{}, std::polar(1.0, std::numbers::pi / 4.0)});
    }
    if (name == "tdg") {
        return applyMatrix1(state, q[0], 0,
                            {1.0, Complex{}, Complex{}, std::polar(1.0, -std::numbers::pi / 4.0)});
    }
    if (name == "sx") {
        return applyMatrix1(state, q[0], 0,
                            {Complex{0.5, 0.5}, Complex{0.5, -0.5}, Complex{0.5, -0.5},
                             Complex{0.5, 0.5}});
    }
    if (name == "sxdg") {
        return applyMatrix1(state, q[0], 0,
                            {Complex{0.5, -0.5}, Complex{0.5, 0.5}, Complex{0.5, 0.5},
                             Complex{0.5, -0.5}});
    }
    if (name == "rx") return applyMatrix1(state, q[0], 0, rxMatrix(p[0]));
    if (name == "ry") return applyMatrix1(state, q[0], 0, ryMatrix(p[0]));
    if (name == "rz") return applyMatrix1(state, q[0], 0, rzMatrix(p[0]));
    if (name == "p" || name == "u1") return applyMatrix1(state, q[0], 0, phaseMatrix(p[0]));
    if (name == "u2") return applyMatrix1(state, q[0], 0, u3Matrix(std::numbers::pi / 2.0, p[0], p[1]));
    if (name == "u3") return applyMatrix1(state, q[0], 0, u3Matrix(p[0], p[1], p[2]));
    if (name == "cx") return state.applyExchange(q[1], bitMask(q[0]));
    if (name == "cy") return applyMatrix1(state, q[1], bitMask(q[0]), matrixY());
    if (name == "cz") return applyMatrix1(state, q[1], bitMask(q[0]), {1.0, Complex{}, Complex{}, -1.0});
    if (name == "ch") return applyMatrix1(state, q[1], bitMask(q[0]), matrixH());
    if (name == "crx") return applyMatrix1(state, q[1], bitMask(q[0]), rxMatrix(p[0]));
    if (name == "cry") return applyMatrix1(state, q[1], bitMask(q[0]), ryMatrix(p[0]));
    if (name == "crz") return applyMatrix1(state, q[1], bitMask(q[0]), rzMatrix(p[0]));
    if (name == "cp") return applyMatrix1(state, q[1], bitMask(q[0]), phaseMatrix(p[0]));
    if (name == "swap") return state.applySwap(q[0], q[1], 0, {1.0, 0.0});
    if (name == "ccx") return state.applyExchange(q[2], controlMask(q, 2));
    if (name == "mcphase") {
        return state.applyDiagonal(q.back(), controlMask(q, q.size() - 1), {1.0, 0.0},
                                   std::polar(1.0, p[0]));
    }
    throw std::invalid_argument("'" + name + "' kapısı GPU yürütücüsünde yok");
}

RunResult runShots(const Circuit& circuit, std::size_t shots, std::uint64_t seed) {
    validate(circuit);
    if (circuit.clbitCount == 0) {
        throw std::invalid_argument("devrede klasik kayıt yok; histogram üretilemez");
    }
    RunResult result;
    result.shots = shots;
    result.seed = seed;
    if (shots == 0) return result;

    std::mt19937_64 rng(seed);
    DeviceState state(circuit.qubitCount);
    std::vector<int> clbits(circuit.clbitCount, 0);
    if (hasTerminalMeasurements(circuit)) {
        std::vector<std::pair<std::size_t, std::size_t>> measures;
        for (const Instruction& instruction : circuit.instructions) {
            if (instruction.kind == Instruction::Kind::Gate) applyGate(state, instruction);
            if (instruction.kind == Instruction::Kind::Measure) {
                measures.emplace_back(instruction.clbit, instruction.qubits[0]);
            }
        }
        checkNorm(state);
        for (const std::size_t basis : state.sample(shots, rng)) {
            std::fill(clbits.begin(), clbits.end(), 0);
            for (const auto& [clbit, qubit] : measures) {
                clbits[clbit] = static_cast<int>((basis >> qubit) & 1);
            }
            ++result.counts[bitString(clbits)];
        }
        return result;
    }

    for (std::size_t shot = 0; shot < shots; ++shot) {
        state.reset();
        std::fill(clbits.begin(), clbits.end(), 0);
        for (const Instruction& instruction : circuit.instructions) {
            if (instruction.kind == Instruction::Kind::Gate) {
                applyGate(state, instruction);
            } else if (instruction.kind == Instruction::Kind::Measure) {
                clbits[instruction.clbit] =
                    static_cast<int>(state.measure(instruction.qubits[0], rng));
            }
        }
        ++result.counts[bitString(clbits)];
    }
    return result;
}

}  // namespace quantim::gpu
