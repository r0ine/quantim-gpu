// SPDX-License-Identifier: PolyForm-Small-Business-1.0.0
#include "runner.hpp"
#include "kernels.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <exception>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numbers>
#include <numeric>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace {

using quantim::gpu::DeviceState;
using quantim::quantum::NormCheck;
using quantim::quantum::qasm::Circuit;
using quantim::quantum::qasm::Instruction;
using quantim::quantum::qasm::RunResult;
using Complex = std::complex<double>;
namespace qasm = quantim::quantum::qasm;
namespace detail = quantim::quantum::detail;

struct GateSpec {
    std::string_view name;
    std::size_t qubits;
    std::size_t params;
};

constexpr std::array<GateSpec, 29> gateSpecs{{
    {"x", 1, 0},       {"y", 1, 0},       {"z", 1, 0},       {"h", 1, 0},
    {"s", 1, 0},       {"sdg", 1, 0},     {"t", 1, 0},       {"tdg", 1, 0},
    {"id", 1, 0},      {"sx", 1, 0},      {"sxdg", 1, 0},    {"rx", 1, 1},
    {"ry", 1, 1},      {"rz", 1, 1},      {"p", 1, 1},       {"u1", 1, 1},
    {"u2", 1, 2},      {"u3", 1, 3},      {"cx", 2, 0},      {"cy", 2, 0},
    {"cz", 2, 0},      {"ch", 2, 0},      {"swap", 2, 0},    {"crx", 2, 1},
    {"cry", 2, 1},     {"crz", 2, 1},     {"cp", 2, 1},      {"mcphase", 0, 1},
    {"ccx", 3, 0},
}};

std::size_t checks = 0;
std::size_t failures = 0;

void check(bool condition, const std::string& label) {
    ++checks;
    if (!condition) {
        ++failures;
        std::cerr << "BAŞARISIZ: " << label << '\n';
    }
}

void compareAmplitudes(const Circuit& circuit, const std::string& label) {
    DeviceState gpu(circuit.qubitCount);
    for (const Instruction& instruction : circuit.instructions) {
        if (instruction.kind == Instruction::Kind::Gate) {
            quantim::gpu::applyGate(gpu, instruction);
        }
    }
    const std::vector<Complex> actual = gpu.download();
    const std::vector<Complex> expected =
        qasm::simulate(circuit, 0, NormCheck::AtMeasure, 1, false);
    const bool sameSize = actual.size() == expected.size();
    const bool sameBits = sameSize &&
                          std::memcmp(actual.data(), expected.data(),
                                      actual.size() * sizeof(Complex)) == 0;
    if (!sameBits) {
        const std::size_t common = std::min(actual.size(), expected.size());
        std::size_t first = common;
        double largest = 0.0;
        for (std::size_t index = 0; index < common; ++index) {
            if (first == common &&
                std::memcmp(&actual[index], &expected[index], sizeof(Complex)) != 0) {
                first = index;
            }
            largest = std::max(largest, std::abs(actual[index] - expected[index]));
        }
        std::cerr << std::setprecision(17) << label << ": ilk indeks=" << first
                  << ", en büyük fark=" << largest;
        if (!sameSize) {
            std::cerr << ", GPU boyu=" << actual.size() << ", CPU boyu=" << expected.size();
        }
        std::cerr << '\n';
    }
    check(sameBits, label + " genlikleri");
}

void compareCounts(const Circuit& circuit, std::size_t shots, std::uint64_t seed,
                   const std::string& label) {
    const RunResult actual = quantim::gpu::runShots(circuit, shots, seed);
    const RunResult expected =
        qasm::runShots(circuit, shots, seed, NormCheck::AtMeasure, 1, false);
    if (actual.counts != expected.counts) {
        auto gpuIt = actual.counts.begin();
        auto cpuIt = expected.counts.begin();
        while (gpuIt != actual.counts.end() || cpuIt != expected.counts.end()) {
            std::string key;
            std::size_t gpuCount = 0;
            std::size_t cpuCount = 0;
            if (cpuIt == expected.counts.end() ||
                (gpuIt != actual.counts.end() && gpuIt->first < cpuIt->first)) {
                key = gpuIt->first;
                gpuCount = gpuIt->second;
                ++gpuIt;
            } else if (gpuIt == actual.counts.end() || cpuIt->first < gpuIt->first) {
                key = cpuIt->first;
                cpuCount = cpuIt->second;
                ++cpuIt;
            } else {
                key = gpuIt->first;
                gpuCount = gpuIt->second;
                cpuCount = cpuIt->second;
                ++gpuIt;
                ++cpuIt;
            }
            if (gpuCount != cpuCount) {
                std::cerr << label << ": ilk anahtar=" << key << ", GPU=" << gpuCount
                          << ", CPU=" << cpuCount << '\n';
                break;
            }
        }
    }
    check(actual.counts == expected.counts, label + " sayımları");
    check(actual.shots == expected.shots, label + " atış sayısı");
    check(actual.seed == expected.seed, label + " tohum");
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

std::string angleText(double angle) {
    std::ostringstream text;
    text << std::setprecision(std::numeric_limits<double>::max_digits10) << angle;
    return text.str();
}

std::string header(std::size_t qubits, std::size_t clbits = 0) {
    std::ostringstream text;
    text << "OPENQASM 2.0;\ninclude \"qelib1.inc\";\nqreg q[" << qubits << "];\n";
    if (clbits != 0) text << "creg c[" << clbits << "];\n";
    return text.str();
}

std::string gateLine(const GateSpec& gate, const std::vector<std::size_t>& qubits,
                     const std::vector<std::string>& params) {
    std::ostringstream text;
    text << gate.name;
    if (!params.empty()) {
        text << '(';
        for (std::size_t index = 0; index < params.size(); ++index) {
            if (index != 0) text << ',';
            text << params[index];
        }
        text << ')';
    }
    text << ' ';
    for (std::size_t index = 0; index < qubits.size(); ++index) {
        if (index != 0) text << ',';
        text << "q[" << qubits[index] << ']';
    }
    text << ";\n";
    return text.str();
}

std::vector<std::size_t> anchoredQubits(const GateSpec& gate, std::size_t qubits,
                                        std::size_t anchor) {
    const std::size_t arity = gate.qubits == 0 ? std::min<std::size_t>(qubits, 4) : gate.qubits;
    std::vector<std::size_t> operands;
    operands.reserve(arity);
    if (arity == 1 || gate.name == "swap") operands.push_back(anchor);
    for (std::size_t offset = 1; operands.size() + 1 < arity; ++offset) {
        operands.push_back((anchor + offset) % qubits);
    }
    if (arity > 1) operands.push_back(anchor);
    if (gate.name == "swap") operands.back() = (anchor + 1) % qubits;
    return operands;
}

void gateCases(std::mt19937_64& rng) {
    std::uniform_real_distribution<double> angle(-std::numbers::pi, std::numbers::pi);
    for (const auto& qubits : std::array<std::size_t, 3>{3, 6, 12}) {
        for (const auto& anchor : std::array<std::size_t, 3>{0, qubits / 2, qubits - 1}) {
            for (const GateSpec& gate : gateSpecs) {
                const std::array<std::string, 4> angles{
                    "0", "pi", "-pi/3", angleText(angle(rng))};
                const std::size_t variants = gate.params == 0 ? 1 : angles.size();
                for (std::size_t variant = 0; variant < variants; ++variant) {
                    std::vector<std::string> params;
                    for (std::size_t index = 0; index < gate.params; ++index) {
                        params.push_back(angles[(variant + index) % angles.size()]);
                    }
                    std::ostringstream source;
                    source << header(qubits) << "h q;\nrz(pi/7) q[0];\n"
                           << "ry(pi/5) q[" << qubits / 2 << "];\n"
                           << "p(-pi/9) q[" << qubits - 1 << "];\n"
                           << gateLine(gate, anchoredQubits(gate, qubits, anchor), params);
                    const std::string label = std::string(gate.name) + " n=" +
                                              std::to_string(qubits) + " hedef=" +
                                              std::to_string(anchor) + " varyant=" +
                                              std::to_string(variant);
                    compareAmplitudes(qasm::parseString(source.str()), label);
                }
            }
        }
    }
}

std::string randomAngle(std::mt19937_64& rng) {
    switch (rng() % 4) {
    case 0: return "0";
    case 1: return "pi";
    case 2: return "-pi/3";
    default:
        return angleText(std::uniform_real_distribution<double>(-std::numbers::pi,
                                                                 std::numbers::pi)(rng));
    }
}

std::string randomGateLine(const GateSpec& gate, std::size_t qubits, std::mt19937_64& rng) {
    std::vector<std::size_t> operands(qubits);
    std::iota(operands.begin(), operands.end(), 0);
    std::shuffle(operands.begin(), operands.end(), rng);
    const std::size_t arity = gate.qubits == 0
                                  ? std::min<std::size_t>(qubits, 3 + rng() % 3)
                                  : gate.qubits;
    operands.resize(arity);
    std::vector<std::string> params;
    for (std::size_t index = 0; index < gate.params; ++index) {
        params.push_back(randomAngle(rng));
    }
    return gateLine(gate, operands, params);
}

void randomCases(std::mt19937_64& rng) {
    for (std::size_t circuitIndex = 0; circuitIndex < 30; ++circuitIndex) {
        const std::size_t qubits =
            std::array<std::size_t, 6>{2, 4, 7, 10, 14, 18}[circuitIndex % 6];
        const std::size_t gateCount = 20 + rng() % 101;
        std::vector<const GateSpec*> available;
        for (const GateSpec& gate : gateSpecs) {
            if (gate.qubits <= qubits && (gate.name != "mcphase" || qubits >= 3)) {
                available.push_back(&gate);
            }
        }

        std::ostringstream bare;
        std::ostringstream mid;
        bare << header(qubits, qubits + 3);
        mid << header(qubits, qubits + 3);
        const std::size_t repeatedQubit = rng() % qubits;
        const std::size_t midCount = 1 + rng() % 3;
        for (std::size_t step = 0; step < gateCount; ++step) {
            const GateSpec& gate = step == 0 ? gateSpecs[8]
                                   : step == 1 ? gateSpecs[22]
                                   : step == 2 ? gateSpecs[circuitIndex % gateSpecs.size()]
                                               : *available[rng() % available.size()];
            const std::string line = randomGateLine(gate, qubits, rng);
            bare << line;
            mid << line;
            if (step == gateCount / 3) {
                bare << "barrier q;\n";
                mid << "barrier q;\n";
            }
            if (step == gateCount / 2) {
                for (std::size_t measurement = 0; measurement < midCount; ++measurement) {
                    const std::size_t measured = measurement == 0 ? repeatedQubit : rng() % qubits;
                    mid << "measure q[" << measured << "] -> c["
                        << qubits + measurement << "];\n";
                }
            }
        }
        mid << "measure q[" << repeatedQubit << "] -> c[0];\n";
        const Circuit unmeasured = qasm::parseString(bare.str());
        const std::string label = "rastgele " + std::to_string(circuitIndex) +
                                  " n=" + std::to_string(qubits);
        compareAmplitudes(unmeasured, label);

        std::ostringstream terminal;
        terminal << bare.str();
        for (std::size_t qubit = 0; qubit < qubits; ++qubit) {
            terminal << "measure q[" << qubit << "] -> c[" << qubit << "];\n";
        }
        compareCounts(qasm::parseString(terminal.str()), 2048, 5000 + circuitIndex,
                      label + " son ölçüm");
        compareCounts(qasm::parseString(mid.str()), 200, 6000 + circuitIndex,
                      label + " ara ölçüm");

        std::ostringstream partial;
        partial << bare.str() << "measure q[0] -> c[" << qubits << "];\n"
                << "measure q[" << qubits - 1 << "] -> c[" << qubits + 2 << "];\n"
                << "measure q[0] -> c[" << qubits + 1 << "];\n";
        compareCounts(qasm::parseString(partial.str()), 2048, 7000 + circuitIndex,
                      label + " kısmi/tekrarlı ölçüm");
    }
}

std::string ghz(std::size_t qubits) {
    std::ostringstream source;
    source << header(qubits, qubits) << "h q[0];\n";
    for (std::size_t qubit = 1; qubit < qubits; ++qubit) {
        source << "cx q[0],q[" << qubit << "];\n";
    }
    source << "measure q -> c;\n";
    return source.str();
}

std::string qft(std::size_t qubits) {
    std::ostringstream source;
    source << header(qubits, qubits) << "x q[0];\nx q[" << qubits - 1 << "];\n";
    for (std::size_t target = 0; target < qubits; ++target) {
        source << "h q[" << target << "];\n";
        for (std::size_t control = target + 1; control < qubits; ++control) {
            source << "cp(pi/" << (std::size_t{1} << (control - target))
                   << ") q[" << control << "],q[" << target << "];\n";
        }
    }
    for (std::size_t first = 0; first < qubits / 2; ++first) {
        source << "swap q[" << first << "],q[" << qubits - 1 - first << "];\n";
    }
    source << "measure q -> c;\n";
    return source.str();
}

std::string layered(std::size_t qubits) {
    std::ostringstream source;
    source << header(qubits, qubits);
    for (std::size_t layer = 0; layer < 3; ++layer) {
        for (std::size_t qubit = 0; qubit < qubits; ++qubit) {
            source << "u3(pi/" << 3 + (qubit + layer) % 5 << ","
                   << (layer == 1 ? "-pi/7" : "pi/7") << ",pi/11) q[" << qubit
                   << "];\n";
        }
        for (std::size_t qubit = 0; qubit + 1 < qubits; ++qubit) {
            source << "cx q[" << qubit << "],q[" << qubit + 1 << "];\n";
        }
    }
    source << "measure q -> c;\n";
    return source.str();
}

void namedCases() {
    for (const auto& qubits : std::array<std::size_t, 3>{3, 10, 20}) {
        compareCounts(qasm::parseString(ghz(qubits)), 4096, 8100 + qubits,
                      "GHZ n=" + std::to_string(qubits));
    }
    for (const auto& qubits : std::array<std::size_t, 2>{5, 12}) {
        compareCounts(qasm::parseString(qft(qubits)), 4096, 8200 + qubits,
                      "QFT n=" + std::to_string(qubits));
    }
    for (const auto& qubits : std::array<std::size_t, 2>{8, 16}) {
        compareCounts(qasm::parseString(layered(qubits)), 4096, 8300 + qubits,
                      "katmanlı n=" + std::to_string(qubits));
    }
}

void edgeCases() {
    const Circuit empty = qasm::parseString(header(2, 3) + "measure q[0] -> c[2];\n");
    compareCounts(empty, 0, 9011, "sıfır atış");
    const Circuit unmeasured = qasm::parseString(header(2, 3) + "h q[0];\n");
    compareCounts(unmeasured, 11, 9013, "ölçümsüz atış");
    const Circuit noCreg = qasm::parseString(header(2) + "h q[0];\n");
    expectInvalid("klasik kayıtsız GPU", [&] {
        (void)quantim::gpu::runShots(noCreg, 10, 9012);
    });
    expectInvalid("klasik kayıtsız CPU", [&] {
        (void)qasm::runShots(noCreg, 10, 9012, NormCheck::AtMeasure, 1, false);
    });
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
        std::mt19937_64 rng(20261102);
        gateCases(rng);
        randomCases(rng);
        namedCases();
        edgeCases();
    } catch (const std::exception& error) {
        ++failures;
        std::cerr << "BAŞARISIZ: " << error.what() << '\n';
    }

    if (failures != 0) {
        std::cerr << failures << " hata, " << checks << " kontrol\n";
        return 1;
    }
    std::cout << "GPU QASM testleri: " << checks << " kontrol geçti\n";
    return 0;
}
