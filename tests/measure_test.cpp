// SPDX-License-Identifier: PolyForm-Small-Business-1.0.0
#include "device_state.hpp"
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
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

using quantim::gpu::DeviceState;
using Complex = std::complex<double>;
using Amplitudes = std::vector<Complex>;
namespace detail = quantim::quantum::detail;

int checks = 0;
int failures = 0;

void check(bool condition, const std::string& label) {
    ++checks;
    if (!condition) {
        ++failures;
        std::cerr << "BAŞARISIZ: " << label << '\n';
    }
}

void checkState(const Amplitudes& actual, const Amplitudes& expected,
                const std::string& label) {
    const bool sameSize = actual.size() == expected.size();
    const bool sameBits = sameSize &&
                          std::memcmp(actual.data(), expected.data(),
                                      actual.size() * sizeof(Complex)) == 0;
    if (sameSize && !sameBits) {
        std::size_t first = actual.size();
        double largest = 0.0;
        for (std::size_t index = 0; index < actual.size(); ++index) {
            if (first == actual.size() &&
                std::memcmp(&actual[index], &expected[index], sizeof(Complex)) != 0) {
                first = index;
            }
            largest = std::max(largest, std::abs(actual[index] - expected[index]));
        }
        std::cerr << std::setprecision(17) << label << ": ilk fark=" << first
                  << ", en büyük mutlak fark=" << largest << '\n';
    }
    check(sameBits, label + (sameSize ? " bit farkı" : " boyut farkı"));
}

void checkSamples(const std::vector<std::size_t>& actual,
                  const std::vector<std::size_t>& expected, const std::string& label) {
    if (actual != expected) {
        const std::size_t common = std::min(actual.size(), expected.size());
        std::size_t first = 0;
        while (first < common && actual[first] == expected[first]) ++first;
        std::cerr << label << ": ilk fark=" << first;
        if (first < common) {
            std::cerr << ", GPU=" << actual[first] << ", CPU=" << expected[first];
        } else {
            std::cerr << ", GPU boyu=" << actual.size() << ", CPU boyu=" << expected.size();
        }
        std::cerr << '\n';
    }
    check(actual == expected, label + " örnek dizisi");
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

template <typename Work>
void expectRuntime(const std::string& label, Work work) {
    try {
        work();
        check(false, label + " kabul edildi");
    } catch (const std::runtime_error&) {
        check(true, label);
    } catch (const std::exception& error) {
        check(false, label + ": " + error.what());
    }
}

std::size_t bit(std::size_t qubit) { return std::size_t{1} << qubit; }

Amplitudes randomState(std::size_t qubits, std::mt19937_64& rng,
                       std::size_t first = 0, std::size_t count = 0) {
    Amplitudes amplitudes(bit(qubits));
    if (count == 0) count = amplitudes.size();
    std::uniform_real_distribution<double> value(-1.0, 1.0);
    double norm = 0.0;
    for (std::size_t index = first; index < first + count; ++index) {
        amplitudes[index] = {value(rng), value(rng)};
        norm += std::norm(amplitudes[index]);
    }
    const double scale = 1.0 / std::sqrt(norm);
    for (std::size_t index = first; index < first + count; ++index) {
        amplitudes[index] *= scale;
    }
    return amplitudes;
}

detail::Matrix2 randomUnitary2(std::mt19937_64& rng) {
    constexpr double pi = 3.14159265358979323846;
    std::uniform_real_distribution<double> angle(-pi, pi);
    std::uniform_real_distribution<double> theta(0.0, pi);
    const double half = theta(rng) / 2.0;
    const double phi = angle(rng);
    const double lambda = angle(rng);
    return {Complex{std::cos(half), 0.0}, -std::polar(std::sin(half), lambda),
            std::polar(std::sin(half), phi), std::polar(std::cos(half), phi + lambda)};
}

detail::Matrix4 randomUnitary4(std::mt19937_64& rng) {
    const detail::Matrix2 first = randomUnitary2(rng);
    const detail::Matrix2 second = randomUnitary2(rng);
    std::uniform_real_distribution<double> angle(-3.14159265358979323846,
                                                  3.14159265358979323846);
    const Complex phase = std::polar(1.0, angle(rng));
    detail::Matrix4 matrix{};
    for (std::size_t row = 0; row < 4; ++row) {
        for (std::size_t column = 0; column < 4; ++column) {
            matrix[4 * row + column] =
                first[2 * (row >> 1) + (column >> 1)] *
                second[2 * (row & 1u) + (column & 1u)];
            if (row == 3) matrix[4 * row + column] *= phase;
        }
    }
    return matrix;
}

std::vector<std::pair<std::size_t, std::size_t>> pairsFor(std::size_t qubits) {
    if (qubits == 24) return {{0, 1}, {22, 23}, {0, 23}};
    std::vector<std::pair<std::size_t, std::size_t>> pairs;
    const auto add = [&](std::size_t first, std::size_t second) {
        const std::pair<std::size_t, std::size_t> pair{first, second};
        if (first < qubits && second < qubits && first != second &&
            std::find(pairs.begin(), pairs.end(), pair) == pairs.end()) {
            pairs.push_back(pair);
        }
    };
    add(0, 1);
    add(1, 0);
    add(0, qubits - 1);
    add(qubits / 2, qubits / 2 + 1);
    add(qubits - 1, 0);
    return pairs;
}

std::vector<std::size_t> controlsFor(std::size_t qubits, std::size_t first,
                                     std::size_t second) {
    std::vector<std::size_t> controls{0};
    if (qubits == 24) return controls;
    std::vector<std::size_t> available;
    for (std::size_t qubit = 0; qubit < qubits; ++qubit) {
        if (qubit != first && qubit != second) available.push_back(qubit);
    }
    if (!available.empty()) controls.push_back(bit(available.front()));
    if (available.size() >= 2) {
        controls.push_back(bit(available.front()) | bit(available.back()));
    }
    return controls;
}

void matrixCases(std::size_t qubits, std::mt19937_64& rng) {
    DeviceState gpu(qubits);
    const Amplitudes initial = randomState(qubits, rng);
    for (const auto& [first, second] : pairsFor(qubits)) {
        for (const std::size_t controls : controlsFor(qubits, first, second)) {
            const detail::Matrix4 matrix = randomUnitary4(rng);
            Amplitudes cpu = initial;
            gpu.upload(initial);
            detail::applyMatrix4(detail::Target(reinterpret_cast<double*>(cpu.data()),
                                                cpu.size()),
                                 first, second, controls, matrix);
            gpu.applyMatrix4(first, second, controls, matrix);
            checkState(gpu.download(), cpu,
                       "4x4 n=" + std::to_string(qubits) + " çift=" +
                           std::to_string(first) + "," + std::to_string(second) +
                           " kontrol=" + std::to_string(controls));
        }
    }
}

// Vector-based copy of the anonymous measureQubit function in qasm.cpp.
std::size_t referenceMeasure(Amplitudes& amplitudes, std::size_t qubit,
                             std::mt19937_64& rng) {
    const std::size_t mask = bit(qubit);
    constexpr std::size_t blockSize = 1 << 14;
    const std::size_t blockCount = (amplitudes.size() + blockSize - 1) / blockSize;
    std::vector<std::array<double, 2>> partials(blockCount);
    for (std::size_t block = 0; block < blockCount; ++block) {
        const std::size_t end = std::min((block + 1) * blockSize, amplitudes.size());
        double blockTotal = 0.0;
        double blockOne = 0.0;
        for (std::size_t index = block * blockSize; index < end; ++index) {
            const double probability = std::norm(amplitudes[index]);
            blockTotal += probability;
            if ((index & mask) != 0) blockOne += probability;
        }
        partials[block] = {blockTotal, blockOne};
    }

    double total = 0.0;
    double one = 0.0;
    for (const auto& partial : partials) {
        total += partial[0];
        one += partial[1];
    }
    if (!std::isfinite(total) || std::abs(total - 1.0) > DeviceState::kNormTolerance) {
        throw std::runtime_error("durum vektörünün normu bozuldu: " + std::to_string(total));
    }
    const double probabilityOne = std::clamp(one / total, 0.0, 1.0);
    const double sample = std::generate_canonical<double, 53>(rng);
    const bool outcome = sample < probabilityOne;
    const double kept = outcome ? one : total - one;
    if (!(kept > 0.0)) throw std::runtime_error("ölçüm sonrası olasılık kalmadı");

    const double scale = 1.0 / std::sqrt(kept);
    for (std::size_t block = 0; block < blockCount; ++block) {
        const std::size_t end = std::min((block + 1) * blockSize, amplitudes.size());
        for (std::size_t index = block * blockSize; index < end; ++index) {
            const bool set = (index & mask) != 0;
            amplitudes[index] = set == outcome ? amplitudes[index] * scale : Complex{};
        }
    }
    return outcome ? 1 : 0;
}

void compareMeasurement(DeviceState& gpu, Amplitudes& cpu, std::size_t qubit,
                        std::mt19937_64& gpuRng, std::mt19937_64& cpuRng,
                        const std::string& label) {
    const std::size_t expected = referenceMeasure(cpu, qubit, cpuRng);
    const std::size_t actual = gpu.measure(qubit, gpuRng);
    check(actual == expected, label + " sonuç");
    checkState(gpu.download(), cpu, label + " çökertme");
    check(gpuRng() == cpuRng(), label + " RNG konumu");
}

void measureCases(std::size_t qubits, std::mt19937_64& setup) {
    DeviceState gpu(qubits);
    Amplitudes cpu = randomState(qubits, setup);
    gpu.upload(cpu);
    std::vector<std::size_t> order(qubits);
    std::iota(order.begin(), order.end(), 0);
    std::shuffle(order.begin(), order.end(), setup);
    std::mt19937_64 gpuRng(20261102 + qubits);
    std::mt19937_64 cpuRng(20261102 + qubits);
    for (const std::size_t qubit : order) {
        compareMeasurement(gpu, cpu, qubit, gpuRng, cpuRng,
                           "ölçüm n=" + std::to_string(qubits) +
                               " kübit=" + std::to_string(qubit));
    }

    const std::size_t target = qubits - 1;
    for (const std::size_t outcome : std::array<std::size_t, 2>{0, 1}) {
        cpu.assign(bit(qubits), Complex{});
        cpu[outcome == 0 ? 0 : bit(target)] = Complex{1.0, 0.0};
        gpu.upload(cpu);
        gpuRng.seed(700 + 2 * qubits + outcome);
        cpuRng.seed(700 + 2 * qubits + outcome);
        const std::string label = "kesin ölçüm n=" + std::to_string(qubits) +
                                  " sonuç=" + std::to_string(outcome);
        const std::size_t expected = referenceMeasure(cpu, target, cpuRng);
        const std::size_t actual = gpu.measure(target, gpuRng);
        check(actual == outcome, label + " kesin sonuç");
        check(actual == expected, label + " CPU sonucu");
        checkState(gpu.download(), cpu, label + " çökertme");
        check(gpuRng() == cpuRng(), label + " RNG konumu");
    }
}

// Preserves the block and selection order of qasm.cpp's terminal sampler.
std::vector<std::size_t> referenceSample(const Amplitudes& amplitudes, std::size_t shots,
                                         std::mt19937_64& rng) {
    constexpr std::size_t blockSize = 1 << 14;
    const std::size_t blockCount = (amplitudes.size() + blockSize - 1) / blockSize;
    std::vector<double> partials(blockCount);
    std::vector<std::size_t> lastNonzero(blockCount, 0);
    for (std::size_t block = 0; block < blockCount; ++block) {
        const std::size_t end = std::min((block + 1) * blockSize, amplitudes.size());
        double sum = 0.0;
        std::size_t finalBasis = 0;
        for (std::size_t basis = block * blockSize; basis < end; ++basis) {
            const double probability = std::norm(amplitudes[basis]);
            sum += probability;
            if (probability > 0.0) finalBasis = basis;
        }
        partials[block] = sum;
        lastNonzero[block] = finalBasis;
    }

    std::vector<double> prefix(blockCount + 1, 0.0);
    std::size_t finalBasis = 0;
    for (std::size_t block = 0; block < blockCount; ++block) {
        prefix[block + 1] = prefix[block] + partials[block];
        finalBasis = std::max(finalBasis, lastNonzero[block]);
    }
    const double total = prefix.back();
    std::vector<double> samples(shots);
    for (double& sample : samples) {
        sample = std::generate_canonical<double, 53>(rng) * total;
    }
    std::sort(samples.begin(), samples.end());

    std::vector<std::size_t> sampleEnds(blockCount);
    std::size_t next = 0;
    for (std::size_t block = 0; block < blockCount; ++block) {
        while (next < shots && samples[next] < prefix[block + 1]) ++next;
        sampleEnds[block] = next;
    }

    std::vector<std::size_t> bases(shots, finalBasis);
    for (std::size_t block = 0; block < blockCount; ++block) {
        std::size_t sample = block == 0 ? 0 : sampleEnds[block - 1];
        const std::size_t sampleEnd = sampleEnds[block];
        if (sample == sampleEnd) continue;
        const std::size_t end = std::min((block + 1) * blockSize, amplitudes.size());
        double local = 0.0;
        for (std::size_t basis = block * blockSize; basis < end && sample < sampleEnd;
             ++basis) {
            local += std::norm(amplitudes[basis]);
            const double cumulative = prefix[block] + local;
            while (sample < sampleEnd && samples[sample] < cumulative) {
                bases[sample++] = basis;
            }
        }
    }
    return bases;
}

void compareSampling(DeviceState& gpu, const Amplitudes& amplitudes, std::size_t shots,
                     std::uint64_t seed, const std::string& label) {
    gpu.upload(amplitudes);
    std::mt19937_64 gpuRng(seed);
    std::mt19937_64 cpuRng(seed);
    const std::vector<std::size_t> expected = referenceSample(amplitudes, shots, cpuRng);
    const std::vector<std::size_t> actual = gpu.sample(shots, gpuRng);
    checkSamples(actual, expected, label);
    checkState(gpu.download(), amplitudes, label + " durum korunumu");
    check(gpuRng() == cpuRng(), label + " RNG konumu");
}

void sampleCases(std::size_t qubits, std::mt19937_64& setup) {
    DeviceState gpu(qubits);
    const Amplitudes random = randomState(qubits, setup);
    const std::string sizeLabel = "örnekleme n=" + std::to_string(qubits);
    if (qubits == 24) {
        compareSampling(gpu, random, 1000, 3000 + qubits, sizeLabel + " rastgele 1000");
    } else {
        for (const std::size_t shots : std::array<std::size_t, 5>{0, 1, 7, 1000, 100000}) {
            compareSampling(gpu, random, shots, 3000 + qubits + shots,
                            sizeLabel + " rastgele " + std::to_string(shots));
        }
    }

    if (qubits == 15 || qubits == 20 || qubits == 24) {
        const std::size_t first = (bit(qubits) / DeviceState::kSampleBlock - 1) *
                                  DeviceState::kSampleBlock;
        const Amplitudes sparse = randomState(qubits, setup, first, DeviceState::kSampleBlock);
        compareSampling(gpu, sparse, 1000, 4000 + qubits, sizeLabel + " tek blok");
    }

    Amplitudes basis(bit(qubits));
    basis[bit(qubits) - 1] = Complex{1.0, 0.0};
    compareSampling(gpu, basis, 1000, 5000 + qubits, sizeLabel + " baz durumu");
}

void errorCases() {
    DeviceState gpu(2);
    const detail::Matrix4 identity{1.0, 0.0, 0.0, 0.0,
                                   0.0, 1.0, 0.0, 0.0,
                                   0.0, 0.0, 1.0, 0.0,
                                   0.0, 0.0, 0.0, 1.0};
    std::mt19937_64 rng(123);
    expectInvalid("weigh kübit sınırı", [&] { (void)gpu.weigh(2); });
    expectInvalid("measure kübit sınırı", [&] { (void)gpu.measure(2, rng); });
    expectInvalid("4x4 aynı kübit", [&] { gpu.applyMatrix4(0, 0, 0, identity); });
    expectInvalid("4x4 ilk kübit kontrolü", [&] { gpu.applyMatrix4(0, 1, bit(0), identity); });
    expectInvalid("4x4 ikinci kübit kontrolü", [&] { gpu.applyMatrix4(0, 1, bit(1), identity); });
    const Amplitudes normTwo{Complex{1.0, 0.0}, Complex{1.0, 0.0},
                             Complex{}, Complex{}};
    gpu.upload(normTwo);
    expectRuntime("normu iki olan durum", [&] { (void)gpu.measure(0, rng); });
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
        std::mt19937_64 setup(20261102);
        for (const std::size_t qubits : std::array<std::size_t, 7>{2, 3, 5, 10, 14, 20, 24}) {
            matrixCases(qubits, setup);
        }
        for (const std::size_t qubits : std::array<std::size_t, 5>{1, 3, 10, 15, 20}) {
            measureCases(qubits, setup);
        }
        for (const std::size_t qubits : std::array<std::size_t, 6>{1, 4, 14, 15, 20, 24}) {
            sampleCases(qubits, setup);
        }
        errorCases();
    } catch (const std::exception& error) {
        ++failures;
        std::cerr << "BAŞARISIZ: " << error.what() << '\n';
    }

    if (failures != 0) {
        std::cerr << failures << " hata, " << checks << " kontrol\n";
        return 1;
    }
    std::cout << "GPU ölçüm testleri: " << checks << " kontrol geçti\n";
    return 0;
}
