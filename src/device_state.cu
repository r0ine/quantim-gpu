// SPDX-License-Identifier: PolyForm-Small-Business-1.0.0
#include "device_state.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <utility>

namespace quantim::gpu {

namespace {

struct Coefficient {
    double re;
    double im;
};

struct Rotation {
    Coefficient m0, m1, m2, m3;
};

struct Quad {
    Coefficient m[16];
    std::size_t offsets[4];
};

// Positions of the bits an item index skips, ascending, and the bits every
// produced index has set: item k maps to the k-th index of the subspace.
struct Spread {
    unsigned count = 0;
    unsigned char bits[DeviceState::kMaxQubits]{};
    unsigned long long fixed = 0;
};

constexpr unsigned kBlock = 256;

void check(cudaError_t status, const char* where) {
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string("CUDA hatası (") + where + "): " +
                                 cudaGetErrorString(status));
    }
}

Coefficient split(std::complex<double> value) {
    return {value.real(), value.imag()};
}

bool isOne(std::complex<double> value) {
    return value == std::complex<double>{1.0, 0.0};
}

Spread spreadOver(std::size_t excluded, std::size_t fixed) {
    Spread result;
    for (unsigned bit = 0; bit < DeviceState::kMaxQubits; ++bit) {
        if ((excluded >> bit) & 1u) result.bits[result.count++] = static_cast<unsigned char>(bit);
    }
    result.fixed = fixed;
    return result;
}

__device__ std::size_t indexOf(std::size_t item, const Spread& spread) {
    for (unsigned slot = 0; slot < spread.count; ++slot) {
        const unsigned bit = spread.bits[slot];
        const std::size_t low = item & ((std::size_t{1} << bit) - 1);
        item = ((item >> bit) << (bit + 1)) | low;
    }
    return item | spread.fixed;
}

__device__ double2 times(Coefficient m, double2 v) {
    return {m.re * v.x - m.im * v.y, m.re * v.y + m.im * v.x};
}

// Same grouping as rotatePair in the CPU kernels.
__global__ void rotateKernel(double2* __restrict__ data, std::size_t begin, std::size_t count,
                             Spread spread, std::size_t partner, Rotation m) {
    const std::size_t item = blockIdx.x * std::size_t{blockDim.x} + threadIdx.x;
    if (item >= count) return;
    const std::size_t zero = indexOf(begin + item, spread);
    const double2 a = data[zero];
    const double2 b = data[zero | partner];
    data[zero] = {(m.m0.re * a.x - m.m0.im * a.y) + (m.m1.re * b.x - m.m1.im * b.y),
                  (m.m0.re * a.y + m.m0.im * a.x) + (m.m1.re * b.y + m.m1.im * b.x)};
    data[zero | partner] = {(m.m2.re * a.x - m.m2.im * a.y) + (m.m3.re * b.x - m.m3.im * b.y),
                            (m.m2.re * a.y + m.m2.im * a.x) + (m.m3.re * b.y + m.m3.im * b.x)};
}

__global__ void diagonalKernel(double2* __restrict__ data, std::size_t begin, std::size_t count,
                               Spread spread, std::size_t partner, Coefficient zeroPhase,
                               Coefficient onePhase, bool touchZero, bool touchOne) {
    const std::size_t item = blockIdx.x * std::size_t{blockDim.x} + threadIdx.x;
    if (item >= count) return;
    const std::size_t zero = indexOf(begin + item, spread);
    if (touchZero) data[zero] = times(zeroPhase, data[zero]);
    if (touchOne) data[zero | partner] = times(onePhase, data[zero | partner]);
}

__global__ void exchangeKernel(double2* __restrict__ data, std::size_t begin, std::size_t count,
                               Spread spread, std::size_t partner) {
    const std::size_t item = blockIdx.x * std::size_t{blockDim.x} + threadIdx.x;
    if (item >= count) return;
    const std::size_t zero = indexOf(begin + item, spread);
    const double2 a = data[zero];
    data[zero] = data[zero | partner];
    data[zero | partner] = a;
}

// `spread` yields the indices with second = 1 and first = 0; `flip` turns one
// into its partner.
__global__ void swapKernel(double2* __restrict__ data, std::size_t begin, std::size_t count,
                           Spread spread, std::size_t flip, Coefficient phase, bool plain) {
    const std::size_t item = blockIdx.x * std::size_t{blockDim.x} + threadIdx.x;
    if (item >= count) return;
    const std::size_t here = indexOf(begin + item, spread);
    const std::size_t there = here ^ flip;
    double2 a = data[there];
    double2 b = data[here];
    if (!plain) {
        a = times(phase, a);
        b = times(phase, b);
    }
    data[here] = a;
    data[there] = b;
}

// Same grouping as transformQuad in the CPU kernels.
__global__ void quadKernel(double2* __restrict__ data, std::size_t begin, std::size_t count,
                           Spread spread, Quad quad) {
    const std::size_t item = blockIdx.x * std::size_t{blockDim.x} + threadIdx.x;
    if (item >= count) return;
    const std::size_t base = indexOf(begin + item, spread);
    double2 v[4];
    for (int column = 0; column < 4; ++column) v[column] = data[base | quad.offsets[column]];
    for (int row = 0; row < 4; ++row) {
        const Coefficient* r = quad.m + 4 * row;
        data[base | quad.offsets[row]] = {
            ((r[0].re * v[0].x - r[0].im * v[0].y) + (r[1].re * v[1].x - r[1].im * v[1].y)) +
                ((r[2].re * v[2].x - r[2].im * v[2].y) + (r[3].re * v[3].x - r[3].im * v[3].y)),
            ((r[0].re * v[0].y + r[0].im * v[0].x) + (r[1].re * v[1].y + r[1].im * v[1].x)) +
                ((r[2].re * v[2].y + r[2].im * v[2].x) + (r[3].re * v[3].y + r[3].im * v[3].x))};
    }
}

// The measurement sums walk a block serially, one thread per block, because
// that is the order the CPU adds them in.
constexpr unsigned kSerialThreads = 64;
constexpr std::size_t kSerialBlocksPerLaunch = 2048;

__global__ void weighKernel(const double2* __restrict__ data, std::size_t firstBlock,
                            std::size_t blocks, std::size_t span, std::size_t mask,
                            double2* __restrict__ partials) {
    const std::size_t block = firstBlock + blockIdx.x * std::size_t{blockDim.x} + threadIdx.x;
    if (block >= blocks) return;
    double total = 0.0;
    double one = 0.0;
    for (std::size_t index = block * span; index < (block + 1) * span; ++index) {
        const double2 value = data[index];
        const double probability = value.x * value.x + value.y * value.y;
        total += probability;
        if ((index & mask) != 0) one += probability;
    }
    partials[block] = {total, one};
}

__global__ void collapseKernel(double2* __restrict__ data, std::size_t begin, std::size_t count,
                               std::size_t mask, bool outcome, double scale) {
    const std::size_t index = begin + blockIdx.x * std::size_t{blockDim.x} + threadIdx.x;
    if (index >= begin + count) return;
    const double2 value = data[index];
    data[index] = ((index & mask) != 0) == outcome ? double2{value.x * scale, value.y * scale}
                                                    : double2{0.0, 0.0};
}

// Per block: the serial sum of |a|^2 and the last index with non-zero mass.
__global__ void massKernel(const double2* __restrict__ data, std::size_t firstBlock,
                           std::size_t blocks, std::size_t span, double* __restrict__ sums,
                           unsigned long long* __restrict__ last) {
    const std::size_t block = firstBlock + blockIdx.x * std::size_t{blockDim.x} + threadIdx.x;
    if (block >= blocks) return;
    double sum = 0.0;
    std::size_t final = 0;
    for (std::size_t index = block * span; index < (block + 1) * span; ++index) {
        const double2 value = data[index];
        const double probability = value.x * value.x + value.y * value.y;
        sum += probability;
        if (probability > 0.0) final = index;
    }
    sums[block] = sum;
    last[block] = final;
}

struct SampleTask {
    unsigned long long block;
    unsigned long long first;
    unsigned long long end;
    double prefix;
};

// Walks one block in the CPU's order and hands out the sorted samples that
// fall inside it.
__global__ void pickKernel(const double2* __restrict__ data, std::size_t span,
                           const SampleTask* __restrict__ tasks, std::size_t taskBegin,
                           std::size_t taskCount, const double* __restrict__ samples,
                           unsigned long long* __restrict__ bases) {
    const std::size_t slot = blockIdx.x * std::size_t{blockDim.x} + threadIdx.x;
    if (slot >= taskCount) return;
    const SampleTask task = tasks[taskBegin + slot];
    std::size_t sample = task.first;
    double local = 0.0;
    for (std::size_t index = task.block * span; index < (task.block + 1) * span && sample < task.end;
         ++index) {
        const double2 value = data[index];
        local += value.x * value.x + value.y * value.y;
        const double cumulative = task.prefix + local;
        while (sample < task.end && samples[sample] < cumulative) bases[sample++] = index;
    }
}

// Device scratch memory, checked against cudaMemGetInfo before allocation.
template <typename T>
class Scratch {
public:
    explicit Scratch(std::size_t count) : count_(count) {
        if (count == 0) return;
        std::size_t freeBytes = 0;
        std::size_t totalBytes = 0;
        check(cudaMemGetInfo(&freeBytes, &totalBytes), "bellek bilgisi");
        if (freeBytes < sizeof(T) * count + DeviceState::kReserveBytes) {
            throw std::runtime_error("VRAM yetersiz: geçici alan için " +
                                     std::to_string(sizeof(T) * count) + " bayt gerekli");
        }
        check(cudaMalloc(reinterpret_cast<void**>(&data_), sizeof(T) * count), "cudaMalloc");
    }
    ~Scratch() {
        if (data_ != nullptr) cudaFree(data_);
    }
    Scratch(const Scratch&) = delete;
    Scratch& operator=(const Scratch&) = delete;

    T* get() const noexcept { return data_; }
    void upload(const std::vector<T>& values) {
        check(cudaMemcpy(data_, values.data(), sizeof(T) * values.size(), cudaMemcpyHostToDevice),
              "yükleme");
    }
    std::vector<T> download() const {
        std::vector<T> values(count_);
        check(cudaMemcpy(values.data(), data_, sizeof(T) * count_, cudaMemcpyDeviceToHost),
              "indirme");
        return values;
    }

private:
    std::size_t count_;
    T* data_ = nullptr;
};

template <typename Kernel, typename... Arguments>
void launchSerial(const double2* data, std::size_t blocks, Kernel kernel,
                  Arguments... arguments) {
    for (std::size_t first = 0; first < blocks; first += kSerialBlocksPerLaunch) {
        const std::size_t count = std::min(kSerialBlocksPerLaunch, blocks - first);
        const auto grid = static_cast<unsigned>((count + kSerialThreads - 1) / kSerialThreads);
        kernel<<<grid, kSerialThreads>>>(data, first, arguments...);
        check(cudaGetLastError(), "ölçüm çekirdeği");
    }
}

constexpr unsigned kSumThreads = 256;
constexpr unsigned kSumBlocksPerLaunch = 256;

// One block per `span` amplitudes: each thread adds a strided share in order,
// then the block halves its shared array; the tree is the same every call.
__global__ void normKernel(const double2* __restrict__ data, std::size_t firstBlock,
                           std::size_t span, double* __restrict__ partials) {
    __shared__ double sums[kSumThreads];
    const std::size_t block = firstBlock + blockIdx.x;
    const double2* here = data + block * span;
    double sum = 0.0;
    for (std::size_t index = threadIdx.x; index < span; index += kSumThreads) {
        const double2 value = here[index];
        sum += value.x * value.x + value.y * value.y;
    }
    sums[threadIdx.x] = sum;
    __syncthreads();
    for (unsigned stride = kSumThreads / 2; stride > 0; stride /= 2) {
        if (threadIdx.x < stride) sums[threadIdx.x] += sums[threadIdx.x + stride];
        __syncthreads();
    }
    if (threadIdx.x == 0) partials[block] = sums[0];
}

template <typename Kernel, typename... Arguments>
void launch(double* data, std::size_t items, Kernel kernel, Arguments... arguments) {
    auto* pairs = reinterpret_cast<double2*>(data);
    for (std::size_t begin = 0; begin < items; begin += DeviceState::kItemsPerLaunch) {
        const std::size_t count = std::min(DeviceState::kItemsPerLaunch, items - begin);
        const auto blocks = static_cast<unsigned>((count + kBlock - 1) / kBlock);
        kernel<<<blocks, kBlock>>>(pairs, begin, count, arguments...);
        check(cudaGetLastError(), "çekirdek başlatma");
    }
}

std::size_t itemsOf(std::size_t qubits, const Spread& spread) {
    return std::size_t{1} << (qubits - spread.count);
}

}  // namespace

DeviceInfo deviceInfo() {
    int count = 0;
    const cudaError_t status = cudaGetDeviceCount(&count);
    if (status != cudaSuccess || count == 0) {
        throw std::runtime_error(std::string("CUDA aygıtı bulunamadı: ") +
                                 (status != cudaSuccess ? cudaGetErrorString(status)
                                                        : "aygıt sayısı 0"));
    }
    cudaDeviceProp properties{};
    check(cudaGetDeviceProperties(&properties, 0), "aygıt bilgisi");
    DeviceInfo info;
    info.name = properties.name;
    info.major = properties.major;
    info.minor = properties.minor;
    check(cudaMemGetInfo(&info.freeBytes, &info.totalBytes), "bellek bilgisi");
    return info;
}

DeviceState::DeviceState(std::size_t qubits) : qubits_(qubits) {
    if (qubits == 0 || qubits > kMaxQubits) {
        throw std::invalid_argument("GPU kübit sayısı 1 ile " + std::to_string(kMaxQubits) +
                                    " arasında olmalı: " + std::to_string(qubits));
    }
    if (qubits > kUnguardedQubits) {
        const char* allowed = std::getenv("QUANTIM_GPU_LARGE");
        if (allowed == nullptr || std::string(allowed) != "1") {
            throw std::invalid_argument(std::to_string(kUnguardedQubits) +
                                        " kübitin üstü için QUANTIM_GPU_LARGE=1 gerekli");
        }
    }
    const std::size_t bytes = (sizeof(double2) << qubits) + sizeof(double) * sumBlocks();
    std::size_t freeBytes = 0;
    std::size_t totalBytes = 0;
    check(cudaMemGetInfo(&freeBytes, &totalBytes), "bellek bilgisi");
    if (freeBytes < bytes + kReserveBytes) {
        throw std::runtime_error("VRAM yetersiz: " + std::to_string(bytes) + " bayt + " +
                                 std::to_string(kReserveBytes) + " bayt pay gerekli, boş " +
                                 std::to_string(freeBytes) + " bayt");
    }
    check(cudaMalloc(reinterpret_cast<void**>(&data_), sizeof(double2) << qubits), "cudaMalloc");
    try {
        check(cudaMalloc(reinterpret_cast<void**>(&partials_), sizeof(double) * sumBlocks()),
              "cudaMalloc");
        reset();
    } catch (...) {
        if (partials_ != nullptr) cudaFree(partials_);
        cudaFree(data_);
        throw;
    }
}

DeviceState::~DeviceState() {
    if (partials_ != nullptr) cudaFree(partials_);
    if (data_ != nullptr) cudaFree(data_);
}

DeviceState::DeviceState(DeviceState&& other) noexcept
    : qubits_(other.qubits_),
      data_(std::exchange(other.data_, nullptr)),
      partials_(std::exchange(other.partials_, nullptr)) {}

DeviceState& DeviceState::operator=(DeviceState&& other) noexcept {
    if (this != &other) {
        if (partials_ != nullptr) cudaFree(partials_);
        if (data_ != nullptr) cudaFree(data_);
        qubits_ = other.qubits_;
        data_ = std::exchange(other.data_, nullptr);
        partials_ = std::exchange(other.partials_, nullptr);
    }
    return *this;
}

std::size_t DeviceState::sumBlocks() const noexcept {
    return dimension() > kSumSpan ? dimension() / kSumSpan : 1;
}

void DeviceState::reset() {
    check(cudaMemset(data_, 0, sizeof(double2) << qubits_), "sıfırlama");
    const double one = 1.0;
    check(cudaMemcpy(data_, &one, sizeof(one), cudaMemcpyHostToDevice), "sıfırlama");
}

void DeviceState::upload(std::span<const std::complex<double>> amplitudes) {
    if (amplitudes.size() != dimension()) {
        throw std::invalid_argument("genlik sayısı durum boyuna eşit olmalı");
    }
    check(cudaMemcpy(data_, amplitudes.data(), sizeof(double2) * amplitudes.size(),
                     cudaMemcpyHostToDevice),
          "yükleme");
}

std::vector<std::complex<double>> DeviceState::download() const {
    std::vector<std::complex<double>> amplitudes(dimension());
    check(cudaMemcpy(amplitudes.data(), data_, sizeof(double2) * amplitudes.size(),
                     cudaMemcpyDeviceToHost),
          "indirme");
    return amplitudes;
}

std::complex<double> DeviceState::amplitude(std::size_t index) const {
    if (index >= dimension()) throw std::invalid_argument("genlik indeksi durum dışında");
    std::complex<double> value;
    check(cudaMemcpy(&value, data_ + 2 * index, sizeof(value), cudaMemcpyDeviceToHost),
          "genlik okuma");
    return value;
}

double DeviceState::normSquared() const {
    const std::size_t blocks = sumBlocks();
    const std::size_t span = dimension() / blocks;
    const auto* pairs = reinterpret_cast<const double2*>(data_);
    for (std::size_t first = 0; first < blocks; first += kSumBlocksPerLaunch) {
        const auto count = static_cast<unsigned>(std::min<std::size_t>(kSumBlocksPerLaunch,
                                                                      blocks - first));
        normKernel<<<count, kSumThreads>>>(pairs, first, span, partials_);
        check(cudaGetLastError(), "norm çekirdeği");
    }
    std::vector<double> partials(blocks);
    check(cudaMemcpy(partials.data(), partials_, sizeof(double) * blocks, cudaMemcpyDeviceToHost),
          "norm okuma");
    double total = 0.0;
    for (const double partial : partials) total += partial;
    return total;
}

void DeviceState::checkQubit(std::size_t qubit, std::size_t controls, const char* what) const {
    if (qubit >= qubits_ || (controls >> qubits_) != 0 || ((controls >> qubit) & 1u) != 0) {
        throw std::invalid_argument(std::string(what) + ": geçersiz kübit ya da kontrol");
    }
}

void DeviceState::applyMatrix(std::size_t target, std::size_t controls, const Matrix2& matrix) {
    checkQubit(target, controls, "matris");
    const std::size_t partner = std::size_t{1} << target;
    const Spread spread = spreadOver(partner | controls, controls);
    const Rotation m{split(matrix[0]), split(matrix[1]), split(matrix[2]), split(matrix[3])};
    launch(data_, itemsOf(qubits_, spread), rotateKernel, spread, partner, m);
}

void DeviceState::applyDiagonal(std::size_t target, std::size_t controls,
                                std::complex<double> zero, std::complex<double> one) {
    checkQubit(target, controls, "köşegen");
    const bool touchZero = !isOne(zero);
    const bool touchOne = !isOne(one);
    if (!touchZero && !touchOne) return;
    const std::size_t partner = std::size_t{1} << target;
    const Spread spread = spreadOver(partner | controls, controls);
    launch(data_, itemsOf(qubits_, spread), diagonalKernel, spread, partner, split(zero),
           split(one), touchZero, touchOne);
}

void DeviceState::applyExchange(std::size_t target, std::size_t controls) {
    checkQubit(target, controls, "değiştirme");
    const std::size_t partner = std::size_t{1} << target;
    const Spread spread = spreadOver(partner | controls, controls);
    launch(data_, itemsOf(qubits_, spread), exchangeKernel, spread, partner);
}

void DeviceState::applySwap(std::size_t first, std::size_t second, std::size_t controls,
                            std::complex<double> phase) {
    checkQubit(first, controls, "takas");
    checkQubit(second, controls, "takas");
    if (first == second) throw std::invalid_argument("takas: iki kübit aynı olamaz");
    const std::size_t firstMask = std::size_t{1} << first;
    const std::size_t secondMask = std::size_t{1} << second;
    const Spread spread = spreadOver(firstMask | secondMask | controls, controls | secondMask);
    launch(data_, itemsOf(qubits_, spread), swapKernel, spread, firstMask | secondMask,
           split(phase), isOne(phase));
}

void DeviceState::applyMatrix4(std::size_t first, std::size_t second, std::size_t controls,
                               const Matrix4& matrix) {
    checkQubit(first, controls, "4x4");
    checkQubit(second, controls, "4x4");
    if (first == second) throw std::invalid_argument("4x4: iki kübit aynı olamaz");
    const std::size_t firstMask = std::size_t{1} << first;
    const std::size_t secondMask = std::size_t{1} << second;
    const Spread spread = spreadOver(firstMask | secondMask | controls, controls);
    Quad quad{};
    for (std::size_t index = 0; index < 16; ++index) quad.m[index] = split(matrix[index]);
    quad.offsets[0] = 0;
    quad.offsets[1] = secondMask;
    quad.offsets[2] = firstMask;
    quad.offsets[3] = firstMask | secondMask;
    launch(data_, itemsOf(qubits_, spread), quadKernel, spread, quad);
}

Weights DeviceState::weigh(std::size_t qubit) const {
    checkQubit(qubit, 0, "ölçüm");
    const std::size_t span = std::min(dimension(), kSampleBlock);
    const std::size_t blocks = dimension() / span;
    Scratch<double2> partials(blocks);
    const auto* pairs = reinterpret_cast<const double2*>(data_);
    for (std::size_t first = 0; first < blocks; first += kSerialBlocksPerLaunch) {
        const std::size_t count = std::min(kSerialBlocksPerLaunch, blocks - first);
        const auto grid = static_cast<unsigned>((count + kSerialThreads - 1) / kSerialThreads);
        weighKernel<<<grid, kSerialThreads>>>(pairs, first, blocks, span, std::size_t{1} << qubit,
                                              partials.get());
        check(cudaGetLastError(), "ölçüm çekirdeği");
    }
    Weights weights;
    for (const double2 partial : partials.download()) {
        weights.total += partial.x;
        weights.one += partial.y;
    }
    return weights;
}

void DeviceState::collapse(std::size_t qubit, bool outcome, double scale) {
    checkQubit(qubit, 0, "ölçüm");
    auto* pairs = reinterpret_cast<double2*>(data_);
    const std::size_t mask = std::size_t{1} << qubit;
    for (std::size_t begin = 0; begin < dimension(); begin += 2 * kItemsPerLaunch) {
        const std::size_t count = std::min(2 * kItemsPerLaunch, dimension() - begin);
        const auto grid = static_cast<unsigned>((count + kBlock - 1) / kBlock);
        collapseKernel<<<grid, kBlock>>>(pairs, begin, count, mask, outcome, scale);
        check(cudaGetLastError(), "ölçüm çekirdeği");
    }
}

std::size_t DeviceState::measure(std::size_t qubit, std::mt19937_64& rng) {
    const Weights weights = weigh(qubit);
    if (!std::isfinite(weights.total) || std::abs(weights.total - 1.0) > kNormTolerance) {
        throw std::runtime_error("durum vektörünün normu bozuldu: " +
                                 std::to_string(weights.total));
    }
    const double probabilityOne = std::clamp(weights.one / weights.total, 0.0, 1.0);
    const bool outcome = std::generate_canonical<double, 53>(rng) < probabilityOne;
    const double kept = outcome ? weights.one : weights.total - weights.one;
    if (!(kept > 0.0)) throw std::runtime_error("ölçüm sonrası olasılık kalmadı");
    collapse(qubit, outcome, 1.0 / std::sqrt(kept));
    return outcome ? 1 : 0;
}

std::vector<std::size_t> DeviceState::sample(std::size_t shots, std::mt19937_64& rng) const {
    const std::size_t span = std::min(dimension(), kSampleBlock);
    const std::size_t blocks = dimension() / span;
    const auto* pairs = reinterpret_cast<const double2*>(data_);
    std::vector<double> sums;
    std::vector<unsigned long long> last;
    {
        Scratch<double> deviceSums(blocks);
        Scratch<unsigned long long> deviceLast(blocks);
        launchSerial(pairs, blocks, massKernel, blocks, span, deviceSums.get(), deviceLast.get());
        sums = deviceSums.download();
        last = deviceLast.download();
    }
    // Then everything the CPU does on the host, in the same order.
    std::vector<double> prefix(blocks + 1, 0.0);
    std::size_t finalBasis = 0;
    for (std::size_t block = 0; block < blocks; ++block) {
        prefix[block + 1] = prefix[block] + sums[block];
        finalBasis = std::max<std::size_t>(finalBasis, last[block]);
    }
    const double total = prefix.back();
    std::vector<double> samples(shots);
    for (double& value : samples) value = std::generate_canonical<double, 53>(rng) * total;
    std::sort(samples.begin(), samples.end());

    std::vector<SampleTask> tasks;
    std::size_t next = 0;
    for (std::size_t block = 0; block < blocks; ++block) {
        const std::size_t first = next;
        while (next < shots && samples[next] < prefix[block + 1]) ++next;
        if (next != first) tasks.push_back({block, first, next, prefix[block]});
    }

    std::vector<unsigned long long> bases(shots, finalBasis);
    if (!tasks.empty()) {
        Scratch<SampleTask> deviceTasks(tasks.size());
        Scratch<double> deviceSamples(shots);
        Scratch<unsigned long long> deviceBases(shots);
        deviceTasks.upload(tasks);
        deviceSamples.upload(samples);
        deviceBases.upload(bases);
        for (std::size_t first = 0; first < tasks.size(); first += kSerialBlocksPerLaunch) {
            const std::size_t count = std::min(kSerialBlocksPerLaunch, tasks.size() - first);
            const auto grid = static_cast<unsigned>((count + kSerialThreads - 1) / kSerialThreads);
            pickKernel<<<grid, kSerialThreads>>>(pairs, span, deviceTasks.get(), first, count,
                                                 deviceSamples.get(), deviceBases.get());
            check(cudaGetLastError(), "örnekleme çekirdeği");
        }
        bases = deviceBases.download();
    }
    return {bases.begin(), bases.end()};
}

void DeviceState::synchronize() const {
    check(cudaDeviceSynchronize(), "eşzamanlama");
}

}  // namespace quantim::gpu
