// SPDX-License-Identifier: PolyForm-Small-Business-1.0.0
#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <cstdio>
#include <cstdlib>

#ifdef QUANTIM_HAVE_NVML
#include <nvml.h>
#endif

#define CHECK(call) do { \
    cudaError_t status = (call); \
    if (status != cudaSuccess) { \
        std::fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(status)); \
        std::exit(1); \
    } \
} while (false)

namespace {

constexpr size_t copyBytes = size_t{256} << 20;
constexpr size_t chunkElements = size_t{1} << 20;
constexpr int trials = 7;
constexpr int warmups = 3;

__global__ void copyKernel(double* destination, const double* source, size_t first, size_t count) {
    for (size_t offset = blockIdx.x * blockDim.x + threadIdx.x;
         offset < count; offset += gridDim.x * blockDim.x) {
        destination[first + offset] = source[first + offset];
    }
}

__global__ void triadKernel(double* destination, const double* source, size_t first, size_t count) {
    for (size_t offset = blockIdx.x * blockDim.x + threadIdx.x;
         offset < count; offset += gridDim.x * blockDim.x) {
        const size_t index = first + offset;
        destination[index] = source[index] + 0.5 * destination[index];
    }
}

template <class Launch>
double measure(Launch launch) {
    cudaEvent_t start;
    cudaEvent_t stop;
    CHECK(cudaEventCreate(&start));
    CHECK(cudaEventCreate(&stop));

    std::array<double, trials> elapsed{};
    for (int run = -warmups; run < trials; ++run) {
        CHECK(cudaEventRecord(start));
        launch();
        CHECK(cudaEventRecord(stop));
        CHECK(cudaEventSynchronize(stop));
        if (run >= 0) {
            float milliseconds = 0.0f;
            CHECK(cudaEventElapsedTime(&milliseconds, start, stop));
            elapsed[run] = milliseconds / 1000.0;
        }
    }

    CHECK(cudaEventDestroy(stop));
    CHECK(cudaEventDestroy(start));
    std::sort(elapsed.begin(), elapsed.end());
    return elapsed[trials / 2];
}

void printBandwidth(const char* label, double bytes, double seconds) {
    std::printf("%s: %.2f GB/s\n", label, bytes / seconds / 1e9);
}

void printTelemetry() {
#ifdef QUANTIM_HAVE_NVML
    if (nvmlInit_v2() == NVML_SUCCESS) {
        nvmlDevice_t device;
        unsigned int temperature;
        unsigned int smClock;
        unsigned int memoryClock;
        if (nvmlDeviceGetHandleByIndex_v2(0, &device) == NVML_SUCCESS) {
            if (nvmlDeviceGetTemperature(device, NVML_TEMPERATURE_GPU, &temperature) == NVML_SUCCESS) {
                std::printf("sıcaklık: %u °C\n", temperature);
            } else {
                std::printf("sıcaklık: okunamadı\n");
            }
            if (nvmlDeviceGetClockInfo(device, NVML_CLOCK_SM, &smClock) == NVML_SUCCESS) {
                std::printf("SM saati: %u MHz\n", smClock);
            }
            if (nvmlDeviceGetClockInfo(device, NVML_CLOCK_MEM, &memoryClock) == NVML_SUCCESS) {
                std::printf("bellek saati (NVML): %u MHz\n", memoryClock);
            }
        } else {
            std::printf("sıcaklık: okunamadı\n");
        }
        nvmlShutdown();
        return;
    }
#endif
    std::printf("sıcaklık: okunamadı\n");
}

void launchCopy(double* destination, const double* source, size_t count) {
    constexpr int threads = 256;
    for (size_t first = 0; first < count; first += chunkElements) {
        const size_t chunk = std::min(chunkElements, count - first);
        const int blocks = static_cast<int>((chunk + threads - 1) / threads);
        copyKernel<<<blocks, threads>>>(destination, source, first, chunk);
        CHECK(cudaGetLastError());
    }
}

void launchTriad(double* destination, const double* source, size_t count) {
    constexpr int threads = 256;
    for (size_t first = 0; first < count; first += chunkElements) {
        const size_t chunk = std::min(chunkElements, count - first);
        const int blocks = static_cast<int>((chunk + threads - 1) / threads);
        triadKernel<<<blocks, threads>>>(destination, source, first, chunk);
        CHECK(cudaGetLastError());
    }
}

} // namespace

int main() {
    int device = 0;
    CHECK(cudaGetDevice(&device));
    cudaDeviceProp properties{};
    CHECK(cudaGetDeviceProperties(&properties, device));

    size_t freeBytes = 0;
    size_t totalBytes = 0;
    CHECK(cudaMemGetInfo(&freeBytes, &totalBytes));
    int memoryClock = 0;
    int busWidth = 0;
    CHECK(cudaDeviceGetAttribute(&memoryClock, cudaDevAttrMemoryClockRate, device));
    CHECK(cudaDeviceGetAttribute(&busWidth, cudaDevAttrGlobalMemoryBusWidth, device));
    int driverVersion = 0;
    int runtimeVersion = 0;
    CHECK(cudaDriverGetVersion(&driverVersion));
    CHECK(cudaRuntimeGetVersion(&runtimeVersion));

    std::printf("aygıt: %s\n", properties.name);
    std::printf("hesaplama yeteneği: %d.%d\n", properties.major, properties.minor);
    std::printf("SM sayısı: %d\n", properties.multiProcessorCount);
    std::printf("toplam VRAM: %.2f GiB\n", double(totalBytes) / (1ull << 30));
    std::printf("boş VRAM: %.2f GiB\n", double(freeBytes) / (1ull << 30));
    std::printf("bellek saati: %.0f MHz\n", memoryClock / 1000.0);
    std::printf("bellek veri yolu: %d bit\n", busWidth);
    std::printf("kuramsal bant genişliği: %.2f GB/s\n",
                double(memoryClock) * 1000.0 * 2.0 * busWidth / 8.0 / 1e9);
    std::printf("sürücü sürümü: %d.%d\n", driverVersion / 1000, (driverVersion % 1000) / 10);
    std::printf("CUDA çalışma zamanı: %d.%d\n", runtimeVersion / 1000, (runtimeVersion % 1000) / 10);
    printTelemetry();

    constexpr size_t largestBytes = (size_t{1} << 26) * sizeof(double);
    constexpr size_t neededBytes = 2 * largestBytes;
    if (freeBytes < neededBytes + (size_t{1} << 30)) {
        std::fprintf(stderr, "ölçüm için 1 GiB aygıt belleği ve 1 GiB güvenlik payı gerekli\n");
        return 1;
    }

    void* pinned = nullptr;
    double* source = nullptr;
    double* destination = nullptr;
    CHECK(cudaHostAlloc(&pinned, copyBytes, cudaHostAllocDefault));
    CHECK(cudaMalloc(&source, largestBytes));
    CHECK(cudaMalloc(&destination, largestBytes));
    CHECK(cudaMemset(source, 0, largestBytes));
    CHECK(cudaMemset(destination, 0, largestBytes));

    auto* hostBytes = static_cast<unsigned char*>(pinned);
    std::fill_n(hostBytes, copyBytes, 0);
    const auto copyChunks = [](void* target, const void* origin, cudaMemcpyKind direction) {
        constexpr size_t transferBytes = size_t{32} << 20;
        for (size_t first = 0; first < copyBytes; first += transferBytes) {
            CHECK(cudaMemcpyAsync(static_cast<unsigned char*>(target) + first,
                                  static_cast<const unsigned char*>(origin) + first,
                                  transferBytes, direction));
        }
    };

    printBandwidth("host → aygıt kopyası", copyBytes,
                   measure([&] { copyChunks(source, pinned, cudaMemcpyHostToDevice); }));
    printBandwidth("aygıt → host kopyası", copyBytes,
                   measure([&] { copyChunks(pinned, source, cudaMemcpyDeviceToHost); }));

    for (int exponent : {24, 26}) {
        const size_t count = size_t{1} << exponent;
        const double bytes = double(count) * sizeof(double);
        char label[80];
        std::snprintf(label, sizeof(label), "aygıt kopyası (2^%d double)", exponent);
        printBandwidth(label, 2.0 * bytes,
                       measure([&] { launchCopy(destination, source, count); }));
        std::snprintf(label, sizeof(label), "triad okuma+yazma (2^%d double)", exponent);
        printBandwidth(label, 3.0 * bytes,
                       measure([&] { launchTriad(destination, source, count); }));
    }

    CHECK(cudaFree(destination));
    CHECK(cudaFree(source));
    CHECK(cudaFreeHost(pinned));
    return 0;
}
