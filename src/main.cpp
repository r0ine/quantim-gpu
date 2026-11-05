// SPDX-License-Identifier: PolyForm-Small-Business-1.0.0
// quantim-gpu run <file.qasm | -> [--shots N] [--seed S] [--json] [--no-fuse] [--device cuda]
// Output matches `quantim run`; the device goes to standard error and, with
// --json, into a "device" field.
#include "runner.hpp"

#include <algorithm>
#include <charconv>
#include <cstdint>
#include <exception>
#include <iomanip>
#include <iostream>
#include <new>
#include <sstream>
#include <stdexcept>
#include <string>
#include <system_error>
#include <utility>
#include <vector>

namespace {

using quantim::quantum::qasm::Circuit;
using quantim::quantum::qasm::RunResult;

constexpr const char* kVersion = "0.1.0";

struct Options {
    std::string path;
    std::size_t shots = 1024;
    std::uint64_t seed = 20261022;
    bool json = false;
};

void printUsage() {
    std::cout << "Kullanım:\n"
              << "  quantim-gpu run <dosya.qasm> [--shots N] [--seed S] [--json] [--no-fuse]"
                 " [--device cuda]\n"
              << "  quantim-gpu run - [...]  (devre standart girdiden okunur)\n"
              << "  quantim-gpu --version\n";
}

std::uint64_t parseNumber(const std::string& text, const std::string& what) {
    std::uint64_t value = 0;
    const auto result = std::from_chars(text.data(), text.data() + text.size(), value);
    if (result.ec != std::errc{} || result.ptr != text.data() + text.size()) {
        throw std::invalid_argument(what + " negatif olmayan bir tam sayı olmalı: " + text);
    }
    return value;
}

void printHistogram(const RunResult& result) {
    std::vector<std::pair<std::string, std::size_t>> rows(result.counts.begin(),
                                                          result.counts.end());
    std::sort(rows.begin(), rows.end(), [](const auto& left, const auto& right) {
        if (left.second != right.second) return left.second > right.second;
        return left.first < right.first;
    });
    std::cout << "atış: " << result.shots << "  tohum: " << result.seed << '\n';
    std::cout << std::fixed << std::setprecision(4);
    for (const auto& row : rows) {
        const double share = result.shots == 0 ? 0.0
                                               : static_cast<double>(row.second) /
                                                     static_cast<double>(result.shots);
        std::cout << row.first << "  " << row.second << "  " << share << '\n';
    }
    std::cout.unsetf(std::ios::floatfield);
}

void printJson(const RunResult& result, const std::string& device) {
    std::cout << "{\"shots\":" << result.shots << ",\"seed\":" << result.seed
              << ",\"device\":\"" << device << "\",\"counts\":{";
    bool first = true;
    for (const auto& [key, count] : result.counts) {
        std::cout << (first ? "" : ",") << '"' << key << "\":" << count;
        first = false;
    }
    std::cout << "}}\n";
}

}  // namespace

int main(int argc, char** argv) {
    const std::vector<std::string> arguments(argv + 1, argv + argc);
    if (arguments.empty()) {
        printUsage();
        return 2;
    }
    if (arguments.front() == "--version" || arguments.front() == "version") {
        std::cout << "quantim-gpu " << kVersion << " (PolyForm-Small-Business-1.0.0)\n";
        return 0;
    }
    if (arguments.front() == "--help" || arguments.front() == "-h" || arguments.front() == "help") {
        printUsage();
        return 0;
    }
    if (arguments.front() != "run") {
        std::cerr << "Hata: bilinmeyen komut: " << arguments.front() << '\n';
        printUsage();
        return 2;
    }

    Options options;
    for (std::size_t index = 1; index < arguments.size(); ++index) {
        const std::string& argument = arguments[index];
        if (argument == "--help" || argument == "-h") {
            printUsage();
            return 0;
        }
        if (argument == "--json") {
            options.json = true;
            continue;
        }
        // There is no fusion on the GPU yet, so --no-fuse changes nothing.
        if (argument == "--no-fuse") continue;
        if (argument == "--shots" || argument == "--seed" || argument == "--device") {
            if (index + 1 >= arguments.size()) {
                std::cerr << "Hata: " << argument << " için bir değer gerekli\n";
                return 2;
            }
            const std::string& value = arguments[++index];
            try {
                if (argument == "--device") {
                    if (value != "cuda") {
                        throw std::invalid_argument("quantim-gpu yalnız --device cuda destekler; "
                                                    "CPU için quantim kullan");
                    }
                } else if (argument == "--shots") {
                    options.shots = static_cast<std::size_t>(parseNumber(value, argument));
                } else {
                    options.seed = parseNumber(value, argument);
                }
            } catch (const std::exception& error) {
                std::cerr << "Hata: " << error.what() << '\n';
                return 2;
            }
            continue;
        }
        if (argument != "-" && !argument.empty() && argument.front() == '-') {
            std::cerr << "Hata: bilinmeyen seçenek: " << argument << '\n';
            printUsage();
            return 2;
        }
        if (!options.path.empty()) {
            std::cerr << "Hata: yalnızca bir devre dosyası verilebilir\n";
            return 2;
        }
        options.path = argument;
    }
    if (options.path.empty()) {
        std::cerr << "Hata: devre dosyası gerekli\n";
        printUsage();
        return 2;
    }

    try {
        Circuit circuit;
        if (options.path == "-") {
            std::ostringstream buffer;
            buffer << std::cin.rdbuf();
            if (std::cin.bad()) throw std::runtime_error("dosya okunamadı: <stdin>");
            circuit = quantim::quantum::qasm::parseString(buffer.str());
            circuit.name = "<stdin>";
        } else {
            circuit = quantim::quantum::qasm::parseFile(options.path);
        }
        const quantim::gpu::DeviceInfo info = quantim::gpu::deviceInfo();
        std::cerr << "aygıt: cuda:0 " << info.name << " (CC " << info.major << '.' << info.minor
                  << "), boş VRAM " << (info.freeBytes >> 20) << " / " << (info.totalBytes >> 20)
                  << " MiB\n";
        std::cerr << "durum vektörü belleği: "
                  << ((std::uint64_t{1} << circuit.qubitCount) * 16) << " bayt\n";
        const RunResult result = quantim::gpu::runShots(circuit, options.shots, options.seed);
        if (options.json) {
            printJson(result, "cuda:0 " + info.name);
        } else {
            printHistogram(result);
        }
        return 0;
    } catch (const std::bad_alloc&) {
        std::cerr << "Hata: devre için bellek ayrılamadı\n";
        return 1;
    } catch (const std::exception& error) {
        std::cerr << "Hata: " << error.what() << '\n';
        return 1;
    }
}
