# SPDX-License-Identifier: PolyForm-Small-Business-1.0.0
# Usage: bash tools/gpu_guard.sh <log-file> <benchmark-pid>
# Logs GPU status every two seconds; stops the given process above 85 °C.

set -u

if [[ $# -ne 2 || ! $2 =~ ^[0-9]+$ ]]; then
    echo "Kullanım: $0 <günlük-dosyası> <benchmark-pid>" >&2
    exit 2
fi

log_file=$1
benchmark_pid=$2

while kill -0 "$benchmark_pid" 2>/dev/null; do
    if ! reading=$(nvidia-smi --query-gpu=temperature.gpu,clocks.sm,clocks.mem,memory.used,power.draw --format=csv,noheader); then
        echo "nvidia-smi okunamadı" >&2
        exit 1
    fi

    printf '%s %s\n' "$(date --iso-8601=seconds)" "$reading" >> "$log_file"
    temperature=${reading%%,*}
    if [[ $temperature =~ ^[0-9]+$ ]] && (( temperature >= 85 )); then
        echo "GPU sıcaklığı ${temperature} °C; süreç ${benchmark_pid} durduruluyor" >> "$log_file"
        kill "$benchmark_pid"
        exit 0
    fi
    sleep 2
done
