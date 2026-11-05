# SPDX-License-Identifier: PolyForm-Small-Business-1.0.0
"""Time the Quantim CPU and GPU CLIs in alternating rounds."""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import runpy
import shutil
import statistics
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CIRCUITS = ROOT / "extern" / "quantim" / "cpp" / "quantum" / "bench" / "circuits.py"
GENERATOR = ROOT / "tools" / "random_qasm.py"
FAMILIES = ("layered", "qft", "ghz", "random")
FALLBACK = {"layered": "all", "qft": "diag", "ghz": "clifford", "random": "all"}
TELEMETRY = "temperature.gpu,clocks.sm,memory.used,power.draw"


def sizes_from(text: str) -> list[int]:
    try:
        sizes = [int(part.strip()) for part in text.split(",")]
    except ValueError as error:
        raise ValueError("--sizes virgülle ayrılmış tam sayılar içermeli") from error
    if not sizes or any(size < 3 or size > 28 for size in sizes):
        raise ValueError("kübit sayıları 3–28 arasında olmalı")
    return sizes


def binary_name(value: str) -> str:
    return str(Path(value).resolve()) if os.sep in value or "/" in value else value


def qasm_source(qubits: int, gates: list) -> str:
    lines = ["OPENQASM 2.0;", 'include "qelib1.inc";', f"qreg q[{qubits}];",
             f"creg c[{qubits}];"]
    for name, operands, angles in gates:
        parameters = f"({', '.join(repr(angle) for angle in angles)})" if angles else ""
        targets = ", ".join(f"q[{qubit}]" for qubit in operands)
        lines.append(f"{name}{parameters} {targets};")
    lines.append("measure q -> c;")
    return "\n".join(lines) + "\n"


def make_circuit(family: str, qubits: int, path: Path, recipes: dict | None) -> int:
    if recipes is not None:
        gates = recipes[family](qubits)
        path.write_text(qasm_source(qubits, gates), encoding="utf-8")
        return len(gates)
    gate_count = 20 * qubits
    generated = subprocess.run(
        [sys.executable, str(GENERATOR), "--qubits", str(qubits), "--gates", str(gate_count),
         "--seed", str(qubits), "--measure", "--families", FALLBACK[family]],
        text=True, capture_output=True, check=True,
    )
    path.write_text(generated.stdout, encoding="utf-8")
    return gate_count


def gpu_reading(smi: str, environment: dict) -> dict:
    command = [smi, f"--query-gpu={TELEMETRY}", "--format=csv,noheader"]
    completed = subprocess.run(command, text=True, capture_output=True, check=True, env=environment)
    lines = completed.stdout.strip().splitlines()
    if len(lines) != 1:
        raise ValueError("tek GPU için bir telemetri satırı bekleniyordu")
    parts = lines[0].split(",")
    if len(parts) != 4:
        raise ValueError(f"nvidia-smi çıktısı çözülemedi: {lines[0]}")
    numbers = []
    for part in parts:
        match = re.search(r"[-+]?\d+(?:\.\d+)?", part)
        if not match:
            raise ValueError(f"nvidia-smi değeri çözülemedi: {part}")
        numbers.append(float(match.group()))
    return dict(zip(("temperature_c", "clock_mhz", "memory_mib", "power_w"), numbers))


def timed_run(command: list[str], environment: dict) -> float:
    start = time.perf_counter()
    completed = subprocess.run(command, text=True, capture_output=True, env=environment)
    elapsed = time.perf_counter() - start
    if completed.returncode:
        detail = completed.stderr.strip() or completed.stdout.strip() or "çıktı yok"
        raise RuntimeError(f"{command[0]} çıkış kodu {completed.returncode}: {detail}")
    return elapsed


def measured(samples: list[float]) -> str:
    return f"{statistics.median(samples):.4f}"


def span(samples: list[float]) -> str:
    return f"{min(samples):.4f}–{max(samples):.4f}"


def main() -> int:
    parser = argparse.ArgumentParser(description="CPU/GPU dönüşümlü süre ölçümü yapar.")
    parser.add_argument("--cpu-bin", default="quantim", help="CPU ikilisi")
    parser.add_argument("--gpu-bin", default="quantim-gpu", help="GPU ikilisi")
    parser.add_argument("--sizes", default="20,22,24", help="virgülle ayrılmış kübit sayıları")
    parser.add_argument("--rounds", type=int, default=5, help="ölçülen tur sayısı")
    parser.add_argument("--warmup", type=int, default=1, help="ısınma turu sayısı")
    parser.add_argument("--cooldown", type=float, default=10, help="turlar arası saniye")
    parser.add_argument("--shots", type=int, default=1, help="çalıştırma başına atış")
    parser.add_argument("--seed", type=int, default=7, help="ölçüm tohumu")
    parser.add_argument("--log", type=Path, help="JSONL günlük yolu")
    args = parser.parse_args()
    try:
        sizes = sizes_from(args.sizes)
    except ValueError as error:
        parser.error(str(error))
    if args.rounds < 1 or args.warmup < 0 or args.cooldown < 0 or not math.isfinite(args.cooldown):
        parser.error("tur sayısı pozitif, ısınma ve soğuma süresi negatif olmayan sayı olmalı")
    if args.shots < 1 or args.seed < 0:
        parser.error("atış pozitif, tohum sıfır veya pozitif olmalı")

    recipes = runpy.run_path(str(CIRCUITS))["SPEED_FAMILIES"] if CIRCUITS.is_file() else None
    source = str(CIRCUITS.relative_to(ROOT)) if recipes is not None else (
        "tools/random_qasm.py (layered→all, qft→diag, ghz→clifford, random→all; "
        "özgün aile devreleri değil)"
    )
    environment = os.environ.copy()
    if not shutil.which("nvidia-smi", path=environment.get("PATH")):
        environment["PATH"] = f"/usr/lib/wsl/lib{os.pathsep}{environment.get('PATH', '')}"
    smi = shutil.which("nvidia-smi", path=environment["PATH"])
    cpu_bin, gpu_bin = binary_name(args.cpu_bin), binary_name(args.gpu_bin)
    log_path = args.log or ROOT / "bench" / f"ab_{datetime.now():%Y%m%d_%H%M%S_%f}.jsonl"
    log_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Devre kaynağı: {source}")
    print(f"Günlük: {log_path}")
    print("Süreler saniye; oranlar CPU/GPU. Süreye süreç açılışı ve QASM okuma dahildir.")
    print("| Aile | n | Kapı | CPU 1 iş parçacığı | CPU 4 iş parçacığı | GPU | "
          "CPU 1/GPU | CPU 4/GPU | CPU 1 min–max | CPU 4 min–max | GPU min–max |")
    print("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    cases = [(family, qubits) for family in FAMILIES for qubits in sizes]
    aborted = False
    with log_path.open("w", encoding="utf-8") as log, tempfile.TemporaryDirectory(prefix="quantim-ab-") as workdir:
        log.write(json.dumps({"type": "setup", "time_utc": datetime.now(timezone.utc).isoformat(),
                              "source": source, "cpu_bin": cpu_bin, "gpu_bin": gpu_bin,
                              "sizes": sizes, "rounds": args.rounds, "warmup": args.warmup,
                              "cooldown_s": args.cooldown, "shots": args.shots, "seed": args.seed,
                              "fusion": "CLI varsayılanı", "norm": "CLI varsayılanı"},
                             ensure_ascii=False) + "\n")
        log.flush()
        for family, qubits in cases:
            if aborted:
                print(f"| {family} | {qubits} | — | " + " | ".join(["çalışmadı"] * 8) + " |")
                continue
            gate_count = None
            try:
                if smi is None:
                    raise RuntimeError("nvidia-smi bulunamadı")
                path = Path(workdir) / f"{family}_{qubits}.qasm"
                gate_count = make_circuit(family, qubits, path, recipes)
                commands = {
                    "cpu1": [cpu_bin, "run", str(path), "--shots", str(args.shots),
                             "--seed", str(args.seed), "--threads", "1"],
                    "cpu4": [cpu_bin, "run", str(path), "--shots", str(args.shots),
                             "--seed", str(args.seed), "--threads", "4"],
                    "gpu": [gpu_bin, "run", str(path), "--shots", str(args.shots), "--seed", str(args.seed)],
                }
                samples = {"cpu1": [], "cpu4": [], "gpu": []}
                for turn in range(-args.warmup, args.rounds):
                    order = ("cpu1", "gpu", "gpu", "cpu4")
                    if turn % 2:
                        order = tuple(reversed(order))
                    for position, target in enumerate(order):
                        before = gpu_reading(smi, environment) if target == "gpu" else None
                        if before is not None:
                            log.write(json.dumps({"type": "gpu_before", "family": family,
                                                  "qubits": qubits, "turn": turn,
                                                  "position": position, **before}) + "\n")
                            log.flush()
                        if before is not None and before["temperature_c"] >= 80:
                            raise RuntimeError(f"GPU sıcaklığı {before['temperature_c']:g} °C")
                        elapsed = timed_run(commands[target], environment)
                        after = gpu_reading(smi, environment) if target == "gpu" else None
                        record = {"type": "run", "family": family, "qubits": qubits,
                                  "gates": gate_count, "turn": turn, "warmup": turn < 0,
                                  "position": position, "target": target, "seconds": elapsed,
                                  "gpu_before": before, "gpu_after": after}
                        log.write(json.dumps(record, ensure_ascii=False) + "\n")
                        log.flush()
                        if after is not None and after["temperature_c"] >= 80:
                            raise RuntimeError(f"GPU sıcaklığı {after['temperature_c']:g} °C")
                        if turn >= 0:
                            samples[target].append(elapsed)
                    if turn < args.rounds - 1:
                        time.sleep(args.cooldown)
                cpu1, cpu4, gpu = (samples[name] for name in ("cpu1", "cpu4", "gpu"))
                values = [family, str(qubits), str(gate_count), measured(cpu1), measured(cpu4),
                          measured(gpu), f"{statistics.median(cpu1) / statistics.median(gpu):.2f}",
                          f"{statistics.median(cpu4) / statistics.median(gpu):.2f}",
                          span(cpu1), span(cpu4), span(gpu)]
                print("| " + " | ".join(values) + " |", flush=True)
            except (OSError, ValueError, KeyError, RuntimeError, subprocess.CalledProcessError) as error:
                aborted = True
                print(f"| {family} | {qubits} | {gate_count if gate_count is not None else '—'} | "
                      + " | ".join(["çalışmadı"] * 8) + " |", flush=True)
                log.write(json.dumps({"type": "error", "family": family, "qubits": qubits,
                                      "message": str(error)}, ensure_ascii=False) + "\n")
                log.flush()
                print(f"Ölçüm durdu ({family}, {qubits}): {error}", file=sys.stderr)
    return int(aborted)


if __name__ == "__main__":
    raise SystemExit(main())
