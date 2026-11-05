# SPDX-License-Identifier: PolyForm-Small-Business-1.0.0
"""Compare CPU and GPU CLI output on the same QASM circuits."""

from __future__ import annotations

import argparse
import json
import math
import re
import subprocess
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
GENERATOR = ROOT / "tools" / "random_qasm.py"
EXAMPLES = ROOT / "extern" / "quantim" / "cpp" / "quantum" / "examples" / "qasm"
FAMILIES = {"all", "clifford", "diag", "ctrl"}


def cases_from(text: str) -> list[tuple[str, int, int]]:
    cases = []
    for entry in text.split(","):
        parts = entry.strip().split(":")
        if len(parts) != 3 or parts[0] not in FAMILIES:
            raise ValueError(f"geçersiz devre: {entry!r}; biçim aile:kübit:tohum")
        try:
            qubits, seed = map(int, parts[1:])
        except ValueError as error:
            raise ValueError(f"geçersiz kübit veya tohum: {entry!r}") from error
        if not 1 <= qubits <= 28 or seed < 0:
            raise ValueError(f"kübit 1–28, tohum sıfır veya pozitif olmalı: {entry!r}")
        cases.append((parts[0], qubits, seed))
    return cases


def measured_qubits(source: str) -> int:
    match = re.search(r"\bqreg\s+\w+\[(\d+)\]\s*;", source)
    if not match:
        raise ValueError("örnekte qreg bulunamadı")
    return int(match.group(1))


def read_run(binary: str, circuit: Path, shots: int, seed: int) -> dict:
    command = [binary, "run", str(circuit), "--shots", str(shots), "--seed", str(seed), "--json"]
    completed = subprocess.run(command, text=True, capture_output=True, check=True)
    payload = json.loads(completed.stdout)
    if not isinstance(payload, dict) or payload.get("shots") != shots or payload.get("seed") != seed:
        raise ValueError("JSON içindeki shots/seed istenen değerlerle uyuşmuyor")
    counts = payload.get("counts")
    if not isinstance(counts, dict) or any(
        not isinstance(key, str) or type(value) is not int or value < 0
        for key, value in counts.items()
    ) or sum(counts.values()) != shots:
        raise ValueError("JSON counts alanı geçersiz veya atış toplamı farklı")
    return payload


def samples_distance(left: dict, right: dict, shots: int) -> tuple[float, float, float, float]:
    counts_a, counts_b = left["counts"], right["counts"]
    outcomes = counts_a.keys() | counts_b.keys()
    tvd = sum(abs(counts_a.get(key, 0) - counts_b.get(key, 0)) for key in outcomes) / (2 * shots)
    chi_square = sum(
        (counts_a.get(key, 0) - counts_b.get(key, 0)) ** 2
        / (counts_a.get(key, 0) + counts_b.get(key, 0))
        for key in outcomes
    )
    degrees = max(1, len(outcomes) - 1)
    tail = math.log(1000)
    chi_limit = degrees + 2 * math.sqrt(degrees * tail) + 2 * tail
    tvd_limit = min(1.0, math.sqrt(chi_limit / (2 * shots)))
    return tvd, tvd_limit, chi_square, chi_limit


def numeric_entries(values: object, complex_values: bool) -> dict[str, complex | float]:
    if isinstance(values, list):
        entries = {str(index): value for index, value in enumerate(values)}
    elif isinstance(values, dict):
        entries = values
    else:
        raise ValueError("olasılık/genlik alanı dizi veya nesne olmalı")

    parsed = {}
    for key, value in entries.items():
        if not isinstance(key, str):
            raise ValueError("olasılık/genlik anahtarı metin olmalı")
        if complex_values:
            try:
                if isinstance(value, list) and len(value) == 2:
                    value = complex(*value)
                elif isinstance(value, dict) and {"real", "imag"} <= value.keys():
                    value = complex(value["real"], value["imag"])
                else:
                    raise ValueError("genlik [gerçel, sanal] biçiminde olmalı")
            except (TypeError, OverflowError) as error:
                raise ValueError("geçersiz genlik") from error
            if not math.isfinite(value.real) or not math.isfinite(value.imag):
                raise ValueError("sonlu olmayan genlik")
        else:
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError("geçersiz olasılık")
        parsed[key] = value
    return parsed


def exact_distance(left: dict, right: dict) -> float | None:
    differences = []
    for field in ("probabilities", "amplitudes"):
        if field not in left or field not in right:
            continue
        values_a = numeric_entries(left[field], field == "amplitudes")
        values_b = numeric_entries(right[field], field == "amplitudes")
        if values_a.keys() != values_b.keys():
            raise ValueError(f"{field} indisleri uyuşmuyor")
        differences.append(max((abs(values_a[key] - values_b[key]) for key in values_a), default=0.0))
    return max(differences) if differences else None


def main() -> int:
    parser = argparse.ArgumentParser(description="Quantim CPU/GPU histogramlarını karşılaştırır.")
    parser.add_argument("--cpu-bin", default="quantim", help="CPU ikilisi (varsayılan: quantim)")
    parser.add_argument("--gpu-bin", default="quantim-gpu", help="GPU ikilisi (varsayılan: quantim-gpu)")
    parser.add_argument("--cases", default="all:3:7,clifford:5:11,diag:6:13,ctrl:6:17",
                        help="aile:kübit:tohum listesi")
    parser.add_argument("--gates", type=int, default=80, help="üretilen devrede kapı sayısı")
    parser.add_argument("--shots", type=int, default=4096, help="devre başına atış sayısı")
    parser.add_argument("--seed", type=int, default=7, help="örnek QASM devrelerinin ölçüm tohumu")
    parser.add_argument("--tol-prob", type=float, default=2e-9, help="tam olasılık/genlik farkı sınırı")
    args = parser.parse_args()
    try:
        cases = cases_from(args.cases)
    except ValueError as error:
        parser.error(str(error))
    if args.gates < 0 or args.shots < 1 or args.seed < 0 or not math.isfinite(args.tol_prob) or args.tol_prob < 0:
        parser.error("kapı/tohum negatif, atış sıfır veya tolerans geçersiz olamaz")

    failed = False
    print("| Devre | n | TVD / sınır | χ² / sınır | En büyük tam fark | Durum |")
    print("|---|---:|---:|---:|---:|---|")
    with tempfile.TemporaryDirectory(prefix="quantim-compare-") as workdir:
        circuits = []
        for family, qubits, seed in cases:
            circuits.append((f"{family}:{qubits}:{seed}", qubits, seed,
                             Path(workdir) / f"{family}_{qubits}_{seed}.qasm", family))
        if EXAMPLES.is_dir():
            for path in sorted(EXAMPLES.glob("*.qasm")):
                try:
                    qubits = measured_qubits(path.read_text(encoding="utf-8"))
                except ValueError as error:
                    print(f"Örnek {path.name}: {error}", file=sys.stderr)
                    failed = True
                    continue
                circuits.append((path.name, qubits, args.seed, path, None))

        for name, qubits, seed, path, family in circuits:
            try:
                if family is not None:
                    generated = subprocess.run(
                        [sys.executable, str(GENERATOR), "--qubits", str(qubits), "--gates",
                         str(args.gates), "--seed", str(seed), "--measure", "--families", family],
                        text=True, capture_output=True, check=True,
                    )
                    path.write_text(generated.stdout, encoding="utf-8")
                cpu = read_run(args.cpu_bin, path, args.shots, seed)
                gpu = read_run(args.gpu_bin, path, args.shots, seed)
                tvd, tvd_limit, chi_square, chi_limit = samples_distance(cpu, gpu, args.shots)
                exact = exact_distance(cpu, gpu)
                passed = (tvd <= tvd_limit and chi_square <= chi_limit
                          and (exact is None or exact <= args.tol_prob))
                failed |= not passed
                exact_text = f"{exact:.3g}" if exact is not None else "—"
                status = "geçti" if passed else "aştı"
                print(f"| {name} | {qubits} | {tvd:.4g} / {tvd_limit:.4g} | "
                      f"{chi_square:.4g} / {chi_limit:.4g} | {exact_text} | {status} |", flush=True)
            except (OSError, subprocess.CalledProcessError, ValueError, json.JSONDecodeError) as error:
                failed = True
                print(f"| {name} | {qubits} | — | — | — | çalışmadı |", flush=True)
                print(f"{name}: {error}", file=sys.stderr)
    print("χ² sınırı: gözlenen destek için yaklaşık 0,001 kuyruk eşiği; seyrek sonuçlarda kesin test değildir.")
    return int(failed)


if __name__ == "__main__":
    raise SystemExit(main())
