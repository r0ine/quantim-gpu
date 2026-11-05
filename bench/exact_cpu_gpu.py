# SPDX-License-Identifier: PolyForm-Small-Business-1.0.0
"""Compare quantim-gpu exactly with CPU quantim (Scalar, unfused) on the same circuits.

python3 bench/exact_cpu_gpu.py <cpu-quantim> <quantim-gpu> <work-directory>
"""

import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "extern" / "quantim" / "cpp" / "quantum" / "bench"))
import circuits  # noqa: E402


def to_qasm(n, gates):
    lines = ["OPENQASM 2.0;", 'include "qelib1.inc";', f"qreg q[{n}];", f"creg c[{n}];"]
    for name, qubits, params in gates:
        args = ", ".join(f"q[{q}]" for q in qubits)
        lines.append(f"{name}({', '.join(repr(p) for p in params)}) {args};" if params
                     else f"{name} {args};")
    lines.append("measure q -> c;")
    return "\n".join(lines) + "\n"


def run(binary, path, shots, seed, scalar):
    env = dict(os.environ)
    command = [binary, "run", str(path), "--shots", str(shots), "--seed", str(seed), "--json"]
    if scalar:
        env["QUANTIM_SIMD"] = "scalar"
        command.append("--no-fuse")
    out = subprocess.run(command, env=env, text=True, capture_output=True, check=True)
    return json.loads(out.stdout)


def main(cpu, gpu, work):
    work = Path(work)
    work.mkdir(parents=True, exist_ok=True)
    cases = []
    for family, build in circuits.SPEED_FAMILIES.items():
        for n in (5, 12, 20):
            path = work / f"{family}_{n}.qasm"
            path.write_text(to_qasm(n, build(n)))
            cases.append((f"{family} {n}", path))
    generator = ROOT / "tools" / "random_qasm.py"
    for family, n, seed in (("all", 6, 3), ("clifford", 10, 5), ("diag", 12, 8), ("ctrl", 14, 13),
                            ("all", 18, 21)):
        path = work / f"rq_{family}_{n}_{seed}.qasm"
        text = subprocess.run([sys.executable, str(generator), "--qubits", str(n), "--gates", "120",
                               "--seed", str(seed), "--measure", "--families", family],
                              text=True, capture_output=True, check=True).stdout
        path.write_text(text)
        cases.append((f"random_qasm {family} {n}", path))
    for name in ("bell", "ghz", "grover2", "qft3", "deutsch_jozsa2"):
        cases.append((f"örnek {name}",
                      ROOT / "extern" / "quantim" / "cpp" / "quantum" / "examples" / "qasm" / f"{name}.qasm"))

    failures = 0
    print("| devre | atış | CPU Scalar --no-fuse = GPU |")
    print("|---|---|---|")
    for label, path in cases:
        shots = 4096
        reference = run(cpu, path, shots, 7, True)
        observed = run(gpu, path, shots, 7, False)
        same = reference["counts"] == observed["counts"] and observed["shots"] == shots
        failures += not same
        print(f"| {label} | {shots} | {'evet' if same else 'HAYIR'} |", flush=True)
    print(f"{len(cases)} devre, {failures} fark")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main(*sys.argv[1:4]))
