# SPDX-License-Identifier: PolyForm-Small-Business-1.0.0
"""Generate random OpenQASM 2.0 circuits accepted by the qasm.cpp parser.

Uses only the Python standard library. A given seed produces the same output.
Gate names come from the built-in table in extern/quantim/cpp/quantum/src/qasm.cpp.
"""

from __future__ import annotations

import argparse
import random

# StateVector::kMaxQubits in qasm.cpp
MAX_QUBITS = 28

SINGLE = ("h", "x", "y", "z", "s", "sdg", "t", "tdg", "sx")
SINGLE_ANGLE = ("rx", "ry", "rz", "p", "u3")
TWO_PLAIN = ("cx", "cz", "swap", "ch")
TWO_ANGLE = ("cp", "crz", "cry")
THREE = ("ccx",)

ALL_GATES = SINGLE + SINGLE_ANGLE + TWO_PLAIN + TWO_ANGLE + THREE

# Gate name -> (qubit count, angle parameter count)
SHAPE = {
    "h": (1, 0),
    "x": (1, 0),
    "y": (1, 0),
    "z": (1, 0),
    "s": (1, 0),
    "sdg": (1, 0),
    "t": (1, 0),
    "tdg": (1, 0),
    "sx": (1, 0),
    "rx": (1, 1),
    "ry": (1, 1),
    "rz": (1, 1),
    "p": (1, 1),
    "u3": (1, 3),
    "cx": (2, 0),
    "cz": (2, 0),
    "swap": (2, 0),
    "ch": (2, 0),
    "cp": (2, 1),
    "crz": (2, 1),
    "cry": (2, 1),
    "ccx": (3, 0),
}

CLIFFORD = ("h", "x", "y", "z", "s", "sdg", "sx", "cx", "cz", "swap", "ch")
DIAGONAL = ("z", "s", "sdg", "t", "tdg", "rz", "p", "cz", "cp", "crz")
CONTROLLED = ("cx", "cz", "ch", "cp", "crz", "cry", "ccx")

FAMILIES = {
    "all": ALL_GATES,
    "clifford": CLIFFORD,
    "diag": DIAGONAL,
    "ctrl": CONTROLLED,
}

# Favor single-qubit gates; make three-qubit gates less frequent.
ARITY_WEIGHT = {1: 4, 2: 3, 3: 2}

# The qasm.cpp expression parser supports + - * /, parentheses, and pi.
PI_FORMS = (
    "pi/8",
    "pi/6",
    "pi/4",
    "pi/3",
    "pi/2",
    "2*pi/3",
    "3*pi/4",
    "5*pi/8",
    "7*pi/4",
)


def _angle(rng: random.Random) -> str:
    """Generate an angle from the seed, usually as a pi expression."""
    if rng.random() < 0.6:
        sign = "-" if rng.random() < 0.3 else ""
        return sign + rng.choice(PI_FORMS)
    return f"{rng.uniform(-2.0, 2.0):.6f}"


def _target(rng: random.Random, qubits: int) -> int:
    """Choose a target, favoring bit 0, the middle bit, and the highest bit."""
    if qubits == 1:
        return 0
    anchors = tuple(dict.fromkeys((0, qubits // 2, qubits - 1)))
    if rng.random() < 0.45:
        return rng.choice(anchors)
    return rng.randrange(qubits)


def _other(rng: random.Random, qubits: int, excluded: int) -> int:
    value = rng.randrange(qubits - 1)
    return value + 1 if value >= excluded else value


def _pair(rng: random.Random, qubits: int) -> tuple[int, int]:
    """Return (control, target), mixing adjacent, distant, and arbitrary pairs."""
    target = _target(rng, qubits)
    if qubits == 2:
        return 1 - target, target

    roll = rng.random()
    if roll < 0.5:
        if target == 0:
            control = 1
        elif target == qubits - 1:
            control = qubits - 2
        else:
            control = target + rng.choice((-1, 1))
    elif roll < 0.75:
        far = [
            qubit
            for qubit in range(qubits)
            if qubit != target and 2 * abs(qubit - target) >= qubits
        ]
        control = rng.choice(far) if far else _other(rng, qubits, target)
    else:
        control = _other(rng, qubits, target)
    return control, target


def _triple(rng: random.Random, qubits: int) -> tuple[int, int, int]:
    """Return (control, control, target)."""
    target = _target(rng, qubits)
    others = [qubit for qubit in range(qubits) if qubit != target]
    first = rng.choice(others)
    second = rng.choice([qubit for qubit in others if qubit != first])
    return first, second, target


def _statement(rng: random.Random, gate: str, qubits: int) -> str:
    arity, parameter_count = SHAPE[gate]
    if arity == 1:
        used = (_target(rng, qubits),)
    elif arity == 2:
        used = _pair(rng, qubits)
    else:
        used = _triple(rng, qubits)

    parameters = ""
    if parameter_count:
        parameters = "(" + ", ".join(_angle(rng) for _ in range(parameter_count)) + ")"

    operands = ", ".join(f"q[{qubit}]" for qubit in used)
    return f"{gate}{parameters} {operands};"


def generate(
    qubits: int,
    gates: int,
    seed: int,
    measure: bool = False,
    family: str = "all",
) -> str:
    if not 1 <= qubits <= MAX_QUBITS:
        raise ValueError(f"kübit sayısı 1 ile {MAX_QUBITS} arasında olmalı: {qubits}")
    if gates < 0:
        raise ValueError(f"kapı sayısı negatif olamaz: {gates}")
    if family not in FAMILIES:
        raise ValueError(f"bilinmeyen aile: {family}")

    rng = random.Random(seed)
    allowed = [gate for gate in FAMILIES[family] if SHAPE[gate][0] <= qubits]
    if gates > 0 and not allowed:
        raise ValueError(
            f"'{family}' ailesinde {qubits} kübit için uygun kapı yok; "
            "--qubits değerini artırın"
        )

    lines = [
        "OPENQASM 2.0;",
        'include "qelib1.inc";',
        "",
        f"// rastgele devre: {qubits} kubit, {gates} kapi, tohum {seed}, aile {family}",
        f"qreg q[{qubits}];",
    ]
    if measure:
        lines.append(f"creg c[{qubits}];")
    lines.append("")

    weights = [ARITY_WEIGHT[SHAPE[gate][0]] for gate in allowed]
    for _ in range(gates):
        gate = rng.choices(allowed, weights=weights, k=1)[0]
        lines.append(_statement(rng, gate, qubits))

    if measure:
        lines.extend(["", "measure q -> c;"])

    return "\n".join(lines) + "\n"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="random_qasm.py",
        description="qasm.cpp okuyucusuna uyan rastgele OpenQASM 2.0 devresi üretir.",
        epilog="Örnek: random_qasm.py --qubits 6 --gates 40 --seed 7 --measure",
    )
    parser.add_argument(
        "--qubits",
        type=int,
        required=True,
        metavar="N",
        help=f"kübit sayısı (1-{MAX_QUBITS})",
    )
    parser.add_argument(
        "--gates",
        type=int,
        required=True,
        metavar="G",
        help="kapı sayısı (negatif olamaz)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        required=True,
        metavar="S",
        help="rastgelelik tohumu; aynı tohum aynı çıktıyı verir",
    )
    parser.add_argument(
        "--measure",
        action="store_true",
        help="devrenin sonunda tüm kübitleri ölç",
    )
    parser.add_argument(
        "--families",
        choices=("all", "clifford", "diag", "ctrl"),
        default="all",
        help="kapı ailesi (varsayılan: all)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        text = generate(args.qubits, args.gates, args.seed, args.measure, args.families)
    except ValueError as error:
        parser.error(str(error))
    print(text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
