"""Three-Address Code (TAC) Intermediate Representation and Virtual Machine (Track F).

Provides:
1. TAC Instruction definition (`TACInstr`)
2. Documented cycle cost model per opcode
3. TAC Virtual Machine (`TACVM`) executing bytecode with exact cycle tracking,
   instruction counting, wall-clock timing via perf_counter, and integer semantics.
"""

from __future__ import annotations
from dataclasses import dataclass
import time
from typing import Any, Dict, List, Optional, Tuple, Union


# -----------------------------------------------------------------------------
# Opcode Execution Cost Model
# -----------------------------------------------------------------------------
# Hardware-calibrated RISC cycle cost model:
# - Simple ALU / bitwise / assign operations: 1 cycle
# - Multiply: 3 cycles (pipelined multiplier)
# - Divide / Modulo: 10 cycles (multi-cycle hardware divider)
# - Control flow jumps and branches: 2 cycles (pipeline flush / branch penalty)
# - Pseudo-operations (LABEL): 0 cycles (metadata only, no hardware execution)
# -----------------------------------------------------------------------------
OPCODE_CYCLES: Dict[str, int] = {
    "ASSIGN": 1,
    "ADD": 1,
    "SUB": 1,
    "SHL": 1,
    "SHR": 1,
    "LT": 1,
    "GT": 1,
    "LE": 1,
    "GE": 1,
    "EQ": 1,
    "NE": 1,
    "MUL": 3,
    "DIV": 10,
    "MOD": 10,
    "JUMP": 2,
    "JUMP_IF_ZERO": 2,
    "LABEL": 0,
    "PRINT": 1,
    "RETURN": 1,
}


@dataclass
class TACInstr:
    """Represents a single Three-Address Code instruction."""
    op: str
    dest: Optional[str]
    src1: Optional[Union[str, int, float]]
    src2: Optional[Union[str, int, float]] = None

    def __repr__(self) -> str:
        if self.op == "LABEL":
            return f"{self.dest}:"
        elif self.op == "ASSIGN":
            return f"  {self.dest} = {self.src1}"
        elif self.op in ("ADD", "SUB", "MUL", "DIV", "MOD", "SHL", "SHR",
                         "LT", "GT", "LE", "GE", "EQ", "NE"):
            op_sym = {
                "ADD": "+", "SUB": "-", "MUL": "*", "DIV": "/", "MOD": "%",
                "SHL": "<<", "SHR": ">>", "LT": "<", "GT": ">",
                "LE": "<=", "GE": ">=", "EQ": "==", "NE": "!="
            }.get(self.op, self.op)
            return f"  {self.dest} = {self.src1} {op_sym} {self.src2}"
        elif self.op == "JUMP":
            return f"  goto {self.dest}"
        elif self.op == "JUMP_IF_ZERO":
            return f"  if {self.src1} == 0 goto {self.dest}"
        elif self.op == "PRINT":
            return f"  print {self.src1}"
        elif self.op == "RETURN":
            return f"  return {self.src1}"
        return f"  {self.op} {self.dest} {self.src1} {self.src2}"


class TACVM:
    """Virtual Machine executing Three-Address Code instructions."""

    def __init__(self, instructions: List[TACInstr]):
        self.instructions = instructions
        self.labels: Dict[str, int] = {}
        self.env: Dict[str, Union[int, float]] = {}
        self.output: List[Union[int, float]] = []
        self.instructions_executed = 0
        self.total_cycles = 0

        # Build label table
        for idx, instr in enumerate(self.instructions):
            if instr.op == "LABEL" and instr.dest:
                self.labels[instr.dest] = idx

    def _resolve(self, operand: Union[str, int, float, None]) -> Union[int, float]:
        if operand is None:
            return 0
        if isinstance(operand, (int, float)):
            return operand
        if isinstance(operand, str):
            if operand in self.env:
                return self.env[operand]
            try:
                if "." in operand:
                    return float(operand)
                return int(operand)
            except ValueError:
                return 0
        return 0

    def run(self, max_steps: int = 500_000) -> Dict[str, Any]:
        """Executes instructions with strict cycle cost tracking and wall-clock measurement."""
        t0 = time.perf_counter()
        pc = 0
        n_instr = len(self.instructions)
        self.instructions_executed = 0
        self.total_cycles = 0

        while pc < n_instr and self.instructions_executed < max_steps:
            instr = self.instructions[pc]
            cost = OPCODE_CYCLES.get(instr.op, 1)
            self.total_cycles += cost

            if instr.op == "LABEL":
                pc += 1
                continue

            self.instructions_executed += 1

            if instr.op == "ASSIGN":
                val = self._resolve(instr.src1)
                self.env[instr.dest] = val
                pc += 1

            elif instr.op == "ADD":
                v1 = self._resolve(instr.src1)
                v2 = self._resolve(instr.src2)
                self.env[instr.dest] = v1 + v2
                pc += 1

            elif instr.op == "SUB":
                v1 = self._resolve(instr.src1)
                v2 = self._resolve(instr.src2)
                self.env[instr.dest] = v1 - v2
                pc += 1

            elif instr.op == "MUL":
                v1 = self._resolve(instr.src1)
                v2 = self._resolve(instr.src2)
                self.env[instr.dest] = v1 * v2
                pc += 1

            elif instr.op == "DIV":
                v1 = self._resolve(instr.src1)
                v2 = self._resolve(instr.src2)
                if v2 == 0:
                    raise ZeroDivisionError(f"Division by zero in TACVM at PC {pc}: {instr}")
                # C-style integer truncation towards zero
                if isinstance(v1, int) and isinstance(v2, int):
                    self.env[instr.dest] = int(v1 / v2)
                else:
                    self.env[instr.dest] = v1 / v2
                pc += 1

            elif instr.op == "MOD":
                v1 = self._resolve(instr.src1)
                v2 = self._resolve(instr.src2)
                if v2 == 0:
                    raise ZeroDivisionError(f"Modulo by zero in TACVM at PC {pc}: {instr}")
                # C-style remainder matching truncation
                if isinstance(v1, int) and isinstance(v2, int):
                    self.env[instr.dest] = v1 - int(v1 / v2) * v2
                else:
                    self.env[instr.dest] = v1 % v2
                pc += 1

            elif instr.op == "SHL":
                v1 = int(self._resolve(instr.src1))
                v2 = int(self._resolve(instr.src2))
                if v2 < 0 or v2 >= 32:
                    raise ValueError(f"Invalid shift count {v2} at PC {pc}")
                self.env[instr.dest] = v1 << v2
                pc += 1

            elif instr.op == "SHR":
                v1 = int(self._resolve(instr.src1))
                v2 = int(self._resolve(instr.src2))
                if v2 < 0 or v2 >= 32:
                    raise ValueError(f"Invalid shift count {v2} at PC {pc}")
                self.env[instr.dest] = v1 >> v2
                pc += 1

            elif instr.op == "LT":
                v1 = self._resolve(instr.src1)
                v2 = self._resolve(instr.src2)
                self.env[instr.dest] = 1 if v1 < v2 else 0
                pc += 1

            elif instr.op == "GT":
                v1 = self._resolve(instr.src1)
                v2 = self._resolve(instr.src2)
                self.env[instr.dest] = 1 if v1 > v2 else 0
                pc += 1

            elif instr.op == "LE":
                v1 = self._resolve(instr.src1)
                v2 = self._resolve(instr.src2)
                self.env[instr.dest] = 1 if v1 <= v2 else 0
                pc += 1

            elif instr.op == "GE":
                v1 = self._resolve(instr.src1)
                v2 = self._resolve(instr.src2)
                self.env[instr.dest] = 1 if v1 >= v2 else 0
                pc += 1

            elif instr.op == "EQ":
                v1 = self._resolve(instr.src1)
                v2 = self._resolve(instr.src2)
                self.env[instr.dest] = 1 if v1 == v2 else 0
                pc += 1

            elif instr.op == "NE":
                v1 = self._resolve(instr.src1)
                v2 = self._resolve(instr.src2)
                self.env[instr.dest] = 1 if v1 != v2 else 0
                pc += 1

            elif instr.op == "JUMP":
                if instr.dest in self.labels:
                    pc = self.labels[instr.dest] + 1
                else:
                    break

            elif instr.op == "JUMP_IF_ZERO":
                cond = self._resolve(instr.src1)
                if cond == 0 and instr.dest in self.labels:
                    pc = self.labels[instr.dest] + 1
                else:
                    pc += 1

            elif instr.op == "PRINT":
                val = self._resolve(instr.src1)
                self.output.append(val)
                pc += 1

            elif instr.op == "RETURN":
                val = self._resolve(instr.src1)
                self.output.append(val)
                break

            else:
                pc += 1

        elapsed = time.perf_counter() - t0
        return {
            "instructions_executed": self.instructions_executed,
            "total_cycles": self.total_cycles,
            "code_size": n_instr,
            "final_env": copy_clean_env(self.env),
            "output": self.output.copy(),
            "wall_time_us": elapsed * 1e6,
        }


def copy_clean_env(env: Dict[str, Any]) -> Dict[str, Any]:
    """Filters out internal compiler temporaries starting with t_."""
    return {k: v for k, v in env.items() if not k.startswith("t_")}
