"""Three-Address Code (TAC) Intermediate Representation, Lexer, Parser, and Virtual Machine (Track F).

Provides:
1. Toy Language Parser & TAC Generator
2. TAC Instructions: ASSIGN, BINOP, LABEL, JUMP, JUMP_IF_ZERO, PRINT, RETURN
3. TAC Virtual Machine executing bytecode with cycle and instruction tracking
"""

from dataclasses import dataclass
import time
from typing import Any, Dict, List, Optional, Tuple, Union


@dataclass
class TACInstr:
    """Represents a single Three-Address Code instruction."""
    op: str                 # 'ASSIGN', 'ADD', 'SUB', 'MUL', 'DIV', 'MOD', 'LABEL', 'JUMP', 'JUMP_IF_ZERO', 'PRINT', 'RETURN'
    dest: Optional[str]     # Destination variable / temporary
    src1: Optional[Union[str, int, float]]   # First operand
    src2: Optional[Union[str, int, float]] = None # Second operand (for binary ops)

    def __repr__(self) -> str:
        if self.op == "LABEL":
            return f"{self.dest}:"
        elif self.op == "ASSIGN":
            return f"  {self.dest} = {self.src1}"
        elif self.op in ("ADD", "SUB", "MUL", "DIV", "MOD", "SHL"):
            op_sym = {"ADD": "+", "SUB": "-", "MUL": "*", "DIV": "/", "MOD": "%", "SHL": "<<"}.get(self.op, self.op)
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
        
        # Build label index
        for idx, instr in enumerate(self.instructions):
            if instr.op == "LABEL" and instr.dest:
                self.labels[instr.dest] = idx

    def _resolve(self, operand: Union[str, int, float, None]) -> Union[int, float]:
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

    def run(self, max_steps: int = 100_000) -> Dict[str, Any]:
        """Executes instructions until completion or max_steps limit."""
        t0 = time.perf_counter()
        pc = 0
        n_instr = len(self.instructions)
        
        while pc < n_instr and self.instructions_executed < max_steps:
            instr = self.instructions[pc]
            self.instructions_executed += 1
            
            if instr.op == "LABEL":
                pc += 1
                continue
                
            elif instr.op == "ASSIGN":
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
                self.env[instr.dest] = v1 // v2 if isinstance(v1, int) and isinstance(v2, int) and v2 != 0 else (v1 / v2 if v2 != 0 else 0)
                pc += 1
                
            elif instr.op == "SHL":
                v1 = int(self._resolve(instr.src1))
                v2 = int(self._resolve(instr.src2))
                self.env[instr.dest] = v1 << v2
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
            "final_env": copy_clean_env(self.env),
            "output": self.output.copy(),
            "runtime_us": elapsed * 1e6
        }


def copy_clean_env(env: Dict[str, Any]) -> Dict[str, Any]:
    """Filters out internal compiler temporaries starting with t_."""
    return {k: v for k, v in env.items() if not k.startswith("t_")}
