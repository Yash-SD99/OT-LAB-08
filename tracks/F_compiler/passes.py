"""Optimization Passes for Three-Address Code (Track F).

Includes 6 optimization passes:
1. Constant Folding & Propagation (evaluates constant arithmetic at compile time with safe integer semantics)
2. Common Subexpression Elimination (CSE, eliminates redundant identical computations)
3. Dead Code Elimination (DCE, eliminates assignments to unused variables via liveness analysis)
4. Strength Reduction (converts expensive operations into cheap ones under strict integer semantics)
5. Loop Invariant Code Motion (LICM, hoists loop-invariant computations out of loops)
6. Loop Unrolling (unrolls loop bodies by factor u, reducing branch penalty and cycle overhead)
"""

from __future__ import annotations
import copy
from typing import Any, Dict, List, Optional, Set, Tuple, Union
from tracks.F_compiler.ir_compiler import TACInstr


def is_constant(val: Any) -> bool:
    """Checks whether an operand is a numeric literal."""
    if isinstance(val, (int, float)):
        return True
    if isinstance(val, str):
        try:
            float(val)
            return True
        except ValueError:
            return False
    return False


def to_const(val: Any) -> Union[int, float]:
    """Converts a constant operand to int or float."""
    if isinstance(val, (int, float)):
        return val
    try:
        if "." in str(val):
            return float(val)
        return int(val)
    except ValueError:
        return 0


class ConstantFoldingAndPropagationPass:
    """Pass 1: Evaluates constant expressions at compile time and propagates values.
    Enforces strict integer semantics (zero-division protection, shift bounds, truncation).
    """

    def run(self, instructions: List[TACInstr]) -> List[TACInstr]:
        optimized = []
        known_constants: Dict[str, Union[int, float]] = {}

        for instr in instructions:
            # Control flow boundaries invalidate known constants across basic blocks
            if instr.op in ("LABEL", "JUMP", "JUMP_IF_ZERO"):
                known_constants.clear()
                optimized.append(instr)
                continue

            # Substitute known constants into source operands
            s1 = known_constants.get(str(instr.src1), instr.src1) if instr.src1 is not None else None
            s2 = known_constants.get(str(instr.src2), instr.src2) if instr.src2 is not None else None

            # Constant Folding for Binary Operations
            bin_ops = ("ADD", "SUB", "MUL", "DIV", "MOD", "SHL", "SHR",
                       "LT", "GT", "LE", "GE", "EQ", "NE")

            if instr.op in bin_ops and is_constant(s1) and is_constant(s2):
                c1 = to_const(s1)
                c2 = to_const(s2)
                can_fold = True
                res_val: Union[int, float] = 0

                if instr.op == "ADD":
                    res_val = c1 + c2
                elif instr.op == "SUB":
                    res_val = c1 - c2
                elif instr.op == "MUL":
                    res_val = c1 * c2
                elif instr.op == "DIV":
                    if c2 == 0:
                        can_fold = False  # Preserve runtime division-by-zero
                    else:
                        res_val = int(c1 / c2) if isinstance(c1, int) and isinstance(c2, int) else (c1 / c2)
                elif instr.op == "MOD":
                    if c2 == 0:
                        can_fold = False
                    else:
                        res_val = c1 - int(c1 / c2) * c2 if isinstance(c1, int) and isinstance(c2, int) else (c1 % c2)
                elif instr.op == "SHL":
                    ic2 = int(c2)
                    if 0 <= ic2 < 32:
                        res_val = int(c1) << ic2
                    else:
                        can_fold = False
                elif instr.op == "SHR":
                    ic2 = int(c2)
                    if 0 <= ic2 < 32:
                        res_val = int(c1) >> ic2
                    else:
                        can_fold = False
                elif instr.op == "LT":
                    res_val = 1 if c1 < c2 else 0
                elif instr.op == "GT":
                    res_val = 1 if c1 > c2 else 0
                elif instr.op == "LE":
                    res_val = 1 if c1 <= c2 else 0
                elif instr.op == "GE":
                    res_val = 1 if c1 >= c2 else 0
                elif instr.op == "EQ":
                    res_val = 1 if c1 == c2 else 0
                elif instr.op == "NE":
                    res_val = 1 if c1 != c2 else 0

                if can_fold:
                    new_instr = TACInstr(op="ASSIGN", dest=instr.dest, src1=res_val)
                    optimized.append(new_instr)
                    if instr.dest:
                        known_constants[instr.dest] = res_val
                    continue

            # Assignments of constants
            if instr.op == "ASSIGN":
                if is_constant(s1):
                    val = to_const(s1)
                    optimized.append(TACInstr(op="ASSIGN", dest=instr.dest, src1=val))
                    if instr.dest:
                        known_constants[instr.dest] = val
                else:
                    optimized.append(TACInstr(op="ASSIGN", dest=instr.dest, src1=s1))
                    if instr.dest in known_constants:
                        del known_constants[instr.dest]
            else:
                new_instr = TACInstr(op=instr.op, dest=instr.dest, src1=s1, src2=s2)
                optimized.append(new_instr)
                if instr.dest and instr.dest in known_constants:
                    del known_constants[instr.dest]

        return optimized


class CommonSubexpressionEliminationPass:
    """Pass 2: Identifies and eliminates redundant identical subexpressions within basic blocks."""

    def run(self, instructions: List[TACInstr]) -> List[TACInstr]:
        optimized = []
        available_expressions: Dict[Tuple[str, str, str], str] = {}

        bin_ops = ("ADD", "SUB", "MUL", "DIV", "MOD", "SHL", "SHR",
                   "LT", "GT", "LE", "GE", "EQ", "NE")

        for instr in instructions:
            if instr.op in ("LABEL", "JUMP", "JUMP_IF_ZERO"):
                available_expressions.clear()
                optimized.append(instr)
                continue

            # Invalidate any previously cached expressions whose operands or source values are overwritten
            if instr.dest:
                d_str = str(instr.dest)
                available_expressions = {
                    k: v for k, v in available_expressions.items()
                    if k[1] != d_str and k[2] != d_str and v != d_str
                }

            if instr.op in bin_ops:
                s1_str = str(instr.src1)
                s2_str = str(instr.src2)
                # Canonical ordering for commutative operations (ADD, MUL, EQ, NE)
                if instr.op in ("ADD", "MUL", "EQ", "NE") and s1_str > s2_str:
                    s1_str, s2_str = s2_str, s1_str

                expr_key = (instr.op, s1_str, s2_str)

                if expr_key in available_expressions:
                    # Reuse previously computed value
                    prev_dest = available_expressions[expr_key]
                    optimized.append(TACInstr(op="ASSIGN", dest=instr.dest, src1=prev_dest))
                else:
                    available_expressions[expr_key] = str(instr.dest)
                    optimized.append(instr)
            else:
                optimized.append(instr)

        return optimized


class DeadCodeEliminationPass:
    """Pass 3: Backward fixed-point liveness analysis removing dead temporary and unused variable assignments.
    Iterates backward sweeps to fixed point across control-flow back-edges to ensure variables
    used inside loops or subsequent iterations are correctly recognized as live.
    """

    def run(self, instructions: List[TACInstr]) -> List[TACInstr]:
        used_vars: Set[str] = set()

        # Fixed-point iteration over the instruction stream to correctly account for loop back-edges
        for _ in range(10):
            prev_len = len(used_vars)
            for instr in reversed(instructions):
                if instr.op in ("PRINT", "RETURN", "JUMP_IF_ZERO"):
                    if instr.src1 and isinstance(instr.src1, str) and not is_constant(instr.src1):
                        used_vars.add(instr.src1)
                elif instr.dest and instr.dest in used_vars:
                    # Instruction destination is read downstream, so its inputs are live
                    if instr.src1 and isinstance(instr.src1, str) and not is_constant(instr.src1):
                        used_vars.add(instr.src1)
                    if instr.src2 and isinstance(instr.src2, str) and not is_constant(instr.src2):
                        used_vars.add(instr.src2)
            if len(used_vars) == prev_len:
                break

        # Forward filter: keep instructions that are live or produce control flow/side-effects
        optimized = []
        for instr in instructions:
            if instr.op in ("LABEL", "JUMP", "JUMP_IF_ZERO", "PRINT", "RETURN"):
                optimized.append(instr)
            elif instr.dest:
                if instr.dest in used_vars:
                    optimized.append(instr)
                else:
                    # Dead assignment eliminated!
                    continue
            else:
                optimized.append(instr)

        return optimized


class StrengthReductionPass:
    """Pass 4: Algebraic simplification and strength reduction with safe integer semantics.
    Transforms:
    - x * 2^k -> x << k (for k in [0, 30])
    - x * 0 -> 0, x * 1 -> x
    - x + 0 -> x, x - 0 -> x
    - x / 1 -> x
    Notice: Does NOT transform x / 2^k -> x >> k because C integer truncation differs from
    arithmetic right shift on negative dividends (-7 / 2 = -3 vs -7 >> 1 = -4).
    """

    def run(self, instructions: List[TACInstr]) -> List[TACInstr]:
        optimized = []

        for instr in instructions:
            if instr.op == "MUL":
                s1, s2 = instr.src1, instr.src2
                # x * 0 -> 0
                if s2 == 0 or s2 == "0" or s1 == 0 or s1 == "0":
                    optimized.append(TACInstr(op="ASSIGN", dest=instr.dest, src1=0))
                    continue
                # x * 1 -> x
                if s2 == 1 or s2 == "1":
                    optimized.append(TACInstr(op="ASSIGN", dest=instr.dest, src1=s1))
                    continue
                if s1 == 1 or s1 == "1":
                    optimized.append(TACInstr(op="ASSIGN", dest=instr.dest, src1=s2))
                    continue
                # x * 2^k -> x << k (Strength Reduction to bit shift)
                if is_constant(s2):
                    c2 = to_const(s2)
                    if isinstance(c2, int) and c2 > 0 and (c2 & (c2 - 1)) == 0:
                        shift_k = c2.bit_length() - 1
                        if 0 <= shift_k <= 30:
                            optimized.append(TACInstr(op="SHL", dest=instr.dest, src1=s1, src2=shift_k))
                            continue
                if is_constant(s1):
                    c1 = to_const(s1)
                    if isinstance(c1, int) and c1 > 0 and (c1 & (c1 - 1)) == 0:
                        shift_k = c1.bit_length() - 1
                        if 0 <= shift_k <= 30:
                            optimized.append(TACInstr(op="SHL", dest=instr.dest, src1=s2, src2=shift_k))
                            continue

            elif instr.op == "ADD":
                s1, s2 = instr.src1, instr.src2
                # x + 0 -> x
                if s2 == 0 or s2 == "0":
                    optimized.append(TACInstr(op="ASSIGN", dest=instr.dest, src1=s1))
                    continue
                if s1 == 0 or s1 == "0":
                    optimized.append(TACInstr(op="ASSIGN", dest=instr.dest, src1=s2))
                    continue

            elif instr.op == "SUB":
                s1, s2 = instr.src1, instr.src2
                # x - 0 -> x
                if s2 == 0 or s2 == "0":
                    optimized.append(TACInstr(op="ASSIGN", dest=instr.dest, src1=s1))
                    continue

            elif instr.op == "DIV":
                s1, s2 = instr.src1, instr.src2
                # x / 1 -> x
                if s2 == 1 or s2 == "1":
                    optimized.append(TACInstr(op="ASSIGN", dest=instr.dest, src1=s1))
                    continue

            optimized.append(instr)

        return optimized


class LoopInvariantCodeMotionPass:
    """Pass 5: Hoists loop-invariant computations out of loop bodies into pre-headers.
    Identifies instructions whose operands are constants or defined strictly outside
    the loop body, and whose destination is not modified elsewhere inside the loop.
    """

    def run(self, instructions: List[TACInstr]) -> List[TACInstr]:
        # Detect loops bounded by (LABEL L_start ... JUMP L_start)
        label_positions: Dict[str, int] = {}
        for idx, instr in enumerate(instructions):
            if instr.op == "LABEL" and instr.dest:
                label_positions[instr.dest] = idx

        loop_regions: List[Tuple[int, int]] = []
        for idx, instr in enumerate(instructions):
            if instr.op == "JUMP" and instr.dest in label_positions:
                start_idx = label_positions[instr.dest]
                if start_idx < idx:
                    loop_regions.append((start_idx, idx))

        if not loop_regions:
            return instructions

        # Process loops (inner-most or first detected)
        result = list(instructions)
        for start_idx, end_idx in loop_regions:
            # Recompute modified variables inside the loop
            modified_in_loop: Set[str] = set()
            for i in range(start_idx, end_idx + 1):
                if result[i].dest:
                    modified_in_loop.add(str(result[i].dest))

            hoisted_instructions: List[TACInstr] = []
            loop_body_indices: Set[int] = set()

            # Iterate to fixed point for chained invariant expressions
            changed = True
            while changed:
                changed = False
                for i in range(start_idx + 1, end_idx):
                    if i in loop_body_indices:
                        continue
                    instr = result[i]
                    if instr.op in ("ADD", "SUB", "MUL", "DIV", "MOD", "SHL", "SHR",
                                   "LT", "GT", "LE", "GE", "EQ", "NE", "ASSIGN"):
                        # Safe arithmetic: check operands
                        s1_inv = is_constant(instr.src1) or (str(instr.src1) not in modified_in_loop)
                        s2_inv = (instr.src2 is None) or is_constant(instr.src2) or (str(instr.src2) not in modified_in_loop)

                        if s1_inv and s2_inv:
                            # Verify destination is not modified elsewhere in loop
                            dest_count = sum(1 for j in range(start_idx, end_idx + 1) if result[j].dest == instr.dest)
                            if dest_count == 1:
                                hoisted_instructions.append(instr)
                                loop_body_indices.add(i)
                                # Once hoisted, its destination is now external/invariant
                                modified_in_loop.discard(str(instr.dest))
                                changed = True

            if hoisted_instructions:
                # Reconstruct program with hoisted instructions before start_idx
                new_prog = []
                for idx, instr in enumerate(result):
                    if idx == start_idx:
                        new_prog.extend(hoisted_instructions)
                    if idx not in loop_body_indices:
                        new_prog.append(instr)
                result = new_prog
                break  # Complete one loop transformation per pass

        return result


class LoopUnrollPass:
    """Pass 6: Unrolls loop bodies by factor u (e.g. u=2, u=4).
    Replicates the loop body to amortize loop branch overhead and enable cross-iteration
    optimizations like CSE and Constant Folding across unrolled steps.
    """

    def __init__(self, unroll_factor: int = 2):
        self.unroll_factor = unroll_factor

    def run(self, instructions: List[TACInstr]) -> List[TACInstr]:
        if self.unroll_factor <= 1:
            return instructions

        # Detect canonical while-loops:
        # LABEL L_start
        # ... condition evaluation producing t_cond
        # JUMP_IF_ZERO t_cond L_end
        # ... body ...
        # JUMP L_start
        # LABEL L_end
        n = len(instructions)
        for i in range(n):
            if instructions[i].op == "LABEL":
                start_label = instructions[i].dest
                # Search for matching JUMP start_label
                jump_pos = -1
                for j in range(i + 1, n):
                    if instructions[j].op == "JUMP" and instructions[j].dest == start_label:
                        jump_pos = j
                        break
                if jump_pos != -1:
                    # Verify branch at start + 2
                    if i + 2 < jump_pos and instructions[i + 2].op == "JUMP_IF_ZERO":
                        end_label = instructions[i + 2].dest
                        cond_instr = instructions[i + 1]
                        br_instr = instructions[i + 2]
                        body = instructions[i + 3:jump_pos]

                        # Check if body is simple straight-line code (no internal labels/branches)
                        has_internal_cf = any(instr.op in ("LABEL", "JUMP", "JUMP_IF_ZERO") for instr in body)
                        if not has_internal_cf and len(body) > 0:
                            # Replicate body u times inside the loop with intermediate guarded exit checks
                            unrolled_loop = [instructions[i], cond_instr, br_instr]
                            for step in range(self.unroll_factor):
                                unrolled_loop.extend(copy.deepcopy(body))
                                if step < self.unroll_factor - 1:
                                    # Intermediate condition check prevents overshooting on non-multiple trip counts
                                    unrolled_loop.append(copy.deepcopy(cond_instr))
                                    unrolled_loop.append(copy.deepcopy(br_instr))
                            unrolled_loop.append(instructions[jump_pos]) # JUMP L_start
                            
                            # Replace in instructions
                            new_instructions = instructions[:i] + unrolled_loop + instructions[jump_pos + 1:]
                            return new_instructions

        return instructions


class OptimizationPipeline:
    """Multi-level compiler optimization driver (-O0, -O1, -O2, -O3)."""

    def __init__(self, level: str = "O3", unroll_factor: int = 1):
        self.level = level
        self.unroll_factor = unroll_factor
        self.const_fold = ConstantFoldingAndPropagationPass()
        self.cse = CommonSubexpressionEliminationPass()
        self.dce = DeadCodeEliminationPass()
        self.strength_red = StrengthReductionPass()
        self.licm = LoopInvariantCodeMotionPass()
        self.unroll = LoopUnrollPass(unroll_factor=unroll_factor)

    def run(self, instructions: List[TACInstr]) -> List[TACInstr]:
        if self.level == "O0":
            return instructions

        curr = copy.deepcopy(instructions)

        if self.level == "O1":
            curr = self.const_fold.run(curr)
            curr = self.dce.run(curr)
            return curr

        if self.level == "O2":
            curr = self.const_fold.run(curr)
            curr = self.cse.run(curr)
            curr = self.dce.run(curr)
            return curr

        # -O3: Full optimization suite with LICM, Strength Reduction, and fixed-point iteration
        if self.unroll_factor > 1:
            curr = self.unroll.run(curr)

        curr = self.licm.run(curr)

        # Iterate until fixed point (up to 4 rounds)
        for _ in range(4):
            prev_len = len(curr)
            curr = self.const_fold.run(curr)
            curr = self.strength_red.run(curr)
            curr = self.cse.run(curr)
            curr = self.dce.run(curr)
            if len(curr) == prev_len:
                break

        return curr
