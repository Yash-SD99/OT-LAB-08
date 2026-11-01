"""Optimization Passes for Three-Address Code (Track F).

Includes 4 optimization passes:
1. Constant Folding & Propagation (evaluates constant arithmetic at compile time)
2. Common Subexpression Elimination (CSE, eliminates redundant identical computations)
3. Dead Code Elimination (DCE, eliminates assignments to unused variables via liveness analysis)
4. Strength Reduction (converts expensive operations like x * 2^k into shift, removes algebraic identities)
"""

from typing import Any, Dict, List, Set, Tuple, Union
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
    """Pass 1: Evaluates constant expressions at compile time and propagates values."""
    def run(self, instructions: List[TACInstr]) -> List[TACInstr]:
        optimized = []
        known_constants: Dict[str, Union[int, float]] = {}
        
        for instr in instructions:
            # Clear known constants on control-flow jumps/labels to ensure safety across basic blocks
            if instr.op in ("LABEL", "JUMP", "JUMP_IF_ZERO"):
                known_constants.clear()
                optimized.append(instr)
                continue
                
            # Replace operands with known constant values if available
            s1 = known_constants.get(str(instr.src1), instr.src1)
            s2 = known_constants.get(str(instr.src2), instr.src2) if instr.src2 is not None else None
            
            # Constant Folding for Binary Operations
            if instr.op in ("ADD", "SUB", "MUL", "DIV", "SHL") and is_constant(s1) and is_constant(s2):
                c1 = to_const(s1)
                c2 = to_const(s2)
                res_val = 0
                if instr.op == "ADD":
                    res_val = c1 + c2
                elif instr.op == "SUB":
                    res_val = c1 - c2
                elif instr.op == "MUL":
                    res_val = c1 * c2
                elif instr.op == "DIV":
                    res_val = c1 // c2 if isinstance(c1, int) and isinstance(c2, int) and c2 != 0 else (c1 / c2 if c2 != 0 else 0)
                elif instr.op == "SHL":
                    res_val = int(c1) << int(c2)
                    
                # Replace with ASSIGN of folded constant
                new_instr = TACInstr(op="ASSIGN", dest=instr.dest, src1=res_val)
                optimized.append(new_instr)
                if instr.dest:
                    known_constants[instr.dest] = res_val
                    
            elif instr.op == "ASSIGN":
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
        
        for instr in instructions:
            if instr.op in ("LABEL", "JUMP", "JUMP_IF_ZERO"):
                available_expressions.clear()
                optimized.append(instr)
                continue
                
            if instr.op in ("ADD", "SUB", "MUL", "DIV", "SHL"):
                s1_str = str(instr.src1)
                s2_str = str(instr.src2)
                # Canonical ordering for commutative operations (ADD, MUL)
                if instr.op in ("ADD", "MUL") and s1_str > s2_str:
                    s1_str, s2_str = s2_str, s1_str
                    
                expr_key = (instr.op, s1_str, s2_str)
                
                if expr_key in available_expressions:
                    # Reuse previously computed temporary
                    prev_dest = available_expressions[expr_key]
                    optimized.append(TACInstr(op="ASSIGN", dest=instr.dest, src1=prev_dest))
                else:
                    available_expressions[expr_key] = str(instr.dest)
                    optimized.append(instr)
                    
            else:
                optimized.append(instr)
                # Invalidate expressions containing destination if overwritten
                if instr.dest:
                    d_str = str(instr.dest)
                    available_expressions = {
                        k: v for k, v in available_expressions.items()
                        if k[1] != d_str and k[2] != d_str and v != d_str
                    }
                    
        return optimized


class DeadCodeEliminationPass:
    """Pass 3: Backward liveness analysis removing dead temporary assignments."""
    def run(self, instructions: List[TACInstr]) -> List[TACInstr]:
        # Identify variables whose values are eventually read by subsequent instructions
        used_vars: Set[str] = set()
        
        # Backward sweep to compute variable usage
        for instr in reversed(instructions):
            if instr.op in ("PRINT", "RETURN", "JUMP_IF_ZERO"):
                if instr.src1 and isinstance(instr.src1, str) and not is_constant(instr.src1):
                    used_vars.add(instr.src1)
            elif instr.op in ("ADD", "SUB", "MUL", "DIV", "SHL", "ASSIGN"):
                if instr.dest in used_vars:
                    # Destination is read! Source operands are live
                    if instr.src1 and isinstance(instr.src1, str) and not is_constant(instr.src1):
                        used_vars.add(instr.src1)
                    if instr.src2 and isinstance(instr.src2, str) and not is_constant(instr.src2):
                        used_vars.add(instr.src2)
                        
        # Forward filter: keep instructions that are live or have external side-effects
        optimized = []
        for instr in instructions:
            if instr.op in ("LABEL", "JUMP", "JUMP_IF_ZERO", "PRINT", "RETURN"):
                optimized.append(instr)
            elif instr.dest:
                if instr.dest in used_vars:
                    optimized.append(instr)
                else:
                    # Dead code eliminated!
                    continue
            else:
                optimized.append(instr)
                
        return optimized


class StrengthReductionPass:
    """Pass 4: Algebraic simplification and strength reduction."""
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
                    if c2 > 0 and (c2 & (c2 - 1)) == 0: # Power of 2
                        shift_k = int(c2).bit_length() - 1
                        optimized.append(TACInstr(op="SHL", dest=instr.dest, src1=s1, src2=shift_k))
                        continue
                if is_constant(s1):
                    c1 = to_const(s1)
                    if c1 > 0 and (c1 & (c1 - 1)) == 0:
                        shift_k = int(c1).bit_length() - 1
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
