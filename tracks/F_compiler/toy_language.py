"""
Toy Source Language: Lexer, Parser (AST), and TAC Code Generator (Track F).

Provides a structured language supporting:
- Variable declarations: `var x = 10;`
- Assignments: `x = x + 1;`
- Binary expressions: `+`, `-`, `*`, `/`, `%`, `<<`, `>>`, `<`, `<=`, `>`, `>=`, `==`, `!=`
- Conditional branching: `if (cond) { ... } else { ... }`
- Iteration: `while (cond) { ... }`
- Returns: `return expr;`
- Lowering directly into Three-Address Code (`List[TACInstr]`).
"""

from __future__ import annotations
import re
from dataclasses import dataclass
from typing import List, Optional, Union, Any
from tracks.F_compiler.ir_compiler import TACInstr


# ---------------------------------------------------------
# 1. Lexer & Tokens
# ---------------------------------------------------------

@dataclass
class Token:
    type: str
    value: Any
    line: int


TOKEN_SPECIFICATION = [
    ('COMMENT',   r'//[^\n]*'),
    ('NUMBER',    r'\d+'),
    ('VAR',       r'\bvar\b'),
    ('IF',        r'\bif\b'),
    ('ELSE',      r'\belse\b'),
    ('WHILE',     r'\bwhile\b'),
    ('RETURN',    r'\breturn\b'),
    ('PRINT',     r'\bprint\b'),
    ('ID',        r'[a-zA-Z_][a-zA-Z0-9_]*'),
    ('LE',        r'<='),
    ('GE',        r'>='),
    ('EQ',        r'=='),
    ('NE',        r'!='),
    ('SHL',       r'<<'),
    ('SHR',       r'>>'),
    ('LT',        r'<'),
    ('GT',        r'>'),
    ('ASSIGN',    r'='),
    ('PLUS',      r'\+'),
    ('MINUS',     r'-'),
    ('MUL',       r'\*'),
    ('DIV',       r'/'),
    ('MOD',       r'%'),
    ('LPAREN',    r'\('),
    ('RPAREN',    r'\)'),
    ('LBRACE',    r'\{'),
    ('RBRACE',    r'\}'),
    ('SEMICOLON', r';'),
    ('SKIP',      r'[ \t\r\n]+'),
    ('MISMATCH',  r'.'),
]

TOKEN_REGEX = '|'.join(f'(?P<{name}>{pattern})' for name, pattern in TOKEN_SPECIFICATION)


def tokenize(code: str) -> List[Token]:
    tokens = []
    line_num = 1
    for match in re.finditer(TOKEN_REGEX, code):
        kind = match.lastgroup
        value = match.group()
        if kind == 'NUMBER':
            tokens.append(Token('NUMBER', int(value), line_num))
        elif kind == 'SKIP':
            line_num += value.count('\n')
        elif kind == 'COMMENT':
            pass
        elif kind == 'MISMATCH':
            raise SyntaxError(f"Unexpected character {value!r} on line {line_num}")
        else:
            tokens.append(Token(kind, value, line_num))
    tokens.append(Token('EOF', '', line_num))
    return tokens


# ---------------------------------------------------------
# 2. Abstract Syntax Tree (AST) Nodes
# ---------------------------------------------------------

class ASTNode:
    pass


@dataclass
class Program(ASTNode):
    statements: List[ASTNode]


@dataclass
class VarDecl(ASTNode):
    name: str
    expr: ASTNode


@dataclass
class Assignment(ASTNode):
    name: str
    expr: ASTNode


@dataclass
class BinaryExpr(ASTNode):
    op: str
    left: ASTNode
    right: ASTNode


@dataclass
class Literal(ASTNode):
    value: int


@dataclass
class Variable(ASTNode):
    name: str


@dataclass
class IfStatement(ASTNode):
    condition: ASTNode
    then_body: List[ASTNode]
    else_body: Optional[List[ASTNode]] = None


@dataclass
class WhileStatement(ASTNode):
    condition: ASTNode
    body: List[ASTNode]


@dataclass
class ReturnStatement(ASTNode):
    expr: ASTNode


@dataclass
class PrintStatement(ASTNode):
    expr: ASTNode


# ---------------------------------------------------------
# 3. Recursive Descent Parser
# ---------------------------------------------------------

class Parser:
    def __init__(self, tokens: List[Token]):
        self.tokens = tokens
        self.pos = 0

    def current(self) -> Token:
        return self.tokens[self.pos]

    def consume(self, expected_type: str) -> Token:
        tok = self.current()
        if tok.type != expected_type:
            raise SyntaxError(f"Expected token {expected_type}, got {tok.type} ({tok.value}) on line {tok.line}")
        self.pos += 1
        return tok

    def parse(self) -> Program:
        statements = []
        while self.current().type != 'EOF':
            statements.append(self.parse_statement())
        return Program(statements)

    def parse_statement(self) -> ASTNode:
        tok = self.current()
        if tok.type == 'VAR':
            self.consume('VAR')
            name = self.consume('ID').value
            self.consume('ASSIGN')
            expr = self.parse_expression()
            self.consume('SEMICOLON')
            return VarDecl(name, expr)
        elif tok.type == 'IF':
            self.consume('IF')
            self.consume('LPAREN')
            cond = self.parse_expression()
            self.consume('RPAREN')
            self.consume('LBRACE')
            then_body = []
            while self.current().type != 'RBRACE':
                then_body.append(self.parse_statement())
            self.consume('RBRACE')
            else_body = None
            if self.current().type == 'ELSE':
                self.consume('ELSE')
                self.consume('LBRACE')
                else_body = []
                while self.current().type != 'RBRACE':
                    else_body.append(self.parse_statement())
                self.consume('RBRACE')
            return IfStatement(cond, then_body, else_body)
        elif tok.type == 'WHILE':
            self.consume('WHILE')
            self.consume('LPAREN')
            cond = self.parse_expression()
            self.consume('RPAREN')
            self.consume('LBRACE')
            body = []
            while self.current().type != 'RBRACE':
                body.append(self.parse_statement())
            self.consume('RBRACE')
            return WhileStatement(cond, body)
        elif tok.type == 'RETURN':
            self.consume('RETURN')
            expr = self.parse_expression()
            self.consume('SEMICOLON')
            return ReturnStatement(expr)
        elif tok.type == 'PRINT':
            self.consume('PRINT')
            expr = self.parse_expression()
            self.consume('SEMICOLON')
            return PrintStatement(expr)
        elif tok.type == 'ID':
            name = self.consume('ID').value
            self.consume('ASSIGN')
            expr = self.parse_expression()
            self.consume('SEMICOLON')
            return Assignment(name, expr)
        else:
            raise SyntaxError(f"Unexpected statement starting with {tok.type} on line {tok.line}")

    def parse_expression(self) -> ASTNode:
        return self.parse_comparison()

    def parse_comparison(self) -> ASTNode:
        node = self.parse_shift()
        while self.current().type in ('LT', 'GT', 'LE', 'GE', 'EQ', 'NE'):
            op_tok = self.current()
            self.consume(op_tok.type)
            right = self.parse_shift()
            node = BinaryExpr(op_tok.value, node, right)
        return node

    def parse_shift(self) -> ASTNode:
        node = self.parse_additive()
        while self.current().type in ('SHL', 'SHR'):
            op_tok = self.current()
            self.consume(op_tok.type)
            right = self.parse_additive()
            node = BinaryExpr(op_tok.value, node, right)
        return node

    def parse_additive(self) -> ASTNode:
        node = self.parse_multiplicative()
        while self.current().type in ('PLUS', 'MINUS'):
            op_tok = self.current()
            self.consume(op_tok.type)
            right = self.parse_multiplicative()
            node = BinaryExpr(op_tok.value, node, right)
        return node

    def parse_multiplicative(self) -> ASTNode:
        node = self.parse_primary()
        while self.current().type in ('MUL', 'DIV', 'MOD'):
            op_tok = self.current()
            self.consume(op_tok.type)
            right = self.parse_primary()
            node = BinaryExpr(op_tok.value, node, right)
        return node

    def parse_primary(self) -> ASTNode:
        tok = self.current()
        if tok.type == 'NUMBER':
            self.consume('NUMBER')
            return Literal(tok.value)
        elif tok.type == 'ID':
            self.consume('ID')
            return Variable(tok.value)
        elif tok.type == 'LPAREN':
            self.consume('LPAREN')
            expr = self.parse_expression()
            self.consume('RPAREN')
            return expr
        else:
            raise SyntaxError(f"Unexpected primary expression token {tok.type} on line {tok.line}")


# ---------------------------------------------------------
# 4. AST to TAC Lowering (Code Generator)
# ---------------------------------------------------------

class TACLowerer:
    """Lowers high-level AST into List[TACInstr]."""

    def __init__(self):
        self.instructions: List[TACInstr] = []
        self.temp_counter = 0
        self.label_counter = 0

    def new_temp(self) -> str:
        self.temp_counter += 1
        return f"t_{self.temp_counter}"

    def new_label(self, prefix: str = "L") -> str:
        self.label_counter += 1
        return f"{prefix}_{self.label_counter}"

    def lower(self, program: Program) -> List[TACInstr]:
        self.instructions = []
        for stmt in program.statements:
            self.lower_statement(stmt)
        return self.instructions

    def lower_statement(self, stmt: ASTNode):
        if isinstance(stmt, VarDecl):
            val = self.lower_expression(stmt.expr)
            self.instructions.append(TACInstr(op='ASSIGN', dest=stmt.name, src1=val))
        elif isinstance(stmt, Assignment):
            val = self.lower_expression(stmt.expr)
            self.instructions.append(TACInstr(op='ASSIGN', dest=stmt.name, src1=val))
        elif isinstance(stmt, ReturnStatement):
            val = self.lower_expression(stmt.expr)
            self.instructions.append(TACInstr(op='RETURN', dest=None, src1=val))
        elif isinstance(stmt, PrintStatement):
            val = self.lower_expression(stmt.expr)
            self.instructions.append(TACInstr(op='PRINT', dest=None, src1=val))
        elif isinstance(stmt, IfStatement):
            cond_val = self.lower_expression(stmt.condition)
            else_lbl = self.new_label("else") if stmt.else_body is not None else None
            end_lbl = self.new_label("endif")

            target_lbl = else_lbl if else_lbl else end_lbl
            self.instructions.append(TACInstr(op='JUMP_IF_ZERO', dest=target_lbl, src1=cond_val))

            for s in stmt.then_body:
                self.lower_statement(s)

            if else_lbl:
                self.instructions.append(TACInstr(op='JUMP', dest=end_lbl, src1=None))
                self.instructions.append(TACInstr(op='LABEL', dest=else_lbl, src1=None))
                for s in stmt.else_body:
                    self.lower_statement(s)

            self.instructions.append(TACInstr(op='LABEL', dest=end_lbl, src1=None))

        elif isinstance(stmt, WhileStatement):
            start_lbl = self.new_label("while_start")
            end_lbl = self.new_label("while_end")

            self.instructions.append(TACInstr(op='LABEL', dest=start_lbl, src1=None))
            cond_val = self.lower_expression(stmt.condition)
            self.instructions.append(TACInstr(op='JUMP_IF_ZERO', dest=end_lbl, src1=cond_val))

            for s in stmt.body:
                self.lower_statement(s)

            self.instructions.append(TACInstr(op='JUMP', dest=start_lbl, src1=None))
            self.instructions.append(TACInstr(op='LABEL', dest=end_lbl, src1=None))
        else:
            raise NotImplementedError(f"Cannot lower statement: {stmt}")

    def lower_expression(self, expr: ASTNode) -> Union[str, int]:
        if isinstance(expr, Literal):
            return expr.value
        elif isinstance(expr, Variable):
            return expr.name
        elif isinstance(expr, BinaryExpr):
            left_val = self.lower_expression(expr.left)
            right_val = self.lower_expression(expr.right)
            t = self.new_temp()

            op_map = {
                '+': 'ADD', '-': 'SUB', '*': 'MUL', '/': 'DIV', '%': 'MOD',
                '<<': 'SHL', '>>': 'SHR',
                '<': 'LT', '>': 'GT', '<=': 'LE', '>=': 'GE', '==': 'EQ', '!=': 'NE'
            }
            ir_op = op_map.get(expr.op)
            if not ir_op:
                raise ValueError(f"Unsupported binary operator {expr.op}")

            self.instructions.append(TACInstr(op=ir_op, dest=t, src1=left_val, src2=right_val))
            return t
        else:
            raise NotImplementedError(f"Cannot lower expression: {expr}")


def compile_source_to_tac(source_code: str) -> List[TACInstr]:
    """End-to-end frontend pipeline: source string -> Lexer -> Parser -> AST -> List[TACInstr]."""
    tokens = tokenize(source_code)
    parser = Parser(tokens)
    ast = parser.parse()
    lowerer = TACLowerer()
    return lowerer.lower(ast)
