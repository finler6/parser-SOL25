"""
Jméno a příjmení: Gleb Litvinchuk
Login: xlitvi02
"""

#!/usr/bin/env python3
import sys
import argparse
import re
import xml.etree.ElementTree as ET
import xml.dom.minidom
from collections import namedtuple

# Definice struktury tokenu
Token = namedtuple("Token", ["type", "value", "pos"])

# =====================================================================
# Výjimky s kódy chyb pro správné ukončení
# =====================================================================

class SOL25Error(Exception):
    """Bázová třída pro všechny výjimky v překladači SOL25."""
    def __init__(self, message, error_code):
        super().__init__(message)
        self.error_code = error_code

class LexicalError(SOL25Error):
    def __init__(self, message="Lexikální chyba"):
        super().__init__(message, 21)

class SyntacticError(SOL25Error):
    def __init__(self, message="Syntaktická chyba", error_code=22):
        super().__init__(message, error_code)

class SemanticError(SOL25Error):
    def __init__(self, message="Sémantická chyba", error_code=35):
        super().__init__(message, error_code)

# =====================================================================
# Definice uzlů AST pro jazyk SOL25
# =====================================================================

class ProgramNode:
    def __init__(self):
        self.classes = []
        self.description = None

class ClassNode:
    def __init__(self, name, parent):
        self.name = name
        self.parent = parent
        self.methods = []

class MethodNode:
    def __init__(self, selector):
        self.selector = selector
        self.block = None

class BlockNode:
    def __init__(self, arity=0):
        self.arity = arity
        self.parameters = []
        self.assignments = []

class AssignmentNode:
    def __init__(self, order, var_name, expr):
        self.order = order
        self.var_name = var_name
        self.expr = expr

# Třída pro reprezentaci uzlů stromu výrazů
class ExpressionNode:
    pass

class LiteralNode(ExpressionNode):
    def __init__(self, literal_class, value):
        self.literal_class = literal_class
        self.value = value

class VarNode(ExpressionNode):
    def __init__(self, name):
        self.name = name

class SendNode(ExpressionNode):
    def __init__(self, selector, receiver, args):
        self.selector = selector
        self.receiver = receiver
        self.args = args

# =====================================================================
# Hlavní třída SOL25Parser: lexikální analýza, parsování a statická sémantika
# =====================================================================

class SOL25Parser:
    def __init__(self, source_code):
        self.source_code = source_code
        self.tokens = []
        self.pos = 0
        self.first_comment = None

    def parse(self):
        self.tokens, self.first_comment = self._lex()
        self.pos = 0
        ast = self._parse_program()
        self._check_semantics(ast)
        self._check_inheritance(ast)
        if self.first_comment:
            ast.description = self.first_comment[1:-1]
        return ast

    def _check_inheritance(self, ast: ProgramNode):
        builtins = {"Object", "Integer", "String", "Nil", "True", "False"}
        defined = {cls.name for cls in ast.classes}
        for cls in ast.classes:
            if cls.parent not in defined and cls.parent not in builtins:
                raise SemanticError(f"Use of undefined class '{cls.parent}'", 32)

    def _debug_token_info(self, token):
        line = self.source_code.count("\n", 0, token.pos) + 1
        return f"line {line}, pos {token.pos}"

    # -----------------------------------------------------------------
    # Lexikální analýza
    # -----------------------------------------------------------------
    def _lex(self):
        token_specification = [
            ('COMMENT', r'"(?:\\.|[^"\\])*"'),
            ('WHITESPACE', r'[ \t\n\u00A0]+'),
            ('LPAR', r'\('),
            ('RPAR', r'\)'),
            ('ASSIGN', r':='),
            ('DOT', r'\.'),
            ('COLON', r':'),
            ('BAR', r'\|'),
            ('LBRACKET', r'\['),
            ('RBRACKET', r'\]'),
            ('LBRACE', r'\{'),
            ('RBRACE', r'\}'),
            ('KEYWORD', r'\b(?:class|self|super|nil|true|false)\b'),
            ('CLASS_ID', r'\b[A-Z][A-Za-z0-9]*\b'),
            ('IDENTIFIER', r'\b[a-z_][A-Za-z0-9_]*\b'),
            ('INTEGER', r'[+-]?\d+'),
            ('STRING', r"'(?:\\.|[^'\\])*'"),
        ]
        tok_regex = '|'.join(f"(?P<{name}>{pattern})" for name, pattern in token_specification) # vytvoření regexu
        get_token = re.compile(tok_regex).match
        pos = 0
        tokens = []
        first_comment = None
        while pos < len(self.source_code):
            m = get_token(self.source_code, pos)
            if m:
                typ = m.lastgroup
                val = m.group(typ)
                if typ == "WHITESPACE":
                    pass
                elif typ == "COMMENT":
                    if first_comment is None:
                        first_comment = val
                else:
                    tokens.append(Token(typ, val, pos))
                pos = m.end()
            else:
                raise LexicalError(f"Unexpected character: {self.source_code[pos]}")
        return tokens, first_comment

    # -----------------------------------------------------------------
    # Pomocné metody pro práci s tokeny
    # -----------------------------------------------------------------
    def _peek(self):
        if self.pos < len(self.tokens):
            return self.tokens[self.pos]
        return None

    def _advance(self): # posun na další token
        token = self.tokens[self.pos]
        self.pos += 1
        return token

    def _expect(self, token_type, value=None): # očekávání konkrétního tokenu
        token = self._peek()
        if token is None:
            raise SyntacticError(f"Expected {token_type} but reached end of input")
        if token.type != token_type:
            raise SyntacticError(f"Expected token type {token_type} but got {token.type}")
        if value is not None and token.value != value:
            raise SyntacticError(f"Expected token value {value} but got {token.value}")
        return self._advance()

    # -----------------------------------------------------------------
    # Syntaktická analýza (rekurzivní sestup)
    # -----------------------------------------------------------------
    def _parse_program(self):
        program_node = ProgramNode()
        while self._peek() is not None:
            class_node = self._parse_class()
            program_node.classes.append(class_node)
        return program_node

    def _parse_class(self):
        self._expect("KEYWORD", "class")
        token = self._expect("CLASS_ID")
        class_name = token.value
        self._expect("COLON")
        token = self._expect("CLASS_ID")
        parent_name = token.value
        self._expect("LBRACE")
        class_node = ClassNode(class_name, parent_name)
        while self._peek() is not None and self._peek().type != "RBRACE":
            method_node = self._parse_method()
            class_node.methods.append(method_node)
        self._expect("RBRACE")
        return class_node

    def _parse_method(self):
        selector_tokens = []
        while self._peek() is not None and self._peek().type != "LBRACKET":
            token = self._advance()
            if token.type == "CLASS_ID":
                raise SyntacticError(f"Method selector cannot contain class identifier '{token.value}'", 22)
            if token.type == "COLON" and not selector_tokens:
                raise SyntacticError("Method selector cannot start with a colon", 22)
            selector_tokens.append(token)
        if not selector_tokens:
            raise SyntacticError("Missing method selector", 22)
        start = selector_tokens[0].pos
        end = selector_tokens[-1].pos + len(selector_tokens[-1].value)
        selector_text = self.source_code[start:end]
        if any(ch.isspace() for ch in selector_text):
            raise SyntacticError("Invalid method selector format", 22)
        selector = "".join(token.value for token in selector_tokens).strip() # vytvoření selektoru
        reserved = {"self", "super", "nil", "true", "false", "class"} # rezervovaná bulit-in slova
        if selector in reserved:
            raise SyntacticError(f"Reserved identifier '{selector}' cannot be used as a method selector", 22)
        block = self._parse_block()
        expected_params = selector.count(":")
        if block.arity != expected_params:
            raise SemanticError(
                f"Method arity mismatch: selector expects {expected_params} parameters, but block has {block.arity}",
                33
            )
        method_node = MethodNode(selector)
        method_node.block = block
        return method_node

    def _parse_block(self):
        lbracket = self._expect("LBRACKET")
        block_node = BlockNode()
        if self._peek() is not None and self._peek().type == "COLON":
            while self._peek() is not None and self._peek().type == "COLON":
                colon_token = self._advance()
                if self._peek() is None or self._peek().pos != (colon_token.pos + len(colon_token.value)):
                    raise SyntacticError("Invalid block parameter format", 22)
                param_token = self._expect("IDENTIFIER")
                block_node.parameters.append(param_token.value.strip())
            if len(block_node.parameters) != len(set(block_node.parameters)):
                raise SemanticError("Duplicate formal parameter", 35)
            self._expect("BAR")
        elif self._peek() is not None and self._peek().type == "BAR":
            self._advance()
        else:
            raise SyntacticError("Expected parameter declarations or BAR in block", 22)
        order = 1
        while self._peek() is not None and self._peek().type != "RBRACKET":
            if self._peek().type == "DOT":
                temp_pos = self.pos
                self._advance()
                if self._peek() is not None and self._peek().type == "RBRACKET":
                    break
                else:
                    self.pos = temp_pos
            assignment = self._parse_assignment(order)
            block_node.assignments.append(assignment)
            order += 1
        self._expect("RBRACKET")
        block_node.arity = len(block_node.parameters)
        return block_node

    def _parse_assignment(self, order):
        token = self._expect("IDENTIFIER")
        var_name = token.value
        next_tok = self._peek()
        if next_tok is not None and next_tok.type == "ASSIGN":
            self._advance()
            expr = self._parse_expression(in_assignment=True)
        else:
            raise SyntacticError(
                f"Expected token type ASSIGN after variable '{var_name}', but got {next_tok.type if next_tok else 'EOF'}"
            )
        self._expect("DOT")
        return AssignmentNode(order, var_name, expr)

    def _parse_expression(self, in_assignment=False, allow_message_send=True):
        expr = self._parse_primary()
        if allow_message_send:
            next_tok = self._peek()
            if next_tok is not None and next_tok.type == "IDENTIFIER":
                if in_assignment:
                    expr = self._parse_message_send(expr)
                else:
                    raise SyntacticError(
                        f"Message send must be enclosed in parentheses when not used directly as an assignment right-hand side at {self._debug_token_info(next_tok)}",
                        22
                    )
        return expr

    def _parse_parenthesized_expression(self):
        lpar = self._expect("LPAR")
        receiver = self._parse_expression(in_assignment=True)
        if self._peek() is not None and self._peek().type == "RPAR":
            self._expect("RPAR")
            return receiver
        selector = ""
        args = []
        while True:
            token = self._peek()
            if token is None or token.type != "IDENTIFIER":
                break
            current_pos = self.pos
            ident_token = self._advance()
            if self._peek() is not None and self._peek().type == "COLON":
                self._advance()
                selector += ident_token.value + ":"
                arg_expr = self._parse_expression(in_assignment=True, allow_message_send=False)
                args.append(arg_expr)
            else:
                if selector:
                    self.pos = current_pos
                    break
                else:
                    selector = ident_token.value
            if self._peek() is None or self._peek().type not in ("IDENTIFIER", "COLON"):
                break
        self._expect("RPAR")
        return SendNode(selector, receiver, args)

    # -----------------------------------------------------------------
    # Zpracování řetězcových literálů: kontrola escape sekvence
    # -----------------------------------------------------------------
    def _process_string(self, raw_string):
        s = raw_string[1:-1]
        if "\n" in s:
            raise LexicalError("Lexikální chyba")
        result = ""
        i = 0
        while i < len(s):
            if s[i] == '\\':
                if i + 1 >= len(s):
                    raise LexicalError("Incomplete escape sequence in string literal")
                next_char = s[i + 1]
                if next_char == "'" or next_char == "\\":
                    result += "\\" + next_char
                elif next_char == "n":
                    result += "\\n"
                else:
                    raise LexicalError(f"Invalid escape sequence \\{next_char}")
                i += 2
            else:
                result += s[i]
                i += 1
        return result

    # -----------------------------------------------------------------
    # Pomocné funkce pro statickou sémantiku
    # -----------------------------------------------------------------
    def _collect_var_names(self, expr):
        names = set()
        if isinstance(expr, VarNode):
            names.add(expr.name)
        elif isinstance(expr, LiteralNode):
            pass
        elif isinstance(expr, SendNode):
            names.update(self._collect_var_names(expr.receiver))
            for arg in expr.args:
                names.update(self._collect_var_names(arg))
        return names

    def _check_block_semantics(self, block: BlockNode, defined=None): # kontrola blokového výrazu
        if defined is None:
            defined = set()
        defined = defined.union(set(block.parameters)).union({"self", "super"})
        for assign in block.assignments:
            if assign.var_name in block.parameters:
                raise SemanticError(f"Assignment to formal parameter '{assign.var_name}' is not allowed", 34)
            used_vars = self._collect_var_names(assign.expr)
            for var in used_vars:
                if var not in defined:
                    raise SemanticError(f"Use of undefined variable '{var}'", 32)
            if isinstance(assign.expr, BlockNode):
                self._check_block_semantics(assign.expr, defined.copy())
            if assign.var_name not in defined:
                defined.add(assign.var_name)

    """Metoda pro získání všech tříd, které dědí danou třídu"""
    def _compute_allowed_methods(self, ast: ProgramNode):
        builtin_allowed = {
            "Integer": {"new", "from:", "equalTo:", "greaterThan:", "plus:", "minus:", "multiplyBy:", "divBy:",
                        "asString", "asInteger", "timesRepeat:"},
            "String": {"new", "read", "print", "equalTo:", "asString", "asInteger", "concatenateWith:",
                       "startsWith:endsBefore:"},
            "Nil": {"asString"},
            "True": {"not", "and:", "or:", "ifTrue:ifFalse:"},
            "False": {"not", "and:", "or:", "ifTrue:ifFalse:"},
            "Object": {"new", "from:", "identicalTo:", "equalTo:", "asString", "isNumber", "isString", "isBlock", "isNil"},
            "Block": {"value", "value:", "value:value:", "value:value:value:"}
        }
        allowed_mapping = {}

        for built_in in {"Integer", "String", "Nil", "True", "False", "Object"}:
            allowed_mapping[built_in] = builtin_allowed[built_in]

        for cls in ast.classes:
            if cls.name in builtin_allowed:
                own_selectors = {method.selector for method in cls.methods}
                allowed_mapping[cls.name] = allowed_mapping[cls.name].union(own_selectors) # přidání vlastních selektorů
            else:
                if cls.parent in allowed_mapping:
                    parent_allowed = allowed_mapping[cls.parent]
                elif cls.parent in builtin_allowed:
                    parent_allowed = builtin_allowed[cls.parent]
                else:
                    parent_allowed = set()
                own_selectors = {method.selector for method in cls.methods}
                allowed_mapping[cls.name] = parent_allowed.union(own_selectors)
        return allowed_mapping

    """Metoda pro kontrolu dědičnosti"""
    def _check_semantics(self, ast: ProgramNode):
        seen = {}
        for cls in ast.classes:
            if cls.name in seen:
                raise SemanticError(f"Class redefinition: {cls.name}", 35)
            seen[cls.name] = cls

        main_class = None
        for cls in ast.classes:
            if cls.name == "Main":
                main_class = cls
                break
        if main_class is None:
            raise SemanticError("Missing Main class", 31)
        run_found = False
        for method in main_class.methods:
            if method.selector.strip() == "run":
                if method.block and method.block.arity == 0:
                    run_found = True
                    break
        if not run_found:
            raise SemanticError("Missing run method in Main class", 31)

        for cls in ast.classes:
            for method in cls.methods:
                if method.block:
                    self._check_block_semantics(method.block)

        self._check_inheritance(ast)
        self._check_circular_inheritance(ast)

        allowed_mapping = self._compute_allowed_methods(ast)
        allowed_classes = {cls.name for cls in ast.classes}.union(
            {"Object", "Integer", "String", "Nil", "True", "False"}) # všechny třídy + built-in třídy

        ast = self._transform_class_literals(ast, allowed_classes)

        for cls in ast.classes:
            seen_methods = set()
            for method in cls.methods:
                if method.selector in seen_methods:
                    raise SemanticError(f"Duplicate method selector: {method.selector}", 35)
                seen_methods.add(method.selector)

        for cls in ast.classes:
            for method in cls.methods:
                if method.block:
                    self._check_expr_class_usage(method.block, allowed_classes, allowed_mapping)

    def _check_circular_inheritance(self, ast: ProgramNode): # kontrola cyklické dědičnosti
        graph = {cls.name: cls.parent for cls in ast.classes} # vytvoření grafu dědičnosti
        for cls in graph:
            visited = set()
            current = cls
            while current in graph:
                if current in visited:
                    raise SemanticError("Circular inheritance detected", 35)
                visited.add(current)
                current = graph[current]

    def _parse_primary(self):
        # Získáme aktuální token bez posunu (nahlédnutí do fronty tokenů)
        token = self._peek()
        # Pokud není žádný token, signalizujeme syntaktickou chybu - neočekávaný konec vstupu
        if token is None:
            raise SyntacticError("Unexpected end of input in primary expression")
        # Zpracování celočíselného literálu
        if token.type == "INTEGER":
            token = self._advance()  # Přejdeme na další token
            return LiteralNode("Integer", token.value)
        # Zpracování řetězcového literálu s následným ošetřením escape sekvencí
        elif token.type == "STRING":
            token = self._advance()
            processed = self._process_string(token.value)
            return LiteralNode("String", processed)
        # Zpracování klíčových slov: literály nil, true, false nebo proměnné self/super
        elif token.type == "KEYWORD":
            # Pokud se jedná o literály nil, true, false
            if token.value in ("nil", "true", "false"):
                token = self._advance()
                if token.value == "nil":
                    return LiteralNode("Nil", token.value)
                elif token.value == "true":
                    return LiteralNode("True", token.value)
                else:
                    return LiteralNode("False", token.value)
            # Pokud se jedná o speciální proměnné self a super
            elif token.value in ("self", "super"):
                token = self._advance()
                return VarNode(token.value)
            else:
                # Nečekané klíčové slovo vede k syntaktické chybě
                raise SyntacticError(f"Unexpected keyword in primary expression: {token.value}")
        # Zpracování identifikátoru třídy – vytváříme literál třídy
        elif token.type == "CLASS_ID":
            token = self._advance()
            return LiteralNode("class", token.value)
        # Zpracování běžného identifikátoru – vracíme jako proměnnou
        elif token.type == "IDENTIFIER":
            token = self._advance()
            return VarNode(token.value)
        # Pokud token značí začátek bloku, předáme řízení metodě pro parsování bloku
        elif token.type == "LBRACKET":
            return self._parse_block()
        # Pokud token značí levou závorku, parsujeme výraz v závorce nebo zprávu
        elif token.type == "LPAR":
            return self._parse_parenthesized_expression()
        else:
            # Pokud token neodpovídá žádné očekávané kategorii, vyvoláme syntaktickou chybu
            raise SyntacticError(f"Unexpected token in primary expression: {token.value}")

    def _transform_class_literals(self, node, allowed):
        # Transformace literálů reprezentujících třídy na speciální literály
        if isinstance(node, VarNode):
            # Pokud název proměnné začíná velkým písmenem, jedná se o literál třídy
            if node.name and node.name[0].isupper():
                # Pokud není třída definovaná, vyvoláme semantickou chybu
                if node.name not in allowed:
                    raise SemanticError(f"Use of undefined variable '{node.name}'", 32)
                # Vracíme transformovaný literál třídy
                return LiteralNode("class", node.name)
            else:
                return node
        # Literály necháváme beze změny
        elif isinstance(node, LiteralNode):
            return node
        # Rekurzivní transformace pro uzly odeslání zprávy (message send)
        elif isinstance(node, SendNode):
            new_receiver = self._transform_class_literals(node.receiver, allowed)
            new_args = [self._transform_class_literals(arg, allowed) for arg in node.args]
            return SendNode(node.selector, new_receiver, new_args)
        # Rekurzivní transformace pro blokové uzly
        elif isinstance(node, BlockNode):
            new_assignments = [self._transform_class_literals(assign, allowed) for assign in node.assignments]
            new_block = BlockNode(node.arity)
            new_block.parameters = node.parameters[:]  # Zkopírujeme seznam parametrů
            new_block.assignments = new_assignments
            return new_block
        # Transformace přiřazovacích uzlů – zpracování pravé strany přiřazení
        elif isinstance(node, AssignmentNode):
            new_expr = self._transform_class_literals(node.expr, allowed)
            return AssignmentNode(node.order, node.var_name, new_expr)
        # Transformace uzlů metod – zpracování bloku metody
        elif isinstance(node, MethodNode):
            new_block = self._transform_class_literals(node.block, allowed)
            new_method = MethodNode(node.selector)
            new_method.block = new_block
            return new_method
        # Transformace uzlů tříd – zpracování metod v rámci třídy
        elif isinstance(node, ClassNode):
            new_methods = [self._transform_class_literals(m, allowed) for m in node.methods]
            new_class = ClassNode(node.name, node.parent)
            new_class.methods = new_methods
            return new_class
        # Transformace kořenového uzlu programu – zpracování všech tříd v programu
        elif isinstance(node, ProgramNode):
            new_classes = [self._transform_class_literals(c, allowed) for c in node.classes]
            new_program = ProgramNode()
            new_program.classes = new_classes
            new_program.description = node.description
            return new_program
        else:
            # Pokud uzel není známého typu, vracíme jej beze změny
            return node

    def _get_effective_type(self, node):
        """
        Vrací tuple (cls_name, kind) pro daný uzel, pokud představuje třídu.
        Pokud je uzel literál třídy (LiteralNode s literal_class „class“), kind == 'class'.
        Pokud je uzel blokový literál (BlockNode), vrací („Block“, „instance“).
        Pokud je uzel SendNode se selektorem „from:“ a jeho příjemce má efektivní typ,
        pak je výsledkem instance této třídy (kind == 'instance').
        V opačném případě vrací None.
        """
        if isinstance(node, LiteralNode) and node.literal_class == "class":
            return (node.value, 'class')
        if isinstance(node, BlockNode):
            return ("Block", "instance")
        if isinstance(node, SendNode) and node.selector == "from:":
            eff = self._get_effective_type(node.receiver)
            if eff is not None:
                return (eff[0], 'instance')
        return None

    def _check_expr_class_usage(self, node, allowed_classes, allowed_mapping):
        """
        Prochází AST a kontroluje, zda jsou třídy použity správně.
        allowed_classes je množina povolených tříd.
        allowed_mapping je slovník, kde klíčem je název třídy a hodnotou množina povolených selektorů.
        """
        if isinstance(node, LiteralNode):
            if node.literal_class in {"class", "Integer", "String", "Nil", "True", "False"}:
                cls_name = node.value if node.literal_class == "class" else node.literal_class
                if cls_name not in allowed_classes:
                    raise SemanticError(f"Use of undefined class '{cls_name}'", 32)
            return
        elif isinstance(node, SendNode):
            eff = self._get_effective_type(node.receiver)
            if eff is not None:
                cls_name, kind = eff
                if cls_name == "Block":
                    pass
                else:
                    if kind == 'instance':
                        allowed_set = allowed_mapping.get(cls_name, set()) - {"new", "from:"}
                    else:
                        allowed_set = allowed_mapping.get(cls_name, set())
                    if node.selector not in allowed_set:
                        if kind == 'class':
                            raise SemanticError(f"Class '{cls_name}' does not support selector '{node.selector}'", 32)
                        else:
                            raise SemanticError(f"Class '{cls_name}' does not support selector '{node.selector}'", 51)
            else:
                if (isinstance(node.receiver, LiteralNode) and
                        node.receiver.literal_class in {"class", "Integer", "String", "Nil", "True", "False"}):
                    cls_name = node.receiver.value if node.receiver.literal_class == "class" else node.receiver.literal_class
                    allowed_set = allowed_mapping.get(cls_name, set())
                    if node.selector not in allowed_set:
                        raise SemanticError(f"Class '{cls_name}' does not support selector '{node.selector}'", 32)
            self._check_expr_class_usage(node.receiver, allowed_classes, allowed_mapping)
            for arg in node.args:
                self._check_expr_class_usage(arg, allowed_classes, allowed_mapping)
        elif isinstance(node, BlockNode):
            for assign in node.assignments:
                self._check_expr_class_usage(assign.expr, allowed_classes, allowed_mapping)
        elif isinstance(node, AssignmentNode):
            self._check_expr_class_usage(node.expr, allowed_classes, allowed_mapping)
        elif isinstance(node, MethodNode):
            self._check_expr_class_usage(node.block, allowed_classes, allowed_mapping)
        elif isinstance(node, ClassNode):
            for method in node.methods:
                self._check_expr_class_usage(method, allowed_classes, allowed_mapping)
        elif isinstance(node, ProgramNode):
            for cls in node.classes:
                self._check_expr_class_usage(cls, allowed_classes, allowed_mapping)

    def _parse_message_send(self, receiver):
        token = self._peek()
        if token is None or token.type != "IDENTIFIER":
            return receiver

        selector_parts = []
        args = []
        ident_token = self._advance()
        last_end = ident_token.pos + len(ident_token.value)

        if self._peek() is not None and self._peek().type == "COLON":
            colon_token = self._peek()
            if self.source_code[last_end:colon_token.pos] != "":
                raise SyntacticError("Invalid message selector format", 22)
            self._advance()
            selector_parts.append(ident_token.value + ":")
            last_end = colon_token.pos + len(colon_token.value)
            arg_expr = self._parse_expression(in_assignment=True, allow_message_send=False)
            args.append(arg_expr)
            while self._peek() is not None and self._peek().type == "IDENTIFIER":
                next_ident = self._advance()
                if self._peek() is not None and self._peek().type == "COLON":
                    colon_token = self._peek()
                    if self.source_code[next_ident.pos + len(next_ident.value):colon_token.pos] != "":
                        raise SyntacticError("Invalid message selector format", 22)
                    self._advance()
                    selector_parts.append(next_ident.value + ":")
                    last_end = colon_token.pos + len(colon_token.value)
                    arg_expr = self._parse_expression(in_assignment=True, allow_message_send=False)
                    args.append(arg_expr)
                else:
                    break
            selector = "".join(selector_parts)
            return SendNode(selector, receiver, args)
        else:
            return SendNode(ident_token.value, receiver, [])


# =====================================================================
# Pomocné funkce pro generování XML
# =====================================================================
def _expr_to_xml(expr) -> ET.Element:
    element = ET.Element("expr") # vytvoření elementu pro výraz
    if isinstance(expr, LiteralNode):
        sub = ET.Element("literal", {"class": expr.literal_class, "value": expr.value}) # vytvoření elementu pro literál
        element.append(sub)
    elif isinstance(expr, VarNode):
        sub = ET.Element("var", {"name": expr.name}) # vytvoření elementu pro proměnnou
        element.append(sub)
    elif isinstance(expr, SendNode):
        send_elem = ET.Element("send", {"selector": expr.selector}) # vytvoření elementu pro odeslání zprávy
        receiver_elem = _expr_to_xml(expr.receiver)
        send_elem.append(receiver_elem)
        arg_order = 1
        for arg in expr.args:
            arg_elem = ET.Element("arg", {"order": str(arg_order)}) # vytvoření elementu pro argument
            arg_expr_elem = _expr_to_xml(arg)
            arg_elem.append(arg_expr_elem)
            send_elem.append(arg_elem)
            arg_order += 1
        element.append(send_elem)
    elif isinstance(expr, BlockNode):
        block_elem = ET.Element("block", {"arity": str(expr.arity)}) # vytvoření elementu pro blok
        order = 1
        for param in expr.parameters:
            param_elem = ET.Element("parameter", {"name": param, "order": str(order)}) # vytvoření elementu pro parametr
            block_elem.append(param_elem)
            order += 1
        for assign in expr.assignments:
            assign_elem = ET.Element("assign", {"order": str(assign.order)}) # vytvoření elementu pro přiřazení
            var_elem = ET.Element("var", {"name": assign.var_name})
            assign_elem.append(var_elem)
            assign_expr_elem = _expr_to_xml(assign.expr)
            assign_elem.append(assign_expr_elem)
            block_elem.append(assign_elem)
        element.append(block_elem)
    else:
        unknown = ET.Element("unknown")
        element.append(unknown)
    return element

# =====================================================================
# Funkce pro generování XML z AST
# =====================================================================
def generate_xml(ast: ProgramNode) -> str:
    """ Funkce generuje XML reprezentaci AST.
        1. Vytvoří kořenový element "program" s atributy 'language' a (pokud existuje) 'description'.
        2. Pro každou třídu v AST vytvoří element "class" s atributy 'name' a 'parent'.
        3. U každé třídy prochází metody, pro každou metodu vytvoří element "method" s atributem 'selector'.
        4. Pokud metoda obsahuje blok, vytvoří element "block" s atributem 'arity' a vloží do něj:
           - Elementy "parameter" s informací o jménech a pořadí parametrů.
           - Elementy "assign" pro každé přiřazení, které obsahují podřízené elementy "var" a XML reprezentaci výrazu.
        5. Nakonec se vzniklé XML zformátuje pomocí xml.dom.minidom pro lepší čitelnost.
    """
    attribs = {"language": "SOL25"}
    if ast.description is not None:
        descr = ast.description.replace("\n", "&#10;")
        attribs["description"] = descr
    root = ET.Element("program", attribs)
    for cls in ast.classes:
        class_elem = ET.SubElement(root, "class", {"name": cls.name, "parent": cls.parent})
        for method in cls.methods:
            method_elem = ET.SubElement(class_elem, "method", {"selector": method.selector})
            if method.block:
                block_elem = ET.SubElement(method_elem, "block", {"arity": str(method.block.arity)})
                order = 1
                for param in method.block.parameters:
                    ET.SubElement(block_elem, "parameter", {"name": param, "order": str(order)})
                    order += 1
                for assign in method.block.assignments:
                    assign_elem = ET.SubElement(block_elem, "assign", {"order": str(assign.order)})
                    ET.SubElement(assign_elem, "var", {"name": assign.var_name})
                    expr_elem = _expr_to_xml(assign.expr)
                    assign_elem.append(expr_elem)
    rough_string = ET.tostring(root, encoding='utf-8')
    reparsed = xml.dom.minidom.parseString(rough_string)
    pretty = reparsed.toprettyxml(indent="    ", encoding="UTF-8").decode("utf-8")
    pretty = pretty.replace("&amp;#10;", "&#10;")
    return pretty

# =====================================================================
# Výstupní funkce nápovědy
# =====================================================================
def print_help():
    help_text = (
        "Usage: parse.py -h [--help]\n"
        "Reads SOL25 source code from standard input, checks its correctness, and outputs an XML representation of the AST."
    )
    print(help_text)

# =====================================================================
# Hlavní funkce
# =====================================================================
def main():
    arg_parser = argparse.ArgumentParser(add_help=False)
    arg_parser.add_argument("-h", "--help", action="store_true", help="Show this help message and exit")
    args, unknown = arg_parser.parse_known_args()

    if unknown or len(sys.argv) > 2:
        print("Error: Invalid arguments", file=sys.stderr)
        sys.exit(10)

    if args.help:
        print_help()
        sys.exit(0)

    source_code = sys.stdin.read()
    try:
        parser_instance = SOL25Parser(source_code)
        ast = parser_instance.parse()
        xml_output = generate_xml(ast)
        print(xml_output)
        sys.exit(0)
    except SOL25Error as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(e.error_code)
    except Exception as e:
        print(f"Internal error: {e}", file=sys.stderr)
        sys.exit(99)

if __name__ == "__main__":
    main()