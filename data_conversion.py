import re
import json
import omegaconf
import sympy
import copy
from sympy import expand
# def add_brackets_to_negatives(formula):
#     formula = re.sub(r'((?<=^)|(?<=[*+/]))-((\d+\.\d+|\d+)|([a-zA-Z_]\w*))', r'(-\2)', formula)
#     return formula

def handle_negatives(formula):
    # Define a string that matches a negative sign at the start, after an operator, or after an opening parenthesis
    negative_sign_pattern = r'((?<=^)|(?<=[*+/])|(?<=\())-'

    # Define a string that matches a number
    number_pattern = r'(\d+\.\d+|\d+)'

    # Define a string that matches a variable
    variable_pattern = r'([a-zA-Z_]\w*)'

    # Combine the patterns into two separate patterns
    negative_number_pattern = negative_sign_pattern + number_pattern
    negative_variable_pattern = negative_sign_pattern + variable_pattern

    # Replace negative numbers with 1
    formula = re.sub(negative_number_pattern, r'1', formula)

    # Replace negative variables with 1*variable
    formula = re.sub(negative_variable_pattern, r'1*\2', formula)

    return formula


# def expand_power(expression):
#     # fixme 目前不支持三角函数的指数运算
#     # print('expression:',expression)
#     def expand(match):
#         base = match.group(1)
#         exponent = int(match.group(2))
#         return '*'.join([base] * exponent)
#
#     def recursive_expand(expr):
#         pattern = re.compile(r'\(([^()]+)\)\*\*(\d+)')
#         mark=False
#         while '**' in expr:
#             expr = re.sub(pattern, expand, expr)
#             expr = re.sub(r'([a-zA-Z_]+\d*)\*\*(\d+)', expand, expr)
#             mark=True
#         if mark:
#             expr=str(sympy.simplify(expr))
#         return expr
#
#     return recursive_expand(expression)
# def expand_power(expression):
#     def expand_binomial(match):
#         # base = match.group(1)
#         # exponent = int(match.group(2))
#         # expanded_expr = str(expand(f'({base})**{exponent}'))
#         expanded_expr=str(expand(expression))
#         return expanded_expr
#
#     def recursive_expand(expr):
#         pattern = re.compile(r'\(([^()]+)\)\*\*(\d+)')
#         binomial_pattern = re.compile(r'\(([a-zA-Z_0-9\s+\-*]+)\)\*\*(\d+)')
#         expr = re.sub(binomial_pattern, expand_binomial, expr)
#         return expr
#
#     return recursive_expand(expression)

def expand_power(expression):
    return str(expand(expression))
def expand_repeat(expression):
    def expand(match):
        base = match.group(1)
        exponent = int(match.group(2))
        return '*'.join([base] * exponent)

    def recursive_expand(expr):
        pattern = re.compile(r'\(([^()]+)\)\*\*(\d+)')
        simple_pattern = re.compile(r'([a-zA-Z_]*\d*)\*\*(\d+)')
        while '**' in expr:
            # for expressions in parentheses
            expr = re.sub(pattern, expand, expr)
            # for simple expressions without parentheses
            expr = re.sub(simple_pattern, expand, expr)
        return expr

    return recursive_expand(expression)

# def expand_power(expression):
#     def expand_binomial(match):
#         base = match.group(1)
#         exponent = int(match.group(2))
#         expanded_expr = str(expand(f'({base})**{exponent}'))
#         return expanded_expr
#
#     def recursive_expand(expr):
#         pattern = re.compile(r'\(([^()]+)\)\*\*(\d+)')
#         binomial_pattern = re.compile(r'\(([a-zA-Z_0-9\s+\-*]+)\)\*\*(\d+)')
#         expr = re.sub(binomial_pattern, expand_binomial, expr)
#         return expr
#
#     return recursive_expand(expression)
def reverse_add_sub(expr):
    # Helper function to find the index of the character that closes the first open parenthesis.
    def find_closing_paren(expr, open_pos):
        counter = 1
        for i in range(open_pos + 1, len(expr)):
            if expr[i] == '(':
                counter += 1
            elif expr[i] == ')':
                counter -= 1
                if counter == 0:
                    return i
        return -1  # If no closing parenthesis is found

    # Find the leading negative sign and determine the boundary of its term.
    if expr.lstrip().startswith('-'):
        # Skip whitespace and the leading negative sign
        pos = next((i for i, ch in enumerate(expr) if not ch.isspace()), None) + 1
        # We'll store the index positions where operators appear
        operators = []
        while pos < len(expr):
            char = expr[pos]
            if char == '(':
                # Find the matching closing parenthesis and skip the content inside
                closing_pos = find_closing_paren(expr, pos)
                if closing_pos < 0:
                    # No closing parenthesis found so break
                    break
                else:
                    # Skip past the closing parenthesis
                    pos = closing_pos
            elif char in '+-':
                # Found an operator, so we remember its position
                operators.append(pos)
                break
            # Move to the next character position
            pos += 1

        # If operators are found, determine the expression before and after the found operator
        if operators:
            op_pos = operators[0]
            return expr[op_pos + 1:] + ' - ' + expr[:op_pos].lstrip('-')
        else:
            # If there's no '+' or '-' after the leading '-', simply return the expression as it is
            return expr

    # If the expression does not start with a negative sign, return it unchanged
    return expr

def infix_to_prefix(expression):
    operators = set(['+', '-', '*', '/', 'sin', 'cos', 'tan','exp', 'ln','asin','acos','atan'])

    def is_operator(char):
        return char in operators

    def get_precedence(operator):
        precedence = {'+': 1, '-': 1, '*': 2, '/': 2, 'sin': 3, 'cos': 3, 'tan': 3,'exp': 3, 'ln': 3,'asin': 3,'acos': 3,'atan': 3}
        return precedence.get(operator, 0)

    def infix_to_prefix_recursive(expr):
        stack = []
        output = []
        tokens = re.findall(r'([a-zA-Z_]+\d*|\*\*|\d+\.\d+|\d+|\S)', expr)
        # chatgpt改进之后的
        # tokens = re.findall(r'([a-zA-Z_]+\d*|\*+|\d+\.\d+|\d+|\S)', expr)

        for i, token in enumerate(reversed(tokens)):
            if re.match(r'[a-zA-Z_]+\d*', token):
                output.append(token)
            elif re.match(r'\d+\.\d+|\d+', token):
                output.append(token)
            elif token == ')':
                stack.append(token)
            elif token == '(':
                while stack and stack[-1] != ')':
                    output.append(stack.pop())
                stack.pop()
            elif is_operator(token):
                while stack and get_precedence(stack[-1]) > get_precedence(token):
                    output.append(stack.pop())
                stack.append(token)
            elif token == '**':
                output.append(token)
            elif token == '-' and (i == 0 or (i > 0 and (tokens[i - 1] in operators or tokens[i - 1] == '('))):
                # Handle negative sign as a unary operator for the entire number
                output.append('(-')
                stack.append(')')
            elif token == '-' and i > 0 and re.match(r'\d+\.\d+|\d+', tokens[i - 1]):
                # Handle negative constant
                if i < len(tokens) - 1 and tokens[i + 1] == '(':
                    # Subtract within parentheses
                    output.append('-')
                else:
                    output.append('*')
                    output.append('(-')
                    stack.append(')')

        while stack:
            output.append(stack.pop())

        return ''.join(reversed(output))

    return infix_to_prefix_recursive(expression)


# def split_expr(expr):
#     # tokens = re.findall(r'\(-\d+\.\d+\)|\d+\.\d+|-?\d+|x_[12]|sin|cos|tan|[/*\-+^]', expr)
#     tokens = re.findall(r'\(-\d+\.\d+\)|\d+\.\d+|-?\d+|x_[12]|sin|cos|tan|c|[/*\-+^]', expr)
#     result = []
#     for token in tokens:
#         if token.startswith('(') and token.endswith(')'):  # If the token is a negative number
#             token = token[1:-1]  # exclude the parentheses '(' and ')'
#         try:
#             result.append(float(token))
#         except ValueError:  # If it's not a number, keep it as is
#             result.append(token)
#
#     return result


# def split_expr(expr):
#     # tokens = re.findall(r'([a-zA-Z_]+\d*|\*\*|\d+\.\d+|\d+|\S)', expr)
#     # tokens = re.findall(r'([a-zA-Z_]+\d*|\d+\.\d+|\d+|\*|\S)', expr)
#
#     functions = ['tan', 'sin', 'cos']
#     # Create a regex pattern for the functions
#     func_pattern = '|'.join(functions)
#     # The full regex pattern includes the functions
#     pattern = fr'({func_pattern}|[a-zA-Z_]+\d*|\d+\.\d+|\d+|\*|\S)'
#     tokens = re.findall(pattern, expr)
#     # tokens = re.findall(r'([a-zA-Z_]+\d*|\*+|\d+\.\d+|\d+|\S)', expr)
# #     I need to remove the brackets from the constants
#     result = []
#     for token in tokens:
#         if token.startswith('(') and token.endswith(')'):  # If the token is a negative number
#             token = token[1:-1]  # exclude the parentheses '(' and ')'
#         try:
#             result.append(float(token))
#         except ValueError:  # If it's not a number, keep it as is
#             result.append(token)
#
#     return result

def split_expr(expr):
    functions = ['sin', 'cos', 'tan','exp', 'ln', 'sqrt', 'pow', 'log', 'abs', 'sign', 'floor', 'ceil', 'round', 'trunc','asin','acos','atan']
    allowed_vars = ['x_0', 'x_1', 'x_2', 'x_3', 'c_0', 'c_1', 'c_2', 'c_3', 'c_4', 'c_5', 'c_6', 'c_7']

    # Create patterns for the functions and allowed variables, and escape them
    funcs_pattern = '|'.join(functions)
    allowed_vars_pattern = '|'.join(map(re.escape, allowed_vars))

    # Full regex pattern to match functions, allowed variables and general tokens
    pattern = fr'({funcs_pattern}|{allowed_vars_pattern}|[a-zA-Z]+(?!\d)|\d+\.\d+|\d+|\*|\S)'
    tokens = re.findall(pattern, expr)
    return tokens



def convert_inverse_prim(prim, args):
    """
    Convert inverse prims according to:
    [Dd]iv(a,b) -> Mul[a, 1/b]
    [Ss]ub(a,b) -> Add[a, -b]
    We achieve this by overwriting the corresponding format method of the sub and div prim.
    """
    prim = copy.copy(prim)
    #prim.name = re.sub(r'([A-Z])', lambda pat: pat.group(1).lower(), prim.name)    # lower all capital letters

    converter = {
        'sub': lambda *args_: "Add({}, Mul(-1,{}))".format(*args_),
        'protectedDiv': lambda *args_: "Mul({}, Pow({}, -1))".format(*args_),
        'mul': lambda *args_: "Mul({},{})".format(*args_),
        'add': lambda *args_: "Add({},{})".format(*args_)
    }
    prim_formatter = converter.get(prim.name, prim.format)

    return prim_formatter(*args)

def stringify_for_sympy(f):
    """Return the expression in a human readable string.
    """
    string = ""
    stack = []
    for node in f:
        stack.append((node, []))
        while len(stack[-1][1]) == stack[-1][0].arity:
            prim, args = stack.pop()
            string = convert_inverse_prim(prim, args)
            if len(stack) == 0:
                break  # If stack is empty, all nodes should have been seen
            stack[-1][1].append(string)
    return string

