import math
def protectedMul(left, right):
    try:
        return left*right
    except OverflowError:
        return 1e7
# def protectedDiv(left, right):
#     try:
#         return left / right
#     except ZeroDivisionError:
#         return 1
def protectedDiv(left, right):
    if right == 0:
        return left
    res = left / right
    if res > 1e7:
        return 1e7
    if res < -1e7:
        return -1e7
    return res
def protectedExp(arg):
    if arg > 10:
        arg = 10
    return math.exp(arg)

def protectedLog(arg):
    if abs(arg) < 1e-5:
        arg = 1e-5
    return math.log(abs(arg))


def protectedAsin(x):
    if x < -1.0 or x > 1.0:
        return 99999
    else:
        return math.asin(x)


def protectedAcos(x):
    if x < -1.0 or x > 1.0:
        return 99999
    else:
        return math.asin(x)

def protectedAtan(x):
    try:
        # Calculate atan
        return math.atan(x)
    except Exception as e:
        # Handle exceptions (e.g., invalid input)
        print(f"Error: {e}")
        # Return a default value or handle the error as needed
        return None


#================================================================ ======================================

def back_mul(a, b, rlt_posi):
    "a * x = b."
    if a == 0 and b != a:
        return b / math.sqrt(1 + a * a)
    else:
        return b / a if b != a else 1

def back_sub(a, b, rlt_posi):
    "a - x = b or x - a = b."
    if rlt_posi == 0:#x-a
        return a + b
    else:#a-x
        return a - b

def back_truediv(a, b, rlt_posi):
    "a / x = b or x / a = b."
    if rlt_posi == 0:#x/a
        return a * b
    else:#a/x

        if b == 0 and b != a:
            return a / math.sqrt(1 + b * b)
        else:
            return a / b if b != a else 1

def back_add(a, b, rlt_posi):
    "a + x = b"
    return b - a

def back_neg(a, rlt_posi):
    "x=-a"
    return -a

'''return NAN if a out of [-1, 1] '''
def back_sin(a, rlt_posi):
    "x=sin a"
    if -1 <= a <= 1:
        return protectedAsin(a)
    else:
        return float('nan')

def back_tan(a, rlt_posi):
    "x=sin a"
    if -1 <= a <= 1:
        return protectedAtan(a)
    else:
        return float('nan')

def back_cos(a, rlt_posi):
    "x=cos a"
    if -1 <= a <= 1:
        return protectedAcos(a)
    else:
        return float('nan')

def back_exp(a, rlt_posi):
    "x=exp a"
    if a>0:
        return protectedLog(a)
    else:
        return float('nan')
def back_log(a, rlt_posi):
    "x=log a"
    return protectedExp(a)
def back_asin(a, rlt_posi):
    "x=asin a"
    return math.sin(a)
def back_acos(a, rlt_posi):
    "x=acos a"
    return math.cos(a)
def back_atan(a, rlt_posi):
    "x=atan a"
    return math.tan(a)
def back_pow(a, b, rlt_posi):
    "x ** a = b or a ** x = b."
    try:
        if rlt_posi == 0:#x ** a = b
            return b ** (1./a) if a != 0 else math.nan
        else:#a ** x = b
            return math.log(b, a) if b > 0 and a > 0 and a != 1 else math.nan
    except OverflowError:
        return float('nan')

def back_sqrt(a, rlt_posi):
    "x ** 2 = a."
    return a**2

oper_set = {'mul': back_mul, 'div': back_truediv, 'sub': back_sub, 'add': back_add,
            'neg': back_neg, 'sin': back_sin, 'cos': back_cos, 'tan': back_tan, 'exp': back_exp, 'ln': back_log,
            'asin': back_asin, 'acos': back_acos, 'atan': back_atan, 'pow': back_pow}