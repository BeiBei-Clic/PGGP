import math
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
        return math.asin(a)
    else:
        return -1

def back_tan(a, rlt_posi):
    "x=sin a"
    if -1 <= a <= 1:
        return math.atan(a)
    else:
        return -1

def back_cos(a, rlt_posi):
    "x=cos a"
    if -1 <= a <= 1:
        return math.acos(a)
    else:
        return -1

def back_tan(b, rlt_posi):
    #tan x = b
    return math.atan(b)

def back_pow(a, b, rlt_posi):
    "x ** a = b or a ** x = b."
    if rlt_posi == 0:#x ** a = b
        return b ** (1./a) if a != 0 else math.nan
    else:#a ** x = b
        return math.log(b, a) if b > 0 and a > 0 and a != 1 else math.nan

def back_exp(a, rlt_posi):
    "x=exp a"
    if a>0:
        return math.log(a)
    else:
        return -1
def back_log(a, rlt_posi):
    "x=log a"
    return math.exp(a)


def back_asin(b, rlt_posi):
    #asin x = b
    return math.sin(b)

def back_acos(b, rlt_posi):
    #acos x = b
    return math.cos(b)

def back_atan(b, rlt_posi):
    #atan x = b
    return math.tan(b)

oper_set = {'mul': back_mul, 'div': back_truediv, 'sub': back_sub, 'add': back_add,
            'neg': back_neg, 'sin': back_sin, 'cos': back_cos, 'tan': back_tan, 'exp': back_exp, 'ln': back_log, 'pow': back_pow, 'asin': back_asin,
            'acos': back_acos, 'atan': back_atan}