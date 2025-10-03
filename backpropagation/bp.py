
import numpy as np
from .oper_nan_new import oper_set
from deap import gp

def deseries(indiv, childs, parent, idx):
    '''deep first search'''
    node = indiv[idx]
    cur_idx = idx

    for i in range(node.arity):
        childs[cur_idx].append(idx + 1)
        parent[idx + 1] = cur_idx
        idx = deseries(indiv, childs, parent, idx + 1)

    assert(len(childs[cur_idx]) == node.arity)
    return idx



# def back_exp(expr, pset):
#     oper = {'mul': back_mul, 'truediv': back_truediv, 'sub': back_sub, 'add': back_add, 'neg': back_neg}
#     code = str(expr)
#     print(pset.arguments, code)
#     if len(pset.arguments) > 0:
#         args = ",".join(arg for arg in pset.arguments)
#         code = "lambda {args}: {code}".format(args=args, code=code)
#         print(code, oper)
#         return eval(code, oper, {})
def get_subtr_rg(indiv, begin):
    end = begin + 1
    total = indiv[begin].arity
    while total > 0:
        total += indiv[end].arity - 1
        end += 1
    return (begin, end)
def rebuild_rlt(individual):
    '''get childs and parent of each node from list'''
    parent=[[] for i in range(len(individual))]
    childs=[[] for i in range(len(individual))]
    deseries(individual, childs, parent, 0)
    parent[0] = -1
    return (parent, childs)

def search_without_partial(individual, subtr_root):
    '''
    get a random tree node from an individual with a given subtree removed
    @param individual:  A PrimitiveTree type var
    @param subtr_root:  The given subtree id which will be exclued
    @return: a random node idx of the individual without the given subtree
    '''
    subtr_rg = get_subtr_rg(individual, subtr_root)
    return np.random.choice(individual[0:subtr_rg[0]] + individual[subtr_rg[1]:], 1) if len(individual) - 1 > subtr_rg[1] - subtr_rg[0] else None

def evaluate(individual,pset, dataset):
    (X, y) = dataset
    '''prog evaluate'''
    func = gp.compile(expr=individual, pset=pset)
    predict_vals = []
    for data in X:
        if len(data)>1:
            predict_vals.append(func(data[0],data[1]))
        else:
            predict_vals.append(func(data[0]))
    return predict_vals

    # print('tg_subtree', gp.PrimitiveTree(individual[tg_subtree]))
def is_complex(number):
    """
    Determine whether the given number is a complex number.

    :param number: The number to be checked.
    :return: True if the number is complex, False otherwise.
    """
    # A number is complex if it has a non-zero imaginary part

    if isinstance(number, list):
        for n in number:
            if isinstance(n, complex) and n.imag != 0:
                return True
    else:
        return isinstance(number, complex) and number.imag != 0

def backpropogation(individual, pset, dataset, target_idx):
    """
    semantic backpropagation
    @param individual:  A PrimitiveTree type var
    @param pset: PrimitiveSet
    @param dataset: train dataset as type (X, y)
    @param target_idx: the selected mutation point of the input individual
    @return: (predict_vals, smt_subtg): predict_vals is the predict output of the individual, smt_subtg is the subtarget semantic
    """
    # print('individual', individual)
    (X, y) = dataset

    '''subtree semantic prepare'''
    (parent, childs) = rebuild_rlt(individual)
    idx = target_idx
    slices = []
    compute_seq = []
    rlt_posis=[]

    while parent[idx] != -1:
        pr = parent[idx]
        rlt_posi = childs[pr].index(idx)
        rlt_posis.append(rlt_posi)
        childs[pr].pop(rlt_posi)
        slices.append([gp.PrimitiveTree(individual[individual.searchSubtree(child)])for child in childs[pr]])
        idx=parent[idx]
        compute_seq.append(pr)
    slices.reverse()
    compute_seq.reverse()
    rlt_posis.reverse()

    smts = []
    for subtrs in slices:
        smt = [[]for i in range(len(X))]
        for i, subtr in enumerate(subtrs):
            # print(subtr, type(subtr))
            try:
                func = gp.compile(expr=subtr, pset=pset)
            except SyntaxError:
                return 'nan'
            for j, data in enumerate(X):
              if is_complex(data):
                  return 'nan'
              try:
                  smt[j].append(func(*data))
              except:
                  return 'nan'
        smts.append(smt)

    # print(smts)
    '''backpropagation'''
    smt_subtg = y
    for i, idx in enumerate(compute_seq):
        # print('1:',individual[idx].name)
        try:
            smt_subtg = [oper_set[individual[idx].name](*smts[i][j], y_val, rlt_posis[i]) for j, y_val in enumerate(smt_subtg)]
        except TypeError:
            # print('TypeError: unsupported operand type(s) for /: \'int\' and \'list\'')
            return 'nan'
        except KeyError:
            return 'nan'

        if any(np.isnan(smt_subtg)):
            # smt_subtg = last_smt_subtg
            return 'nan'

        # final_idx = idx

    return  smt_subtg



