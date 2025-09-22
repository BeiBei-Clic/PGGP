
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
    # 定义语义反向传播函数，用于计算在指定节点处需要的语义值
    # individual: DEAP中的PrimitiveTree对象，表示一个个体（数学表达式）
    # pset: 原语集，包含函数和终端符号
    # dataset: 训练数据集，格式为(X, y)
    # target_idx: 目标变异点在个体中的索引
    # 返回: 预测值和子目标语义值

    # print('individual', individual)
    (X, y) = dataset
    # 解包数据集为输入特征X和目标值y

    '''subtree semantic prepare'''
    # 准备子树语义计算
    (parent, childs) = rebuild_rlt(individual)
    # 重建个体中各节点的父子关系
    # parent: 每个节点的父节点索引列表
    # childs: 每个节点的子节点索引列表

    idx = target_idx
    # 初始化当前索引为目标节点索引

    slices = []
    # 初始化切片列表，用于存储需要计算语义的子树

    compute_seq = []
    # 初始化计算序列，用于存储需要反向传播的节点序列

    rlt_posis = []
    # 初始化相对位置列表，用于存储子节点在父节点中的位置

    while parent[idx] != -1:
        # 当当前节点不是根节点时循环（-1表示无父节点，即根节点）
        pr = parent[idx]
        # 获取当前节点的父节点索引

        rlt_posi = childs[pr].index(idx)
        # 获取当前节点在其父节点子节点列表中的位置

        rlt_posis.append(rlt_posi)
        # 将位置添加到相对位置列表中

        childs[pr].pop(rlt_posi)
        # 从父节点的子节点列表中移除当前节点（为计算其他兄弟节点做准备）

        slices.append([gp.PrimitiveTree(individual[individual.searchSubtree(child)])for child in childs[pr]])
        # 将父节点的其他子节点（兄弟节点）构造成子树并添加到切片列表中

        idx = parent[idx]
        # 将当前索引更新为父节点索引，向上遍历

        compute_seq.append(pr)
        # 将父节点添加到计算序列中

    slices.reverse()
    # 反转切片列表，使计算顺序从底层到顶层

    compute_seq.reverse()
    # 反转计算序列，使计算顺序从底层到顶层

    rlt_posis.reverse()
    # 反转相对位置列表，使位置顺序与计算顺序一致

    smts = []
    # 初始化语义值列表

    for subtrs in slices:
        # 遍历切片列表中的每个子树集合
        smt = [[]for i in range(len(X))]
        # 为每个输入样本初始化一个语义值列表

        for i, subtr in enumerate(subtrs):
            # 遍历当前集合中的每个子树
            # print(subtr, type(subtr))
            func = gp.compile(expr=subtr, pset=pset)
            # 将子树编译为可执行的Python函数

            for j, data in enumerate(X):
                # 遍历每个输入样本
                if is_complex(data):
                    # 如果数据是复数
                    return 'nan'
                    # 返回'nan'表示计算失败

                try:
                    smt[j].append(func(*data))
                    # 计算子树在当前数据点的输出并添加到语义列表中
                except:
                    return 'nan'
                    # 如果计算出错，返回'nan'

        smts.append(smt)
        # 将计算得到的语义值添加到语义值列表中

    # print(smts)
    '''backpropagation'''
    # 开始反向传播计算
    smt_subtg = y
    # 初始化子目标语义为目标值y

    for i, idx in enumerate(compute_seq):
        # 遍历计算序列中的每个节点索引
        # print('1:',individual[idx].name)
        try:
            smt_subtg = [oper_set[individual[idx].name](*smts[i][j], y_val, rlt_posis[i]) for j, y_val in enumerate(smt_subtg)]
            # 使用操作符集合中的反向函数计算新的语义目标值
            # individual[idx].name: 当前节点的操作符名称
            # smts[i][j]: 当前节点兄弟子树在样本j处的语义值
            # y_val: 当前语义目标值
            # rlt_posis[i]: 当前节点在其父节点中的位置
        except TypeError:
            # print('TypeError: unsupported operand type(s) for /: \'int\' and \'list\'')
            return 'nan'
            # 如果类型错误，返回'nan'

        except KeyError:
            return 'nan'
            # 如果操作符名称不存在，返回'nan'

        if any(np.isnan(smt_subtg)):
            # smt_subtg = last_smt_subtg
            return 'nan'
            # 如果语义目标值中包含NaN，返回'nan'

        # final_idx = idx

    return smt_subtg
    # 返回计算得到的语义目标值



