import time
from src.nesymres.architectures.model import Model
from data_conversion import *
import numpy as np
import argparse
import operator
import math
import os
import random
from functools import partial
from deap import algorithms
from deap import base
from deap import creator
from deap import tools
from deap import gp
import torch
import json, re
import omegaconf
import sympy
from src.nesymres.dclasses import FitParams, BFGSParams
from backpropagation import *
from sklearn.metrics import mean_squared_error
import warnings

warnings.filterwarnings("ignore" )


def root_mean_squared_error(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))


def transformer_init():
    with open('jupyter/100M/eq_setting.json', 'r') as json_file:
        eq_setting = json.load(json_file)

    cfg = omegaconf.OmegaConf.load("100M/config.yaml")
    return eq_setting, cfg


def get_res_transformer(X, y, BFGS, first_call=False):
    # 将输入数据转换为NumPy数组
    input_X = np.array(X)
    input_Y = np.array(y)
    # 将NumPy数组转换为PyTorch张量
    X = torch.from_numpy(input_X)
    y = torch.from_numpy(input_Y)

    # 初始化Transformer模型设置和配置
    eq_setting, cfg = transformer_init()

    # 配置BFGS参数，用于优化拟合过程
    bfgs = BFGSParams(
        # 是否激活BFGS优化
        activated=cfg.inference.bfgs.activated,
        # BFGS重启次数
        n_restarts=cfg.inference.bfgs.n_restarts,
        # 如果系数不存在是否添加系数
        add_coefficients_if_not_existing=cfg.inference.bfgs.add_coefficients_if_not_existing,
        # 输出标准化参数
        normalization_o=cfg.inference.bfgs.normalization_o,
        # 移除索引参数
        idx_remove=cfg.inference.bfgs.idx_remove,
        # 标准化类型
        normalization_type=cfg.inference.bfgs.normalization_type,
        # 停止时间
        stop_time=cfg.inference.bfgs.stop_time,
    )

    # 设置模型拟合参数
    params_fit = FitParams(
        # 词到ID的映射
        word2id=eq_setting["word2id"],
        # ID到词的映射
        id2word={int(k): v for k, v in eq_setting["id2word"].items()},
        # 一元运算符列表
        una_ops=eq_setting["una_ops"],
        # 二元运算符列表
        bin_ops=eq_setting["bin_ops"],
        # 变量列表
        total_variables=list(eq_setting["total_variables"]),
        # 系数列表
        total_coefficients=list(eq_setting["total_coefficients"]),
        # 重写函数列表
        rewrite_functions=list(eq_setting["rewrite_functions"]),
        # BFGS参数
        bfgs=bfgs,
        # beam搜索大小，这是准确性和拟合时间之间的权衡
        beam_size=cfg.inference.beam_size
    )
    
    # 设置预训练模型权重路径
    weights_path = "weights/100M.ckpt"
    # 加载预训练模型
    model = Model.load_from_checkpoint(weights_path, cfg=cfg.architecture)
    # 设置模型为评估模式
    model.eval()
    # 如果CUDA可用，将模型移到GPU上
    if torch.cuda.is_available():
        model.cuda()
    # 创建部分函数，用于模型拟合
    fitfunc = partial(model.fitfunc, cfg_params=params_fit)

    # 如果是第一次调用
    if first_call:
        # 使用BFGS优化进行第一次拟合
        _ = fitfunc(X, y, BFGS=True)

        # 获取拟合后的方程
        final_equation = model.get_equation()

        # 不使用BFGS优化获取前缀符号列表
        prefix_symbol_list = fitfunc(X, y, BFGS=False)

        # 返回符号列表和最终方程
        return prefix_symbol_list, final_equation

    # 如果启用BFGS优化
    if BFGS:
        try:
            # 使用BFGS优化获取前缀符号列表
            prefix_symbol_list = fitfunc(X, y, BFGS)
        except ValueError:
            # 如果出现ValueError，返回None
            return [None, None]

        # 获取预测方程
        final_equation = model.get_equation()

        # 返回符号列表、预测方程、总系数和总BFGS时间
        return prefix_symbol_list, final_equation, model.total_c, model.total_bfgs_time
    else:
        # 不使用BFGS优化获取前缀符号列表
        prefix_symbol_list = fitfunc(X, y, BFGS)
        # 返回符号列表
        return prefix_symbol_list



def protectedMul(left, right):
    try:
        return left*right
    except OverflowError:
        return 1e7

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
    if is_complex(arg):
        return 99999
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

def protectedSqrt(x):
    if x < 0:
        return 99999
    else:
        return math.sqrt(x)

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


def convert_to_list(trimmed_eq, accurate_constant=False):

    token_list= []
    for i in range(n_variables):
        token_list.append('x_'+str(i+1))

    for i, x in enumerate(trimmed_eq):
        if x == 'constant' :
            if not accurate_constant:
                trimmed_eq[i]='rand505'

        elif  x.startswith('x_'):
            index=int(x[-1])
            trimmed_eq[i]='x_'+str(index-1)
        else:

            trimmed_eq[i] = x

    return trimmed_eq



def get_creator():
    # 创建一个原语集，命名为"MAIN"，变量数量为n_variables
    pset = gp.PrimitiveSet("MAIN", n_variables)
    # 创建参数重命名映射，将默认的ARG0, ARG1等重命名为x_0, x_1等
    rename_kwargs = {"ARG{}".format(i): 'x_'+str(i) for i in range(n_variables)}
    # 更新原语集中的参数名称映射
    for k, v in rename_kwargs.items():
        pset.mapping[k].name = v
    # 应用参数重命名
    pset.renameArguments(**rename_kwargs)
    # 添加基本数学运算符到原语集
    # 加法运算
    pset.addPrimitive(operator.add, 2)
    # 减法运算
    pset.addPrimitive(operator.sub, 2)
    # 受保护的乘法运算（防止溢出）
    pset.addPrimitive(protectedMul, 2, name='mul')
    # 受保护的除法运算（避免除零错误）
    pset.addPrimitive(protectedDiv, 2, name='div')
    # 受保护的指数运算（防止结果过大）
    pset.addPrimitive(protectedExp, 1, name="exp")
    # 受保护的对数运算（处理负数和零）
    pset.addPrimitive(protectedLog, 1, name="ln")
    # 受保护的平方根运算（处理负数）
    pset.addPrimitive(protectedSqrt, 1, name="sqrt")
    # 幂运算
    pset.addPrimitive(operator.pow, 2, name="pow")
    # 绝对值运算
    pset.addPrimitive(operator.abs, 1, name="abs")
    # 正弦函数
    pset.addPrimitive(math.sin, 1)
    # 余弦函数
    pset.addPrimitive(math.cos, 1)
    # 正切函数
    pset.addPrimitive(math.tan, 1)
    # 受保护的反正弦函数（处理定义域外的值）
    pset.addPrimitive(protectedAsin, 1,name='asin')
    # 受保护的反余弦函数（处理定义域外的值）
    pset.addPrimitive(protectedAcos, 1,name='acos')
    # 受保护的反正切函数
    pset.addPrimitive(protectedAtan, 1, name='atan')
    # 添加临时常数，在-5到5之间随机生成
    pset.addEphemeralConstant("rand505", partial(random.uniform, -5, 5))

    # 创建适应度类，目标是最小化适应度值
    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    # 创建个体类，个体是原语树结构，具有适应度属性
    creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMin)

    # 返回原语集和创建器
    return pset, creator

def init_individual(pset, creator, trimmed_eq):

    plist=[]

    for t in trimmed_eq:
        if t in pset.mapping:
            plist.append(pset.mapping[t])

        elif t in [ '-3','-2','-1','0', '1', '2', '3', '4', '5']:

            if t not in pset.terminals[pset.ret]:
                pset.addTerminal(float(t), name=t)
                term = pset.terminals[pset.ret][-1]

            else:
                for i, term in enumerate(pset.terminals[pset.ret]):
                    if term.name == t:
                        break
                term = pset.terminals[pset.ret][i]()

            plist.append(term)


        elif t=='rand505':

            index_rand505=n_variables
            term = pset.terminals[pset.ret][index_rand505]()

            plist.append(term)

        else:
            value = float(t)

            pset.addTerminal(value, name=t)

            plist.append(pset.terminals[pset.ret][-1])

    individual = creator.Individual(gp.PrimitiveTree(plist))

    return individual

def is_complex(number):
    """
    Determine whether the given number is a complex number.

    :param number: The number to be checked.
    :return: True if the number is complex, False otherwise.
    """
    # A number is complex if it has a non-zero imaginary part
    return isinstance(number, complex) and number.imag != 0


def evalSymbReg(individual, pset, toolbox):

    func = toolbox.compile(expr=individual)
    sqerrors=[]
    wrong_mark=999999999
    for i, x in enumerate(input_X):
        try:
            try:
                try:
                    result=func(*x)
                except AttributeError:
                    print('AttributeError: NoneType object has no attribute __import__')
                    return wrong_mark,
                except ValueError:
                    print('ValueError: math domain error')
                    return wrong_mark,
                except ZeroDivisionError:
                    print('ZeroDivisionError: float division by zero')
                    return wrong_mark,

                if is_complex(result):
                    return wrong_mark,
                else:
                    tmp = (func(*x) - input_Y[i]) ** 2
                    sqerrors.append(tmp)
            except TypeError:
                print('TypeError: cannot unpack non-iterable float object')

                return wrong_mark,

        except OverflowError:
            return wrong_mark,
    try:
        res = math.sqrt(math.fsum(sqerrors) / len(input_X))
    except TypeError:
        print("TypeError: cannot convert complex to float")
        return  wrong_mark,

    return res,

def mutReplace(individual, pset, toolbox, creator):

    global  input_X
    node_index = random.randrange(len(individual))
    if len(individual) == 1:
        return gp.mutUniform(individual, expr=toolbox.expr_mut, pset=pset)
    while individual[node_index].arity == 0:
        node_index = random.randrange(len(individual))
    try:
        semantic = backpropogation(individual, pset, (input_X, input_Y), node_index)
    except OverflowError:
        return gp.mutUniform(individual, expr=toolbox.expr_mut, pset=pset)
    if semantic == 'nan':
        return gp.mutUniform(individual, expr=toolbox.expr_mut, pset=pset)

    try:
        symbol_list, prediceted_equation,total_c, bfgs_time= get_res_transformer(input_X, semantic, BFGS=True)

    except ValueError:
        return gp.mutUniform(individual, expr=toolbox.expr_mut, pset=pset)

    if symbol_list is None:
        return gp.mutUniform(individual, expr=toolbox.expr_mut, pset=pset)



    symbol_list=convert_to_list(symbol_list,accurate_constant=True)


    new_subtree=init_individual(pset, creator, symbol_list)

    CT_slice = individual.searchSubtree(node_index)

    individual[CT_slice] = new_subtree

    return individual,

def mutate(individual, pset, creator, toolbox, p_subtree=0.05):

    if random.random() < p_subtree:

        return mutReplace(individual, pset=pset, toolbox=toolbox, creator=creator)
    else:

        return gp.mutUniform(individual, expr=toolbox.expr_mut, pset=pset)


def main(filename):
    # 定义训练数据集和测试数据集的路径
    dataset_path = "benchmark_dataset/"
    
    # 构建训练数据文件的完整路径
    file_path = os.path.join(dataset_path, filename + '.txt')
    
    # 声明全局变量，用于存储变量数量和输入数据
    global n_variables
    global input_X
    global input_Y
    
    # 初始化文件内容字典和输入数据列表
    file_contents = {}
    input_X = []
    input_Y = []

    # 检查训练数据文件是否存在
    if os.path.isfile(file_path):
        try:
            # 打开并读取训练数据文件
            with open(file_path, 'r') as file:
                # 逐行读取文件内容
                for line in file:
                    # 去除行首行尾空白字符并按空格分割
                    data = line.strip().split()
                    
                    # 将除最后一个元素外的所有元素转换为浮点数作为输入特征
                    x = list(map(float, data[:-1]))
                    # 将最后一个元素作为目标值
                    y = float(data[-1])
                    # 将输入特征和目标值分别添加到对应列表中
                    input_X.append(x)
                    input_Y.append(y)

        # 捕获文件读取过程中的异常
        except Exception as e:
            file_contents[filename] = f"Error reading file: {e}"
    else:
        # 如果训练数据文件不存在，打印错误信息并返回
        print(f"找不到训练数据集文件: {file_path}")
        return

    # 检查训练数据是否为空
    if len(input_X) == 0:
        print("训练数据集为空，请检查文件格式是否正确")
        return

    # 获取输入变量的数量
    n_variables= len(input_X[0])

    # 定义测试数据集路径
    test_dataset_path = "benchmark_test/"
    # 构建测试数据文件的完整路径
    file_path = os.path.join(test_dataset_path, filename + '.txt')

    # 初始化测试数据列表
    test_X = []
    test_Y = []

    # 检查测试数据文件是否存在
    if os.path.isfile(file_path):
        try:
            # 打开并读取测试数据文件
            with open(file_path, 'r') as file:
                # 逐行读取文件内容
                for line in file:
                    # 去除行首行尾空白字符并按空格分割
                    data = line.strip().split()
                    
                    # 将除最后一个元素外的所有元素转换为浮点数作为输入特征
                    x = list(map(float, data[:-1]))
                    # 将最后一个元素作为目标值
                    y = float(data[-1])
                    # 将输入特征和目标值分别添加到对应列表中
                    test_X.append(x)
                    test_Y.append(y)

        # 捕获文件读取过程中的异常
        except Exception as e:
            file_contents[filename] = f"Error reading file: {e}"
    else:
        # 如果测试数据文件不存在，打印错误信息并返回
        print(f"找不到测试数据集文件: {file_path}")
        return

    # 将测试数据转换为NumPy数组
    test_X = np.array(test_X)
    test_Y = np.array(test_Y)

    # 获取GP的基本设置和创建器
    pset, creator = get_creator()

    # 使用Transformer模型获取初始符号列表和方程
    symbol_list, equation, =  get_res_transformer(input_X, input_Y, BFGS=False, first_call=True)

    # 根据变量数量生成变量名列表
    total_variables = ["x_1", "x_2", "x_3"][:n_variables]

    # 构建变量字典，用于后续方程计算
    X_dict = {x: test_X[:, idx] for idx, x in enumerate(total_variables)}
    # 使用SymPy将符号方程转换为可执行函数并计算预测值
    y_pred = np.array(sympy.lambdify(",".join(total_variables), equation)(**X_dict))

    # 计算Transformer模型在测试集上的RMSE
    transformer_rmse = root_mean_squared_error(test_Y.ravel(), y_pred.ravel())

    # 如果RMSE非常小，则提前结束程序
    if transformer_rmse < 1e-10:
        print('early stop')
        return

    # 将符号列表转换为可用于初始化个体的格式
    trimmed_eq = convert_to_list(symbol_list, accurate_constant=False)

    # 设置GP工具箱
    toolbox = base.Toolbox()
    # 注册表达式生成函数
    toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=2, max_=6)
    # 注册随机个体生成函数
    toolbox.register("random_individual", tools.initIterate, creator.Individual, toolbox.expr)
    # 注册个体生成函数，使用Transformer生成的表达式作为引导
    toolbox.register("individual", init_individual, pset, creator, trimmed_eq=trimmed_eq)
    # 注册种群生成函数
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    # 注册随机种群生成函数
    toolbox.register("random_population", tools.initRepeat, list, toolbox.random_individual)
    # 注册变异时使用的表达式生成函数
    toolbox.register("expr_mut", gp.genFull, min_=0, max_=2)
    # 注册编译函数，用于将个体转换为可执行函数
    toolbox.register("compile", gp.compile, pset=pset)
    # 注册评估函数
    toolbox.register("evaluate", evalSymbReg, pset=pset, toolbox=toolbox)
    # 注册选择函数
    toolbox.register("select", tools.selTournament, tournsize=3)
    # 注册交叉函数
    toolbox.register("mate", gp.cxOnePoint)
    # 注册变异函数
    toolbox.register("mutate", mutate, pset=pset, creator=creator, toolbox=toolbox, p_subtree=0.025)
    # 添加交叉和变异操作的高度限制装饰器
    toolbox.decorate("mate", gp.staticLimit(key=operator.attrgetter("height"), max_value=17))
    toolbox.decorate("mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value=17))

    # 设置随机种子以确保结果可重现
    np.random.seed(8346)
    random.seed(8346)

    # 初始化种群
    pop = []
    # 使用Transformer生成的表达式初始化20个相同的个体
    transformer_init=toolbox.population(n=1)
    for i in range(20):
        pop+=copy.deepcopy(transformer_init)

    # 用随机生成的个体填充剩余种群空间
    pop+=toolbox.random_population(n=200-len(pop))

    # 创建名人堂用于保存最佳个体
    hof = tools.HallOfFame(1)
    # 创建统计信息对象
    stats_fit = tools.Statistics(lambda ind: ind.fitness.values)
    stats_size = tools.Statistics(len)
    mstats = tools.MultiStatistics(fitness=stats_fit, size=stats_size)
    # 注册统计信息计算方法
    mstats.register("avg", np.mean)
    mstats.register("std", np.std)
    mstats.register("min", np.min)
    mstats.register("max", np.max)

    # 记录开始时间
    start_time = time.time()
    # 运行遗传算法进化过程
    pop, log, rmse_mid, test_rmse_mid, generations = algorithms.eaSimple(n_variables,pop, pset, toolbox, test_X,test_Y,0.5, 0.2, ngen=300, stats=mstats,
                                   halloffame=hof, verbose=True)
    # 记录结束时间
    end_time = time.time()

    # 计算训练时间
    training_time=end_time - start_time

    # 获取最后一代种群中的最小适应度值
    last_generation_fitness = np.min([ind.fitness.values[0] for ind in pop])

    # 编译名人堂中的最佳个体
    func = toolbox.compile(expr=hof[0])

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Transformer-GP')
    parser.add_argument('dataset_name', type=str, default='Simple-1', nargs='?')
    args = parser.parse_args()
    main(args.dataset_name)
