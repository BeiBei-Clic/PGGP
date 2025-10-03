# Transformer辅助遗传编程用于符号回归

本项目实现了Transformer辅助遗传编程（PGGP）方法，旨在通过预训练模型辅助遗传编程（GP）的初始化和变异过程，以提高符号回归任务的效率和准确性。

![image](https://github.com/xiaoxuh/pggp_codes/blob/main/framework.png)

### PGGP方法
预训练模型引导的遗传编程（PGGP）方法旨在辅助GP的初始化和变异。它利用Transformer模型的能力来生成或改进符号表达式，从而在搜索空间中更有效地找到最优解。

### 快速开始

1.  **下载预训练模型权重**：
    首先，您需要创建一个名为 `weights` 的文件夹，并从以下链接下载Transformer模型的预训练权重：
    `https://github.com/SymposiumOrganization/NeuralSymbolicRegressionThatScales/tree/main`

2.  **运行单次实验**：
    下载权重后，您可以通过运行 `main.py` 来执行单次符号回归实验。您可以指定数据集名称：
    ```bash
    python main.py --dataset_name 1.6.2
    ```
    如果不指定 `--dataset_name`，将使用默认的Feynman数据集 `I.6.2`。

3.  **运行批量实验**：
    为了在多个数据集和随机种子上进行批量实验，并收集统计结果，请运行 `run_experiments.py`：
    ```bash
    python run_experiments.py
    ```
    该脚本将自动在预定义的数据集列表（`1.6.2`, `1.6.2a`, `1.12.1`, `1.12.5`, `1.14.4`, `1.25.13`）上，为每个数据集运行10个不同的随机种子（0-9）。

### 实验原理

#### `main.py`
`main.py` 是核心的符号回归实现文件，它负责：

*   **数据加载与预处理**：从指定的数据集文件（例如 `benchmark_dataset/1.6.2.txt`）中读取数据，并将其分割为训练集和测试集（80%训练，20%测试）。所有数据（输入特征 `X` 和目标变量 `Y`）都会进行标准化处理，以确保模型在统一的尺度上进行学习和预测。
*   **Transformer模型集成**：利用预训练的Transformer模型（通过 `get_res_transformer` 函数）来辅助GP过程。在GP进化开始前，Transformer会进行一次初步预测，生成一个初始的符号表达式。
*   **遗传编程（GP）进化**：使用DEAP库实现遗传编程框架，包括：
    *   **个体表示**：符号表达式以树结构表示。
    *   **适应度评估**：通过 `evalSymbReg` 函数评估个体的适应度，该函数计算标准化测试数据上的均方误差（MSE）。**所有MSE计算均使用标准化后的数据进行，以确保结果的一致性和可比性。**
    *   **遗传操作**：包括选择（锦标赛选择）、交叉（单点交叉）和变异。变异操作中，`mutReplace` 函数会利用Transformer模型来替换子树，从而引入模型引导的智能变异。
    *   **深度限制**：为了防止生成过于复杂的表达式，对表达式的深度进行了限制（初始生成深度限制为4，交叉和变异操作后的深度限制为10）。
*   **结果输出**：在进化过程中，每隔一定代数会输出当前种群的最佳适应度。最终，返回测试集上的MSE、适应度趋势、训练时间以及最佳个体（符号表达式）。

#### `run_experiments.py`
`run_experiments.py` 脚本用于自动化和管理批量实验，它执行以下操作：

*   **数据集遍历**：遍历预定义的数据集列表。
*   **多种子运行**：对于每个数据集，它会使用10个不同的随机种子调用 `main.py`，以减少随机性对结果的影响，并获得更稳健的统计数据。
*   **结果收集与分析**：收集每次运行的 `test_mse`、`fitness_trend`、`training_time` 和 `best_individual` 等结果。然后，计算 `test_mse` 的25%、50%（中位数）和75%分位数。
*   **结果保存**：将每个数据集的详细实验结果（包括所有运行的原始数据和分位数对应的最佳实验）保存到 `results/` 目录下的JSON文件中。这使得后续的分析和比较变得容易。

### 结果文件说明

`results/` 文件夹中会生成以数据集名称命名的JSON文件（例如 `results/1.6.2.json`）。每个JSON文件包含：

*   `dataset`：数据集名称。
*   `total_runs`：总共运行的实验次数（通常为10）。
*   `quantiles`：`test_mse` 的25%、50%和75%分位数。
*   `selected_experiments`：对应于25%、50%和75%分位数的实验的详细信息，包括使用的随机种子、`test_mse`、适应度趋势、训练时间以及找到的最佳符号表达式。

### 引用

```
@article{han2025transformer,
  title={Transformer-Assisted Genetic Programming for Symbolic Regression [Research Frontier]},
  author={Han, Xiaoxu and Zhong, Jinghui and Ma, Zhitong and Mu, Xin and Gligorovski, Nikola},
  journal={IEEE Computational Intelligence Magazine},
  volume={20},
  number={2},
  pages={58--79},
  year={2025},
  publisher={IEEE}
}
```

数据集链接: https://space.mit.edu/home/tegmark/aifeynman.html
