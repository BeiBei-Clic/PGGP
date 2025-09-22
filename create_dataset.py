import os
import numpy as np

def create_directories():
    """创建必要的目录"""
    dirs = ["benchmark_dataset", "benchmark_test"]
    for dir_name in dirs:
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)
            print(f"创建目录: {dir_name}")

def create_simple_dataset(filename, num_samples=100):
    """
    创建一个简单的数据集，表示函数 y = x^2 + 2*x + 1
    """
    # 生成x值
    x_values = np.linspace(-5, 5, num_samples)
    
    # 计算对应的y值 (y = x^2 + 2*x + 1)
    y_values = x_values**2 + 2*x_values + 1
    
    # 保存训练数据
    with open(f"benchmark_dataset/{filename}.txt", "w") as f:
        for x, y in zip(x_values, y_values):
            f.write(f"{x:.6f} {y:.6f}\n")
    
    # 生成测试数据 (稍有不同的范围)
    test_x_values = np.linspace(-6, 6, 30)
    test_y_values = test_x_values**2 + 2*test_x_values + 1
    
    # 保存测试数据
    with open(f"benchmark_test/{filename}.txt", "w") as f:
        for x, y in zip(test_x_values, test_y_values):
            f.write(f"{x:.6f} {y:.6f}\n")
    
    print(f"数据集 '{filename}' 已创建")
    print(f"函数关系: y = x^2 + 2*x + 1")

def create_complex_dataset(filename, num_samples=100):
    """
    创建一个更复杂的数据集，表示函数 y = sin(x) + 0.5*cos(2*x)
    """
    # 生成x值
    x_values = np.linspace(-2*np.pi, 2*np.pi, num_samples)
    
    # 计算对应的y值 (y = sin(x) + 0.5*cos(2*x))
    y_values = np.sin(x_values) + 0.5*np.cos(2*x_values)
    
    # 保存训练数据
    with open(f"benchmark_dataset/{filename}.txt", "w") as f:
        for x, y in zip(x_values, y_values):
            f.write(f"{x:.6f} {y:.6f}\n")
    
    # 生成测试数据
    test_x_values = np.linspace(-3*np.pi, 3*np.pi, 30)
    test_y_values = np.sin(test_x_values) + 0.5*np.cos(2*test_x_values)
    
    # 保存测试数据
    with open(f"benchmark_test/{filename}.txt", "w") as f:
        for x, y in zip(test_x_values, test_y_values):
            f.write(f"{x:.6f} {y:.6f}\n")
    
    print(f"复杂数据集 '{filename}' 已创建")
    print(f"函数关系: y = sin(x) + 0.5*cos(2*x)")

if __name__ == "__main__":
    create_directories()
    create_simple_dataset("Simple-1")
    create_complex_dataset("Complex-1")
    print("\n数据集创建完成！你现在可以运行:")
    print("python main.py Simple-1")
    print("或者")
    print("python main.py Complex-1")