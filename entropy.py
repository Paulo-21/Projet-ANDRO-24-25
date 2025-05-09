
import pandas as pd
import numpy as np
import glob
import os
import matplotlib.pyplot as plt

TOTAL_CELLS = 20000 * 20000 * 20000

def compute_entropy(coords, bins_per_dim=10):
    """
    计算空间分布熵。

    参数:
        coords (numpy array): 三维坐标数组
        bins_per_dim (int): 每个维度划分的 bin 数量

    返回:
        float: 熵值
    """
    # 创建 histogram 的 bin 划分
    hist, _ = np.histogramdd(coords, bins=bins_per_dim, range=[[-1, 1], [-1, 1], [-1, 1]])
    # 计算概率分布（忽略 0 的 bin）
    probabilities = hist.flatten()
    probabilities = probabilities[probabilities > 0]
    probabilities = probabilities / probabilities.sum()

    # 计算熵
    entropy = -np.sum(probabilities * np.log(probabilities))
    return entropy

def compute_coverage(coords):
    """
    计算覆盖率。

    参数:
        coords (numpy array): 三维坐标数组

    返回:
        float: 覆盖率值
    """
    unique_points = len(np.unique(coords, axis=0))
    coverage = unique_points / TOTAL_CELLS
    return coverage

def process_csv(file_path, bins_per_dim=10):
    """
    处理单个 CSV 文件，计算熵和覆盖率（所有点和成功点）。

    参数:
        file_path (str): CSV 文件路径

    返回:
        dict: 包含熵和覆盖率的字典
    """
    df = pd.read_csv(file_path)

    # 提取行为描述符并转换为数组
    df['behavior_descriptor'] = df['behavior_descriptor'].apply(
        lambda x: np.fromstring(x.strip("[]").replace("\n", " "), sep=" ")
    )

    # 提取所有点和成功点的坐标
    all_coords = np.vstack(df['behavior_descriptor'].values)
    success_coords = np.vstack(df[df['fitness'] == 1]['behavior_descriptor'].values)

    # 计算熵和覆盖率
    entropy_all = compute_entropy(all_coords, bins_per_dim)
    entropy_success = compute_entropy(success_coords, bins_per_dim)
    coverage_all = compute_coverage(all_coords)
    coverage_success = compute_coverage(success_coords)

    return {
        "entropy_all": entropy_all,
        "entropy_success": entropy_success,
        "coverage_all": coverage_all,
        "coverage_success": coverage_success
    }

def plot_entropy_and_coverage(entropy_all, entropy_success, coverage_all, coverage_success, operators):
    """
    绘制熵和覆盖率折线图，分别包含所有点和成功点的对比。
    """
    plt.figure(figsize=(12, 6))

    # 绘制熵折线图
    plt.subplot(1, 2, 1)
    plt.plot(operators, entropy_all, marker='o', label='All Points', linestyle='-', color='blue')
    plt.plot(operators, entropy_success, marker='x', label='Success Points', linestyle='--', color='green')
    plt.xlabel("Mutation Operator")
    plt.ylabel("Entropy")
    plt.title("Entropy Comparison (All vs Success)")
    plt.legend()
    plt.grid(True)

    # 绘制覆盖率折线图
    plt.subplot(1, 2, 2)
    plt.plot(operators, coverage_all, marker='o', label='All Points', linestyle='-', color='blue')
    plt.plot(operators, coverage_success, marker='x', label='Success Points', linestyle='--', color='green')
    plt.xlabel("Mutation Operator")
    plt.ylabel("Coverage")
    plt.title("Coverage Comparison (All vs Success)")
    plt.legend()
    plt.grid(True)

    # 保存图像
    plt.tight_layout()
    plt.savefig("entropy_coverage_comparison.png")
    plt.show()

def main(folder_path="data/analysis_qd_action"):
    """
    主函数，遍历文件并计算熵和覆盖率。
    """
    operators = ['random', 'gaussian', 'es', 'sa', 'cov']
    entropy_all = []
    entropy_success = []
    coverage_all = []
    coverage_success = []

    for operator in operators:
        file_pattern = f"{folder_path}/*_{operator}_*.csv"
        files = glob.glob(file_pattern)

        if not files:
            print(f"没有找到符合模式的文件: {file_pattern}")
            continue

        # 只取第一个文件（假设一个算子只有一个文件）
        file_path = files[0]
        print(f"处理文件: {file_path}")

        results = process_csv(file_path)

        entropy_all.append(results["entropy_all"])
        entropy_success.append(results["entropy_success"])
        coverage_all.append(results["coverage_all"])
        coverage_success.append(results["coverage_success"])

    plot_entropy_and_coverage(entropy_all, entropy_success, coverage_all, coverage_success, operators)

if __name__ == "__main__":
    main()
