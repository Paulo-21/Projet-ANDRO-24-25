import pandas as pd
import matplotlib.pyplot as plt
import glob
import os

# 匹配模式：你可以根据实际情况调整路径
csv_files = glob.glob("data/3000 population 0.001-0.01/genesis_object_100658_robot_end_effector_close_finger_*.csv")

scale_mutations = []
coverages = []

if not csv_files:
    print("❌ 没有找到任何符合格式的 CSV 文件。")
else:
    for file in csv_files:
        try:
            # 从文件名中提取 scale_mutation（最后一个下划线之后的数字）
            scale = float(file.split("_")[-1].replace(".csv", ""))
        except ValueError:
            print(f"⚠️ 文件名无法解析 scale_mutation: {file}")
            continue

        df = pd.read_csv(file)

        # 统计尝试过的行为格子总数（不管是否成功）
        total_bins = df["behavior_descriptor"].nunique()

        # 统计成功的格子数
        success_bins = df[df["fitness"] == 1]["behavior_descriptor"].nunique()

        coverage = success_bins / total_bins if total_bins > 0 else 0

        scale_mutations.append(scale)
        coverages.append(coverage)

    if scale_mutations:
        # 按照 scale_mutation 排序
        sorted_pairs = sorted(zip(scale_mutations, coverages))
        scale_mutations, coverages = zip(*sorted_pairs)

        # 绘图
        plt.figure(figsize=(10, 6))
        plt.plot(scale_mutations, coverages, marker='o')
        plt.xlabel("Scale Mutation")
        plt.ylabel("Coverage (Success Bins / Tried Bins)")
        plt.title("Coverage vs. Scale Mutation")
        plt.grid(True)
        plt.tight_layout()
        plt.show()
    else:
        print("⚠️ 没有提取到有效的 scale_mutation 和 coverage 数据。")
