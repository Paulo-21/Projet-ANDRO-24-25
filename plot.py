import os
import glob
import re
import matplotlib.pyplot as plt

# CSV 文件所在目录
csv_folder = "data/3000 population 0.001-0.01"

# 匹配所有 .csv 文件
csv_files = glob.glob(os.path.join(csv_folder, "*.csv"))

scale_list = []
success_counts = []

for file in csv_files:
    # 提取 scale_mutation 值，比如从文件名中提取 _0.09.csv
    match = re.search(r"_(0\.\d+)(?:\.csv)?$", file)
    if not match:
        continue
    scale = float(match.group(1))

    # 初始化成功点数量
    success_count = 0

    with open(file, "r") as f:
        lines = f.readlines()[1:]  # 跳过第一行 header
        for line in lines:
            parts = line.strip().split(",")
            if len(parts) >= 2:
                try:
                    fitness = int(parts[1])
                    if fitness == 1:
                        success_count += 1
                except ValueError:
                    continue

    scale_list.append(scale)
    success_counts.append(success_count)

# 排序
sorted_pairs = sorted(zip(scale_list, success_counts))
sorted_scales, sorted_counts = zip(*sorted_pairs)

# 画图
plt.figure(figsize=(8, 5))
plt.plot(sorted_scales, sorted_counts, marker='o')
plt.xlabel("Scale Mutation")
plt.ylabel("Number of Successful Case(fitness = 1)")
plt.title("Success Case vs. Mutation Scale")
plt.grid(True)
plt.tight_layout()
plt.savefig("success_vs_scale.png")
plt.show()
