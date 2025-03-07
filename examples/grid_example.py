import numpy as np
import matplotlib.pyplot as plt
from src.model.grid_generator import GridGenerator

# 配置matplotlib支持中文显示
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['mathtext.fontset'] = 'cm'
plt.rcParams['axes.formatter.use_mathtext'] = True

# 创建图形
fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# 1. 均匀网格
grid_uniform = GridGenerator.uniform_grid(
    x_range=(-50, 50),
    z_range=(0, 100),
    nx=21,
    nz=11
)
grid_uniform.plot(ax=axes[0, 0])
axes[0, 0].set_title('均匀网格')

# 2. 几何级数网格（边界加密）
grid_geometric = GridGenerator.geometric_grid(
    x_range=(-50, 50),
    z_range=(0, 100),
    nx=21,
    nz=11,
    x_ratio=1.2,
    z_ratio=1.2
)
grid_geometric.plot(ax=axes[0, 1])
axes[0, 1].set_title('几何级数网格')

# 3. 局部加密网格
refined_regions = [
    # 在中心区域加密
    ((-20, 20), (30, 70), 21, 21)
]
grid_refined = GridGenerator.refined_grid(
    x_range=(-50, 50),
    z_range=(0, 100),
    nx=21,
    nz=11,
    refined_regions=refined_regions
)
grid_refined.plot(ax=axes[1, 0])
axes[1, 0].set_title('局部加密网格')

# 4. 组合使用（几何级数 + 局部加密）
grid_combined = GridGenerator.geometric_grid(
    x_range=(-50, 50),
    z_range=(0, 100),
    nx=21,
    nz=11,
    x_ratio=1.1,
    z_ratio=1.1
)
grid_combined.plot(ax=axes[1, 1])
axes[1, 1].set_title('组合网格（几何级数）')

# 调整布局
plt.tight_layout()
plt.show()