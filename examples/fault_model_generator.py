import numpy as np
import matplotlib.pyplot as plt
from src.model.model import LayeredEarthModel
from src.visualization.visualization import plot_model

# 配置matplotlib支持中文显示
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False    # 用来正常显示负号

# 创建三层地球模型
layer_depths = [-100, 25, 30, np.inf]  # 层界面深度 (m)
layer_resistivities = [1, 10, 10]   # 各层电阻率 (Ω·m)

# 初始化模型
model = LayeredEarthModel.from_depths_resistivities(layer_depths, layer_resistivities)

# 创建图形并设置大小
plt.figure(figsize=(10, 8))

# 绘制地层模型
plot_model(model, depth_range=(-100, 150))

# 设置x轴范围
plt.gca().set_xlim(-60, 100)

# 设置标题
plt.title('三层水平地层模型')

# 显示图形
plt.show()