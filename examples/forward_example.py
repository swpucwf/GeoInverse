import numpy as np
import matplotlib.pyplot as plt
from src.forward.forward import ForwardSolver
from src.model.model import LayeredEarthModel
from src.visualization.visualization import plot_model, plot_response, EMVisualization

# 配置matplotlib支持中文显示
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False    # 用来正常显示负号
plt.rcParams['mathtext.fontset'] = 'cm'  # 使用Computer Modern字体集
plt.rcParams['axes.formatter.use_mathtext'] = True  # 使用数学文本渲染器

# 设置正演参数
frequencies = [2e3, 4e3, 8e3]  # 工作频率 (Hz)
offsets = [0.5, 1.0, 1.5]      # 发射-接收线圈间距 (m)

# 创建三层地球模型
layer_depths = [0, 2, 5, np.inf]  # 层界面深度 (m)
layer_resistivities = [1, 10, 5]   # 各层电阻率 (Ω·m)

# 初始化模型和正演求解器
model = LayeredEarthModel.from_depths_resistivities(layer_depths, layer_resistivities)
forward = ForwardSolver(frequencies, offsets)

# 计算电磁场响应
responses = forward.compute_response(model)

# 创建可视化图表
fig = plt.figure(figsize=(15, 10))

# 创建子图网格
gs = fig.add_gridspec(2, 2)

# 绘制地层模型
ax1 = fig.add_subplot(gs[0, 0])
plot_model(model, ax=ax1)
ax1.set_title('地层模型')
ax1.set_xlabel('电阻率 (Ω·m)')
ax1.set_ylabel('深度 (m)')

# 绘制幅值响应
ax2 = fig.add_subplot(gs[0, 1])
plot_response(responses, frequencies, offsets, plot_type='amplitude', ax=ax2)
ax2.set_title('幅值响应')

# 绘制相位响应
ax3 = fig.add_subplot(gs[1, 0])
plot_response(responses, frequencies, offsets, plot_type='phase', ax=ax3)
ax3.set_title('相位响应')

# 绘制二维分布图
ax4 = fig.add_subplot(gs[1, 1])
EMVisualization.plot_field_map(np.array(frequencies), np.array(offsets), responses, ax=ax4)
ax4.set_title('幅值分布图')

plt.tight_layout()
plt.show()