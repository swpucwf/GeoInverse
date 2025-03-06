import numpy as np
import matplotlib.pyplot as plt
from src.forward.forward import ForwardSolver
from src.model.model import LayeredEarthModel
from src.visualization.visualization import plot_model, plot_response

# 设置正演参数
frequencies = [2e3, 4e3, 8e3]  # 工作频率 (Hz)
offsets = [0.5, 1.0, 1.5]      # 发射-接收线圈间距 (m)

# 创建三层地球模型
layer_depths = [0, 2, 5, np.inf]  # 层界面深度 (m)
layer_resistivities = [1, 10, 5]   # 各层电阻率 (Ω·m)

# 初始化模型和正演求解器
model = LayeredEarthModel(layer_depths, layer_resistivities)
forward = ForwardSolver(frequencies, offsets)

# 计算电磁场响应
responses = forward.compute_response(model)

# 创建可视化图表
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

# 绘制地层模型
plot_model(model, ax=ax1)
ax1.set_title('地层模型')
ax1.set_xlabel('电阻率 (Ω·m)')
ax1.set_ylabel('深度 (m)')

# 绘制电磁场响应
plot_response(responses, frequencies, offsets, ax=ax2)
ax2.set_title('电磁场响应')
ax2.set_xlabel('频率 (Hz)')
ax2.set_ylabel('幅值 (A/m)')

plt.tight_layout()
plt.show()