import numpy as np
import matplotlib.pyplot as plt
from src.model.model_2d import EMModel2D, Grid2D, EMParameters2D
from src.forward.forward_2d import ForwardSolver2D
from src.visualization.advanced_visualization import plot_forward_results

# 配置matplotlib支持中文显示
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False    # 用来正常显示负号
plt.rcParams['mathtext.fontset'] = 'cm'  # 使用Computer Modern字体集
plt.rcParams['axes.formatter.use_mathtext'] = True  # 使用数学文本渲染器

# 创建网格（使用非均匀网格以提高计算效率）
x = np.concatenate([
    np.linspace(-200, -50, 6),   # 左侧粗网格
    np.linspace(-50, 50, 21),    # 中间细网格（目标区域）
    np.linspace(50, 200, 6)      # 右侧粗网格
])
z = np.concatenate([
    np.linspace(0, 20, 11),      # 浅部细网格
    np.linspace(20, 60, 21),     # 中部中等网格（包含异常体区域）
    np.linspace(60, 200, 8)      # 深部粗网格
])
dx = np.diff(x)                   # x方向网格间距
dz = np.diff(z)                   # z方向网格间距
grid = Grid2D(x=x, z=z, dx=dx, dz=dz)

# 创建电磁场参数
nx = len(x)
nz = len(z)

# 创建一个更复杂的地质模型
conductivity = np.ones((nz, nx)) * 0.0005  # 背景电导率0.0005 S/m（高阻基岩）

# 添加导电异常体
# 上部导电层（风化层）
conductivity[:20, :] = 0.02  # 0.02 S/m

# 中部异常体（矿化带）
conductivity[30:40, 50:70] = 0.2  # 0.2 S/m

# 深部导电层（含水层）
conductivity[60:, :] = 0.05  # 0.05 S/m

# 创建参数对象
parameters = EMParameters2D(
    conductivity=conductivity,
    relative_permittivity=np.ones((nz, nx)),  # 相对介电常数为1
    relative_permeability=np.ones((nz, nx))   # 相对磁导率为1
)

# 设置频率（覆盖更大范围）
frequencies = np.array([10.0, 50.0, 100.0, 500.0, 1000.0])  # 多个频率点

# 创建模型
model = EMModel2D(grid=grid, parameters=parameters, frequencies=frequencies)

# 设置发射源和接收器位置
source_positions = np.array([
    [-150.0, 0.0],  # 左侧发射源
    [0.0, 0.0],     # 中间发射源
    [150.0, 0.0]    # 右侧发射源
])  # 多个发射源位置

receiver_positions = np.array([
    [x, 0.0] for x in np.linspace(-180, 180, 81)
])  # 接收器位置，覆盖模型范围

# 创建求解器
solver = ForwardSolver2D(
    frequencies=frequencies,
    source_positions=source_positions,
    receiver_positions=receiver_positions
)

# 计算响应
response = solver.compute_response(model)

# 清除所有现有图形
plt.close('all')

# 创建第一个图形窗口，展示模型参数
fig1 = plt.figure(figsize=(20, 12))

# 绘制网格分布
plt.subplot(231)
grid.plot(show_grid=True)
plt.title('网格分布')

# 绘制电导率分布
plt.subplot(232)
model.plot(parameter='conductivity')
plt.title('电导率分布')

# 绘制相对介电常数分布
plt.subplot(233)
model.plot(parameter='permittivity')
plt.title('相对介电常数分布')

# 绘制相对磁导率分布
plt.subplot(234)
model.plot(parameter='permeability')
plt.title('相对磁导率分布')

# 绘制发射源和接收器位置
plt.subplot(235)
model.plot(parameter='conductivity')
plt.plot(source_positions[:, 0], source_positions[:, 1], 'r^', label='发射源', markersize=10)
plt.plot(receiver_positions[:, 0], receiver_positions[:, 1], 'kv', label='接收器', markersize=8)
plt.legend()
plt.title('观测系统布置')



# 调整子图间距
plt.subplots_adjust(wspace=0.4, hspace=0.4)
plt.tight_layout()
plt.show()

# 创建第二个图形窗口，展示正演结果
fig2 = plt.figure(figsize=(20, 16))
plot_forward_results(
    model=model,
    response=response,
    frequencies=frequencies,
    offsets=receiver_positions[:, 0],
    show_model=True
)

# 调整子图间距，增加间距以避免重叠
plt.subplots_adjust(wspace=0.6, hspace=0.8)
plt.tight_layout()
plt.show()