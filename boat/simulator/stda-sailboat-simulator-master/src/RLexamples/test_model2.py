from scipy.io import loadmat
from scipy.interpolate import griddata
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
# from environment import FlowEnvironment
from environment3 import FlowEnvironment
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
import numpy as np

dir_flow = './flow/cylinder2D_Re100_ref.mat'
model_file = './t0/model_1995.zip'
model_file2 = './t0/model2_1995.zip'
# 20230730
model_file3 = './t0/8'
# 20230731
model_file4 = './t0/model2_190.zip'
model_file5 = './t0/0to5'
# 20230801
model_file6 = './t0/model2_500'
model_file7 = './t0/model2_770'

env = FlowEnvironment(dir_name=dir_flow)
test_env = DummyVecEnv([lambda: env])

# 加载预训练的模型
model = PPO.load(model_file7, render_mode='human')

obs = test_env.reset()
test_env.render()
time_set = env.return_time()
rewards_array = []
done = False
while not done:
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, _ = test_env.step(action)
    rewards_array.append(reward)
    test_env.render()
print(rewards_array)
test_env.close()

target_p = env.target_point
print(target_p)
time_record = env.render_time
position_record = env.render_positions

# 将matlab格式用python加载
data = loadmat(dir_flow)
t_star = data['t_star'][0]  # 301个时间步长
X_star = data['X_star']  # (34944, 301)
Y_star = data['Y_star']  # (34944, 301)
U_star = data['U_star']  # (34944, 301)
V_star = data['V_star']  # (34944, 301)
P_star = data['P_star']  # (34944, 301)

# Define parameters
x_max = np.max(X_star[:, 0])  # 取所有行第一列的最大值
x_min = np.min(X_star[:, 0])  # 取所有行第一列的最小值
y_max = np.max(Y_star[:, 0])  # 取所有行第一列的最大值
y_min = np.min(Y_star[:, 0])  # 取所有行第一列的最小值
grid_points_x = 300
grid_points_y = 150
step_x = (x_max - x_min) / (grid_points_x - 1)  # 定义步长
step_y = (y_max - y_min) / (grid_points_y - 1)
x = np.arange(x_min, x_max + step_x, step_x)  # x_min到x_max，步长为step_x的等差数列
y = np.arange(y_min, y_max + step_y, step_y)
X, Y = np.meshgrid(x, y)  # 绘制x,y的坐标系

# Create figure绘图定义
fig, axs = plt.subplots(1, 3, figsize=(20, 8))  # 3个(20, 8)的子图

trajectory_x = []
trajectory_y = []

# 记录运动点的坐标
init_x = 0.0  # 初始x坐标
init_y = 0.0  # 初始y坐标


# 定义更新函数
def update_plot(i):
    t = time_record[i]
    print(f't = {t}')

    # Interpolation
    u_grid = griddata((X_star[:, t], Y_star[:, t]), U_star[:, t], (X, Y), method='linear')
    v_grid = griddata((X_star[:, t], Y_star[:, t]), V_star[:, t], (X, Y), method='linear')
    p_grid = griddata((X_star[:, t], Y_star[:, t]), P_star[:, t], (X, Y), method='linear')

    # 原点位置确定
    # 半径为0.5
    u_grid[X ** 2 + Y ** 2 < 0.25] = np.nan
    v_grid[X ** 2 + Y ** 2 < 0.25] = np.nan
    p_grid[X ** 2 + Y ** 2 < 0.25] = np.nan

    # 清除之前绘制的轨迹
    axs[0].cla()
    axs[1].cla()
    axs[2].cla()

    # Plot velocity and pressure fields
    axs[0].imshow(u_grid, extent=(x_min, x_max, y_min, y_max), cmap='jet', origin='lower')
    axs[0].set_title('u velocity', fontsize=14)

    axs[1].imshow(v_grid, extent=(x_min, x_max, y_min, y_max), cmap='jet', origin='lower')
    axs[1].set_title('v velocity', fontsize=14)

    axs[2].imshow(p_grid, extent=(x_min, x_max, y_min, y_max), cmap='jet', origin='lower')
    axs[2].set_title('pressure', fontsize=14)

    axs[0].scatter(target_p[0], target_p[1], s=4, color='blue')
    axs[1].scatter(target_p[0], target_p[1], s=4, color='black')
    axs[2].scatter(target_p[0], target_p[1], s=4, color='red')

    # 绘制轨迹
    trajectory_x.append(position_record[i][0])
    trajectory_y.append(position_record[i][1])
    axs[0].plot(trajectory_x, trajectory_y, color='blue')
    axs[1].plot(trajectory_x, trajectory_y, color='black')
    axs[2].plot(trajectory_x, trajectory_y, color='red')

    axs[0].set_title(f'u velocity (t={t})', fontsize=14)
    axs[1].set_title(f'v velocity (t={t})', fontsize=14)
    axs[2].set_title(f'pressure (t={t})', fontsize=14)


# 设置动画中帧数（时间步长数）
num_frames = len(time_record) - 1

# 创建动画
animation = FuncAnimation(fig, update_plot, frames=num_frames, interval=200)

# 将动画保存为mp4视频文件
writer = FFMpegWriter(fps=5)  # 设置视频帧率
animation.save('flow_animation.gif', writer='pillow')

plt.show()

# 流场绘图
"""""""""
# Plot flow fields at each time step
for i in range(len(time_record) - 1):
    t = i + time_record[0]
    print(f't = {t}')

    # Interpolation
    u_grid = griddata((X_star[:, t], Y_star[:, t]), U_star[:, t], (X, Y), method='linear')
    v_grid = griddata((X_star[:, t], Y_star[:, t]), V_star[:, t], (X, Y), method='linear')
    p_grid = griddata((X_star[:, t], Y_star[:, t]), P_star[:, t], (X, Y), method='linear')

    # 原点位置确定
    # 半径为0.5
    u_grid[X ** 2 + Y ** 2 < 0.25] = np.nan
    v_grid[X ** 2 + Y ** 2 < 0.25] = np.nan
    p_grid[X ** 2 + Y ** 2 < 0.25] = np.nan

    # Plot velocity and pressure fields
    axs[0].imshow(u_grid, extent=(x_min, x_max, y_min, y_max), cmap='jet', origin='lower')
    axs[0].set_title('u velocity', fontsize=14)

    axs[1].imshow(v_grid, extent=(x_min, x_max, y_min, y_max), cmap='jet', origin='lower')
    axs[1].set_title('v velocity', fontsize=14)

    axs[2].imshow(p_grid, extent=(x_min, x_max, y_min, y_max), cmap='jet', origin='lower')
    axs[2].set_title('pressure', fontsize=14)

    axs[0].scatter(target_p[0], target_p[1], s=4, color='blue')
    axs[1].scatter(target_p[0], target_p[1], s=4, color='red')
    axs[2].scatter(target_p[0], target_p[1], s=4, color='green')

    # 绘制轨迹
    trajectory_x.append(position_record[i][0])
    trajectory_y.append(position_record[i][1])
    axs[0].plot(trajectory_x, trajectory_y, color='blue')
    axs[1].plot(trajectory_x, trajectory_y, color='red')
    axs[2].plot(trajectory_x, trajectory_y, color='green')

    axs[0].set_title(f'u velocity (t={t})', fontsize=14)
    axs[1].set_title(f'v velocity (t={t})', fontsize=14)
    axs[2].set_title(f'pressure (t={t})', fontsize=14)

    plt.pause(0.2)


filename = f'flow3_{time_set}.png'
# 保存最后一张图片
plt.savefig('./pic/' + filename)
plt.show()
"""

# 成功率测试
"""""""""
sc = 0
for i in range(100):
    # 评估模型并记录总奖励
    obs = test_env.reset()
    time_set = env.return_time()
    rewards_array = []
    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, _ = test_env.step(action)
        rewards_array.append(reward)
        test_env.render()
    test_env.close()
    print(sum(rewards_array))
    point = np.sum(rewards_array)
    if point > 198:
        sc += 1

print("成功次数:")
print(sc)
"""

# 直接绘图
"""""""""
cut_points = np.array(env.cut_points)
# 创建空列表用于存储轨迹坐标
trajectory_x = []
trajectory_y = []

for i in range(len(env.render_positions) - 1):
    # 清空图形
    plt.clf()

    # 打印self.cut_points中的所有点
    cut_points = np.array(env.cut_points)
    plt.scatter(cut_points[:, 0], cut_points[:, 1], color='blue', label='Cut Points')
    # 将self.target_point点用红颜色标出
    plt.scatter(env.target_point[0], env.target_point[1], color='red', label='Target Point')
    # 将当前点的位置用绿色标出
    plt.scatter(env.render_positions[i][0], env.render_positions[i][1], color='green', label='Current Point')

    # 添加当前点的坐标到轨迹列表中
    trajectory_x.append(env.render_positions[i][0])
    trajectory_y.append(env.render_positions[i][1])

    # 绘制轨迹线
    plt.plot(trajectory_x, trajectory_y, color='red', label='Trajectory')

    # 添加图例和标题
    plt.legend()
    plt.title('Environment t = {}'.format(i+time_set))

    # 显示图形并暂停一段时间
    plt.pause(0.1)

filename = f'model3_image_{time_set}.png'
# 保存最后一张图片
plt.savefig('./pic/' + filename)
# 关闭图形窗口
plt.close()
"""
