from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from BoatEnvironment import Boat_Environment
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
import numpy as np
from matplotlib.animation import FuncAnimation
from PIL import Image
import matplotlib.patches as patches
import pandas as pd

pic_action = './model_output/动作记录.png'
traj_plot = './model_output/轨迹图.png'
move_plot = './model_output/动态图.gif'


# 动作绘图函数
###############################
def plot_action(array, xlabel="time", ylabel="rudder_angle", title="action_plot", save_dir=None):
    array = np.array(array)  # 将列表转换为NumPy数组
    array = array.flatten()  # 将数组调整为一维形状
    plt.figure(figsize=(8, 6))  # 设置图表大小
    plt.plot(array, marker='o', color='b', linestyle='-', linewidth=2, markersize=8)  # 绘制折线图
    plt.xlabel(xlabel)  # 设置X轴标签
    plt.ylabel(ylabel)  # 设置Y轴标签
    plt.title(title)  # 设置图表标题
    plt.grid(True)  # 显示网格
    plt.savefig(save_dir)


# 静态绘图函数
###############################
def plot_series(points, end_point=None, fig=None, ax=None, N_subplot=1, n_subplot=1, title=None, xlabel=None,
                ylabel=None, label=None,
                legend=False, wind=None, save_dir=None):
    x_pos = [x for x, y in points]
    y_pos = [y for x, y in points]
    if fig is None:
        fig = plt.figure()
    if ax is None:
        ax = fig.add_subplot(N_subplot, 1, n_subplot)
    ax.plot(x_pos, y_pos, label=label)
    if end_point is not None:
        end_x, end_y = end_point
        ax.scatter(end_x, end_y, color='red', label='End Point')  # 在图上标注终点坐标
        ax.scatter(0, 0, color='blue', label='Start Point')  # 在图上标注终点坐标
    if wind is not None:
        arrow_start = (0, 0)
        arrow_end = (wind.x / 5, wind.y / 5)
        arrow = patches.FancyArrowPatch(arrow_start, arrow_end, color='green', arrowstyle='fancy', mutation_scale=5,
                                        linewidth=1)
        ax.add_patch(arrow)

    if not xlabel is None:
        ax.set_xlabel(xlabel)
    if not ylabel is None:
        ax.set_ylabel(ylabel)
    ax.grid(True)

    if legend:
        ax.legend()
    fig.savefig(save_dir)
    return fig, ax


# 动态绘图函数
###############################
def create_animation(position_record, action_array, target_p, current_wind=None, save_dir=None):
    position_record = position_record[:-1]
    fig, ax = plt.subplots()
    ax.set_xlim(min(x for x, y in position_record) - 2, max(x for x, y in position_record) + 2)
    ax.set_ylim(min(y for x, y in position_record) - 2, max(y for x, y in position_record) + 2)

    def update(frame):
        ax.clear()
        ax.set_xlim(min(x for x, y in position_record) - 2, max(x for x, y in position_record) + 2)
        ax.set_ylim(min(y for x, y in position_record) - 2, max(y for x, y in position_record) + 2)
        ax.plot([x for x, y in position_record[:frame + 1]], [y for x, y in position_record[:frame + 1]], marker='o',
                color='b')
        ax.plot(*target_p, marker='o', markersize=10, color='r')

        current_action = 180 * action_array[frame] / np.pi
        current_angle = 180 * current_wind[frame][3] / np.pi
        # current_speed = current_wind[frame][2]
        ax.set_title(f'time_step: {frame + 1}/{len(position_record)}, Action: {current_action}, Wind:{current_angle}')

    ani = FuncAnimation(fig, update, frames=len(position_record), repeat=False, interval=100)
    ani.save(save_dir, writer='pillow', fps=10)
    return ani


# 测试
###############################

# model_file = './trained_model/start00end22.zip'
# model_file = './best_model/model_current_best.zip'
model_file = './model/model_30.zip'
env = Boat_Environment()
test_env = DummyVecEnv([lambda: env])
# 加载预训练的模型
model = PPO.load(model_file, render_mode='human')

"""""""""
rewards_array = []
action_array = []
wind_record = []

obs = test_env.reset()
test_env.render()
time_set = env.return_time()

done = False
while not done:
    wind_record.append(env.wind)
    action, _ = model.predict(obs, deterministic=True)
    action_array.append(action)
    obs, reward, done, _ = test_env.step(action)
    rewards_array.append(reward)
    test_env.render()

total_reward = np.sum(rewards_array)
print(rewards_array)
print(wind_record)
test_env.close()

target_p = env.end_target
time_record = env.render_time
position_record = env.render_positions

# 输出测试

# 静态图像
_, _ = plot_series(position_record[:-1], end_point=target_p, wind=env.wind, save_dir=traj_plot)

# 动作图像
plot_action(action_array, save_dir=pic_action)

# 动图
_ = create_animation(position_record, action_array, target_p, wind_record, save_dir=move_plot)
"""


successful_trajectories = []
successful_times = 0

for _ in range(100):  # 重复测试100次
    obs = test_env.reset()
    rewards_array = []
    action_array = []
    wind_array = []
    done = False
    while not done:
        wind_array.append(env.wind)
        action, _ = model.predict(obs, deterministic=True)
        action_array.append(action)
        obs, reward, done, _ = test_env.step(action)
        rewards_array.append(reward)
        test_env.render()

    total_reward = np.sum(rewards_array)
    target_p = env.end_target
    time_record = env.render_time
    position_record = env.render_positions

    if total_reward > 0:
        # 如果total_reward大于0，保存成功的轨迹
        successful_times += 1
        # 如果total_reward大于0，保存成功的轨迹
        successful_trajectories.append({
            'Total_Reward': total_reward,
            'Action_Array': action_array,
            'Wind_angle': [wind[3] for wind in wind_array],
            'Wind_speed': [wind[2] for wind in wind_array]
        })
# 创建包含所有成功轨迹的DataFrame
final_df = pd.DataFrame(successful_trajectories)
# 保存到Excel文件
final_df.to_excel('./model_output/successful_trajectories.xlsx', index=False)
print(successful_times)

