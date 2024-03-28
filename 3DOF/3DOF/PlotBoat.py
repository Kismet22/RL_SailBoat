import matplotlib.pyplot as plt
import numpy as np
from matplotlib import patches
from matplotlib.animation import FuncAnimation
from matplotlib.ticker import MaxNLocator, AutoLocator
from SailBoat_Environment import *


# 轨迹图
def plot_series(points, end_point=None, center=None, radius=None, fig=None, ax=None, N_subplot=1, n_subplot=1,
                title=None, xlabel=None, ylabel=None, label=None, legend=False, wind=None, save_dir=None):
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
        ax.text(end_x, end_y, f'({end_x}, {end_y})', verticalalignment='bottom', horizontalalignment='right',
                fontsize=8, color='red')
        ax.scatter(0, 0, color='blue', label='Start Point')  # 在图上标注起点坐标
        #        circle = patches.Circle(end_point, calculate_distance([0, 0], end_point) / 10,
        #                                edgecolor='red', facecolor='none', linestyle='--')
        circle = patches.Circle(end_point, 4, edgecolor='red', facecolor='none', linestyle='--')
        ax.add_patch(circle)
    if wind is not None:
        arrow_start = (0, 0)
        arrow_end = (wind.x / 5, wind.y / 5)
        arrow = patches.FancyArrowPatch(arrow_start, arrow_end, color='green', arrowstyle='fancy', mutation_scale=5,
                                        linewidth=1)
        ax.add_patch(arrow)
    if center is not None and radius is not None:
        circle = patches.Circle(center, radius, edgecolor='black', facecolor='none', linestyle='--')
        ax.add_patch(circle)

    if not xlabel is None:
        ax.set_xlabel(xlabel)
    if not ylabel is None:
        ax.set_ylabel(ylabel)
    if title is not None:
        ax.set_title(title)  # 设置图像标题
    ax.grid(True)

    if legend:
        ax.legend()

    if save_dir:
        fig.savefig(save_dir)
    return fig, ax


# 成功失败数图
def plot_success_fail(episode_success_time, episode_fail_time, save_dir=None):
    plt.figure(figsize=(10, 6))  # 设置图形大小为 10x6 英寸

    plt.plot(episode_success_time, label='Success')
    plt.plot(episode_fail_time, label='Fail')

    plt.title('Success and Fail Counts Over Episodes')
    plt.xlabel('Episode')
    plt.ylabel('Counts')
    plt.legend()

    # 自适应调整 x 轴的刻度
    plt.gca().xaxis.set_major_locator(AutoLocator())
    # 确保 x 轴的最小分度为整数
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
    if save_dir:
        plt.savefig(save_dir)


# 平均回报图
def plot_rewards_and_outcomes(episode_mean_rewards, save_dir=None):
    # 设置图形大小
    plt.figure(figsize=(10, 6))

    # 绘制奖励均值点
    plt.plot(episode_mean_rewards, 'ro-', label='Mean Rewards', alpha=0.8)  # 使用红色

    # 添加标题和标签
    plt.title('Rewards for each episode')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')

    # 自适应调整 x 轴的刻度
    plt.gca().xaxis.set_major_locator(AutoLocator())
    # 确保 x 轴的最小分度为整数
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
    if save_dir:
        plt.savefig(save_dir)


# 动作绘图函数
def plot_action(array_in, xlabel="steps", ylabel="rudder_angle/\u03C0", title="Action Plot", save_dir=None):
    array_in = np.array(array_in)  # 将列表转换为NumPy数组
    array_in = array_in.flatten()  # 将数组调整为一维形状
    scaled_array = scale_action(array_in)  # 缩放动作值
    plt.figure(figsize=(8, 6))  # 设置图表大小
    plt.plot(scaled_array, marker='o', color='b', linestyle='-', linewidth=2, markersize=8)  # 绘制折线图
    plt.xlabel(xlabel)  # 设置X轴标签
    plt.ylabel(ylabel)  # 设置Y轴标签
    plt.title(title)  # 设置图表标题
    plt.grid(True)  # 显示网格
    if save_dir:
        plt.savefig(save_dir)  # 保存图像


def plot_action_normal(array_in, xlabel="steps", ylabel="rudder_angle/\u03C0", title="Action Plot", save_dir=None):
    array_in = np.array(array_in)  # 将列表转换为NumPy数组
    array_in = array_in.flatten()  # 将数组调整为一维形状
    plt.figure(figsize=(8, 6))  # 设置图表大小
    plt.plot(array_in, marker='o', color='b', linestyle='-', linewidth=2, markersize=8)  # 绘制折线图
    plt.xlabel(xlabel)  # 设置X轴标签
    plt.ylabel(ylabel)  # 设置Y轴标签
    plt.title(title)  # 设置图表标题
    plt.grid(True)  # 显示网格
    if save_dir:
        plt.savefig(save_dir)  # 保存图像


# 动态绘图函数
# \u03C0 代表pi
###############################
def create_animation(position_record, action_array, target_p, target_r, current_wind=None, save_dir=None):
    position_record = position_record[:-1]
    fig, ax = plt.subplots()
    ax.set_xlim(min(x for x, y in position_record) - 2, max(x for x, y in position_record) + 10)
    ax.set_ylim(min(y for x, y in position_record) - 2, max(y for x, y in position_record) + 10)
    ax.set_xlabel('X/m')
    ax.set_ylabel('Y/m')

    def update(frame):
        ax.clear()
        ax.set_xlabel('X/m')
        ax.set_ylabel('Y/m')
        ax.set_xlim(min(x for x, y in position_record) - 2, max(x for x, y in position_record) + 10)
        ax.set_ylim(min(y for x, y in position_record) - 2, max(y for x, y in position_record) + 10)
        ax.plot([x for x, y in position_record[:frame + 1]], [y for x, y in position_record[:frame + 1]], marker='o',
                color='b')
        ax.plot(*target_p, marker='o', markersize=10, color='r')

        # 绘制圆形边界
        circle = patches.Circle(target_p, radius=target_r, edgecolor='red', facecolor='none', linestyle='--')
        ax.add_patch(circle)

        current_action = 180 * scale_action(action_array[frame][0]) / np.pi
        current_angle = 180 * current_wind[frame][3] / np.pi
        # current_speed = current_wind[frame][2]
        ax.set_title(f'time_step: {frame + 1}/{len(position_record)}, Action: {current_action}, '
                     f'Wind:{current_angle}')

    ani = FuncAnimation(fig, update, frames=len(position_record), repeat=False, interval=100)
    ani.save(save_dir, writer='pillow', fps=10)
    return ani


def plot_success(all_position_records, success_time, save_dir):
    fig, ax = plt.subplots()

    # 绘制目标圆
    target_circle = plt.Circle((20, 10), 10, color='red', fill=False, linestyle='--')
    ax.add_artist(target_circle)

    # 设置每条轨迹的颜色和线条属性
    line_color = 'green'  # 轨迹颜色
    line_alpha = 0.9  # 轨迹透明度
    line_width = 0.15  # 轨迹线宽

    # 绘制所有轨迹
    for record in all_position_records:
        x_values = [pos[0] for pos in record]
        y_values = [pos[1] for pos in record]
        ax.plot(x_values, y_values, color=line_color, linestyle='-', linewidth=line_width, alpha=line_alpha)

    ax.set_xlabel('X/m')
    ax.set_ylabel('Y/m')
    ax.set_title(f'Success Time: {success_time}')
    plt.grid(True)
    plt.savefig(save_dir)  # 保存图像

