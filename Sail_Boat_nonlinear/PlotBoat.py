import matplotlib.pyplot as plt
import numpy as np
from matplotlib import patches
from matplotlib.gridspec import GridSpec
from matplotlib.animation import FuncAnimation
from matplotlib.ticker import MaxNLocator, AutoLocator
import math


def radian_to_degree(radian):
    return math.degrees(radian)


def plot_series_with_wind(points, end_point=None, center=None, radius=None, fig=None, ax=None, title=None,
                          xlabel=None, ylabel=None, legend=False, wind=None, save_dir=None, steps=None):
    x_pos = [x for x, y in points]
    y_pos = [y for x, y in points]

    if fig is None:
        fig = plt.figure(figsize=(15, 14))  # 调整整体图像的尺寸

    gs = GridSpec(2, 1, height_ratios=[3.5, 1])  # 创建一个2行1列的网格布局,height_ratios调整两图的占比
    if ax is None:
        ax = fig.add_subplot(gs[0, 0])  # 主图占第一个位置

    ax.plot(x_pos, y_pos, label='Trajectory')  # 主图中的轨迹

    if center is not None and radius is not None:
        # 绘制圆并设置填充颜色
        circle = patches.Circle(center, radius, edgecolor='black', facecolor='lightblue', linestyle='--')
        ax.add_patch(circle)
        # 标注半径的大小
        ax.text(center[0], center[1] + radius / 2, f'R =  {radius}', verticalalignment='bottom',
                horizontalalignment='center', fontsize=8, color='black')
        # 绘制圆心
        ax.scatter(center[0], center[1], color='black', label='Center')

        # 绘制半径线
        ax.plot([center[0], center[0] + radius], [center[1], center[1]], color='black', linestyle='--')

    if end_point is not None:
        end_x, end_y = end_point
        ax.scatter(end_x, end_y, color='red', label='End Point')  # 终点
        ax.text(end_x, end_y, f'({end_x}, {end_y})', verticalalignment='bottom', horizontalalignment='right',
                fontsize=8, color='red')
        ax.scatter(0, 0, color='blue', label='Start Point')  # 起点
        circle = patches.Circle(end_point, 4, edgecolor='red', facecolor='none', linestyle='--')
        ax.add_patch(circle)

    if not xlabel is None:
        ax.set_xlabel(xlabel)
    if not ylabel is None:
        ax.set_ylabel(ylabel)
    if title is not None:
        if steps is not None:
            title += f', Total Steps: {steps}'
        ax.set_title(title)
    ax.grid(True)

    if legend:
        ax.legend()

    if wind is not None:
        ax2 = fig.add_subplot(gs[1, 0])  # 风速子图占第一个位置
        ax3 = ax2.twinx()  # 添加一个新的坐标轴，与ax2共享X轴
        step_time = 0
        time = []
        speed = []
        direction = []
        for step in wind:
            time.append(step_time)
            speed.append(step.strength)  # 获取风速数据
            direction.append(radian_to_degree(step.direction))  # 将风向从弧度转换为角度
            step_time += 1
        ax2.plot(time, speed, color='blue', label='Wind Speed')  # 风速
        ax2.set_title('Wind')
        ax2.set_xlabel('Step')
        ax2.set_ylabel('Speed/m/s')
        ax2.legend(loc='upper left')
        ax2.grid(True)
        # 设置X轴刻度
        ax2.set_xticks(np.arange(min(time), max(time) + 1, 50))

        ax3.plot(time, direction, color='green', label='Wind Direction')  # 风向
        ax3.set_ylabel('Direction/°')

        # 将子图的图例放置在右上角
        ax3.legend(loc='upper right')

    if save_dir:
        fig.savefig(save_dir)

    return fig, ax


def plot_show(points, end_point=None, center=None, radius=None, fig=None, ax=None, N_subplot=1, n_subplot=1,
              title=None, xlabel=None, ylabel=None, label=None, legend=False, wind=None, save_dir=None, steps=None):
    x_pos = [x for x, y in points]
    y_pos = [y for x, y in points]
    if fig is None:
        fig = plt.figure()
    if ax is None:
        ax = fig.add_subplot(N_subplot, 1, n_subplot)

    if not xlabel is None:
        ax.set_xlabel(xlabel)
    if not ylabel is None:
        ax.set_ylabel(ylabel)
    if title is not None:
        """""""""
        if steps is not None:
            title += f', Total Steps: {steps}'
        """
        ax.set_title(title)  # 设置图像标题

    ax.plot(x_pos, y_pos, label=label)

    if wind is not None:
        arrow_start = (0, 0)
        arrow_end = (wind.x / 5, wind.y / 5)
        arrow = patches.FancyArrowPatch(arrow_start, arrow_end, color='green', arrowstyle='fancy', mutation_scale=5,
                                        linewidth=1)
        ax.add_patch(arrow)

    if center is not None and radius is not None:
        # 绘制圆并设置填充颜色
        circle = patches.Circle(center, radius, edgecolor='black', facecolor='lightblue', linestyle='--')
        ax.add_patch(circle)
        # 标注半径的大小
        ax.text(center[0], center[1] + radius / 2, f'R =  {radius}', verticalalignment='bottom',
                horizontalalignment='center', fontsize=8, color='black')
        # 绘制圆心
        ax.scatter(center[0], center[1], color='black', label='Center')

        # 绘制半径线
        ax.plot([center[0], center[0] + radius], [center[1], center[1]], color='black', linestyle='--')

    if end_point is not None:
        end_x, end_y = end_point
        ax.scatter(end_x, end_y, color='red', label='End Point')  # 在图上标注终点坐标
        # ax.text(end_x, end_y, f'({end_x}, {end_y})', verticalalignment='bottom', horizontalalignment='right', fontsize=8, color='red')
        ax.scatter(0, 0, color='blue', label='Start Point')  # 在图上标注起点坐标
        circle = patches.Circle(end_point, 4, edgecolor='red', facecolor='none', linestyle='--')
        ax.add_patch(circle)

    ax.grid(True)

    if legend:
        ax.legend()

    if save_dir:
        fig.savefig(save_dir)
    return fig, ax


# 动作绘图函数
def plot_action(array_in, xlabel="steps", ylabel="rudder_angle/°", title="Action Plot", save_dir=None):
    array_in = np.array(array_in)  # 将列表转换为NumPy数组
    array_in = array_in.flatten()  # 将数组调整为一维形状
    scaled_array = np.degrees(array_in)  # 将弧度转换为角度制
    plt.figure(figsize=(8, 6))  # 设置图表大小
    plt.plot(scaled_array, marker='o', color='b', linestyle='-', linewidth=2, markersize=8)  # 绘制折线图
    plt.xlabel(xlabel)  # 设置X轴标签
    plt.ylabel(ylabel)  # 设置Y轴标签
    plt.title(title)  # 设置图表标题
    plt.grid(True)  # 显示网格
    if save_dir:
        plt.savefig(save_dir)  # 保存图像


def plot_speed(array_in, xlabel="steps", ylabel="speed/m/s", title="Speed Plot", save_dir=None):
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


def plot_action_normal(array_in, xlabel="steps", ylabel="rudder_angle/°", title="Action Plot", save_dir=None):
    array_in = np.array(array_in)  # 将列表转换为NumPy数组
    array_in = array_in.flatten()  # 将数组调整为一维形状
    array_in = np.degrees(array_in)  # 将弧度转换为角度制
    plt.figure(figsize=(8, 6))  # 设置图表大小
    plt.plot(array_in, marker='o', color='b', linestyle='-', linewidth=2, markersize=8)  # 绘制折线图
    plt.xlabel(xlabel)  # 设置X轴标签
    plt.ylabel(ylabel)  # 设置Y轴标签
    plt.title(title)  # 设置图表标题
    plt.grid(True)  # 显示网格
    if save_dir:
        plt.savefig(save_dir)  # 保存图像


def plot_multi_action_normal(arrays_in, labels=None, x_label="steps",
                             y_label="rudder_angle/°", title="Action Plot",
                             save_dir=None):
    plt.figure(figsize=(8, 6))  # 设置图表大小

    # 如果没有传入标签，自动为每个数组生成标签
    if labels is None:
        labels = [f"Array {i + 1}" for i in range(len(arrays_in))]

    # 对每个输入的数组进行处理并绘制
    for i, array_in in enumerate(arrays_in):
        array_in = np.array(array_in).flatten()  # 将列表转换为NumPy数组并展平成一维
        array_in = np.degrees(array_in)  # 将弧度转换为角度制
        plt.plot(array_in, marker='o', linestyle='-', linewidth=2, markersize=8, label=labels[i])  # 绘制折线图

    plt.xlabel(x_label)  # 设置X轴标签
    plt.ylabel(y_label)  # 设置Y轴标签
    plt.title(title)  # 设置图表标题
    plt.grid(True)  # 显示网格
    plt.legend()  # 显示图例

    # 保存图像或显示图像
    if save_dir:
        plt.savefig(save_dir)  # 保存图像
    else:
        plt.show()  # 显示图像


# 动态绘图函数
# \u03C0 代表pi
###############################
def create_animation(position_record, action_array, target_p, target_r, current_wind=None, save_dir=None):
    position_record = position_record[:-1]
    fig, ax = plt.subplots()
    ax.set_xlim(min(x for x, y in position_record) - 10, max(x for x, y in position_record) + 10)
    ax.set_ylim(min(y for x, y in position_record) - 10, max(y for x, y in position_record) + 10)
    ax.set_xlabel('X/m')
    ax.set_ylabel('Y/m')

    def update(frame):
        ax.clear()
        ax.set_xlabel('X/m')
        ax.set_ylabel('Y/m')
        ax.set_xlim(min(x for x, y in position_record) - 10, max(x for x, y in position_record) + 10)
        ax.set_ylim(min(y for x, y in position_record) - 10, max(y for x, y in position_record) + 10)
        ax.plot([x for x, y in position_record[:frame + 1]], [y for x, y in position_record[:frame + 1]], marker='o',
                color='b')
        ax.plot(*target_p, marker='o', markersize=10, color='r')

        # 绘制圆形边界
        circle = patches.Circle(target_p, radius=target_r, edgecolor='red', facecolor='none', linestyle='--')
        ax.add_patch(circle)

        current_action = 180 * action_array[frame][0] / np.pi
        current_angle = 180 * current_wind[frame][3] / np.pi
        # current_speed = current_wind[frame][2]
        ax.set_title(f'time_step: {frame + 1}/{len(position_record)}, Action: {current_action}, '
                     f'Wind:{current_angle}')

    ani = FuncAnimation(fig, update, frames=len(position_record), repeat=False, interval=100)
    ani.save(save_dir, writer='pillow', fps=10)
    return ani


def plot_success(all_position_records, success_time, total_time, x_center, y_center, success_flag, method_name=None,
                 save_dir=None):
    fig, ax = plt.subplots()

    min_x = -5
    max_x = x_center + 10
    min_y = -5
    max_y = y_center + 10

    # 计算 x 轴和 y 轴的范围，增加一些余量以确保图像能够完整显示轨迹
    x_margin = 2
    y_margin = 2
    ax.set_xlim(min_x - x_margin, max_x + x_margin)
    ax.set_ylim(min_y - y_margin, max_y + y_margin)

    # 绘制目标圆
    target_circle = plt.Circle((x_center, y_center), 10, color='black', fill=False, linestyle='--')
    ax.add_artist(target_circle)

    # 设置每条轨迹的颜色和线条属性
    line_alpha = 0.8  # 轨迹透明度
    line_width = 0.2  # 轨迹线宽

    # 绘制所有轨迹
    for record, success_flag in zip(all_position_records, success_flag):
        x_values = [pos[0] for pos in record]
        y_values = [pos[1] for pos in record]
        if success_flag:  # 如果轨迹成功
            line_color = 'green'
        else:  # 如果轨迹失败
            line_color = 'red'
        ax.plot(x_values, y_values, color=line_color, linestyle='-', linewidth=line_width, alpha=line_alpha)

    ax.set_xlabel('X/m')
    ax.set_ylabel('Y/m')
    ax.set_title(f'Method:{method_name} success rate:{success_time}/{total_time}')
    plt.grid(True)
    plt.show()
    if save_dir:
        plt.savefig(save_dir)  # 保存图像


def plot_success_wind(all_position_records, success_time, save_dir, x_center, y_center):
    fig, ax = plt.subplots()

    # 绘制目标圆
    target_circle = plt.Circle((x_center, y_center), 4, color='red', fill=False, linestyle='--')
    ax.add_artist(target_circle)

    if x_center > y_center:
        edge = x_center
    else:
        edge = y_center

    ax.set_xlim(0, edge + 10)  # 设置 x 轴范围
    ax.set_ylim(-2, edge + 10)  # 设置 y 轴范围

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
    ax.set_title(f'Success Rate: {success_time}%')
    plt.grid(True)
    plt.savefig(save_dir)  # 保存图像


# 轨迹图
def plot_series(x_points, y_points, end_point=None, center=None, radius=None, fig=None, ax=None, N_subplot=1,
                n_subplot=1,
                title=None, xlabel=None, ylabel=None, label=None, legend=False, wind=None, save_dir=None, steps=None):
    x_pos = [x for x in x_points]
    y_pos = [y for y in y_points]
    if fig is None:
        fig = plt.figure(figsize=(15, 14))

    gs = GridSpec(2, 1, height_ratios=[3.5, 1])
    if ax is None:
        ax = fig.add_subplot(gs[0, 0])
    ax.plot(x_pos, y_pos, label=label)

    if end_point is not None:
        end_x, end_y = end_point
        ax.scatter(end_x, end_y, color='red', label='End Point')  # 在图上标注终点坐标
        ax.text(end_x, end_y, f'({end_x}, {end_y})', verticalalignment='bottom', horizontalalignment='right',
                fontsize=20, color='red')
        ax.scatter(0, 0, color='blue', label='Start Point')  # 在图上标注起点坐标
        circle = patches.Circle(end_point, 4, edgecolor='red', facecolor='none', linestyle='--')
        ax.add_patch(circle)
    if wind is not None:
        ax2 = fig.add_subplot(gs[1, 0])
        ax3 = ax2.twinx()
        step_time = 0
        time = []
        speed = []
        direction = []
        for step in wind:
            time.append(step_time)
            speed.append(step.strength)
            direction.append(radian_to_degree(step.direction))
            step_time += 1
        ax2.plot(time, speed, color='blue', label='Wind Speed')
        ax2.set_title('Wind')
        ax2.set_xlabel('Step')
        ax2.set_ylabel('Speed/m/s')
        ax2.legend(loc='upper left')
        ax2.grid(True)
        ax2.set_xticks(np.arange(min(time), max(time) + 1, 50))

        ax3.plot(time, direction, color='green', label='Wind Direction')
        ax3.set_ylabel('Direction/°')

        ax3.legend(loc='upper right')
    if center is not None and radius is not None:
        circle = patches.Circle(center, radius, edgecolor='black', facecolor='none', linestyle='--')
        ax.add_patch(circle)

    if not xlabel is None:
        ax.set_xlabel(xlabel)
    if not ylabel is None:
        ax.set_ylabel(ylabel)
    if title is not None:
        if steps is not None:
            title += f' Total Steps:{steps}'
        ax.set_title(title)  # 设置图像标题
    ax.grid(True)

    if legend:
        ax.legend(['Trajectory', 'End Point', 'Start Point'])

    if save_dir:
        fig.savefig(save_dir)
    return fig, ax


def plot_boxplot(data, title=None, x_label=None, save_dir=None):
    plt.figure(figsize=(8, 6))
    plt.boxplot(data)
    plt.title(title)
    plt.ylabel('value')
    plt.xlabel(x_label)
    plt.grid(True)
    if save_dir:
        plt.savefig(save_dir)


def plot_times(x, y, title=None, x_label=None, save_dir=None):
    plt.figure(figsize=(8, 6))
    plt.plot(x, y, marker='o')
    plt.xlabel(x_label)
    plt.ylabel('Success Rate/%')
    plt.title(title)
    plt.grid(True)
    if save_dir:
        plt.savefig(save_dir)


def plot_rewards_and_actions(action_in, reward_in, save_dir=None):
    action_in = np.array(action_in)  # 将列表转换为NumPy数组
    reward_in = np.array(reward_in)
    action_in = action_in.flatten()  # 将数组调整为一维形状
    action_in = np.degrees(action_in)  # 将弧度转换为角度制
    # 创建图像和轴对象
    fig, ax1 = plt.subplots()

    # 绘制奖励曲线
    color = 'tab:red'
    ax1.set_xlabel('steps')
    ax1.set_ylabel('reward', color=color)
    ax1.plot(reward_in[:-1], color=color)
    ax1.tick_params(axis='y', labelcolor=color)

    # 创建第二个坐标轴并绘制动作曲线
    ax2 = ax1.twinx()
    color = 'tab:blue'
    ax2.set_ylabel('action', color=color)
    ax2.plot(action_in[:-1], color=color)
    ax2.tick_params(axis='y', labelcolor=color)

    # 添加标题
    plt.title('Rewards&Actions Plot')

    # 添加图例并设置大小
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left', prop={'size': 12})  # 设置字体

    if save_dir:
        plt.savefig(save_dir)

    # 显示图像
    plt.show()


def plot_rewards_and_actions_RL(action_in, reward_in, save_dir=None):
    action_in = np.array(action_in)  # 将列表转换为NumPy数组
    reward_in = np.array(reward_in)
    action_in = action_in.flatten()  # 将数组调整为一维形状
    action_in = np.degrees(action_in)  # 将弧度转换为角度制
    # 创建图像和轴对象
    fig, ax1 = plt.subplots()

    # 绘制奖励曲线
    color = 'tab:red'
    ax1.set_xlabel('steps')
    ax1.set_ylabel('reward', color=color)
    ax1.plot(reward_in[:-1], color=color)
    ax1.tick_params(axis='y', labelcolor=color)

    # 创建第二个坐标轴并绘制动作曲线
    ax2 = ax1.twinx()
    color = 'tab:blue'
    ax2.set_ylabel('action', color=color)
    ax2.plot(action_in[:-1], color=color)
    ax2.tick_params(axis='y', labelcolor=color)

    # 添加标题
    plt.title('Rewards&Actions Plot')

    # 添加图例并设置大小
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left', prop={'size': 12})  # 设置字体

    if save_dir:
        plt.savefig(save_dir)

    # 显示图像
    plt.show()


def plot_times_with_std(x, y, y_err, title=None, x_label=None, save_dir=None):
    plt.figure(figsize=(8, 6))
    plt.errorbar(x, y, yerr=y_err, fmt='o', capsize=5)
    plt.plot(x, y, marker='o')

    if isinstance(y_err, (int, float)):  # 如果 y_err 是单个值
        y_err = [y_err] * len(y)  # 将 y_err 扩展成与 y 具有相同长度的列表

    plt.fill_between(x, np.array(y) - np.array(y_err), np.array(y) + np.array(y_err), alpha=0.2)

    plt.xlabel(x_label)
    plt.ylabel('Success Rate/%')
    plt.title(title)
    plt.grid(True)
    if save_dir:
        plt.savefig(save_dir)


def plot_multi_series_with_wind(points_list, end_point=None, centers=None, radii=None, fig=None, ax=None, title=None,
                                xlabel=None, ylabel=None, legend=False, legend_labels=None, wind=None, save_dir=None,
                                steps=None):
    if fig is None:
        fig = plt.figure(figsize=(15, 14))

    if wind is not None:
        gs = GridSpec(2, 1, height_ratios=[3.5, 1])  # 2行布局
    else:
        gs = GridSpec(1, 1)  # 只有1行布局
    if ax is None:
        ax = fig.add_subplot(gs[0, 0])

    for i, points in enumerate(points_list):
        x_pos = [x for x, y in points]
        y_pos = [y for x, y in points]
        if legend_labels is not None:
            ax.plot(x_pos, y_pos, label=legend_labels[i])
        else:
            ax.plot(x_pos, y_pos, label=f'Trajectory {i + 1}')

    if centers is not None and radii is not None:
        for center, radius in zip(centers, radii):
            circle = plt.Circle(center, radius, edgecolor='black', facecolor='lightblue', linestyle='--')
            ax.add_patch(circle)
            ax.text(center[0], center[1] + radius / 2, f'R =  {radius}', verticalalignment='bottom',
                    horizontalalignment='center', fontsize=8, color='black')
            ax.scatter(center[0], center[1], color='black', label='Center')
            ax.plot([center[0], center[0] + radius], [center[1], center[1]], color='black', linestyle='--')

    if end_point is not None:
        end_x, end_y = end_point
        ax.scatter(end_x, end_y, color='red', label='End Point')
        ax.text(end_x, end_y, f'({end_x}, {end_y})', verticalalignment='bottom', horizontalalignment='right',
                fontsize=20, color='red')
        ax.scatter(0, 0, color='blue', label='Start Point')
        circle = plt.Circle(end_point, 4, edgecolor='red', facecolor='none', linestyle='--')
        ax.add_patch(circle)

    if not xlabel is None:
        ax.set_xlabel(xlabel)
    if not ylabel is None:
        ax.set_ylabel(ylabel)
    if title is not None:
        if steps is not None:
            title += f', Total Steps: {steps}'
        ax.set_title(title)
    ax.grid(True)

    if legend:
        ax.legend(loc='upper left', prop={'size': 20})

    if wind is not None:
        ax2 = fig.add_subplot(gs[1, 0])
        ax3 = ax2.twinx()
        step_time = 0
        time = []
        speed = []
        direction = []
        for step in wind:
            time.append(step_time)
            speed.append(step.strength)
            direction.append(radian_to_degree(step.direction))
            step_time += 1
        ax2.plot(time, speed, color='blue', label='Wind Speed')
        ax2.set_title('Wind')
        ax2.set_xlabel('Step')
        ax2.set_ylabel('Speed/m/s')
        ax2.legend(loc='upper left')
        ax2.grid(True)
        ax2.set_xticks(np.arange(min(time), max(time) + 1, 50))

        ax3.plot(time, direction, color='green', label='Wind Direction')
        ax3.set_ylabel('Direction/°')

        ax3.legend(loc='upper right')

    if save_dir:
        fig.savefig(save_dir)

    return fig, ax


def plot_action_reward_vs_distance(action, distance, reward):
    """
    绘制action和reward随distance变化的图像

    参数:
    action (list): action数据列表
    distance (list): distance数据列表
    reward (list): reward数据列表
    """
    # 创建图像和轴对象
    fig, ax1 = plt.subplots(figsize=(8, 6))

    # 绘制distance和action
    ax1.plot(distance, action, color='r', linewidth=1.5, marker='o', markersize=4, label='Action')
    ax1.set_xlabel('Distance/m', fontsize=14)
    ax1.set_ylabel('Action/°', color='r', fontsize=14)

    # 创建另一个y轴并绘制reward
    ax2 = ax1.twinx()
    ax2.plot(distance, reward, color='b', linewidth=1.5, marker='o', markersize=4, label='Reward')
    ax2.set_ylabel('Reward', color='b', fontsize=14)

    # 设置图例
    ax1.legend(loc='upper left')
    ax2.legend(loc='upper right')

    # 设置标题和网格
    plt.title('Action&Reward-Distance Plot', fontsize=16)
    plt.grid(True)

    plt.show()
