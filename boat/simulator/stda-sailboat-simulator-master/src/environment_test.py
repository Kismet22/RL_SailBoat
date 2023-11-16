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

env = Boat_Environment()
test_env = DummyVecEnv([lambda: env])

traj_plot = './model_output/轨迹图.png'


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


def main():
    obs = test_env.reset()
    test_env.render()
    rewards_array = []
    wind_record = []
    state_record = []
    action = np.array([[0]])

    for i in range(10):
        wind_record.append(env.wind)
        state_record.append(obs[0][0])
        obs, reward, done, _ = test_env.step(action)
        rewards_array.append(reward)
        test_env.render()
    position_record = env.render_positions
    target_p = env.end_target
    # 静态图像
    # _, _ = plot_series(position_record[:-1], end_point=target_p, wind=env.wind, save_dir=traj_plot)
    # print(wind_record)
    # print(state_record)
    # print(env.render_winds)
    print(position_record)


if __name__ == "__main__":
    main()
