import gym
import numpy as np
from math import *
from matplotlib import pyplot as plt
from SaiBoat_Environment_1 import Boat_Environment



def test_boat_environment():
    #env = Boat_Environment(render_mode=None, is_fixed_start_and_target=True, sim_wave=False)
    env = Boat_Environment()
    obs, info = env.reset()

    print("Initial Observation:", obs)
    print("Initial Info:", info)

    done = False
    truncated = False
    step_count = 0

    # 用于保存轨迹的列表
    x_positions = []
    y_positions = []

    # 保存初始位置
    #x_positions.append(info["x"])
    #y_positions.append(info["y"])
    #x_positions.append(obs["dx"])
    #y_positions.append(obs["dy"])
    x_positions.append(obs[0])
    y_positions.append(obs[1])


    while not (done or truncated):
        # 采样随机动作
        action = pi / 2
        print(f"Step {step_count}, Action: {action}")

        # 执行一步
        obs, reward, done, truncated, info = env.step(action)

        # 记录轨迹中的x和y
        #x_positions.append(info["x"])
        #y_positions.append(info["y"])
        #x_positions.append(obs["dx"])
        #y_positions.append(obs["dy"])
        x_positions.append(obs[0])
        y_positions.append(obs[1])

        print(f"Observation: {obs}")
        print(f"Reward: {reward}")
        print(f"Info: {info}")
        print(f"Terminated: {done}, Truncated: {truncated}")

        step_count += 1
        if step_count > 500:  # 限制步数以防测试时无限循环
            break

    env.close()

    # 绘制轨迹
    plt.figure()
    plt.plot(x_positions, y_positions, marker='o', color='b', label='Boat Trajectory')
    plt.xlabel('X Position')
    plt.ylabel('Y Position')
    plt.title('Boat Trajectory Over Time')
    plt.grid(True)
    plt.legend()
    plt.show()


if __name__ == "__main__":
    test_boat_environment()
