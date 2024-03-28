from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from PlotBoat import *
import numpy as np
import pandas as pd


def single_test():
    # 加载
    ###############################
    model_file = './trained_model/model_20240327.zip'
    env = Boat_Environment()
    test_env = DummyVecEnv([lambda: env])
    # 加载预训练的模型
    model = PPO.load(model_file, render_mode='human')

    # 单步测试
    ###############################
    pic_action = './model_output/动作记录.png'
    traj_plot = './model_output/轨迹图.png'
    move_plot = './model_output/动态图.gif'
    print("start to evaluate")
    # 评估模型并记录总奖励
    episode_rewards = []
    action_array = []
    state_array = []
    position_record = []
    wind_record = []
    obs = test_env.reset()
    target_p = env.end_target
    print("target_p:", target_p)
    print("本次终点:", env.end_target)
    done = False
    wind_record.append(env.wind)
    # 打印第一个动作
    first_action, _ = model.predict(obs, deterministic=True)  # 输出确定性的动作
    print("第一个动作:", first_action)
    action_array.append(first_action)
    obs, reward, done, _, = test_env.step(first_action)
    episode_rewards.append(reward)
    state_array.append(obs)
    while not done:
        wind_record.append(env.wind)
        action, _ = model.predict(obs, deterministic=True)
        action_array.append(action)
        obs, reward, done, _, = test_env.step(action)
        episode_rewards.append(reward)
        state_array.append(obs)
        if env.position_record != [[0.0, 0.0]]:
            position_record = env.position_record
    test_steps_new = len(episode_rewards)
    print("本次步数为:", test_steps_new)

    total_reward = np.sum(episode_rewards)
    print("test_reward:", total_reward)
    print("############       ############")
    print("\n")

    # 输出测试
    # 静态图像
    _, _ = plot_series(position_record[:-2], end_point=target_p,
                       title='Trajectories', xlabel='X/m', ylabel='Y/m', center=[20, 10], radius=6,
                       wind=env.wind, save_dir=traj_plot)

    # 动作图像
    plot_action(action_array, save_dir=pic_action)

    # 动图
    _ = create_animation(position_record, action_array, target_p, 6, wind_record, save_dir=move_plot)


def multiple_test():
    # 加载
    ###############################
    model_file = './trained_model/model_20240327.zip'
    env = Boat_Environment()
    test_env = DummyVecEnv([lambda: env])
    # 加载预训练的模型
    model = PPO.load(model_file, render_mode='human')

    pic_success = './model_output/成功数测试.png'
    unique_target_positions = set()  # 用于存储唯一的目标位置
    successful_times = 0
    all_position_records = []  # 存储所有的轨迹数据

    for _ in range(100):  # 重复测试100次
        print("start to evaluate")
        # 评估模型并记录总奖励
        episode_rewards = []
        action_array = []
        state_array = []
        position_record = []
        wind_record = []

        # 生成新的目标位置，直到其不在集合中为止
        while True:
            obs = test_env.reset()
            target_p = tuple(env.end_target)  # 将列表转换为元组
            if target_p not in unique_target_positions:
                unique_target_positions.add(target_p)  # 将目标位置添加到集合中
                break  # 如果目标位置唯一，则跳出循环

        print("target_p:", target_p)
        print("本次终点:", env.end_target)
        done = False
        wind_record.append(env.wind)
        # 打印第一个动作
        first_action, _ = model.predict(obs, deterministic=True)  # 输出确定性的动作
        print("第一个动作:", first_action)
        action_array.append(first_action)
        obs, reward, done, _, = test_env.step(first_action)
        episode_rewards.append(reward)
        state_array.append(obs)
        while not done:
            wind_record.append(env.wind)
            action, _ = model.predict(obs, deterministic=True)
            action_array.append(action)
            obs, reward, done, _, = test_env.step(action)
            episode_rewards.append(reward)
            state_array.append(obs)
            if env.position_record != [[0.0, 0.0]]:
                position_record = env.position_record  # 位置记录
        test_steps_new = len(episode_rewards)
        print("本次步数为:", test_steps_new)

        total_reward = np.sum(episode_rewards)  # 总奖励
        print("test_reward:", total_reward)
        all_position_records.append(position_record)

        if total_reward > 0:
            # 如果total_reward大于0，保存成功的轨迹
            successful_times += 1

        print("############       ############")
        print("\n")

    print("总成功数:", successful_times)
    plot_success(all_position_records, success_time=successful_times, save_dir=pic_success)


def fixed_end():
    # 用于多种方法的比较
    # 加载
    ###############################
    model_file = './trained_model/model_20240327.zip'
    env = Boat_Environment_FixedEnd()
    test_env = DummyVecEnv([lambda: env])
    # 加载预训练的模型
    model = PPO.load(model_file, render_mode='human')

    # 单步测试
    ###############################
    pic_action = './model_output/RL动作记录.png'
    traj_plot = './model_output/RL轨迹图.png'
    move_plot = './model_output/RL动态图.gif'
    print("start to evaluate")
    # 评估模型并记录总奖励
    episode_rewards = []
    action_array = []
    state_array = []
    position_record = []
    wind_record = []
    obs = test_env.reset()
    target_p = env.end_target
    print("target_p:", target_p)
    print("本次终点:", env.end_target)
    done = False
    wind_record.append(env.wind)
    # 打印第一个动作
    first_action, _ = model.predict(obs, deterministic=True)  # 输出确定性的动作
    print("第一个动作:", first_action)
    action_array.append(first_action)
    obs, reward, done, _, = test_env.step(first_action)
    episode_rewards.append(reward)
    state_array.append(obs)
    while not done:
        wind_record.append(env.wind)
        action, _ = model.predict(obs, deterministic=True)
        action_array.append(action)
        obs, reward, done, _, = test_env.step(action)
        episode_rewards.append(reward)
        state_array.append(obs)
        if env.position_record != [[0.0, 0.0]]:
            position_record = env.position_record
    test_steps_new = len(episode_rewards)
    print("本次步数为:", test_steps_new)

    total_reward = np.sum(episode_rewards)
    print("test_reward:", total_reward)
    print("############       ############")
    print("\n")

    # 输出测试
    # 静态图像
    _, _ = plot_series(position_record[:-2], end_point=target_p,
                       title='Trajectories', xlabel='X/m', ylabel='Y/m', center=[20, 10], radius=6,
                       wind=env.wind, legend=True, save_dir=traj_plot)

    # 动作图像
    plot_action(action_array, save_dir=pic_action)

    # 动图
    _ = create_animation(position_record, action_array, target_p, 6, wind_record, save_dir=move_plot)


def main():
    # 单次测试
    # single_test()
    # 多次测试
    # multiple_test()
    # 对比测试
    fixed_end()


if __name__ == "__main__":
    main()
