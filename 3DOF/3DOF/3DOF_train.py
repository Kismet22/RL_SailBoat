from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.noise import OrnsteinUhlenbeckActionNoise

from PlotBoat import *
from numpy import *
import matplotlib.pyplot as plt

import torch
import os

deq = solve

dir_pic = './reward_pic/'
dir_model = './model/'
dir_best_model = './best_model/'
traj_plot = './train_traj/'
time_pic = './time_pic/'

# 从默认CUDA设备上删除所有张量
torch.cuda.empty_cache()

''' 
State中包含的信息包括:
    地面坐标位置:POS_X(0), POS_Y(1);
    艏向角:YAW(2);
    前向速度(forward)和漂移速度(leeway):VEL_X(3), VEL_Y(4);
    艏向角变化率:YAW_RATE(5);
    RUDDER_STATE(6), SAIL_STATE(7);
'''

# 自定义回调函数
# 动态调整clip
initial_clip_range = 0.2


class CustomCallback(BaseCallback):
    # 定义一个类变量来存储初始 clip_range 的值
    train_rewards = []  # 整个episode的奖励
    train_success = []
    train_fail = []
    clip_range = initial_clip_range

    def __init__(self, verbose=0):
        super(CustomCallback, self).__init__(verbose)
        self.episode_rewards = []  # 一次训练轮的奖励
        self.episode_actions = []

    def _on_step(self) -> bool:
        # 在每个训练步骤结束后记录奖励
        current_reward = np.mean(self.locals['rewards'])
        self.episode_rewards.append(current_reward)
        self.episode_actions.append(self.locals["actions"])

        # 当 episode 结束时，计算整个 episode 的平均奖励
        if self.locals['dones']:
            episode_sum_reward = np.sum(self.episode_rewards)
            print("当前训练轮的总奖励", episode_sum_reward)
            CustomCallback.train_rewards.append(episode_sum_reward)

            # 根据整个 episode 的平均奖励来动态调整 Clip 范围
            """""""""
            if episode_sum_reward > 300 and CustomCallback.clip_range > 0.12:
                CustomCallback.clip_range = CustomCallback.clip_range - 0.01
            elif episode_sum_reward < 300 and CustomCallback.clip_range < 0.2:
                CustomCallback.clip_range = CustomCallback.clip_range + 0.01
            print("调整后的 clip", CustomCallback.clip_range)
            """

            if self.locals['infos'][0]['success']:
                CustomCallback.train_success.append(episode_sum_reward)
            else:
                CustomCallback.train_fail.append(episode_sum_reward)
            print()

            # 清空奖励记录，准备下一个 episode 的记录
            self.episode_rewards = []
            self.episode_actions = []
            # 设置模型的 Clip 范围
            self.model.policy.clip_range_vf = CustomCallback.clip_range

        return True  # 继续训练


# 设置CUDA设备
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # 指定使用的GPU设备号
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# 从默认CUDA设备上删除所有张量
torch.cuda.empty_cache()

# 创建环境实例
env = Boat_Environment()

# 环境检测
check_env(env)

# 开始训练
train_env = DummyVecEnv([lambda: env])
test_env = DummyVecEnv([lambda: env])

# 创建 PPO 模型

""""
# gae越小，对未来回报的折扣越小
# ent_coef 鼓励探索
# lr学习率
# gamma 注重长期奖励
# n_steps和total_timesteps的关系:模型每n_steps进行一次更新(采样步数),一轮训练共走total_timesteps步，如果中间遇到done就重置环境
model = PPO("MlpPolicy", train_env, verbose=0, device="cuda", n_steps=2048, learning_rate=0.003, gae_lambda=0.95,
            ent_coef=0.0, gamma=0.99, clip_range=0.2)
"""

# PPO训练
model = PPO("MlpPolicy", train_env, verbose=0, device="cuda", n_steps=256, learning_rate=0.0003, gae_lambda=0.95,
            ent_coef=0.0, gamma=0.99, batch_size=64)
# 定义训练参数
total_episode = 5000
# 每次训练步长
timesteps_per_episode = 8192
# 每个episode保存一次数据
save_interval = 1

# 初始化记录变量
episode_mean_rewards = []
episode_success_time = []
episode_fail_time = []

# 开始训练
for episode in range(total_episode):

    # 进行一定数量的时间步训练
    print("\n")
    print("episode:", episode)
    model.learn(total_timesteps=timesteps_per_episode, callback=CustomCallback())
    mean_reward = mean(CustomCallback.train_rewards)
    print(mean_reward)
    success_time = len(CustomCallback.train_success)
    fail_time = len(CustomCallback.train_fail)
    episode_mean_rewards.append(mean_reward)
    print(episode_mean_rewards)

    episode_success_time.append(success_time)
    episode_fail_time.append(fail_time)

    print("mean_episode_reward:", mean_reward)
    print()
    print("train_time", len(CustomCallback.train_rewards))
    print("success_time:", success_time)
    print("fail_time:", fail_time)
    CustomCallback.train_rewards.clear()
    CustomCallback.train_success.clear()
    CustomCallback.train_fail.clear()
    print("\n")
    print("############       ############")

    print("start to evaluate")
    # 评估模型并记录总奖励
    episode_rewards = []
    action_array = []
    state_array = []
    position_record = []
    obs = test_env.reset()
    target_p = env.end_target
    print("target_p:", target_p)
    print("本次终点:", env.end_target)
    done = False
    # 打印第一个动作
    first_action, _ = model.predict(obs, deterministic=True)  # 输出确定性的动作
    print("第一个动作:", first_action)
    action_array.append(first_action)
    obs, reward, done, _, = test_env.step(first_action)
    episode_rewards.append(reward)
    state_array.append(obs)
    while not done:
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
    test_steps_old = test_steps_new
    print("test_reward:", total_reward)
    print("############       ############")
    print("\n")

    # 图像保存
    if episode % save_interval == 0:
        plt.clf()
        # center:区域圆心
        # radius:区域半径
        _, _ = plot_series(position_record[:-2], end_point=target_p,
                           title=f'total_steps:{test_steps_old}', center=[20, 10], radius=6,
                           wind=env.wind, save_dir=traj_plot + f'pos_plot_{episode + 1}.png')
        plt.clf()  # 清除当前的图形
        plot_rewards_and_outcomes(episode_mean_rewards, save_dir=dir_pic + f'reward_plot_{episode + 1}.png')
        plt.clf()  # 清除当前的图形
        plot_success_fail(episode_success_time, episode_fail_time,
                          save_dir=time_pic + f'times_plot_{episode + 1}.png')
        plt.clf()  # 清除当前的图形
        model.save(dir_model + f"model_{episode}.zip")

    # 每个训练周期结束后释放内存
    torch.cuda.empty_cache()
