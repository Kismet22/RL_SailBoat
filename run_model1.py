import gymnasium as gym
from gym.spaces import Box
# matlab数据加载
from scipy.io import loadmat
import numpy as np
from time import process_time
import random
import math
from typing import Union

from stable_baselines3 import PPO
from stable_baselines3.ddpg import DDPG
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.noise import OrnsteinUhlenbeckActionNoise

from heading_controller import heading_controller
from simulation import *
from numpy import *
import matplotlib.pyplot as plt
from sail_angle import sail_angle
import copy

import torch
import os

deq = solve

dir_pic = './pic/'
dir_model = './model/'
dir_best_model = './best_model/'

# 从默认CUDA设备上删除所有张量
torch.cuda.empty_cache()

''' State中包含的信息包括:
    POS_X(0), POS_Y(1), POS_Z(2);
    ROLL(3), PITCH(4), YAW(5);
    VEL_X(6), VEL_Y(7), VEL_Z(8);
    ROLL_RATE(9), PITCH_RATE(10), YAW_RATE(11);
    RUDDER_STATE(12), SAIL_STATE(13);
    YAW是帆船的当前舵角
'''

# 自定义回调函数
# 动态调整clip
# 初始的Clip范围
# self.model.learning_rate 是模型的学习率
# self.model.num_timesteps 是模型的当前时间步

initial_clip_range = 0.2


class CustomCallback(BaseCallback):
    # 定义一个类变量来存储初始 clip_range 的值

    clip_range = initial_clip_range

    def __init__(self, verbose=0):
        super(CustomCallback, self).__init__(verbose)
        self.episode_rewards = []
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
            # 输出 clip_range 的类型
            # print("clip_range 的类型：", type(CustomCallback.clip_range))
            # 根据整个 episode 的平均奖励来动态调整 Clip 范围
            if episode_sum_reward > 500 and CustomCallback.clip_range > 0.12:
                CustomCallback.clip_range = CustomCallback.clip_range - 0.01
            elif episode_sum_reward < 500 and CustomCallback.clip_range < 0.2:
                CustomCallback.clip_range = CustomCallback.clip_range + 0.01

            print("调整后的 clip", CustomCallback.clip_range)
            print("本轮的第一个动作", self.episode_actions[0])

            # 清空奖励记录，准备下一个 episode 的记录
            self.episode_rewards = []
            self.episode_actions = []
            # 设置模型的 Clip 范围
            self.model.policy.clip_range_vf = CustomCallback.clip_range

        return True  # 继续训练


# 常微分方程修改
def deq_sparse(time, state):
    # ignore pitch and heave oscillations (not really needed)
    # 修改常微分方程
    diff = deq(time, state)
    diff[VEL_Z] = 0
    diff[PITCH_RATE] = 0
    diff[ROLL_RATE] = 0
    return diff


# 状态初始化
def init_data_arrays(n_states, N_steps, x0):
    x = zeros((n_states, N_steps + 1))  # n_states行，N_steps + 1列
    r = zeros(N_steps + 1)
    sail = zeros(N_steps + 1)
    t = zeros(N_steps + 1)
    # x0为12个state的初始状态
    x[:, 0] = x0  # 第一列

    ref_heading = zeros(N_steps + 1)
    # 返回状态、时间、奖励、帆状态和前进角度
    return x, t, r, sail, ref_heading


# 微分方程初始化
def init_integrator(x0, sampletime, sparse=False):
    # init integrator
    # 稀疏化处理，只考虑二维上的状态变化
    sparse = True
    # 是否稀疏化处理
    fun = deq if not sparse else deq_sparse
    # 选择模型方法
    # integrator = ode(fun).set_integrator('dopri5', nsteps=4000)
    # 自动设置更新步数
    integrator = ode(fun).set_integrator('dopri5', nsteps=1000)
    # ode(fun)包含当前时间 t 和当前状态 y

    # 初始化状态输入
    integrator.set_initial_value(x0, 0)
    # init heading controller
    controller = heading_controller(sampletime, max_rudder_angle=15 * pi / 180)
    controller.calculate_controller_params(YAW_TIMECONSTANT)

    return integrator, controller


# 平滑处理
def smooth_reference(ref_heading, n_filter=5):
    # potential smoothing of heading reference
    # 拷贝前进角度
    smoothed = ref_heading.copy()
    N = ref_heading.shape[0]
    for i in range(N):
        # 改为整数除法，避免报错
        ind_low = max(0, i - n_filter // 2)
        ind_high = min(N, i + (n_filter - n_filter // 2))
        smoothed[i] = np.mean(ref_heading[ind_low:ind_high])
    return smoothed


# 计算欧式距离
def calculate_distance(old, new):
    distance = abs(sqrt((new[0] - old[0]) ** 2 + (new[1] - old[1]) ** 2))
    return distance


# 随机生成终点
def random_start():
    # 目标区域在一个圆形的范围内
    # 生成随机角度
    angle = random.uniform(0, 2 * math.pi)
    # 生成随机半径，半径小于等于0.5
    radius = random.uniform(0, 1)
    # 将极坐标转换为直角坐标
    x = 5 + radius * math.cos(angle)
    y = 1 + radius * math.sin(angle)
    # x = random.uniform(2.2, 2.4)
    # y = random.uniform(3, 3.2)
    # x = random.choice([3.8, 4])
    # y = random.choice([0.8, 1])
    #x = 15
    #y = 1
    startpoint = [x, y]
    print("目标终点为:", startpoint)
    return startpoint


def random_wind():
    # 生成一个速度和角度都随机的风
    # wind_speed = random.uniform(2, 5)
    # 风速小幅摆动
    # wind_speed = random.uniform(4.9, 5)
    wind_speed = 5
    # wind_angle = random.uniform(0, 2 * pi)
    # 风角小幅摆动
    # wind_angle = random.uniform(0.95 * pi / 2, pi / 2)
    wind_angle = 0.5 * pi
    wind = TrueWind(wind_speed * cos(wind_angle), wind_speed * sin(wind_angle), wind_speed, wind_angle)
    return wind


# 计算两点坐标差
def calculate_angle(current, target):
    angle_radians = math.atan2(target[1] - current[1], target[0] - current[0])
    angle_degrees = math.degrees(angle_radians)
    angle_degrees = pi * angle_degrees / 180
    return angle_degrees


class Boat_Environment(gym.Env):
    def __init__(self, render_mode='human'):

        ############
        # 初始化信息
        self.success_flag = False  # 到达终点的标志
        self.over_flag = False  # 时间用完标志
        self.success_reward = 500
        self.fail_punishment = -50
        self.total_distance = 0

        # 在初始化时设置好目标点
        ############
        # self.end_target = [2, 2]  # 实验，目标坐标点
        self.end_target = [0, 0]
        ############

        self.environment = None  # 实验，周围环境
        self.render_time = []
        self.render_positions = []
        self.render_mode = render_mode
        ############

        ############
        # sb3需要的参数
        self.reward = None
        self.observation = None  # 观测的state
        # self.done = False
        # self.over = False
        self.info = None
        # 定义状态空间和动作空间
        #self.observation_space = gym.spaces.box.Box(low=-np.inf, high=np.inf, shape=(7,), dtype=np.float64)
        self.observation_space = gym.spaces.box.Box(low=-np.inf, high=np.inf, shape=(5,), dtype=np.float64)
        self.action_space: Box = gym.spaces.box.Box(low=-np.pi, high=np.pi, shape=(1,), dtype=np.float32)
        ############

        self.sail_angle = None
        self.state = None  # 实际使用的state
        self.n_states = 12
        if actor_dynamics:
            self.n_states = 14
        self.i = 0

        ############
        # 环境信息
        # TrueWind = namedtuple('TrueWind', 'x, y, strength, direction')
        # self.wind = TrueWind(0, 5, 5, pi / 2)
        # self.wind = TrueWind(1, 3, sqrt(10), atan(3))
        # 随机风
        # self.wind = random_wind()
        self.wind = None
        self.t_end = 150.  # 模拟终止时间
        self.sampletime = .3  # 采样间隔
        self.sail_sampletime = 2.  # 反应时间
        self.N_steps = int(self.t_end / self.sampletime)  # 步长
        self.render_time = []
        self.render_positions = []
        ############

        ############
        # 在RL环境中用不上
        self.ref_heading = None  # 前进方向
        ############

        ############
        # 控制器和积分器初始化
        self.integrator = None
        self.controller = None
        environment[SAIL_ANGLE] = 0. / 180 * pi
        environment[RUDDER_ANGLE] = 0
        environment[WAVE] = Wave(50, 0, 0)  # 振幅、频率、相位
        sim_wave = False
        if sim_wave:
            environment[WAVE] = Wave(length=100., amplitude=.5, direction=0)
            append = "_wave"
        else:
            append = ""
        # environment[TRUE_WIND] = self.wind
        ############

    def reset(self, seed=None, options=None):
        print("环境重置")
        ########
        # 设置风向改变并调整积分器
        self.wind = random_wind()
        environment[TRUE_WIND] = self.wind
        ########

        ########
        self.success_flag = False  # 到达终点的标志
        self.over_flag = False  # 时间用完标志
        ########

        ########
        ########
        # 加入随机设置的起点
        # 后续还需要更改
        self.end_target = random_start()
        target = self.end_target
        # print('目标位置', target)
        self.sail_angle = None
        ########

        ########
        # 状态记录
        self.i = 0
        x0 = zeros(self.n_states)
        x0[VEL_X] = 0.

        # x0[POS_X] = target[0] - 3
        # x0[POS_Y] = target[1] - 2
        # print('起点为', [x0[POS_X], x0[POS_Y]])

        if actor_dynamics:
            x0[SAIL_STATE] = 48 * pi / 180
        integrator, controller = init_integrator(x0, self.sampletime)
        self.integrator = integrator  # 运动更新器
        self.controller = controller  # PID状态控制器
        self.ref_heading = calculate_angle(x0, self.end_target)
        self.state = x0
        # observation相对位置
        x_state = copy.deepcopy(self.state)  # 创建self.state的深拷贝
        ########

        ########
        # 实际使用到的observation
        x_state[0] = self.end_target[0] - x_state[0]  # 将绝对位置坐标改为相对位置坐标
        x_state[1] = self.end_target[1] - x_state[1]
        #observation = np.array([x_state[0], x_state[1], x_state[3], x_state[4], x_state[5], x_state[6], x_state[7]],
        #                       dtype=float64)
        observation = np.array([x_state[0], x_state[1],  x_state[5], x_state[6], x_state[7]],
                               dtype=float64)
        info = {}
        self.observation = observation
        self.total_distance = abs(sqrt(x_state[0] ** 2 + x_state[1] ** 2))
        ########

        return observation, info

    def step(self, action):
        # print(self.i)
        # print(self.state[0], self.state[1])
        ########
        # 每次更新之前重置终止标志
        self.success_flag = False  # 到达终点的标志
        self.over_flag = False  # 时间用完标志
        info = {}
        ########

        #########
        # 计算当前距离
        x = copy.deepcopy(self.state)  # 创建self.state的深拷贝
        x_old = copy.deepcopy(self.state)  # 创建self.state的深拷贝
        # 上一个状态离终点的距离
        distance_old = calculate_distance(x_old, self.end_target)
        #########

        #########
        # 积分器
        integrator = self.integrator
        predefined_sail_angle = None
        #########

        #########
        # 输入上一个状态x
        # print('position', (x[0], x[1]))
        # ref_heading 代表的是期望前往的航向
        # print('heading', self.ref_heading[self.i])
        # print('expected_heading', self.ref_heading)
        #########

        # 运动的动作
        environment[RUDDER_ANGLE] = action

        # PID
        """""""""
        #########
        # PID控制器控制
        speed = sqrt(x[VEL_X] ** 2 + x[VEL_Y] ** 2)
        drift = arctan2(x[VEL_Y], x[VEL_X])
        # x[YAW]中保存当前的航向
        # environment[RUDDER_ANGLE] = self.controller.controll(self.ref_heading[self.i], x[YAW], x[YAW_RATE], speed,
        #                                                     x[ROLL], drift_angle=drift)
        environment[RUDDER_ANGLE] = self.controller.controll(self.ref_heading, x[YAW], x[YAW_RATE], speed,
                                                             x[ROLL], drift_angle=drift)
        #########
        """

        # 随机角度
        """""""""
        #########
        # RL随机角度输入
        #random_rudder_angle = random.uniform(-2*pi, 2*pi)  # 在min_angle和max_angle之间生成随机角度
        #environment[RUDDER_ANGLE] = random_rudder_angle
        #########
        """

        # 船体舵角(action1)
        rudder = environment[RUDDER_ANGLE]
        # print('rudder_angle', rudder)

        # 船体帆角(action2)
        if not predefined_sail_angle is None:
            sail = predefined_sail_angle
            self.sail_angle = sail
            environment[SAIL_ANGLE] = sail  # 目标角度
        # 当前的时间步数是 sail_sampletime（风帆控制采样时间）的整数倍时，控制器会计算新的风帆角度
        else:
            if not self.i % int(self.sail_sampletime / self.sampletime):
                # 添加当前风的影响
                apparent_wind = calculate_apparent_wind(x[YAW], x[VEL_X], x[VEL_Y], self.wind)
                environment[SAIL_ANGLE] = sail_angle(apparent_wind.angle, apparent_wind.speed, SAIL_STRETCHING)

                sail = environment[SAIL_ANGLE]
                self.sail_angle = sail

            else:
                sail = self.sail_angle
        # print('sail_angle:', sail)

        #########
        # 通过积分器更新下一个状态
        # state状态更新
        integrator.integrate(integrator.t + self.sampletime)
        # 状态更新，同时时间也被更新
        x_new = self.integrator.y
        self.state = x_new
        # 保存更新的integrator
        self.integrator = integrator
        x_state = copy.deepcopy(self.state)  # 创建self.state的深拷贝
        #########

        t_step = self.i
        limit = self.t_end / self.sampletime - 2  # 498

        # 环境截断
        # 改变截断位置
        if t_step > limit:
            self.over_flag = True
        # 更新期望角度
        self.ref_heading = calculate_angle(x_new, self.end_target)

        #########
        # 计算更新后的距离
        distance_new = calculate_distance(x_new, self.end_target)
        # 20231109,尝试调大距离范围限制
        if distance_new < 0.5:
            # if distance_new < 0.2:
            self.success_flag = True
        #########

        #########
        # 计算两次运动的距离
        # distance = calculate_distance(x_old, x_new)
        # print('两次运动的距离', distance)
        #########

        #########
        # 更新输出的observation
        x_state[0] = self.end_target[0] - x_state[0]
        x_state[1] = self.end_target[1] - x_state[1]
        # observation仍然是相对位置
        """""""""
        observation = np.array([x_state[0], x_state[1], x_state[3], x_state[4], x_state[5], x_state[6], x_state[7]],
                               dtype=float64)
        """
        observation = np.array([x_state[0], x_state[1], x_state[5], x_state[6], x_state[7]],
                               dtype=float64)
        #########

        #########
        # 计算奖励函数
        time_punish = -self.sampletime  # 时间惩罚项
        distance_reward = 100 * (distance_old - distance_new) / self.wind[2]
        # 考虑用距离除以船速会不会更好
        # proportion = distance_new / self.total_distance
        dynamics_reward = 0
        reward = time_punish + distance_reward + dynamics_reward

        if self.over_flag:
            reward = self.fail_punishment
            print("time runs out", reward)
        elif self.success_flag:
            reward = reward + self.success_reward
            print("success", reward)
        else:
            reward = reward
        #########

        #########
        # save_info
        info['step'] = self.i
        info['action'] = action
        info['success'] = self.success_flag
        info['rudder_angle'] = rudder
        info['sail-angle'] = sail
        info['old_position'] = [x_old[0], x_old[1]]
        info['new_position'] = [self.state[0], self.state[1]]
        info['reward'] = reward
        #########

        #########
        # state_update
        self.i = self.i + 1
        self.observation = observation
        self.reward = reward
        terminated = self.success_flag or self.over_flag
        truncated = self.over_flag
        self.info = info

        # 每次运动后重置风
        ########
        # 设置风向改变并调整积分器
        self.wind = random_wind()
        environment[TRUE_WIND] = self.wind
        ########

        return observation, reward, terminated, truncated, info

    def render(self):
        self.render_time.append(self.i)
        self.render_positions.append((self.state[0], self.state[1]))

    def close(self):
        print("Environment closed.")

    def return_time(self):
        return self.i


# 设置CUDA设备
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # 指定使用的GPU设备号
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# 从默认CUDA设备上删除所有张量
torch.cuda.empty_cache()

# 创建环境实例
env = Boat_Environment()

# 环境检测
# check_env(env)

# 开始训练
train_env = DummyVecEnv([lambda: env])
test_env = DummyVecEnv([lambda: env])
# 创建 PPO 模型
# gae越小，对未来回报的折扣越小
# ent_coef 鼓励探索
# lr学习率
# gamma 注重长期奖励
# n_steps和total_timesteps的关系:模型每n_steps进行一次更新(采样步数),一轮训练共走total_timesteps步，如果中间遇到done就重置环境
# model = PPO("MlpPolicy", train_env, verbose=0, device="cuda", n_steps=2048, learning_rate=0.003, gae_lambda=0.95,
#            ent_coef=0.0, gamma=0.99, clip_range=0.2)

# 创建DDPG的动作噪声
# 创建动作噪音
#action_noise = OrnsteinUhlenbeckActionNoise(mean=np.zeros(1), sigma=float(0.5) * np.ones(1))
#model = DDPG("MlpPolicy", train_env, device="cuda", action_noise=action_noise)

# PPO训练
# 尝试把step数减少
model = PPO("MlpPolicy", train_env, verbose=0, device="cuda", n_steps=64, learning_rate=0.001, gae_lambda=0.95,
            ent_coef=0.0, gamma=0.99, batch_size=64)
# 定义训练参数
total_episode = 5000
# 每次训练步长
timesteps_per_episode = 2048
# 每5个episode保存一次数据
save_interval = 5

# 初始化记录变量
total_rewards = []
test_steps_old = env.t_end / env.sampletime
test_steps_new = env.t_end / env.sampletime
test_min_steps = env.t_end / env.sampletime

# 开始训练
for episode in range(total_episode):
    # 进行一定数量的时间步训练
    print("\n")
    print("episode:", episode)
    print("本次学习率为", model.learning_rate)
    # model.learn(total_timesteps=timesteps_per_episode)
    model.learn(total_timesteps=timesteps_per_episode, callback=CustomCallback())
    print("\n")
    print("############       ############")
    print("start to evaluate")
    # 评估模型并记录总奖励
    episode_rewards = []
    action_array = []
    state_array = []
    obs = test_env.reset()
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
    # test_env.close()
    test_steps_new = len(episode_rewards)
    print("本次步数为:", test_steps_new)

    # 保存当前最佳模型
    if test_steps_new <= test_min_steps:
        test_min_steps = test_steps_new
        model.save(dir_best_model + "model_current_best.zip")

    # 调整学习率
    if test_steps_old > test_steps_new and model.learning_rate > 0.0003:
        print("学习率降低")
        model.learning_rate = model.learning_rate - 0.0001
    elif test_steps_old <= test_steps_new and model.learning_rate < 0.001:
        print("学习率增加")
        model.learning_rate = model.learning_rate + 0.0001
    total_reward = np.sum(episode_rewards)
    test_steps_old = test_steps_new
    print("episode_reward:", total_reward)
    print("############       ############")
    print("\n")
    total_rewards.append(total_reward)

    # 每隔一定周期保存模型
    if episode % save_interval == 0:
        model.save(dir_model + f"model_{episode}.zip")
        plt.plot(total_rewards)
        plt.xlabel("Episode")
        plt.ylabel("Total Reward")
        plt.title("Total Reward over Episodes")
        image_name = f'rewards_plot2_{episode + 1}.png'
        # 保存图像文件
        plt.savefig(dir_pic + image_name)
    # 每个训练周期结束后释放内存
    torch.cuda.empty_cache()
