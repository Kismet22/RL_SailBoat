# gym环境加载
import gymnasium as gym
from gym.spaces import Box
from copy import deepcopy
# 画图
import matplotlib.pyplot as plt
from matplotlib import patches
# 环境加载
from scipy.io import loadmat
import numpy as np
from time import process_time
import random
# 外部文件导入
from heading_controller import heading_controller
from simulation_3DOF import *
from numpy import *
from sail_angle import sail_angle

# 积分器
from scipy.integrate import solve_ivp

deq = solve

''' State中包含的信息包括:
    POS_X(0), POS_Y(1);
    YAW(2);
    VEL_X(3), VEL_Y(4);
    YAW_RATE(5);
    RUDDER_STATE(6), SAIL_STATE(7);
'''


# 画图
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
        ax.scatter(0, 0, color='blue', label='Start Point')  # 在图上标注起点坐标
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


# 微分方程初始化
def init_integrator(x0):
    # init integrator
    fun = deq
    # 选择模型方法
    # nsteps控制的是每一个dt过程中，积分器进行多少次积分
    integrator = ode(fun).set_integrator('dopri5', nsteps=1000)
    # 初始化状态输入
    integrator.set_initial_value(x0, 0)

    return integrator


def init_PID(sampletime):
    # init heading controller
    controller = heading_controller(sampletime, max_rudder_angle=15 * pi / 180)
    controller.calculate_controller_params(YAW_TIMECONSTANT)
    return controller


# 计算欧式距离
def calculate_distance(old, new):
    distance = abs(sqrt((new[0] - old[0]) ** 2 + (new[1] - old[1]) ** 2))
    return distance


# 随机生成起点
def random_start():
    # x = random.uniform(2.2, 2.4)
    # y = random.uniform(3, 3.2)
    # x = random.choice([3.8, 4])
    # y = random.choice([0.8, 1])
    x = 20
    y = 10
    startpoint = [x, y]
    # print("目标终点为:", startpoint)
    return startpoint


# 计算两点坐标差
def calculate_angle(current, target):
    angle_radians = math.atan2(target[1] - current[1], target[0] - current[0])
    angle_degrees = math.degrees(angle_radians)
    angle_degrees = pi * angle_degrees / 180
    return angle_degrees


# 生成一个速度和角度都随机的风
def random_wind():
    # print("random_wind")
    # 风速小幅摆动
    wind_speed = random.uniform(4.9, 5)
    # 风角小幅摆动
    wind_angle = random.uniform(0.95 * pi / 2, pi / 2)
    wind = TrueWind(wind_speed * cos(wind_angle), wind_speed * sin(wind_angle), wind_speed, wind_angle)
    return wind


def fixed_wind():
    # print("fixed_wind")
    wind = TrueWind(0, 5, 5, pi / 2)
    return wind


class Boat_Environment(gym.Env):
    def __init__(self, render_mode='human'):
        print("加载环境")
        ############
        # 损失参数
        self.tim = 1
        self.dis = 2
        self.act = 10
        ############

        ############
        # 初始化信息
        self.success_flag = False  # 到达终点的标志
        self.over_flag = False  # 时间用完标志
        self.success_reward = 500
        self.fail_punishment = -50
        self.total_distance = 0

        # 在初始化时设置好目标点
        ############
        self.end_target = [0, 0]
        ############

        ############
        self.environment = None  # 实验，周围环境
        self.render_time = []
        self.render_positions = []
        self.render_winds = []
        self.render_mode = render_mode
        ############

        ############
        # sb3需要的参数
        self.reward = None
        self.observation = None  # 观测的state
        self.info = None
        # 定义状态空间和动作空间
        self.observation_space = gym.spaces.box.Box(low=-np.inf, high=np.inf, shape=(5,), dtype=np.float64)
        self.action_space: Box = gym.spaces.box.Box(low=- 15 * np.pi / 180, high=- 15 * np.pi / 180, shape=(1,),
                                                    dtype=np.float32)
        ############

        self.sail_angle = None
        self.state = None  # 实际使用的state
        self.n_states = 6
        if actor_dynamics:
            print("actor_dynamics")
            self.n_states = 8
        self.i = 0 # 当前步数

        ############
        # 环境信息
        self.wind = None
        self.t_end = 150.  # 模拟终止时间
        self.sampletime = .3  # 采样间隔
        self.sail_sampletime = 2.  # 反应时间，暂时未使用
        self.N_steps = int(self.t_end / self.sampletime)  # 步长
        self.render_time = []
        self.render_positions = []
        ############

        ############
        # 积分器初始化
        # self.integrator = None
        ############


        ############
        # PID控制器初始化
        self.controller = None
        self.ref_heading = None  # 前进方向
        ############


        environment[SAIL_ANGLE] = 0. / 180 * pi
        environment[RUDDER_ANGLE] = 0
        # 默认静水
        environment[WAVE] = Wave(50, 0, 0)
        sim_wave = False
        if sim_wave:
            print("sim_wave")
            environment[WAVE] = Wave(length=100., amplitude=.5, direction=0)
            append = "_wave"
        else:
            append = ""
        ############

    def reset(self, seed=None, options=None):
        print("环境初始化")
        # self.wind = random_wind()
        self.wind = fixed_wind()
        environment[TRUE_WIND] = self.wind
        print("风速为", self.wind[2])
        ########
        self.success_flag = False  # 到达终点的标志
        self.over_flag = False  # 时间用完标志
        ########

        ########
        # 加入随机设置的起点
        self.end_target = random_start()
        target = self.end_target
        print('目标位置', target)
        self.sail_angle = None
        ########

        ########
        # 状态记录
        self.i = 0
        x0 = zeros(self.n_states)
        x0[VEL_X] = 0.

        if actor_dynamics:
            x0[SAIL_STATE] = 48 * pi / 180
        self.state = x0
        ########

        """""""""
        ########
        # 积分器更新
        integrator = init_integrator(x0)
        self.integrator = integrator  # 运动更新器
        ########
        """


        ########
        # PID控制器
        controller = init_PID(self.sampletime)
        self.controller = controller  # PID状态控制器
        self.ref_heading = calculate_angle(x0, self.end_target)
        ########


        # observation相对位置
        x_state = deepcopy(self.state)  # 创建self.state的深拷贝
        ########

        ########
        # 实际使用到的observation
        x_state[0] = self.end_target[0] - x_state[0]  # 将绝对位置坐标改为相对位置坐标
        x_state[1] = self.end_target[1] - x_state[1]

        # observation = np.array([x_state[POS_X], x_state[POS_Y], x_state[YAW], x_state[VEL_X], x_state[VEL_Y]],
        #                       dtype=float64)
        observation = np.round(
            np.array([x_state[POS_X], x_state[POS_Y], x_state[YAW], x_state[VEL_X], x_state[VEL_Y]], dtype=np.float64),
            5)
        info = {}
        self.observation = observation
        ########

        return observation, info

    def step(self, action):

        clip_action = np.clip(action, self.action_space.low, self.action_space.high)

        ########
        # 每次更新之前重置终止标志
        self.success_flag = False  # 到达终点的标志
        self.over_flag = False  # 时间用完标志
        info = {}
        ########

        #########
        # 计算当前距离
        x = deepcopy(self.state)  # 创建self.state的深拷贝
        x_old = deepcopy(self.state)  # 创建self.state的深拷贝
        # 上一个状态离终点的距离
        distance_old = calculate_distance(x_old, self.end_target)
        #########

        #########
        # 积分器
        # integrator = self.integrator
        predefined_sail_angle = None
        #########

        # 运动的动作
        environment[RUDDER_ANGLE] = clip_action

        # PID

        #########
        # PID控制器控制
        speed = sqrt(x[VEL_X] ** 2 + x[VEL_Y] ** 2)
        drift = arctan2(x[VEL_Y], x[VEL_X])
        environment[RUDDER_ANGLE] = self.controller.controll(self.ref_heading, x[YAW], x[YAW_RATE], speed,
                                                             0, drift_angle=drift)
        #########

        # 船体舵角(action1)
        # 舵角根据强化学习调整
        # environment[RUDDER_ANGLE] = pi * 15 / 180 # 固定角度
        rudder = environment[RUDDER_ANGLE]
        # print('rudder_angle', rudder)

        # 船体帆角(action2)
        # 帆角直接根据最优风帆角计算
        # environment[SAIL_ANGLE] = action[1]
        # sail = environment[SAIL_ANGLE]
        if not predefined_sail_angle is None:
            print("predefined_sail_angle")
            sail = predefined_sail_angle
            self.sail_angle = sail
            environment[SAIL_ANGLE] = sail  # 目标角度


        else:
            # 当前的时间步数是风帆控制采样时间的整数倍时，控制器会计算新的风帆角度
            # if not self.i % int(self.sail_sampletime / self.sampletime):
            # 添加当前风的影响
            # print("sail_angle_change")
            apparent_wind = calculate_apparent_wind(x[YAW], x[VEL_X], x[VEL_Y], self.wind)
            environment[SAIL_ANGLE] = sail_angle(apparent_wind.angle, apparent_wind.speed, SAIL_STRETCHING)

            sail = environment[SAIL_ANGLE]
            self.sail_angle = sail

        # else:
        # sail = self.sail_angle
        # print('sail_angle:', sail)

        #########
        # 通过积分器更新下一个状态
        # state状态更新
        """""""""""
        integrator.integrate(integrator.t + self.sampletime)
        x_new = self.integrator.y
        self.state = x_new
        # 保存更新的integrator
        self.integrator = integrator
        """
        time_span = (self.i * self.sampletime,  self.i * self.sampletime + self.sampletime) # 积分器的时间范围
        solution = solve_ivp(lambda t, y: solve(t, y), time_span, x_old,
                             max_step=self.sampletime / 100) # 积分器更新
        x_new = solution.y[:, -1]
        self.state = x_new
        x_state = deepcopy(self.state)  # 创建self.state的深拷贝
        #########

        t_step = self.i

        limit = self.t_end / self.sampletime - 2  # 498
        # 环境截断
        if t_step > limit:
            self.over_flag = True
        # 更新期望角度
        self.ref_heading = calculate_angle(x_new, self.end_target)

        #########
        # 计算更新后的距离
        distance_new = calculate_distance(x_new, self.end_target)
        # 根据要求扩大范围
        if distance_new < 4:
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
        # observation = np.array([x_state[POS_X], x_state[POS_Y], x_state[YAW], x_state[VEL_X], x_state[VEL_Y]],
        #                       dtype=float64)
        observation = np.round(
            np.array([x_state[POS_X], x_state[POS_Y], x_state[YAW], x_state[VEL_X], x_state[VEL_Y]], dtype=np.float64),
            5)
        #########

        #########
        # 计算奖励函数
        time_punish = - self.tim * self.sampletime  # 时间惩罚项(0.1)
        # print("time_punish", time_punish)

        # 重新设计距离奖励项
        distance_reward = self.dis * (distance_old - distance_new)  # 距离奖励项(0.1)
        # print("distance_reward", distance_reward)

        # 动作限制惩罚
        if not np.array_equal(action, clip_action):
            dynamics_reward = - self.act * float(action) ** 2  # 动作惩罚项(0.01)
        else:
            dynamics_reward = 0
        # print("dynamics_reward", dynamics_reward)

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
        truncated = False
        self.info = info
        #########

        # self.wind = random_wind()
        self.wind = fixed_wind()
        environment[TRUE_WIND] = self.wind

        print(observation)

        return observation, reward, terminated, truncated, info

    def render(self):
        # 风向风速记录
        self.render_winds.append(self.wind)
        # 更新步数记录
        self.render_time.append(self.i)
        # 位置记录
        self.render_positions.append((self.state[0], self.state[1]))

    def close(self):
        print("Environment closed.")

    def return_step(self):
        return self.i


def main():
    state_array = []
    environment = Boat_Environment()
    state, _ = environment.reset()
    temp_position = (environment.state[0], environment.state[1])
    state_array.append(temp_position)
    for i in range(environment.N_steps):
        print(i)
        action = 0
        _, _, success, fail, test_info = environment.step(action)
        if success:
            break
        temp_position = (environment.state[0], environment.state[1])
        state_array.append(temp_position)
    _, _ = plot_series(state_array, end_point=environment.end_target, title='trajectory Plot', xlabel='pos_x',
                       ylabel='pos_y', label='trajectory',
                       legend=True)
    plt.show()


if __name__ == "__main__":
    main()
