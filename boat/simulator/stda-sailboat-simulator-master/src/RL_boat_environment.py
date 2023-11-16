import gymnasium as gym
from gym.spaces import Box
# matlab数据加载
from scipy.io import loadmat
import numpy as np
from time import process_time
import random
from heading_controller import heading_controller
from simulation import *
from numpy import *
import matplotlib.pyplot as plt
from sail_angle import sail_angle
import copy

deq = solve

''' State中包含的信息包括:
    POS_X(0), POS_Y(1), POS_Z(2);
    ROLL(3), PITCH(4), YAW(5);
    VEL_X(6), VEL_Y(7), VEL_Z(8);
    ROLL_RATE(9), PITCH_RATE(10), YAW_RATE(11);
    RUDDER_STATE(12), SAIL_STATE(13);
    YAW是帆船的当前舵角
'''

"""""""""""""""
# 常微分方程类声明
# 暂时用不上
class simple_integrator(object):

    def __init__(self, deq):
        self.t = 0
        self.deq = deq
        self.max_sample_time = .01
        pass

    def set_initial_value(self, x0, t0):
        self.y = x0
        self.t = t0

    def integrate(self, end_time):
        while end_time - self.t > 1.1 * self.max_sample_time:
            self.integrate(self.t + self.max_sample_time)

        self.y += self.deq(self.t, self.y) * (end_time - self.t)
        self.t = end_time
"""""


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
    # 是否稀疏化处理
    fun = deq if not sparse else deq_sparse
    # 选择模型方法
    integrator = ode(fun).set_integrator('dopri5')
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


def plot_series(points, end_point=None, fig=None, ax=None, N_subplot=1, n_subplot=1, title=None, xlabel=None,
                ylabel=None, label=None,
                legend=False):
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
    if not xlabel is None:
        ax.set_xlabel(xlabel)
    if not ylabel is None:
        ax.set_ylabel(ylabel)
    ax.grid(True)

    if legend:
        ax.legend()
    return fig, ax


# 计算欧式距离
def calculate_distance(old, new):
    distance = abs(sqrt((new[0] - old[0]) ** 2 + (new[1] - old[1]) ** 2))
    return distance


# 随机生成起点
def random_start():
    x = random.uniform(-5, 5)
    y = random.uniform(-5, 5)
    startpoint = [x, y]
    return startpoint


# 计算两点坐标差
def calculate_angle(current, target):
    angle_radians = math.atan2(target[1] - current[1], target[0] - current[0])
    angle_degrees = math.degrees(angle_radians)
    angle_degrees = pi * angle_degrees / 180
    return angle_degrees


class Boat_Environment:
    def __init__(self):

        ############
        # 初始化信息
        self.success_flag = False  # 到达终点的标志
        self.over_flag = False  # 时间用完标志
        self.success_reward = 2000
        self.fail_punishment = -2000
        self.end_target = None  # 实验，目标坐标点
        self.environment = None  # 实验，周围环境
        self.render_time = []
        self.render_positions = []
        ############

        ############
        # sb3需要的参数
        self.reward = None
        self.observation = None  # 观测的state
        self.done = False
        self.over = False
        self.info = None
        # 定义状态空间和动作空间
        self.observation_space = gym.spaces.box.Box(low=-np.inf, high=np.inf, shape=(14,), dtype=np.float32)
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
        self.wind = TrueWind(-3, -4, 5, 0)
        # self.wind = TrueWind(2, 2, 0, pi/4)
        #self.wind = TrueWind(5, 0, 5, 0)
        self.t_end = 150.  # 模拟终止时间
        self.sampletime = .3  # 采样间隔
        self.sail_sampletime = 2.  # 反应时间
        self.N_steps = int(self.t_end / self.sampletime)  # 步长
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
        environment[WAVE] = Wave(50, 0, 0)
        sim_wave = False
        if sim_wave:
            environment[WAVE] = Wave(length=100., amplitude=.5, direction=0)
            append = "_wave"
        else:
            append = ""
        environment[TRUE_WIND] = self.wind
        ############

    def reset(self):
        ########
        # 加入随机设置的起点
        target = random_start()
        #target = [2, 2]
        # print('目标位置', target)
        self.end_target = target
        self.sail_angle = None
        ########

        ########
        # 状态记录
        self.i = 0
        x0 = zeros(self.n_states)
        x0[VEL_X] = 0.
        if actor_dynamics:
            x0[SAIL_STATE] = 48 * pi / 180
        integrator, controller = init_integrator(x0, self.sampletime)
        self.integrator = integrator  # 运动更新器
        self.controller = controller  # PID状态控制器
        self.ref_heading = calculate_angle(x0, self.end_target)
        self.state = x0
        # print('self.state1', self.state)
        x_state = copy.deepcopy(self.state)  # 创建self.state的深拷贝
        # print('self.state2', self.state)
        ########

        ########
        # 实际使用到的state
        x_state[0] = self.end_target[0] - x_state[0]  # 将绝对位置坐标改为相对位置坐标
        x_state[1] = self.end_target[1] - x_state[1]
        ########
        observation = x_state
        info = None

        return observation, info


    def step(self, action):
        info = {}
        #########
        # 计算当前距离
        print('self.state', self.state)
        x = copy.deepcopy(self.state)  # 创建self.state的深拷贝
        x_old = copy.deepcopy(self.state)  # 创建self.state的深拷贝
        distance_old = calculate_distance(x_old, self.end_target)
        #print('当前位置距离', distance_old)
        #########

        #########
        # 积分器
        # self.ref_heading = 0
        integrator = self.integrator
        predefined_sail_angle = None
        #########

        #########
        # 输入上一个状态x
        #print('position', (x[0], x[1]))
        # ref_heading 代表的是期望前往的航向
        # print('heading', self.ref_heading[self.i])
        #print('expected_heading', self.ref_heading)
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
        #print('rudder_angle', rudder)

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
        #print('sail_angle:', sail)

        #########
        # 通过积分器更新下一个状态
        # state状态更新
        integrator.integrate(integrator.t + self.sampletime)
        # print('integrator.y after integration:', integrator.y)
        # t = self.integrator.t
        x_new = self.integrator.y
        self.state = x_new
        # 保存更新的integrator
        self.integrator = integrator
        x_state = copy.deepcopy(self.state)  # 创建self.state的深拷贝
        #########

        t_step = self.i
        # 环境截断
        if t_step > 498:
            self.over_flag = True
        # 更新期望角度
        self.ref_heading = calculate_angle(x_new, self.end_target)

        #########
        # 计算更新后的距离
        distance_new = calculate_distance(x_new, self.end_target)
        if distance_new < 0.5:
            self.success_flag = True
        #########

        #########
        # 计算两次运动的距离
        # distance = calculate_distance(x_old, x_new)
        # print('两次运动的距离', distance)
        #########

        # 更新输出的state
        x_state[0] = self.end_target[0] - x_state[0]
        x_state[1] = self.end_target[1] - x_state[1]
        observation = x_new

        #########
        # 计算奖励函数
        time_punish = -self.sampletime  # 时间惩罚项
        distance_reward = 10 * (distance_old - distance_new) / self.sampletime  # 距离奖励项
        dynamics_reward = 0  # 其他力学奖励项
        reward = time_punish + distance_reward + dynamics_reward
        if self.over_flag:
            reward = reward + self.fail_punishment
        elif self.success_flag:
            reward = reward + self.success_reward
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
        info['state'] = observation
        #########

        #########
        # state_update
        self.i = self.i + 1
        self.observation = observation
        self.reward = reward
        terminated = self.success_flag or self.over_flag
        truncated = self.over_flag
        self.over = self.success_flag or self.over_flag
        self.info = info
        #########

        return observation, reward, terminated, truncated, info

    def render(self):
        self.render_time.append(self.i)
        self.render_positions.append((self.end_target[0] - self.observation[0],
                                      self.end_target[1] - self.observation[1]))

    def close(self):
        print("Environment closed.")

    def return_time(self):
        return self.i


def main():

    state_array = []
    environment = Boat_Environment()
    state, _ = environment.reset()
    temp_position = (environment.state[0], environment.state[1])
    state_array.append(temp_position)
    for i in range(environment.N_steps):
        print(i)
        #action = random.uniform(-pi * 45 / 180, pi * 45 / 180)
        #action = pi/2
        action = 0
        _, _, _, _, test_info = environment.step(action)
        print(test_info)
        temp_position = (environment.state[0], environment.state[1])
        state_array.append(temp_position)
    _, _ = plot_series(state_array, end_point=environment.end_target, title='trajectory Plot', xlabel='pos_x',
                          ylabel='pos_y', label='trajectory',
                          legend=True)
    plt.show()



if __name__ == "__main__":
    main()
