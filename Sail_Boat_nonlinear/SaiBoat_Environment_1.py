# gym环境加载
import gymnasium as gym
from gym.spaces import Box
from copy import deepcopy
# 外部文件导入
from sail_angle import sail_angle
from Windcondition import *
from termcolor import colored

deq = solve


# 微分方程初始化
def init_integrator(x0, t0):
    # init integrator
    fun = deq
    # 选择模型方法
    # 'nsteps'控制的是每一个dt过程中，积分器进行多少次积分
    integrator = ode(fun).set_integrator('dopri5', nsteps=1000)
    # 初始化状态输入
    integrator.set_initial_value(x0, t0)
    return integrator


class Boat_Environment(gym.Env):
    def __init__(self, render_mode=None, is_fixed_start_and_target=True, _init_pair=None, sim_wave=False,
                 is_random_wind=False, max_steps=None, _init_wind=None, _init_target_range=None, _init_center=None):

        print("加载环境")
        ############
        # 损失参数
        self.tim = 1
        self.dis = 2
        self.action_low = -15 * pi / 180
        self.action_high = 15 * pi / 180
        ############

        ############
        self.agent_pos_history = []
        self.render_mode = render_mode
        # 随机风标志
        self.wind_condition = is_random_wind
        self._init_wind = _init_wind
        ############

        ############
        # info
        self.reward = 0
        self.action = 0
        self.sail_angle = 0
        # 状态空间
        self.observation_space = gym.spaces.box.Box(low=-np.inf, high=np.inf, shape=(5,), dtype=np.float32)
        # 动作空间
        self.action_space: Box = gym.spaces.box.Box(low=- 15 * np.pi / 180, high=15 * np.pi / 180, shape=(1,),
                                                    dtype=np.float32)
        self.state = None  # 实际使用的state
        self.n_states = 6
        ############

        ############
        # 环境信息
        # 参考:(20, 10), t = 45
        self.wind = None
        if max_steps:
            self.N_steps = max_steps
        else:
            self.N_steps = 250  # 步长
        self.dt = .3  # 仿真间隔
        self.ts_per_step = 1  # 移动步长
        self.t_step = 0  # 当前时间步
        ############

        ############
        # 初始化信息
        # RL所需要的参数
        reward_steps = 250
        self.success_reward = 6 * reward_steps * self.dt  # bonus
        self.fail_punishment = -2 * reward_steps * self.dt  # fail_punish
        ############

        # 起点和终点信息初始化
        self.is_fixed_start_and_target = is_fixed_start_and_target
        if _init_center is not None and _init_center.any():
            self.default_target = _init_center
        else:
            self.default_target = np.array([40, 20])
        self.default_start = np.array([0, 0])
        self.start = self.default_start
        self.agent_pos = self.default_start
        self.target = self.default_target
        self._init_pair = _init_pair
        self.d_start_2_target = 0  # 初始距离
        self.d_2_target = 0  # 与目标终点的距离
        self.d_old_2_target = 0  # 上一个位置与目标终点的距离
        self.target_r = 4  # 到达终点判定的范围
        if _init_target_range:
            self.target_range = _init_target_range
        else:
            self.target_range = 6  # 目标区域的半径范围
        ############

        ############
        # 积分器初始化
        self.integrator = None
        ############

        environment[SAIL_ANGLE] = 0. / 180 * pi
        environment[RUDDER_ANGLE] = 0
        # 默认静水
        environment[WAVE] = Wave(length=50., amplitude=0, direction=0)

        # 模拟波浪
        if sim_wave:
            print("sim_wave")
            environment[WAVE] = Wave(length=100., amplitude=.5, direction=0)
        ############

    def reset(self, seed=None, options=None):
        """
        重设环境，将agent扔回起点，初始化随机风环境，重设随机终点
        :param seed: 供np.random使用的，忽略
        :param options: 好像没用到
        :return: observation, info
        """
        print("环境初始化")
        self.t_step = 0  # 当前时间步

        if self._init_wind:
            self.wind = self._init_wind[self.t_step]
        else:
            self.wind = fixed_wind()

        environment[TRUE_WIND] = self.wind
        ########
        # 起点和终点位置不确定
        if not self.is_fixed_start_and_target:
            self.start = self._rand_in_circle(self.default_start, 0)
            self.target = self._rand_in_circle(self.default_target, self.target_range)
        else:
            # 有初始输入的起点和终点对
            if self._init_pair.any():
                self.start = np.array([self._init_pair[0], self._init_pair[1]])
                self.target = np.array([self._init_pair[2], self._init_pair[3]])
        self.agent_pos = self.start
        self.d_start_2_target = np.linalg.norm(self.agent_pos - self.target)
        self.d_2_target = self.d_start_2_target
        self.d_old_2_target = self.d_2_target
        # 重置轨迹
        self.agent_pos_history = [self.agent_pos]

        # 打印信息
        """""""""
        print("初始化起点", self.agent_pos)
        print("初始化终点", self.target)
        print("初始距离", self.d_start_2_target)
        """
        ########

        ########
        """
        State中包含的信息包括:
        POS_X(0), POS_Y(1);
        YAW(2);
        VEL_X(3), VEL_Y(4);
        YAW_RATE(5);
        RUDDER_STATE(6), SAIL_STATE(7);
        """
        x0 = np.zeros(self.n_states)
        x0[POS_X] = self.agent_pos[0]
        x0[POS_Y] = self.agent_pos[1]
        self.state = x0
        ########

        ########
        # 积分器更新
        integrator = init_integrator(x0, self.t_step * self.dt)
        self.integrator = integrator  # 运动更新器
        ########

        ########
        # RL_state
        _observation = self._get_obs()
        _info = self._get_info()
        return _observation, _info

    def step(self, action):
        # 记录当前的动作
        self.action = action
        _terminated = False
        _truncated = False

        #########
        x_old = deepcopy(self.state)
        # 上一个状态离终点的距离
        distance_old = deepcopy(self.d_old_2_target)
        #########

        #########
        # 积分器
        integrator = self.integrator
        #########

        # 运动的动作
        environment[RUDDER_ANGLE] = action

        # 船体舵角
        # 舵角根据强化学习调整
        rudder = environment[RUDDER_ANGLE]

        # 船体帆角
        # 帆角直接根据最优风帆角计算
        apparent_wind = calculate_apparent_wind(x_old[YAW], x_old[VEL_X], x_old[VEL_Y], self.wind)
        environment[SAIL_ANGLE] = sail_angle(apparent_wind.angle, apparent_wind.speed, SAIL_STRETCHING)
        sail = environment[SAIL_ANGLE]
        self.sail_angle = sail

        #########
        # 通过积分器更新下一个状态
        # state状态更新
        integrator.integrate(integrator.t + self.dt)
        x_new = self.integrator.y
        self.agent_pos = np.array([x_new[POS_X], x_new[POS_Y]])
        self.state = x_new
        # 保存更新的integrator
        self.integrator = integrator
        # 更新时间步
        self.t_step += self.ts_per_step
        # 计算更新后的距离
        self.d_2_target = np.linalg.norm(self.agent_pos - self.target)
        distance_new = deepcopy(self.d_2_target)
        # 更新旧距离
        self.d_old_2_target = distance_new
        # episode_end
        t_step = self.t_step
        limit = self.N_steps

        # 失败条件1:too faraway from target
        if distance_new > 2 * self.d_start_2_target:
            _truncated = True
            _reward = self.fail_punishment
            print("过于远离")
            print(colored("**** Episode Finished **** Too Far.", 'red'))
        else:
            # 失败条件2：超时
            if t_step >= limit:
                _truncated = True
                _reward = self.fail_punishment
                print("超时")
                print(colored("**** Episode Finished **** Reaches Env Time limit.", 'blue'))
            else:
                # 检查终点
                if distance_new <= self.target_r:
                    _terminated = True
                    _reward = self.success_reward
                    print("成功")
                    print(colored("**** Episode Finished **** SUCCESS.", 'green'))
                # 运行途中
                else:
                    time_punish = - self.tim * self.dt  # 时间惩罚项(0.1)
                    # 距离奖励项
                    distance_reward = self.dis * (distance_old - distance_new)  # 距离奖励项(0.1)
                    _reward = time_punish + distance_reward

        #########
        # RL_observation
        _observation = self._get_obs()
        #########

        # 记录当前的奖励
        self.reward = _reward
        self.wind = (self._init_wind[self.t_step] if self._init_wind
                     else random_wind(self.wind) if self.wind_condition else fixed_wind())

        environment[TRUE_WIND] = self.wind
        _info = self._get_info()

        return _observation, _reward, _terminated, _truncated, _info

    @staticmethod
    def _rand_in_circle(center, r):
        d = sqrt(np.random.uniform(0, r ** 2))
        theta = np.random.uniform(0, 2 * pi)
        return center + np.array([d * cos(theta), d * sin(theta)])

    def _get_obs(self):
        """
        返回强化学习需要的状态信息
        """
        delta = self.target - self.agent_pos
        dx = delta[0]
        dy = delta[1]
        yaw = self.state[YAW]
        u = self.state[VEL_X]
        v = self.state[VEL_Y]
        return {"dx": dx, "dy": dy, "yaw": yaw, "u": u, "v": v}

    def _get_info(self):
        """
        更丰富的状态信息
        """
        #########################################
        # step_info
        t_step = self.t_step
        rudder_action = self.action
        sail_action = self.sail_angle
        step_reward = self.reward
        #########################################

        #########################################
        # state_info
        pos_x = self.state[POS_X]
        pos_y = self.state[POS_Y]
        yaw = self.state[YAW]
        u = self.state[VEL_X]
        v = self.state[VEL_Y]
        yaw_rate = self.state[YAW_RATE]
        #########################################

        #########################################
        # environment_info
        wind_speed = self.wind.strength
        wind_angle = self.wind.direction
        #########################################
        return {"step": t_step, "rudder_action": rudder_action, "sail_action": sail_action, "step_reward": step_reward,
                "x": pos_x, "y": pos_y, "yaw": yaw, "u": u, "v": v, "yaw_rate": yaw_rate,
                "wind_speed": wind_speed,
                "wind_angle": wind_angle}

    def render(self):
        pass

    def close(self):
        print("Environment closed.")
