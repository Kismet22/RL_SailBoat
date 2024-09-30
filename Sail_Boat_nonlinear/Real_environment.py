import gymnasium as gym
import numpy as np
from gymnasium.utils import seeding
from gymnasium import spaces
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import math
from termcolor import colored


class DoubleGyreEnvironment(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array", None], "render_fps": 4}

    environment_name = "DoubleGyre"

    def __init__(self, _init_t=None, _init_zero=False, render_mode=None, is_fixed_start_and_target=True, seed=None
                 , swap=False, swim_vel=0.9, _init_pair=None):
        """
        size: size (grid_height and grid_width) of canvas
        dt: time step
        T: total time
        t_step: path periodicity
        """
        ######################### 流场相关参数 #########################
        self.swim_vel = swim_vel
        self.swim_background = 0.9
        self.A = 2 / 3 * self.swim_background
        self.omega = 20 * np.pi * self.swim_background / 3
        self.eps = 0.3
        self.x, self.y = 2, 1
        self.width = 201
        self.height = 101
        self.L = 1
        if _init_t:
            print("Have _init_t")
            self.t_init = _init_t
        else:
            self.t_init = None
        self._is_init_zero = _init_zero
        # 该组参数下，周期约为0.33s，reset中随机初始时间被hardcode为0.33s内随机。
        ######################### 仿真相关参数 #########################
        self.dt = 0.01
        self.t_step_max = 400
        self.frame_pause = 0.01
        ######################### reward shaping #########################
        # self.collide_cost = 100
        # self.success_reward = 200
        # self.lamb = lamb  # weight of movement cost
        # self.rho = rho  # return penalty weight
        ######################### else #########################
        self.is_fixed_start_and_target = is_fixed_start_and_target
        self.default_start = np.array([1.5, 0.5])
        self.default_target = np.array([0.5, 0.5])
        self.start = self.default_start
        self.target = self.default_target
        self._init_pair = _init_pair
        # 用于停止积分器的事件函数。当改函数为0时，积分器停止（即碰撞到某个边界时）
        self.land_event = lambda t, y: y[0] * (y[0] - self.x) * y[1] * (y[1] - self.y)
        self.land_event.terminal = True
        self.target_r = 1 / 50  # 到达终点判定的范围

        # 动作空间设置，1个-pi~pi浮点数，代表冲的角度
        self.action_space = gym.spaces.Box(low=-np.pi, high=np.pi, shape=(1,), dtype=np.float32)

        # 状态空间x,y,u,v
        self.observation_space = spaces.Box(
            low=np.array([0, 0, -5, -5], dtype=np.float32),
            high=np.array([2, 1, 5, 5], dtype=np.float32)
        )
        self.seed(seed)
        self.render_mode = render_mode
        self.agent_pos_history = []
        self.action = None
        self.agent_pos = np.array([0, 0])
        self.prev_dist_to_start = 0
        self.t0 = 0
        self.t_step = 0
        self.d_2_target = 0
        # 一些环境最大最小值，是当前环境下试出来的
        self.o_max = -18.0  # 用来做用o渲染的color bar
        self.o_min = 18.0
        self.swap_start_and_target = swap
        self.ts_per_step = 1
        self.x_min, self.x_max = 0, 2
        self.y_min, self.y_max = 0, 1
        # self.u_min = -1.88  # 用来做归一化
        # self.u_max = 1.88
        # self.v_min = -2.22
        # self.v_max = 3.02

    """""""""
    def step(self, input_action):
        # 裁剪action到取值范围
        pos_cur = self.agent_pos
        action_xy = np.array([self.swim_vel * math.cos(input_action), self.swim_vel * math.sin(input_action)])
        self.action = input_action
        _terminated = False  # success
        _truncated = False  # 截断，超时或出界
        _reward = 0
        _info = self._get_info()
        _observation = self._get_obs(input_info=_info)

        # 仿真dt，返回新的位置 & 是否出界
        pos_new, _truncated = self._solve_ivp(action_xy)

        # 保险，确保ivp没写错，返回的东西在界内
        if pos_new[0] < 0 or pos_new[0] > self.x or pos_new[1] < 0 or pos_new[1] > self.y:
            raise ValueError(pos_new, "Out of bounds!")
        self.agent_pos = pos_new
        # NOTE: 轨迹记录已经在solve_ivp中完成
        self.t_step += 1
        d = np.linalg.norm(pos_new - self.target)

        # 检查出界
        if _truncated:
            _reward -= 0
            print(colored("**** Episode Finished **** Hit Boundary.", 'red'))
        # 没出界
        else:
            # 检查超时
            if self.t_step >= self.t_step_max:
                _truncated = True
                _reward = 0
                print(colored("**** Episode Finished **** Reaches Env Time limit.", 'blue'))
            # 正常情况
            # 检查终点，避免仿真步中冲过终点的小区域，再检查个中点
            if d <= self.target_r * 2.5:
                mid = 0.5 * (pos_new + pos_cur)
                d_mid = np.linalg.norm(mid - self.target)
                if d_mid <= self.target_r or d <= self.target_r:
                    _terminated = True
                    _reward = 200
                    print(colored("**** Episode Finished **** SUCCESS.", 'green'))
            # 没到终点
            else:  # TODO: Reward Shaping
                _reward = -self.dt - 20 * (d - self.d_2_target) / self.swim_vel

        self.d_2_target = d

        if self.render_mode == 'human':
            self._render_frame()

        return _observation, _reward, _terminated, _truncated, _info
    """

    def step(self, input_action):
        input_action = input_action
        pos_cur = self.agent_pos
        action_xy = np.array([self.swim_vel * math.cos(input_action), self.swim_vel * math.sin(input_action)])
        self.action = input_action
        _terminated = False
        _truncated = False
        _reward = 0
        _info = self._get_info()
        _observation = self._get_obs(input_info=_info)

        # 更新位置
        u = _info["u"]
        v = _info["v"]
        new_pos = self.agent_pos + (np.array([u, v]) + action_xy) * self.dt

        self.agent_pos = new_pos
        self.agent_pos_history.append(new_pos.copy())
        self.t_step += self.ts_per_step

        d = np.linalg.norm(new_pos - self.target)

        # 检查出界
        if not (self.x_min <= new_pos[0] <= self.x_max and self.y_min <= new_pos[1] <= self.y_max):
            _truncated = True
            _reward = -0
            print(colored("**** Episode Finished **** Hit Boundary.", 'red'))
        else:
            # 检查超时
            if self.t_step >= self.t_step_max:
                _truncated = True
                _reward = -0
                print(colored("**** Episode Finished **** Reaches Env Time limit.", 'blue'))
            # 正常情况
            else:
                # 检查终点
                if d <= self.target_r * 2.5:
                    mid = 0.5 * (new_pos + pos_cur)
                    d_mid = np.linalg.norm(mid - self.target)
                    if d_mid <= self.target_r or d <= self.target_r:
                        _terminated = True
                        _reward = 200
                        print(colored("**** Episode Finished **** SUCCESS.", 'green'))
                # 运行途中
                # TODO: DoubleGyre Reward Shaping
                else:
                    _reward = -self.dt - 10 * (d - self.d_2_target) / self.swim_vel

        self.d_2_target = d

        if self.render_mode == 'human':
            self._render_frame()

        return _observation, _reward, _terminated, _truncated, _info

    def agent_dynamics_withtime(self, state, action):
        input_action = action
        pos_cur = state
        action_xy = np.array([self.swim_vel * math.cos(input_action), self.swim_vel * math.sin(input_action), 1])
        u = self._get_u(state[0], state[1], state[2])
        v = self._get_v(state[0], state[1], state[2])
        # 更新位置
        new_pos = pos_cur + (np.array([u, v, 0]) + action_xy) * self.dt
        return new_pos

    @staticmethod
    def _normalize(_in, _min, _max, scale=1.0):
        """
        # 进行一个大致的归一化，试图提升网络性能
        """
        return scale * 2 * (_in - _min) / (_max - _min) - 1

    @staticmethod
    def _rand_in_circle(origin, r):
        d = math.sqrt(np.random.uniform(0, r ** 2))
        theta = np.random.uniform(0, 2 * math.pi)
        return origin + np.array([d * math.cos(theta), d * math.sin(theta)])

    def reset(self, seed=None, options=None):
        """
        重设环境，将agent扔回起点。如果需要随机的话随机起终点。
        :param seed: 供np.random使用的，忽略
        :param options: 好像没用到
        :return: observation, info
        """
        # We need the following line to seed self.np_random
        super().reset(seed=seed)
        # 重置时间
        if self.t_init:
            self.t0 = self.t_init
        else:
            if self._is_init_zero:
                self.t0 = 0
            else:
                self.t0 = np.random.uniform(0, 0.33)
        print("初始化时间", self.t0)
        self.t_step = int(self.t0 / self.dt)
        # 重置位置
        if not self.is_fixed_start_and_target:
            self.start = self._rand_in_circle(self.default_start, 1 / 4)
            self.target = self._rand_in_circle(self.default_target, 1 / 4)
            if self.swap_start_and_target:
                if np.random.uniform() > 0.5:
                    self.start = self._rand_in_circle(self.default_target, 1 / 4)
                    self.target = self._rand_in_circle(self.default_start, 1 / 4)
        else:
            if self._init_pair:
                self.start = np.array([self._init_pair[0], self._init_pair[1]])
                self.target = np.array([self._init_pair[2], self._init_pair[3]])
        self.agent_pos = self.start
        self.d_2_target = np.linalg.norm(self.agent_pos - self.target)
        # 重置轨迹
        self.agent_pos_history = [self.agent_pos]

        observation = self._get_obs()
        info = self._get_info()
        if self.render_mode == 'human':
            self._render_frame()
        return observation, info

    def _get_info(self):
        """
        更丰富的信息
        :return:
        """
        delta = self.target - self.agent_pos
        # d = np.linalg.norm(delta)
        # theta = np.arctan2(delta[1], delta[0])
        o = 0

        u = self._get_u(self.agent_pos[0], self.agent_pos[1], self.t_step * self.dt)
        v = self._get_v(self.agent_pos[0], self.agent_pos[1], self.t_step * self.dt)
        o = self._get_o(self.agent_pos[0], self.agent_pos[1], self.t_step * self.dt)
        Vel = math.sqrt(u ** 2 + v ** 2)
        if self.t_step != 0:
            u_last = self._get_u(self.agent_pos[0], self.agent_pos[1], self.t_step - 1)
            v_last = self._get_v(self.agent_pos[0], self.agent_pos[1], self.t_step - 1)
            _du = u - u_last
            _dv = v - v_last
            du = _du / self.dt if _du != 0 else 0
            dv = _dv / self.dt if _du != 0 else 0
        else:
            du = dv = 0

        return {"dx": delta[0], "dy": delta[1], "u": u, "v": v, "du": du, "dv": dv, "o": o, "Vel": Vel}

    def _get_obs(self, input_info=None):
        """
        获得当前时刻的观测，为了速度考虑，如果读过info的话直接从info提取就好了
        :return: 字典，包括{"dx", "dy", "u", "v"}
        """
        if input_info is None:
            _info = self._get_info()
        else:
            _info = input_info
        # TODO: 调整权重
        dx_norm = self._normalize(_info["dx"], self.observation_space.low[0], self.observation_space.high[0], scale=4.0)
        dy_norm = self._normalize(_info["dy"], self.observation_space.low[1], self.observation_space.high[1], scale=4.0)
        u_norm = self._normalize(_info["u"], self.observation_space.low[2], self.observation_space.high[2])
        v_norm = self._normalize(_info["v"], self.observation_space.low[3], self.observation_space.high[3])
        return {"dx": dx_norm, "dy": dy_norm, "u": u_norm, "v": v_norm}

    def render(self):
        self._render_frame()

    def close(self):
        plt.close('all')

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def f(self, x, t):
        return self.eps * np.sin(self.omega * t) * x ** 2 + x - 2 * self.eps * np.sin(self.omega * t) * x

    def df(self, x, t):
        return 2 * self.eps * np.sin(self.omega * t) * x + 1 - 2 * self.eps * np.sin(self.omega * t)

    def ddf(self, x, t):
        return 2 * self.eps * np.sin(self.omega * t)

    def _get_u(self, x, y, t):
        return -np.pi * self.A * np.sin(np.pi * self.f(x, t)) * np.cos(np.pi * y)

    def _get_v(self, x, y, t):
        return np.pi * self.A * np.cos(np.pi * self.f(x, t)) * np.sin(np.pi * y) * self.df(x, t)

    def _get_uv(self, t, loc):
        """
        返回指定时间位置的流场速度[u, v]
        """
        x, y = loc
        u = self._get_u(x, y, t)
        v = self._get_v(x, y, t)
        return np.array([u, v])

    def _get_o(self, x, y, t):
        o = -np.pi ** 2 * self.A * np.sin(np.pi * self.f(x, t)) * np.sin(np.pi * y) * (1 + self.df(x, t) ** 2) + \
            np.pi * self.A * np.cos(np.pi * self.f(x, t)) * np.sin(np.pi * y) * self.ddf(x, t)
        # self.o_min = o.min() if o.min() < self.o_min else self.o_min
        # self.o_max = o.max() if o.max() > self.o_max else self.o_max
        # print(f'o: [{self.o_min}, {self.o_max}]')
        return o

    def get_o_grid(self, t):
        """
        以网格返回涡度，作图用
        """
        X, Y = np.meshgrid(np.linspace(0, self.x, self.width), np.linspace(0, self.y, self.height))
        return self._get_o(X, Y, t)

    def get_Vel_grid(self, t):
        """
        以网格返回流场速度大小，作图用
        """
        X, Y = np.meshgrid(np.linspace(0, self.x, self.width), np.linspace(0, self.y, self.height))
        U = self._get_u(X, Y, t)
        V = self._get_v(X, Y, t)
        # self.u_min = U.min() if U.min() < self.u_min else self.u_min
        # self.u_max = U.max() if U.max() > self.u_max else self.u_max
        # self.v_min = V.min() if V.min() < self.v_min else self.v_min
        # self.v_max = V.max() if V.max() > self.v_max else self.v_max
        # print(f'u: [{self.u_min}, {self.u_max}]; v: [{self.v_min}, {self.v_max}]')
        Vel = np.sqrt(U ** 2 + V ** 2)
        return Vel

    def _solve_ivp(self, action):
        # 确定当前时间，即该步仿真开始时间，时间步（self.t_step）乘仿真步长
        time = self.t0 + self.t_step * self.dt
        # 定义微分方程的右半边，给定时间返回流场速度+agent速度控制量，得到一个[vx, vy]
        # dy/dt = f(t, y)，对t积分
        rhs = lambda t, y: self._get_uv(t, y) + action
        # 定义微分方程求解器，要积分的东西是rhs，积分范围是(time, time + self.dt)
        # event是一个用于判断是否出界的函数，出界为0不出为1。solve_ivp按照时间步长进行积分，如果算着算着发现event从1变0，就立刻停止。
        # 积分器，十步积分
        sol = solve_ivp(rhs, (time, time + self.dt), self.agent_pos, events=self.land_event, max_step=self.dt / 10)
        # sol.t为1*n大小，储存各个积分节点的时间t，开始和终点都会算进去
        # sol.y为2*n大小，储存各个积分节点的位置横纵坐标
        used_time = sol.t[-1] - time

        # 为了不让agent真的冲出去，需要在冲出去之前把积分停掉。
        # if the sensor crashes on land, stop before event so that sensor doesn't crash on land
        # 如果终点冲出去了，那么减小积分时长重新积分。
        while np.any(sol.y[:, -1] < [0, 0]) or np.any(sol.y[:, -1] > [self.x, self.y]):
            # used_time是上一轮积分的时间，到了这个时间冲出去了。所以减少一点，看看还会不会冲出去。
            used_time -= self.dt / 10
            # 如果这个时间已经很短了，那不仿真了，直接宣告结束。
            if used_time < self.dt / 10:
                sol.success = False
                used_time = 0
                # return self._get_obs(), reward, True, {}
                break
            # 如果这个时间还挺长的，那么以used_time为积分周期再来一次，然后再回开头的while判断是否冲出去。
            sol = solve_ivp(rhs, (time, time + used_time), self.agent_pos, events=self.land_event,
                            max_step=self.dt / 10)

        # 最终仿真结束，给出仿真结束的位置。如果这一步仿真成功了，那么时间肯定是dt=0.1，如果没成功，那么在dt=0.1内必定撞墙。
        # 这样的循环可以保证最后的点离撞墙肯定只有dt/10之内
        # 最后的点就是new_pos，只要不是起点，都是积出来的，都是成功的
        new_pos = sol.y[:, -1] if sol.success else self.agent_pos

        if sol.success:
            idx = np.unique(sol.t, return_index=True)[1]
            for i in idx:
                self.agent_pos_history.append([sol.y[0, i], sol.y[1, i]])

        # 返回新的位置，返回积分是否提前终止（即撞墙）。
        return new_pos, abs(used_time - self.dt) > np.finfo(float).eps

    def _render_frame(self):
        if len(self.agent_pos_history) == 0:
            return
        plt.clf()
        cur_time = self.t0 + self.t_step * self.dt
        # 画出流场涡度
        # plt.imshow(self.get_o_grid(cur_time), origin='lower', extent=(0, self.x, 0, self.y),
        #            cmap='coolwarm', aspect='equal', vmin=self.o_min, vmax=self.o_max)
        # 画出流场速度幅值
        plt.imshow(self.get_Vel_grid(cur_time), origin='lower', extent=(0, self.x, 0, self.y),
                   cmap='coolwarm', aspect='equal', vmin=0, vmax=3)
        plt.colorbar()
        X, Y = np.meshgrid(np.linspace(0, self.x, 20), np.linspace(0, self.y, 10))
        UU = self._get_u(X, Y, cur_time)
        VV = self._get_v(X, Y, cur_time)
        plt.quiver(X, Y, UU, VV)
        # 画出agent轨迹及起终点
        agent_positions = np.array(self.agent_pos_history)
        plt.plot(agent_positions[:, 0], agent_positions[:, 1], 'b-')
        plt.scatter(agent_positions[-1, 0], agent_positions[-1, 1], c='blue', label='Agent', s=5)
        plt.scatter(*self.target, c='red', label='Target', s=5)
        plt.scatter(*self.start, c='green', label='Start', s=5)
        circle_center = (float(self.target[0]), float(self.target[1]))
        # 修正圆圈的绘制代码
        circle = plt.Circle(circle_center, radius=self.target_r, color='red', fill=False,
                            linestyle='--')
        plt.gca().add_patch(circle)
        # 画出agent运动方向
        if self.action is not None:
            plt.arrow(float(self.agent_pos[0]), float(self.agent_pos[1]), 0.15 * math.cos(self.action),
                      0.15 * math.sin(self.action), shape='full', color='blue', linewidth=2, head_width=0.02,
                      length_includes_head=True)
        plt.legend()

        plt.xlim(0, self.x)
        plt.ylim(0, self.y)
        plt.title(f'Double Gyre Environment (t={self.t0 + self.t_step * self.dt:.2f})', fontsize=14)
        plt.draw()
        plt.pause(self.frame_pause)
