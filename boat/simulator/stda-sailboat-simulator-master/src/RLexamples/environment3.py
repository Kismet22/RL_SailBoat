import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat
import math
from gym.spaces import Box
import random

""""""""""""""""""""""
%   Data file: cylinder2D_Re100_ref.mat
%   including:
%       t_star: 301 time step, 1x301
%       X_star, Y_star: 2D spatial coordinates, 34944 locations
%                       and 301 time steps
%       U_star, V_star: 2D velocity field, corresponding to 34944 points
%                       and 301 time steps
%       P_star: pressure field, corresponding to 34944 points and 301 time
%               steps
%       O_star: vorticity field, corresponding to 34944 pointsand 301 time
%               steps
% target_within中共有530个点
% start_within中共有592个点
"""""""""


class FlowEnvironment(gym.Env):
    def __init__(self, dir_name, render_mode='human'):
        # 将matlab格式用python加载
        data = loadmat(dir_name)
        self.t_star = data['t_star'][0]  # 301个时间步长
        self.X_star = data['X_star']  # (34944, 301)
        self.Y_star = data['Y_star']  # (34944, 301)
        self.U_star = data['U_star']  # (34944, 301)
        self.V_star = data['V_star']  # (34944, 301)
        self.P_star = data['P_star']  # (34944, 301)
        self.delta_t = self.t_star[2] - self.t_star[1]  # 时间步长
        self.render_mode = render_mode
        self.D = 0.5 * 2  # D
        self.bonus = 200
        self.position = []  # 所有点的集合
        self.render_time = []
        self.render_positions = []
        for i in range(len(self.X_star[:, 0])):
            pos = [self.X_star[i, 0], self.Y_star[i, 0]]
            self.position.append(pos)

        self.r_limit = 2 * self.D  # 指定区域的半径
        self.start_center = [5 * self.D, -2.05 * self.D]  # 定义出发范围的圆心
        self.target_center = [5 * self.D, 2.05 * self.D]  # 定义目标范围的圆心

        self.target_within = self._get_within_range(self.target_center)
        self.start_within = self._get_within_range(self.start_center)
        # 裁剪区域，减少训练量
        self.cut_points = self.cut_range()
        # 定义状态空间和动作空间
        self.observation_space = gym.spaces.box.Box(low=-np.inf, high=np.inf, shape=(4,), dtype=np.float32)
        # self.action_space = gym.spaces.Discrete(8)
        self.action_space: Box = gym.spaces.box.Box(low=0, high=2 * np.pi, shape=(1,), dtype=np.float32)
        # 在初始化时设定好目标点
        self.target_point = self.random_target()
        # 记录当前的环境
        self.current_state = None
        self.position_old = None
        # 初始化当前时间步
        self.t = 0
        self.done = False
        self.over = False
        self.observation = None
        self.reward = None
        self.info = None

    def _get_within_range(self, center):
        within_range = []
        for pos in self.position:
            distance = np.linalg.norm(np.array(pos) - np.array(center))
            if distance <= self.r_limit:
                within_range.append(pos)
        return within_range

    def start_range(self):  # 出发点区域
        return self.start_within

    def target_range(self):  # 目标点区域
        return self.target_within

    # 随机选取一个目标位置
    def random_target(self):
        #random_position = random.choice(self.target_within[:5])  # 随机选取一个目标位置
        #random_position = random.choice(self.target_within[6:11])
        random_position = self.target_within[20]
        return random_position

    # 测试该点周围是否存在D/6范围内的点
    def test_in_range(self, target):
        count = 0
        for point in self.target_within:
            distance = np.linalg.norm(np.array(target) - np.array(point))  # 计算点与目标点之间的欧几里得距离
            if distance <= self.D / 6:
                count += 1
        return count

    # 缩小训练空间大小
    def cut_range(self):
        x_min = 5 * self.D - 5 * self.D
        x_max = 5 * self.D + 5 * self.D
        y_min = -5 * self.D
        y_max = 5 * self.D
        selected_points = [pos for pos in self.position if x_min <= pos[0] <= x_max and y_min <= pos[1] <= y_max]
        return selected_points

    # 环境随机初始化,得到初始的state
    # 初始化位置输入
    def get_og_state(self, target_position):
        # random_position = random.choice(self.start_within)  # 随机选取一个开始位置
        # 选取好初始位置
        random_position = self.start_within[20]
        #random_time = random.choice(self.t_star[:50])  # 随机选取一个开始时间,从前50个中选择，确保训练效果
        random_time = 0
        # 先进行简单的模型测试，确定从t = 0开始
        # random_time = self.t
        # 确定时间步
        pos_index = self.position.index(random_position)  # 位置在总环境中的索引
        # 取第0个位置开始
        # 从选取的时间位置开始
        time_index = np.where(self.t_star == random_time)[0][0]
        self.t = time_index
        u = self.U_star[pos_index, time_index]
        v = self.V_star[pos_index, time_index]
        re_x = target_position[0] - random_position[0]
        re_y = target_position[1] - random_position[1]
        state = [re_x, re_y, u, v]
        self.current_state = state
        self.position_old = random_position
        return state

    # 符合openai的reset()函数
    def reset(self, seed=None, options=None):
        # 调用随机初始位置并得到初始的state
        # self.t = 0
        observation = self.get_og_state(self.target_point)
        info = None
        return observation, info

    # 环境更新函数,输入包括state(一个包含[delta_x, delta_y, u_t, v_t]的矩阵),以及action(theta)
    # 输出包括新的state和reward
    def step(self, action):
        info = {}
        # print(self.delta_t)
        target_position = self.target_point
        theta = action
        # print(theta)
        # 出界标志
        end = False
        # 截断标志
        over = False
        state = self.current_state
        # 计算奖励函数
        # 游泳速度定义为0.8
        swim_speed = 0.4 * 0.3 * self.D / self.delta_t
        #re_x = state[0]
        #re_y = state[1]
        position_old = self.position_old
        # print(position_old)
        # print(math.cos(theta))
        x_swim = 0.12 * self.D * math.cos(theta)  # (0.3D/U)*0.8*U[cos,sin]
        # print(x_swim)
        y_swim = 0.12 * self.D * math.sin(theta)
        x_flow = self.delta_t * state[2]  # 对应位置对应时间的流场x方向运动
        # print(x_flow)
        y_flow = self.delta_t * state[3]  # 对应位置对应时间的流场y方向运动
        position_new = [position_old[0] + x_swim + x_flow, position_old[1] + y_swim + y_flow]
        # print(position_new)
        # 当前时间步的信息
        t_step = self.t
        # 环境截断
        if t_step > 298:
            over = True

        # 检查 position_new 是否在 cut_position 中
        if position_new in self.cut_points and np.linalg.norm(position_new) > 0.5:
            # print(position_new)
            re_x_new = target_position[0] - position_new[0]
            re_y_new = target_position[0] - position_new[1]
            index = self.position.index(position_new)
            u = self.U_star[index, t_step]
            v = self.V_star[index, t_step]
            state = [re_x_new, re_y_new, u, v]

        else:
            # 检查 position_new 是否超过边界
            # 当超出范围时，代表失败，给出一个极大的损失函数
            x_min, x_max = min([pos[0] for pos in self.cut_points]), max([pos[0] for pos in self.cut_points])
            y_min, y_max = min([pos[1] for pos in self.cut_points]), max([pos[1] for pos in self.cut_points])

            # position_new 超过边界，执行相应的操作
            if position_new[0] < x_min or position_new[0] > x_max or position_new[1] < y_min or position_new[1] > y_max \
                    or np.linalg.norm(position_new) < 0.5:
                # print('out')
                # 计算 cut_points 中所有点与 position_new 的距离
                distances = [np.linalg.norm(np.array(pos) - np.array(position_new)) for pos in self.cut_points]

                # 找到距离最近的点的索引（排除与 position_old 相同的点）
                closest_indices = [i for i, dist in enumerate(distances) if self.cut_points[i] != position_old]
                closest_index = min(closest_indices, key=lambda i: distances[i])

                # 使用最近点作为新的 position_new
                position_new = self.cut_points[closest_index]
                re_x_new = target_position[0] - position_new[0]
                re_y_new = target_position[1] - position_new[1]
                index = self.position.index(position_new)
                u = self.U_star[index, t_step]
                v = self.V_star[index, t_step]
                state = [re_x_new, re_y_new, u, v]
                end = True


            else:
                # print('in')
                # 计算 position 中所有点与 position_new 的距离
                # 计算 cut_points 中所有点与 position_new 的距离
                # 计算 cut_points 中所有点与 position_new 的距离
                distances = [np.linalg.norm(np.array(pos) - np.array(position_new)) for pos in self.cut_points]

                # 找到距离最近的点的索引（排除与 position_old 相同的点）
                closest_indices = [i for i, dist in enumerate(distances) if self.cut_points[i] != position_old]
                closest_index = min(closest_indices, key=lambda i: distances[i])

                # 使用最近点作为新的 position_new
                position_new = self.cut_points[closest_index]
                re_x_new = target_position[0] - position_new[0]
                re_y_new = target_position[1] - position_new[1]
                index = self.position.index(position_new)
                u = self.U_star[index, t_step]
                v = self.V_star[index, t_step]
                state = [re_x_new, re_y_new, u, v]
        # 已经截断
        if over:
            done = True
            rewards = -200
        else:
            # 检查是否已经到达终点
            distance = np.linalg.norm(np.array(target_position) - np.array(position_new))
            # 离开边界
            if end:
                # print("end")
                rewards = -200
                done = True

            else:
                if distance < (self.D / 6):
                    print("success")
                    rewards = -self.delta_t + 10 * (np.linalg.norm(np.array(target_position) - np.array(position_old)) -
                                                    np.linalg.norm(
                                                        np.array(target_position) - np.array(
                                                            position_new))) / swim_speed + self.bonus

                    done = True
                else:
                    rewards = -self.delta_t + 10 * (np.linalg.norm(np.array(target_position) - np.array(position_old)) -
                                                    np.linalg.norm(
                                                        np.array(target_position) - np.array(
                                                            position_new))) / swim_speed
                    done = False
        # terminated:结束标志
        # truncated:截断标志
        # 更新当前状态(包括位置和时间步)
        info['old_position'] = position_old
        info['action'] = action
        info['flow_speed'] = [u, v]
        info['step'] = self.t
        # info中保存当前的位置
        info['new_position'] = position_new
        info['done'] = done
        info['reward'] = rewards
        self.t = self.t + 1
        self.current_state = state
        self.position_old = position_new
        observation = state
        self.observation = observation
        reward = rewards
        self.reward = reward
        terminated = done
        self.done = done
        truncated = over
        self.over = end
        self.info = info
        # print(info)
        return observation, reward, terminated, truncated, info


    def render(self):
        # 考虑到实际情况，直接调用render()无法显示图像
        """""""""""
        if self.render_mode == 'human':
            # 打印self.cut_points中的所有点
            cut_points = np.array(self.cut_points)
            plt.scatter(cut_points[:, 0], cut_points[:, 1], color='blue', label='Cut Points')
            # 将self.target_point点用红颜色标出
            plt.scatter(self.target_point[0], self.target_point[1], color='red', label='Target Point')
            # 将self.current_point的点用绿色标出
            plt.scatter(self.target_point[0] - self.current_state[0], self.target_point[1] - self.current_state[1],
                        color='green', label='Current Point')
            # 添加图例和标题
            plt.legend()
            plt.title('Environment')

            plt.ion()  # 开启交互模式
            plt.show()
        """
        # 保存当前点的位置信息到列表中
        self.render_time.append(self.t)
        self.render_positions.append((self.target_point[0] - self.current_state[0],
                                      self.target_point[1] - self.current_state[1]))

    def close(self):
        print("Environment closed.")

    def return_time(self):
        return self.t

    def print_map(self):
        # 创建新的图形窗口
        plt.figure()

        # 获取所有点的坐标
        cut_points = np.array(self.cut_points)
        start_points = np.array(self.start_range())
        target_points = np.array(self.target_range())

        plt.scatter(cut_points[:, 0], cut_points[:, 1], color='blue', label='Cut Points')
        plt.scatter(start_points[:, 0], start_points[:, 1], color='green', label='Start Points')
        plt.scatter(target_points[:, 0], target_points[:, 1], color='red', label='Target Points')

        plt.legend()
        plt.title('Environment')

        plt.savefig('./pic/environment.png')

        # 关闭图形窗口
        plt.close()