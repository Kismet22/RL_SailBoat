import numpy as np
import csv
from hjbppo import HJB_PPO
# from my_ppo_net import Classic_PPO
from icm_ppo import ICM_PPO
from SaiBoat_Environment_1 import Boat_Environment
from Windcondition import read_wind_data
from PlotBoat import *

# hjb_path = './model/hjb_ppo.pth'
# ppo_path = './model/ppo.pth'
wind_path = './wind/long_wind/wind_data1.txt'


def load_position_record_csv(file_name='position_record.csv'):
    with open(file_name, mode='r') as file:
        reader = csv.reader(file)
        return [[float(x.strip().strip('[]')) for x in row] for row in reader]  # 去掉方括号并转换为浮点数


def success_rate_test(env_in, agent_in, test_time, _max_ep_len, _agent_name, _if_plot=False):
    success_times = 0
    count_time = []
    positions_record = []
    all_positions = []
    success_flags = []
    for n in range(test_time):
        state, info = env_in.reset()  # gymnasium库改为在这里设置seed
        for i in range(1, _max_ep_len + 1):
            action = agent_in.select_action(state)
            state, reward, terminated, truncated, info = env_in.step(action)
            positions_record.append([info["x"], info["y"]])
            if terminated:
                print(f"Episode{n} success")
                success_times += 1
                success_flags.append(1)
                count_time.append(env_in.t_step * env_in.dt)
                break
            if truncated:
                print(f"Episode{n} fail")
                success_flags.append(0)
                break
        # clear buffer
        all_positions.append(positions_record)
        positions_record = []
        agent_in.buffer.clear()
        env_in.close()
    average_time = np.mean(count_time)
    if _if_plot:
        plot_success(all_positions, success_time=success_times, total_time=test_time, x_center=env_in.default_target[0],
                     y_center=env_in.default_target[1], success_flag=success_flags, method_name=_agent_name)
    return success_times, average_time


def test():
    print("============================================================================================")
    wind_init = read_wind_data(wind_path)

    max_time_step = 1000
    max_ep_len = max_time_step + 100

    # 判断动作空间是否连续
    # continuous action space; else discrete
    has_continuous_action_space = True

    # update policy for K epochs in one PPO update
    K_epochs = 40

    eps_clip = 0.2  # clip rate for PPO
    gamma = 0.99  # discount factor γ

    lr_actor = 0.0001  # learning rate for actor network
    lr_critic = 0.0002  # learning rate for critic network

    action_std_4_test = 0.002  # set std for action distribution when testing.
    random_seed = 10

    #####################################################
    start_and_end = np.array([0, 0, 40, 20])
    t_c = np.array([60, 30])
    _random_wind = False

    env = Boat_Environment(is_fixed_start_and_target=True, _init_pair=start_and_end, is_random_wind=_random_wind,
                           max_steps=max_time_step, _init_wind=None, _init_center=t_c, _init_target_range=6)

    # 状态空间
    # state space dimension
    state_dim = env.observation_space.shape[0]

    # 动作空间
    action_output_scale = np.array([])
    action_output_bias = np.array([])
    if has_continuous_action_space:
        # action space dimension
        action_dim = env.action_space.shape[0]
        # 输出动作缩放的相关内容
        action_output_scale = (env.action_space.high - env.action_space.low) / 2.0
        action_output_bias = (env.action_space.high + env.action_space.low) / 2.0

    else:
        # 离散动作空间，输出可选的动作数
        action_dim = env.action_space.n

    hjb_ppo_agent = HJB_PPO(state_dim, action_dim, lr_actor, lr_critic, gamma, K_epochs, eps_clip,
                            has_continuous_action_space,
                            action_std_init=action_std_4_test, continuous_action_output_scale=action_output_scale,
                            continuous_action_output_bias=action_output_bias, hjb_lamda=0.5)

    hjb_ppo_agent.load_full(hjb_path)

    ppo_agent = Classic_PPO(state_dim, action_dim, lr_actor, lr_critic, gamma, K_epochs, eps_clip,
                            has_continuous_action_space,
                            action_std_4_test, continuous_action_output_scale=action_output_scale,
                            continuous_action_output_bias=action_output_bias)

    ppo_agent.load_full(ppo_path)

    """""""""
    s_time, av_time = success_rate_test(env, agent_in=ppo_agent, test_time=100, _max_ep_len=max_ep_len,
                                        _if_plot=False,
                                        _agent_name='HJB_PPO')
    print(f"Success rate:{s_time}/100")
    print(f"Average_time:{av_time}")
    """

    ########################### HJB_PPO ###########################
    # 用于保存轨迹的列表
    x_positions_hjb = []
    y_positions_hjb = []
    positions_hjb = []
    actions_hjb = []

    if random_seed:
        state, info = env.reset(seed=random_seed)  # gymnasium库改为在这里设置seed
    else:
        state, info = env.reset()  # gymnasium库改为在这里设置seed

    ep_return_hjb = 0
    total_steps_hjb = 0
    for i in range(1, max_ep_len + 1):
        action = hjb_ppo_agent.select_action(state)
        print(f"Step {i}, Action: {action}")
        actions_hjb.append(action)
        state, reward, terminated, truncated, info = env.step(action)
        ep_return_hjb += reward
        total_steps_hjb = i
        x_positions_hjb.append(info["x"])
        y_positions_hjb.append(info["y"])
        positions_hjb.append([info["x"], info["y"]])
        if terminated:
            break

        if truncated:
            break

        # clear buffer
        hjb_ppo_agent.buffer.clear()
    env.close()

    print("Total_Reward:", ep_return_hjb)
    print("Total_Steps:", total_steps_hjb)

    """""""""
    _, _ = plot_series(x_points=x_positions_hjb, y_points=y_positions_hjb,
                       end_point=[start_and_end[2], start_and_end[3]],
                       wind=wind_init)
    plt.show()
    """

    ########################### PPO ###########################
    # 用于保存轨迹的列表
    x_positions_ppo = []
    y_positions_ppo = []
    positions_ppo = []
    actions_ppo = []

    if random_seed:
        state, info = env.reset(seed=random_seed)  # gymnasium库改为在这里设置seed
    else:
        state, info = env.reset()  # gymnasium库改为在这里设置seed

    ep_return_ppo = 0
    total_steps_ppo = 0
    for i in range(1, max_ep_len + 1):
        action = ppo_agent.select_action(state)
        print(f"Step {i}, Action: {action}")
        actions_ppo.append(action)
        state, reward, terminated, truncated, info = env.step(action)
        ep_return_ppo += reward
        total_steps_ppo = i
        x_positions_ppo.append(info["x"])
        y_positions_ppo.append(info["y"])
        positions_ppo.append([info["x"], info["y"]])
        if terminated:
            break

        if truncated:
            break

        # clear buffer
        ppo_agent.buffer.clear()
    env.close()

    print("Total_Reward:", ep_return_ppo)
    print("Total_Steps:", total_steps_ppo)

    mpc_name = './data/position_record_4020.csv'
    positions_mpc = load_position_record_csv(file_name=mpc_name)
    all_position = [positions_hjb, positions_ppo, positions_mpc]
    all_action = [actions_hjb, actions_ppo]
    legend = True
    legend_labels = ['hjb_ppo', 'ppo', 'mpc']  # 使用你指定的轨迹名称
    # plot_multi_action_normal(arrays_in=all_action, labels=legend_labels)

    wind_init = None
    _, _ = plot_multi_series_with_wind(points_list=all_position, end_point=[start_and_end[2], start_and_end[3]],
                                       legend=legend, legend_labels=legend_labels,
                                       wind=wind_init)
    plt.show()


if __name__ == '__main__':
    test()
