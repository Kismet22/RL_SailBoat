import torch
import numpy as np
import torch.nn as nn
from torch.distributions import MultivariateNormal
import torch.optim as optim
from torch.distributions import MultivariateNormal, Categorical
from torch.optim.lr_scheduler import ExponentialLR

################################## set device ##################################
print("============================================================================================")
# set device to cpu or cuda
device = torch.device('cpu')
if torch.cuda.is_available():
    device = torch.device('cuda:0')
    torch.cuda.empty_cache()
    print("Device set to : " + str(torch.cuda.get_device_name(device)))
else:
    print("Device set to : cpu")
print("============================================================================================")


def force_cpu_as_device():
    global device
    device = torch.device('cpu')
    print("============================================================================================")
    print("Device is FORCED set to : cpu")
    print("============================================================================================")


def init_weights(m):
    if isinstance(m, nn.Linear):
        # 使用正交初始化方法来初始化线性层 m 的权重
        torch.nn.init.orthogonal_(m.weight)


class ICM(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_layer_dim, scale, bias, icm_alpha=0.5):
        super().__init__()
        self.act_dim = action_dim
        self.encode_dim = 64
        self.hidden_layer_dim = hidden_layer_dim
        # scale and bias should be tensor
        self.scale = torch.tensor(scale, device=device)
        self.bias = torch.tensor(bias, device=device)
        self.icm_alpha = icm_alpha

        # 最终输出为(batch_size, encode_dim)
        self.state_encode = nn.Sequential(
            nn.Linear(state_dim, self.hidden_layer_dim),
            nn.Tanh(),
            nn.Linear(self.hidden_layer_dim, self.hidden_layer_dim),
            nn.Tanh(),
            nn.Linear(self.hidden_layer_dim, self.encode_dim),
            nn.Tanh()
        )

        # inverse
        # encode_state_t + encode_state_t+1 -> action_t
        # 最终输出为(batch_size, action_dim)
        self.inverse = nn.Sequential(
            nn.Linear(self.encode_dim * 2, self.hidden_layer_dim),
            nn.Tanh(),
            nn.Linear(self.hidden_layer_dim, self.hidden_layer_dim),
            nn.Tanh(),
            nn.Linear(self.hidden_layer_dim, action_dim),
            nn.Tanh()
        )

        # forward
        # encode_state_t + action_t -> encode_state_t+1
        # 最终输出为(batch_size, encode_dim)
        self.icm_forward_net = nn.Sequential(
            nn.Linear(self.encode_dim + action_dim, self.hidden_layer_dim),
            nn.Tanh(),
            nn.Linear(self.hidden_layer_dim, self.hidden_layer_dim),
            nn.Tanh(),
            nn.Linear(self.hidden_layer_dim, self.encode_dim),
            nn.Tanh()
        )

        self.state_encode.apply(init_weights)
        self.inverse.apply(init_weights)
        self.icm_forward_net.apply(init_weights)

        # # 初始化优化器
        # self.optimizer = torch.optim.Adam([
        #     # self.policy.actor.parameters() 提取actor网络的所有参数
        #     {'params': self.inverse.parameters(), 'lr': 1e-4},
        #     {'params': self.icm_forward_net.parameters(), 'lr': 1e-4}
        # ])

    def icm_forward(self, actions_in, states_in, next_states_in):
        # normalize
        encode_states = self.state_encode(states_in)
        encode_next_states = self.state_encode(next_states_in)

        # TODO:研究这里的维度为什么会发生改变,用.shape or .size()
        # actions进行了一次.squeeze操作，维度为(N,)
        actions_in = actions_in.unsqueeze(-1)

        # 逆模型：根据当前状态和下一状态预测动作
        inverse_in = torch.cat((encode_states, encode_next_states), dim=-1)
        predict_actions = self.inverse(inverse_in)  # 预测连续动作
        predict_actions = predict_actions * self.scale + self.bias

        # 计算逆向损失（MSE损失，针对连续动作）
        inv_loss = nn.MSELoss()(predict_actions, actions_in).mean()

        # 前向模型：根据当前状态和动作预测下一状态
        forward_in = torch.cat((encode_states, actions_in), dim=-1)
        predict_next_states = self.icm_forward_net(forward_in)  # 预测下一状态

        # 基于前向模型的MSE损失
        # [batch_size, t, n]
        icm_forward_loss = 0.5 * nn.MSELoss(reduction='none')(predict_next_states, encode_next_states)
        icm_rewards = self.icm_alpha * icm_forward_loss.mean(dim=-1)  # [batch_size, t]
        icm_forward_loss = icm_forward_loss.mean()  # 标量
        return inv_loss, icm_forward_loss, icm_rewards


################################## IH_PPO Policy ##################################
class RolloutBuffer:
    def __init__(self):
        self.actions = []
        self.states = []
        self.log_probs = []
        self.rewards = []
        self.state_values = []
        self.is_terminals = []
        self.next_states = []

    # 删除整个经验池中的数据
    def clear(self):
        del self.actions[:]
        del self.states[:]
        del self.log_probs[:]
        del self.rewards[:]
        del self.state_values[:]
        del self.is_terminals[:]
        del self.next_states[:]

    # 删除最后一个值
    def del_last(self):
        del self.actions[-1]
        del self.states[-1]
        del self.log_probs[-1]
        del self.rewards[-1]
        del self.state_values[-1]
        del self.is_terminals[-1]
        del self.next_states[-1]

    # 打印每一个元素的长度
    def print_lengths(self):
        print(f"Actions length: {len(self.actions)}")
        print(f"States length: {len(self.states)}")
        print(f"Log_probs length: {len(self.log_probs)}")
        print(f"Rewards length: {len(self.rewards)}")
        print(f"State values length: {len(self.state_values)}")
        print(f"Is terminals length: {len(self.is_terminals)}")
        print(f"Next states length: {len(self.next_states)}")


class ActorCritic(nn.Module):

    def __init__(self, state_dim, action_dim, hidden_layer_dim, has_continuous_action_space, action_std_init):
        super(ActorCritic, self).__init__()

        # 输入网络的矩阵形状为(batch_size, state_dim)
        self.has_continuous_action_space = has_continuous_action_space
        self.hidden_layer_dim = hidden_layer_dim

        # 一维方差列表,长度为动作维度个,值为初始化标准差的平方(方差)
        if has_continuous_action_space:
            self.action_dim = action_dim
            self.action_var = torch.full((action_dim,), action_std_init * action_std_init).to(device)

        # actor
        if has_continuous_action_space:
            self.actor = nn.Sequential(
                nn.Linear(state_dim, self.hidden_layer_dim),
                nn.Tanh(),
                nn.Linear(self.hidden_layer_dim, self.hidden_layer_dim),
                nn.Tanh(),
                nn.Linear(self.hidden_layer_dim, self.hidden_layer_dim),
                nn.Tanh(),
                nn.Linear(self.hidden_layer_dim, action_dim),
                nn.Tanh()  # 最终输出为(batch_size, action_dim)
            )
        else:
            self.actor = nn.Sequential(
                nn.Linear(state_dim, self.hidden_layer_dim),
                nn.Tanh(),
                nn.Linear(self.hidden_layer_dim, self.hidden_layer_dim),
                nn.Tanh(),
                nn.Linear(self.hidden_layer_dim, self.hidden_layer_dim),
                nn.Tanh(),
                nn.Linear(self.hidden_layer_dim, action_dim),
                nn.Softmax(dim=-1)  # 最终输出为(batch_size, action_dim)
                # 依照最后一个维度,在该维度上进行Softmax归一化操作,即在action_dim这个维度上进行归一化
            )

        # critic
        self.critic = nn.Sequential(
            nn.Linear(state_dim, self.hidden_layer_dim),
            nn.Tanh(),
            nn.Linear(self.hidden_layer_dim, self.hidden_layer_dim),
            nn.Tanh(),
            nn.Linear(self.hidden_layer_dim, self.hidden_layer_dim),
            nn.Tanh(),
            nn.Linear(self.hidden_layer_dim, 1)
        )

        # 正交初始化
        self.actor.apply(init_weights)
        self.critic.apply(init_weights)

    def set_action_std(self, new_action_std):
        if self.has_continuous_action_space:
            self.action_var = torch.full((self.action_dim,), new_action_std * new_action_std).to(device)
        else:
            print("--------------------------------------------------------------------------------------------")
            print("WARNING : Calling ActorCritic::set_action_std() on discrete action space policy")
            print("--------------------------------------------------------------------------------------------")

    def forward(self):
        raise NotImplementedError

    def act(self, state):
        """
        给定输入状态，返回一个动作
        :param state:
        :return: 动作、动作的log概率、state状态函数（critic直接输出）
        """
        if self.has_continuous_action_space:

            # torch.diag()创建一个对角阵，对角线上的值为对应位置的方差 (action_dim, action_dim)
            # torch.unsqueeze(dim=n) 在第n个维度上增加一个维度

            # 动作的均值
            action_mean = self.actor(state)  # (batch_size, action_dim)
            # 动作的协方差矩阵
            cov_mat = torch.diag(self.action_var).unsqueeze(dim=0)  # (1, (action_dim, action_dim))
            # 创建一个多元正态分布
            dist = MultivariateNormal(action_mean, cov_mat)
        else:
            action_probs = self.actor(state)
            dist = Categorical(action_probs)

        # 从创建的分布中采样一个动作
        action = dist.sample()
        # 计算采样的动作 action 在概率分布 dist 下的对数概率
        action_log_prob = dist.log_prob(action)
        # 通过critic网络对当前状态 state 进行评价，得到对应状态估值
        state_val = self.critic(state)
        # .detach()该tensor不参与梯度计算
        return action.detach(), action_log_prob.detach(), state_val.detach()

    def evaluate(self, state, action):
        """
        评估state-action pair
        :param state:
        :param action:
        :return: action_log_probs, state_values, dist_entropy
        """
        # action:(batch_size, action_dim)
        if self.has_continuous_action_space:
            action_mean = self.actor(state)
            action_var = self.action_var.expand_as(action_mean)
            cov_mat = torch.diag_embed(action_var).to(device)
            dist = MultivariateNormal(action_mean, cov_mat)

            # For Single Action Environments.
            if self.action_dim == 1:
                # 初始，action:(batch_size, 1)
                # 转化为(batch_size, 1)
                # .reshape(-1, self.action_dim),矩阵根据第二个元素调整
                action = action.reshape(-1, self.action_dim)
        else:
            action_probs = self.actor(state)
            dist = Categorical(action_probs)
        action_log_probs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        state_values = self.critic(state)

        return action_log_probs, state_values, dist_entropy


class IH_PPO:
    def __init__(self, state_dim, action_dim, lr_actor, lr_critic,
                 gamma, K_epochs, eps_clip, has_continuous_action_space, delta_t=0.01,
                 action_std_init=0.6, hidden_layer_dim=128,
                 continuous_action_output_scale=1.0, continuous_action_output_bias=0.0, hjb_lamda=0.5, icm_alpha=0.5):
        """
        PPO constructor.
        :param state_dim: 状态空间维数
        :param action_dim: 动作空间维数
        :param lr_actor: actor网络的学习率
        :param lr_critic: critic网络的学习率
        :param gamma: 奖励值累加时的衰减率 r＋γr_after
        :param K_epochs: 更新优化的轮数epochs
        :param eps_clip: PPO2方法的裁剪率
        :param has_continuous_action_space: 动作空间是否连续
        :param action_std_init: 连续动作空间的分布标准差
        :param continuous_action_output_scale: 连续动作空间时，将默认输出[-1, 1]缩放的倍数
        :param continuous_action_output_bias:  连续动作空间时，将默认输出[-1, 1]缩放后的偏置
        # 解释:网络输出经过归一化，范围为[-1, 1]
        # 实际动作空间范围为[a, b]
        # 映射方式:scale = (b - a)/(1 - (-1)) bias = b - (b - a)/2 = (b + a)/2
        """
        self.has_continuous_action_space = has_continuous_action_space

        if has_continuous_action_space:
            # 连续动作空间时的动作矩阵方差
            self.action_std = action_std_init

        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs
        self.delta_t = delta_t
        self.continuous_action_output_scale = continuous_action_output_scale
        self.continuous_action_output_bias = continuous_action_output_bias

        self.buffer = RolloutBuffer()  # 数据池

        # PPO policy_net
        self.policy = ActorCritic(state_dim, action_dim, hidden_layer_dim, has_continuous_action_space,
                                  action_std_init).to(device)

        self.icm_alpha = icm_alpha
        self.icm = ICM(state_dim=state_dim, action_dim=action_dim, hidden_layer_dim=hidden_layer_dim,
                       scale=continuous_action_output_scale, bias=continuous_action_output_bias, icm_alpha=self.icm_alpha).to(device)

        # 设置优化器，对策略A2C的actor和critic分别进行优化
        # 在强化学习中，优化器会通过反向传播来更新 actor 和 critic 的参数
        # 这里使用的是 Adam 优化器，它是一种基于一阶矩估计的自适应学习率优化算法
        self.optimizer = torch.optim.Adam([
            # self.policy.actor.parameters() 提取actor网络的所有参数
            {'params': self.policy.actor.parameters(), 'lr': lr_actor},
            {'params': self.policy.critic.parameters(), 'lr': lr_critic},
            {'params': self.icm.inverse.parameters(), 'lr': 2e-4},
            {'params': self.icm.icm_forward_net.parameters(), 'lr': 2e-4}
        ])

        # PPO old policy_net
        self.policy_old = ActorCritic(state_dim, action_dim, hidden_layer_dim, has_continuous_action_space,
                                      action_std_init).to(device)

        # 旧策略网络保存上一个策略的所有参数
        # .state_dict():读取网络数据
        # 写入网络数据
        self.policy_old.load_state_dict(self.policy.state_dict())

        # Second Reward :MSE_loss for Actor-Critic Net
        self.MseLoss = nn.MSELoss()
        self.hjb_lamda = hjb_lamda

    def set_action_std(self, new_action_std):
        if self.has_continuous_action_space:
            self.action_std = new_action_std
            self.policy.set_action_std(new_action_std)
            self.policy_old.set_action_std(new_action_std)
        else:
            print("--------------------------------------------------------------------------------------------")
            print("WARNING : Calling PPO::set_action_std() on discrete action space policy")
            print("--------------------------------------------------------------------------------------------")

    def decay_action_std(self, action_std_decay_rate, min_action_std):
        # 动作矩阵方差衰减函数
        # 对动作矩阵的方差进行一定数值上的减少，但不低于最低的方差值
        print("--------------------------------------------------------------------------------------------")
        if self.has_continuous_action_space:
            self.action_std = self.action_std - action_std_decay_rate
            self.action_std = round(self.action_std, 4)
            if self.action_std <= min_action_std:
                self.action_std = min_action_std
                print("setting actor output action_std to min_action_std : ", self.action_std)
            else:
                print("setting actor output action_std to : ", self.action_std)
            self.set_action_std(self.action_std)

        else:
            print("WARNING : Calling PPO::decay_action_std() on discrete action space policy")
        print("--------------------------------------------------------------------------------------------")
        return self.action_std

    def select_action(self, input_state):
        """
        输入state，返回要采取的action，并保存action, action_log_prob, state_val至buffer
        :param input_state:
        :return: 经过线性变换的action
        """

        if isinstance(input_state, dict):
            state = list(input_state.values())
        else:
            state = input_state

        if isinstance(state, np.ndarray):
            state = torch.tensor(state, dtype=torch.float32).to(device)
        elif isinstance(state, list):
            state = torch.tensor(state, dtype=torch.float32).to(device)
        elif isinstance(state, torch.Tensor):
            state = state.to(device)
        else:
            raise TypeError(f"Unsupported state type: {type(state)}")

        # 其余代码保持不变
        # 先处理连续动作情况，使用old_policy做一次act
        if self.has_continuous_action_space:
            # old_policy不保存梯度
            with torch.no_grad():
                # 走一步，保存动作、动作的log概率、value_function
                action, action_log_prob, state_val = self.policy_old.act(state)

            # 在经验池中保存old_policy的决策的相关数据
            self.buffer.states.append(state)
            self.buffer.actions.append(action)
            self.buffer.log_probs.append(action_log_prob)
            self.buffer.state_values.append(state_val)
            # 对action做线性变换
            action_raw = action.detach().cpu().numpy().flatten()
            # action输出的范围为(-1, 1)
            action_out = action_raw * self.continuous_action_output_scale + self.continuous_action_output_bias
            return action_out

        else:
            with torch.no_grad():
                state = torch.FloatTensor(state).to(device)
                action, action_log_prob, state_val = self.policy_old.act(state)

            self.buffer.states.append(state)
            self.buffer.actions.append(action)
            self.buffer.log_probs.append(action_log_prob)
            self.buffer.state_values.append(state_val)

            # .item()用于将包含一个元素的张量转换为 Python 的标量值。这个方法只能用于张量中仅包含一个元素的情况
            return action.item()

    def update(self):
        """
        更新K次网络:buffer中action, action_log_prob, state_val在select_action时保存，reward与terminal从外部加入。
        :return: None
        """
        # convert list to tensor

        # 检查并转换 self.buffer.states 中的每个项
        states_on_device = []
        for state in self.buffer.states:
            if isinstance(state, torch.Tensor):
                # 如果已经是张量，确保它在正确的设备上
                states_on_device.append(state.to(device))
            else:
                # 如果不是张量，将其转换为张量并放到正确的设备上
                states_on_device.append(torch.tensor(state, dtype=torch.float32, device=device))
        old_states = torch.squeeze(torch.stack(states_on_device, dim=0)).detach().to(device)  # 进行堆叠和其他操作

        next_states_on_device = []
        for state in self.buffer.next_states:
            if isinstance(state, torch.Tensor):
                # 如果已经是张量，确保它在正确的设备上
                next_states_on_device.append(state.to(device))
            else:
                # 如果不是张量，将其转换为张量并放到正确的设备上
                next_states_on_device.append(torch.tensor(state, dtype=torch.float32, device=device))
        old_next_states = torch.squeeze(torch.stack(next_states_on_device, dim=0)).detach().to(device)  # 进行堆叠和其他操作

        # 检查并转换 self.buffer.actions 中的每个项
        actions_on_device = []
        for action in self.buffer.actions:
            if isinstance(action, torch.Tensor):
                # 如果已经是张量，确保它在正确的设备上
                actions_on_device.append(action.to(device))
            else:
                # 如果不是张量，将其转换为张量并放到正确的设备上
                actions_on_device.append(torch.tensor(action, dtype=torch.float32, device=device))
        old_actions = torch.squeeze(torch.stack(actions_on_device, dim=0)).detach().to(device)  # 进行堆叠和其他操作

        # 检查并转换 self.buffer.rewards 中的每个项
        # step_rewards用于计算HJB_loss
        step_rewards = []
        for reward in self.buffer.rewards:
            if isinstance(reward, torch.Tensor):
                # 如果已经是张量，确保它在正确的设备上
                step_rewards.append(reward.to(device))
            else:
                # 如果不是张量，将其转换为张量并放到正确的设备上
                step_rewards.append(torch.tensor(reward, dtype=torch.float32, device=device))
        step_rewards = torch.squeeze(torch.stack(step_rewards, dim=0)).detach().to(device)  # 进行堆叠和其他操作

        old_log_probs = torch.squeeze(torch.stack(self.buffer.log_probs, dim=0)).detach().to(device)
        old_state_values = torch.squeeze(torch.stack(self.buffer.state_values, dim=0)).detach().to(device)

        # Step2: 通过icm计算icm_reward
        with torch.no_grad():
            # .detach():防止与原始计算图共享梯度
            old_actions_detached = old_actions.detach()
            old_states_detached = old_states.detach()
            old_next_states_detached = old_next_states.detach()
            _, _, icm_rewards = self.icm.icm_forward(actions_in=old_actions_detached, states_in=old_states_detached,
                                                     next_states_in=old_next_states_detached)

        step_rewards = step_rewards + icm_rewards  # external_reward + internal_reward

        # Monte Carlo estimate of returns
        discount_rewards = []
        discounted_reward = 0

        # 倒着处理buffer，为每一步生成一个monte-carlo的value-function采样，代表在某个状态下一直MC采样到末尾得到的return
        # 折扣回报的计算方式，符合r = r + γΣr
        # reversed()反转操作
        # zip()将两个可迭代的对象配对到一起

        for reward, is_terminal in zip(reversed(step_rewards), reversed(self.buffer.is_terminals)):
            if is_terminal:
                # 当一个时间步是终止状态时，之后的折扣回报将从零开始计算,这里是翻转之后的，则应该表示终止之前的计算
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            discount_rewards.insert(0, discounted_reward)

        # 将数值转化为一个可用于网络计算的tensor形式
        discount_rewards = torch.tensor(discount_rewards, dtype=torch.float32).to(device)
        # Normalizing the rewards:
        # discount_rewards = (discount_rewards - discount_rewards.mean()) / (discount_rewards.std() + 1e-7)

        # buffer中原本是众多连续的一维张量
        # .stack()声明了堆叠方式,将张量在第'dim=n'的方向上进行了堆叠,在这里是第0维(batch_size)维度
        # .squeeze()这个函数用于移除张量中维度为 1 的所有维度 (N,1)会变成(N,)
        # .detach()分离张量，使得它不再参与梯度计算，这是为了确保训练中的状态张量不再影响后续的梯度更新
        # 操作目的，将多个状态一起按同一批次进行处理

        # calculate advantages
        # old_state_values是老的critic网络对每个状态的v值估计，而rewards是经过采样【且归一化】后的真实reward
        # 在一个epoch的更新过程中，优势函数不发生改变，一直使用旧策略的优势函数
        advantages = discount_rewards.detach() - old_state_values.detach()
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-7)

        # Optimize policy for K epochs
        for _ in range(self.K_epochs):
            old_states = old_states.requires_grad_(True)  # 确保 old_states 可计算梯度
            # Evaluating old actions and values，在新的policy下，评价老的数据。拿到一块时间步的log_probs, state_values
            # dist_entropy是网络内部的交叉熵，第三部分损失函数
            log_probs, state_values, dist_entropy = self.policy.evaluate(old_states, old_actions)

            # 计算 `state_values` 对 `old_states` 的梯度
            state_gradients = torch.autograd.grad(
                outputs=state_values,  # 目标输出，即对谁求导
                inputs=old_states,  # 输入，即相对于谁求导
                grad_outputs=torch.ones_like(state_values),  # 如果 `state_values` 不是标量，使用此项计算累积梯度
                create_graph=True  # 如果你还需要计算更高阶的导数，则设置为 True
            )[0]

            # MSE_F
            log_gamma = torch.log(torch.tensor(self.gamma))
            delta_states_f = torch.squeeze(old_next_states - old_states).detach()
            state_values_f = torch.squeeze(state_values).detach()
            state_gradients_f = torch.squeeze(state_gradients).detach()
            rewards_f = torch.squeeze(step_rewards).detach()
            # # delta_states = torch.squeeze(old_states[:, 1:, :] - old_states[:, -1, :]).detach()
            # delta_states_f = torch.squeeze(old_states[1:, :] - old_states[:-1, :]).detach()
            # state_values_f = torch.squeeze(state_values[:-1]).detach()
            # state_gradients_f = torch.squeeze(state_gradients[:-1]).detach()
            # rewards_f = torch.squeeze(step_rewards[:-1]).detach()

            # ▽_xV(x) * (x_t+1 - x_ t) / Δt
            transition_value = torch.sum(state_gradients_f * delta_states_f, dim=1) / self.delta_t
            mse_f = torch.mean((state_values_f * log_gamma + rewards_f + transition_value) ** 2)

            # MSE_U
            # state_values_t = torch.squeeze(state_values[:-1]).detach()
            # state_values_t1 = torch.squeeze(state_values[1:]).detach()  # v(t+1) 对应后面的时间步
            # mse_u = torch.mean((state_values_t - (rewards_f + torch.tensor(self.gamma) * state_values_t1)) ** 2)
            state_values = state_values.squeeze()  # 删除维度为1的轴,使两者的维度一致
            mse_u = self.MseLoss(state_values, discount_rewards)

            # Loss_2:Actor Hjb Loss
            # 0.5 * MSE_u + λ_HJB * MSE_f
            loss_2 = 0.5 * mse_u + self.hjb_lamda * mse_f

            # Loss_1:Critic_loss
            # Finding the ratio (pi_theta / pi_theta__old),因为带着个log所以用exp,评价新老policy的差异
            # exp(log(new)- log(old)) = new/old
            ratios = torch.exp(log_probs - old_log_probs.detach())
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
            loss_1 = -torch.min(surr1, surr2)

            icm_inverse_loss, icm_forward_loss, _ = self.icm.icm_forward(actions_in=old_actions, states_in=old_states,
                                                                         next_states_in=old_next_states)
            icm_loss = icm_inverse_loss + icm_forward_loss  # 两个loss已经进行了.mean()操作
            # PPO Loss
            # 虽然loss写在了一起，但两个网络会各自计算和自己相关的梯度，这没什么问题。
            loss = loss_1 + loss_2 - 0.01 * dist_entropy
            loss = loss.mean() + icm_loss

            # 梯度优化过程
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        # Copy new weights into old policy
        # 保存新的网络数据到policy_old
        self.policy_old.load_state_dict(self.policy.state_dict())

        # clear buffer
        self.buffer.clear()

    def save(self, checkpoint_path):
        # 保存网络数据
        # 将self.policy_old.state_dict()的数据以字典的形式保存到checkpoint_path中去
        torch.save(self.policy_old.state_dict(), checkpoint_path)

    def save_icm(self, checkpoint_path):
        # 保存网络数据
        # 将self.icm.state_dict()的数据以字典的形式保存到checkpoint_path中去
        torch.save(self.icm.state_dict(), checkpoint_path)

    def load(self, checkpoint_path):
        # 加载网络数据
        # 'map_location'参数用于控制模型参数加载到的设备
        # lambda storage, loc: storage是一个匿名函数，将确保模型参数被加载到与保存时相同的设备上，或者将其映射到当前可用的设备
        self.policy_old.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))
        self.policy.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))

    def load_icm(self, checkpoint_path):
        self.icm.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))

    # 加一组更完整的保存的function，能把Adam中自适应学习率的部分也保存好，可以接续训练
    def save_full(self, filename):
        # 不止保存了网络结构数据，同时保存了优化器的参数
        print("=> Saving checkpoint")
        checkpoint = {
            "model_state_dict": self.policy_old.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict()
        }
        torch.save(checkpoint, filename)

    # 加一组更完整的保存的function，能把Adam中自适应学习率的部分也保存好，可以接续训练
    def save_full_icm(self, filename):
        # 不止保存了网络结构数据，同时保存了优化器的参数
        print("=> Saving checkpoint")
        checkpoint = {
            "model_icm_dict": self.icm.state_dict(),
            "optimizer_icm_dict": self.optimizer.state_dict()
        }
        torch.save(checkpoint, filename)

    def load_full(self, filename):
        # 将网络数据和优化器数据都加载出来
        print("=> Loading checkpoint")
        checkpoint = torch.load(filename, weights_only=False)
        self.policy_old.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.policy.load_state_dict(self.policy_old.state_dict())  # Ensure both policies are synced

    def load_full_icm(self, filename):
        # 将网络数据和优化器数据都加载出来
        print("=> Loading checkpoint")
        checkpoint = torch.load(filename, weights_only=False)
        self.icm.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])  # load

