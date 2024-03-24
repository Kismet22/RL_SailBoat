import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal
from torch.distributions import Categorical
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


# 初始化权重
def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.orthogonal_(m.weight)


################################## PPO Policy ##################################
class RolloutBuffer:
    def __init__(self):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.is_terminals = []

    def clear(self):
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.is_terminals[:]


class ActorCritic(nn.Module):
    train = True

    def __init__(self, state_dim, action_dim, hidden_layer_dim, has_continuous_action_space, action_std_init):
        super(ActorCritic, self).__init__()

        self.has_continuous_action_space = has_continuous_action_space
        self.hidden_layer_dim = hidden_layer_dim

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
                nn.Linear(self.hidden_layer_dim, action_dim),
                # nn.Tanh()
            )
        else:
            self.actor = nn.Sequential(
                nn.Linear(state_dim, self.hidden_layer_dim),
                nn.Tanh(),
                nn.Linear(self.hidden_layer_dim, self.hidden_layer_dim),
                nn.Tanh(),
                nn.Linear(self.hidden_layer_dim, action_dim),
                # 离散动作空间用softmax进行归一化
                nn.Softmax(dim=-1)
            )

        # critic
        # 输出状态的价值函数
        self.critic = nn.Sequential(
            nn.Linear(state_dim, self.hidden_layer_dim),
            nn.Tanh(),
            nn.Linear(self.hidden_layer_dim, self.hidden_layer_dim),
            nn.Tanh(),
            nn.Linear(self.hidden_layer_dim, 1)
        )

        self.actor.apply(init_weights)
        self.critic.apply(init_weights)

    # 重置连续动作空间的高斯分布
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
        if self.has_continuous_action_space:
            # 均值
            action_mean = self.actor(state)  # * 20
            # 协方差分布
            cov_mat = torch.diag(self.action_var).unsqueeze(dim=0)
            # 多变量的高斯分布
            dist = MultivariateNormal(action_mean, cov_mat)
        else:
            action_probs = self.actor(state)
            dist = Categorical(action_probs)

        if self.train:
            action = dist.sample()
        else:
            action = dist.mean
        action_logprob = dist.log_prob(action)

        return action.detach(), action_logprob.detach()

    def evaluate(self, state, action):

        if self.has_continuous_action_space:
            # 计算动作均值
            action_mean = self.actor(state)
            # 检查是否存在空值
            if torch.isnan(action_mean).any():
                print(self.actor[0].weight, self.actor[-1].weight)
            # 计算动作方差
            action_var = self.action_var.expand_as(action_mean)
            #  action_var 中的元素作为对角线元素的对角矩阵
            cov_mat = torch.diag_embed(action_var).to(device)
            # 构建正态分布对象
            dist = MultivariateNormal(action_mean, cov_mat)

            # For Single Action Environments.
            if self.action_dim == 1:
                action = action.reshape(-1, self.action_dim)
        else:
            action_probs = self.actor(state)
            dist = Categorical(action_probs)
        # 动作的对数概率
        action_logprobs = dist.log_prob(action)
        # 分布的熵
        dist_entropy = dist.entropy()
        # 状态价值函数
        state_values = self.critic(state)

        return action_logprobs, state_values, dist_entropy


class PPO:
    def __init__(self, state_dim, action_dim, hidden_layer_dim, lr_actor, lr_critic, gamma, K_epochs, eps_clip,
                 has_continuous_action_space, action_std_init=0.6):

        self.has_continuous_action_space = has_continuous_action_space

        # 初始化时设置连续动作空间的动作标准差
        # 连续动作空间的策略通常需要输出动作的均值和标准差,用作策略的探索
        if has_continuous_action_space:
            self.action_std = action_std_init

        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs

        self.buffer = RolloutBuffer()

        self.policy = ActorCritic(state_dim, action_dim, hidden_layer_dim, has_continuous_action_space,
                                  action_std_init).to(device)
        self.optimizer = torch.optim.Adam([
            {'params': self.policy.actor.parameters(), 'lr': lr_actor},
            {'params': self.policy.critic.parameters(), 'lr': lr_critic}
        ])
        # adaptive learning rate
        self.lr_scheduler = ExponentialLR(self.optimizer, gamma=0.99)

        self.policy_old = ActorCritic(state_dim, action_dim, hidden_layer_dim, has_continuous_action_space,
                                      action_std_init).to(device)
        self.policy_old.load_state_dict(self.policy.state_dict())

        self.MseLoss = nn.MSELoss()

    def set_action_std(self, new_action_std):
        # 方差更新
        if self.has_continuous_action_space:
            self.action_std = new_action_std
            self.policy.set_action_std(new_action_std)
            self.policy_old.set_action_std(new_action_std)
        else:
            print("--------------------------------------------------------------------------------------------")
            print("WARNING : Calling PPO::set_action_std() on discrete action space policy")
            print("--------------------------------------------------------------------------------------------")

    def decay_action_std(self, action_std_decay_rate, min_action_std):
        # 逐步降低标准差
        print("--------------------------------------------------------------------------------------------")
        if self.has_continuous_action_space:
            self.action_std = self.action_std - action_std_decay_rate
            # 四舍五入
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

    def select_action(self, state, redo=False):
        if redo:
            # 缓冲区状态清除
            self.buffer.states.pop()
            self.buffer.actions.pop()
            self.buffer.logprobs.pop()

        if self.has_continuous_action_space:
            with torch.no_grad():
                state = torch.FloatTensor(state).to(device)
                # 老策略输出动作和动作概率
                action, action_logprob = self.policy_old.act(state)

            self.buffer.states.append(state)
            self.buffer.actions.append(action)
            self.buffer.logprobs.append(action_logprob)

            return action.detach().cpu().numpy().flatten()
        else:
            with torch.no_grad():
                state = torch.FloatTensor(state).to(device)
                action, action_logprob = self.policy_old.act(state)

            self.buffer.states.append(state)
            self.buffer.actions.append(action)
            self.buffer.logprobs.append(action_logprob)

            return action.item()

    def update(self):
        # Monte Carlo estimate of returns
        # 存储折扣奖励值
        rewards = []
        discounted_reward = 0
        # 将缓冲区中的奖励值和终止标志一一对应
        for reward, is_terminal in zip(reversed(self.buffer.rewards), reversed(self.buffer.is_terminals)):
            if is_terminal:
                discounted_reward = 0
            # 折扣奖励值
            discounted_reward = reward + (self.gamma * discounted_reward)
            # 将折扣奖励置于开头(从前往后叠加单步奖励，从后往前放置期望奖励)
            rewards.insert(0, discounted_reward)

        # Normalizing the rewards
        rewards = torch.tensor(rewards, dtype=torch.float32).to(device)
        # 对奖励值进行标准化处理, 均值0，标准差1
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7)
        # # self.buffer.rewards
        # if return_only:
        #     rewards = torch.tensor([rewards[i] for i in range(len(rewards)) if self.buffer.is_terminals[i]], dtype=torch.float32).to(device)
        #     self.buffer.states =  [self.buffer.states[i] for i in range(len(self.buffer.states)) if self.buffer.is_terminals[i]]
        #     self.buffer.actions =  [self.buffer.actions[i] for i in range(len(self.buffer.actions)) if self.buffer.is_terminals[i]]
        #     self.buffer.logprobs =  [self.buffer.logprobs[i] for i in range(len(self.buffer.logprobs)) if self.buffer.is_terminals[i]]

        # convert list to tensor
        # 将缓冲区数据转换为张量
        old_states = torch.squeeze(torch.stack(self.buffer.states, dim=0)).detach().to(device)
        old_actions = torch.squeeze(torch.stack(self.buffer.actions, dim=0)).detach().to(device)
        old_logprobs = torch.squeeze(torch.stack(self.buffer.logprobs, dim=0)).detach().to(device)

        # Optimize policy for K epochs
        # K_epochs 策略网络进行梯度更新过程中进行更新的次数
        for _ in range(self.K_epochs):
            # Evaluating old actions and values
            # 旧状态和动作的分布函数，状态价值函数和分布熵
            logprobs, state_values, dist_entropy = self.policy.evaluate(old_states, old_actions)

            # match state_values tensor dimensions with rewards tensor
            # 状态价值张量维度和奖励张量维度匹配
            state_values = torch.squeeze(state_values)

            # Finding the ratio (pi_theta / pi_theta__old)
            # 新旧策略的比例
            ratios = torch.exp(logprobs - old_logprobs.detach())

            # Finding Surrogate Loss
            # 计算优势函数，当前状态价值和期望奖励函数的差值
            advantages = rewards - state_values.detach()
            # 无clip版
            surr1 = ratios * advantages
            # 有clip版
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages

            # final loss of clipped objective PPO
            # PPO的损失函数
            # actor_loss + critic_loss + entropy
            loss = -torch.min(surr1, surr2) + 0.5 * self.MseLoss(state_values, rewards) - 0.01 * dist_entropy

            # take gradient step
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()
            # total_norm = 0
            # for p in self.policy.actor.parameters():
            #     param_norm = p.grad.detach().data.norm(2)
            #     total_norm += param_norm.item() ** 2
            # total_norm = total_norm ** 0.5
            # print(total_norm)

        # update lr
        self.lr_scheduler.step()

        # Copy new weights into old policy
        self.policy_old.load_state_dict(self.policy.state_dict())

        # clear buffer
        self.buffer.clear()

    def save(self, checkpoint_path):
        torch.save(self.policy_old.state_dict(), checkpoint_path)

    def load(self, checkpoint_path):
        self.policy_old.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))
        self.policy.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))

    def test_mode(self):
        self.policy_old.train = False
        self.policy.train = False
