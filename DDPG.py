import random
import torch
import gym
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
#超参数
class config():
    def __init__(self):
        #模型中的超参数
        self.repay_buff_size=50000
        self.gamma=0.99
        self.batch_size=128
        self.policy_input_size=3
        self.critic_input_size=4
        self.output_size=1
        self.hidden=512
        self.soft_updata=0.01
        self.actor_lr=0.00004
        self.critic_lr=0.00008
        #高斯噪声的超参数
        self.std=0.5
        self.mu=0
        self.decay_rate=1e-6
        self.init_w=1e-3
config=config()
#搭建actor
class Actor(nn.Module):
    def __init__(self):
        super(Actor, self).__init__()
        self.liner1=nn.Linear(config.policy_input_size,config.hidden)
        self.linear2=nn.Linear(config.hidden,256)
        self.linear3=nn.Linear(256,config.output_size)
        self.linear2.weight.data.uniform_(-config.init_w, config.init_w)
        self.linear2.bias.data.uniform_(-config.init_w, config.init_w)
        self.f1=nn.ReLU()
        self.f2=nn.Tanh()
    def forward(self,x):#输出一个动作
        x=self.f1(self.liner1(x))
        x=self.f1(self.linear2(x))
        x=self.f2(self.linear3(x))
        return x*2
#搭建critic
class Critic(nn.Module):
    def __init__(self):
        super(Critic, self).__init__()
        self.liner1 = nn.Linear(config.critic_input_size, config.hidden)
        self.linear2 = nn.Linear(config.hidden, 256)
        self.linear3 = nn.Linear(256, config.output_size)
        self.linear3.weight.data.uniform_(-config.init_w, config.init_w)
        self.linear3.bias.data.uniform_(-config.init_w, config.init_w)
        self.f1 = nn.ReLU()
        self.f2=nn.Tanh()
    def forward(self,x):
        x = self.f1(self.liner1(x))
        x = self.f2(self.linear2(x))
        x = (self.linear3(x))
        return x
class DDPG():
    def __init__(self):
        self.policy_net=Actor()
        self.target_policy_net=Actor()
        self.updata_Q=Critic()
        self.target_Q=Critic()
        #定义误差函数，优化器
        self.loss_fun=nn.MSELoss(reduction='mean')
        self.Q_optim=torch.optim.Adam(self.updata_Q.parameters(),lr=config.critic_lr)
        self.policy_optim=torch.optim.Adam(self.policy_net.parameters(),lr=config.actor_lr)
        self.repay_buff_size=config.repay_buff_size
        self.mu=config.mu
        self.std=config.std
        self.decay_rate=config.decay_rate
        self.soft_updata=config.soft_updata
        self.buff=[]
        self.env=gym.make('Pendulum-v1')
        self.num=0
        self.num1=0
        self.env.reset(seed=2)
        # 复制参数到目标网络
        for target_param, param in zip(self.target_Q.parameters(), self.updata_Q.parameters()):
            target_param.data.copy_(param.data)
        for target_param, param in zip(self.target_policy_net.parameters(), self.policy_net.parameters()):
            target_param.data.copy_(param.data)
    def noise(self,action):
        action=action.detach().numpy()
        noises=np.random.normal(self.mu,self.std,1)
        #self.std=self.std-self.decay_rate
        noises=np.clip(noises,-0.5,0.5)
        return np.clip(action+noises,-2,2)
    def Buff(self,experience):
        if len(self.buff)<self.repay_buff_size:
            self.buff.append(experience)
        else:
            self.buff.pop(0)
            self.buff.append(experience)
    def random_sample(self):
        for _ in range(100):
            state=self.env.reset()
            while True:
                action=self.env.action_space.sample()
                next_state, reward, done, _ = self.env.step(action)
                self.Buff([state, action, reward, next_state, done])
                if done:
                    break
                state=next_state
    def updata(self):
        self.num+=1
        self.num1+=1
        self.Q_optim.zero_grad()
        train_data=random.sample(self.buff,config.batch_size)
        state=np.array([exp[0] for exp in train_data])
        noise_action=np.array([exp[1] for exp in train_data])
        reward=np.array([[exp[2]] for exp in train_data])
        next_state=np.array([exp[3] for exp in train_data])
        done=np.array([[exp[4]] for exp in train_data])
        #变为张量
        state=torch.from_numpy(state).float()
        noise_action=torch.from_numpy(noise_action).float()
        reward=torch.from_numpy(reward).float()
        next_state=torch.from_numpy(next_state).float()
        done=torch.from_numpy(done).float()
        critic_state_input=torch.cat((state,noise_action),1)
        next_action=self.target_policy_net(next_state)
        critic_nextstate_input=torch.cat((next_state,next_action),1)
        true_action=self.policy_net(state)
        print(true_action.data)
        q_state_value=self.updata_Q(critic_state_input)
        #print(q_state_value.data)
        q_nextstate_value=self.target_Q(critic_nextstate_input)
        target_q=reward+(1-done)*config.gamma*q_nextstate_value
        loss=self.loss_fun(target_q,q_state_value)
        loss.backward()
        self.Q_optim.step()
        #更新策略网络
        if self.num1==2:
            self.policy_optim.zero_grad()
            true_q_input=torch.cat((state,true_action),1)
            true_q_value=self.updata_Q(true_q_input).mean()
            true_q_value=-true_q_value
            true_q_value.backward()
            self.policy_optim.step()
            self.num1=0
        #对网络进行软更新
        if self.num==10:
            for target_param, param in zip(self.target_Q.parameters(), self.updata_Q.parameters()):
                target_param.data.copy_(
                  target_param.data * (1.0 - config.soft_updata) +
                  param.data * config.soft_updata
                )
            for target_param, param in zip(self.target_policy_net.parameters(), self.policy_net.parameters()):
                target_param.data.copy_(
                    target_param.data * (1.0 - config.soft_updata) +
                    param.data * config.soft_updata
                )
            self.num=0


    def train(self):
        self.random_sample()
        state=self.env.reset()
        rewards=0
        count=0
        average=0
        Reward=np.array([])
        while count<=200:
            action=self.policy_net(torch.tensor(state))
            action=self.noise(action)
            next_state,reward,done,_=self.env.step(action)
            self.Buff([state,action,reward,next_state,done])
            if len(self.buff)>=10000:#更新网络
                self.updata()
            state=next_state
            rewards+=reward
            if done:
                count+=1
                state=self.env.reset()
                Reward=np.append(Reward,rewards)
                average=Reward.mean()
                print('第%d回合的总奖励%d' %(count,rewards))
                print('平均奖励%.f' %(average))
                rewards=0
        plt.plot(range(count), Reward)
        plt.xlabel('epoch')
        plt.ylabel('Reward')
        plt.show()
    def test(self):
        state = self.env.reset()
        rewards = 0
        count = 0
        Reward = 0
        Reward = np.array([])
        for _ in range(300):
            state = self.env.reset()
            while True:
                #self.env.render()
                action = self.policy_net(torch.tensor(state))
                action = action.detach().numpy()
                next_state, reward, done, _ = self.env.step(action)
                state = next_state
                rewards += reward
                if done:
                    count += 1
                    Reward =np.append(Reward,rewards)
                    print('第%d回合的总奖励%d' % (count, rewards))
                    rewards = 0
                    break
        plt.plot(range(count), Reward)
        plt.xlabel('epoch')
        plt.ylabel('Reward')
        plt.show()

ddpg=DDPG()
ddpg.train()
#test
ddpg.test()










