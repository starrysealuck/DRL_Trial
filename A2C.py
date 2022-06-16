import random

import torch
import gym
import numpy as np
import torch.nn as nn
env=gym.make('Pendulum-v1')
#print(env.observation_space)
#print(env.action_space.sample())
#A2C算法,策略网络输出从概率分布中采样得到的，所以必须是同策略的
from torch.distributions import Normal

#超参数
data_epoch=4
discount=0.9

#构建全连接神经网络#Actor策略网络
class network(nn.Module):
    def __init__(self):
        super(network, self).__init__()
        self.linear1=nn.Linear(3,256)
        self.linear2=nn.Linear(256,128)
        self.linear3=nn.Linear(128,64)
        self.linear4=nn.Linear(64,1)
        self.activate=nn.Tanh()
        self.activate2=nn.Softplus()
    def forward(self,x):
        x=self.activate(self.linear1(x))
        x=self.activate(self.linear2(x))
        x=self.activate(self.linear3(x))
        mu=self.activate(self.linear4(x))*2
        std=self.activate2(self.linear4(x))+0.1
        m = Normal(mu, std)
        action = m.sample()
        action=torch.clamp(action,-2,2)
        #print(mu.item(),std.item())
        return action,mu,std
class value_net(nn.Module):#critic网络
    def __init__(self):
        super(value_net, self).__init__()
        self.linear1=nn.Linear(3,200)
        self.linear2=nn.Linear(200,64)
        self.linear3=nn.Linear(64,1)
        self.linear3.weight.data.uniform_(-1e-3, 1e-3)
        self.linear3.bias.data.uniform_(-1e-3, 1e-3)
        self.activate=nn.ReLU()
        self.activate1=nn.Tanh()
    def forward(self,x):
        x = self.activate(self.linear1(x))
        x = self.activate1(self.linear2(x))
        x = self.linear3(x)
        return x

class Agent():
    def __init__(self):
        self.policy_net=network()
        self.value_net=value_net()
        self.target_net=value_net()
        self.optim=torch.optim.Adam(self.policy_net.parameters(),lr=0.0004)
        self.optim_value=torch.optim.Adam(self.value_net.parameters(),lr=0.0008)
        self.critic=nn.MSELoss()
        self.rebuff=[]
        for target_param, param in zip(self.target_net.parameters(), self.value_net.parameters()):  # 复制参数到目标网路targe_net
            target_param.data.copy_(param.data)
        self.flag=0
    def buff(self,experience):
        if len(self.rebuff)<100000:
            self.rebuff.append(experience)
        else:
            self.rebuff.pop(0)
            self.rebuff.append(experience)


    def updata(self,states,actions,next_state,rewards,dones):
        self.optim.zero_grad()
        states=torch.tensor(states,dtype=torch.float32)
        actions=torch.tensor(actions,dtype=torch.float32)
        next_state = torch.tensor(next_state, dtype=torch.float32)
        rewards = torch.tensor(rewards, dtype=torch.float32)
        dones=torch.tensor(dones, dtype=torch.float32)
        critic_return=0.9*self.value_net(next_state)*(1-dones)+rewards-self.value_net(states)
        for i in range(len(states)):
            mu,std=self.policy_net(states[i])[1:]
            m=Normal(mu,std)
            losses=-m.log_prob(actions[i])*critic_return[i]
            losses.backward(retain_graph=True)
        self.optim.step()



def collect_data():
    state=env.reset()
    State=[]
    actions=[]
    rewards=[]
    dones=[]
    next_State=[]
    #env.render()
    num=0
    while True:

        action=agent.policy_net(torch.Tensor(state))[0]
        action=action.detach().numpy()
        next_state,reward,done,_=env.step(action)
        agent.buff([state,next_state,reward,done])
        State.append(state)
        actions.append([action])
        rewards.append([reward])
        dones.append([done])
        next_State.append(next_state)
        num+=1
        #critic更新
        if len(agent.rebuff)>=10000:

            agent.optim_value.zero_grad()
            #从经验池中拿数据
            traindata=random.sample(agent.rebuff,100)
            #转换成array
            train_state=np.array([exp[0] for exp in traindata])
            train_nextstate=np.array([exp[1] for exp in traindata])
            train_reward=np.array([[exp[2]] for exp in traindata])
            train_done=np.array([[exp[3]] for exp in traindata])
            train_state=train_state.astype('float32')
            train_nextstate= train_nextstate.astype('float32')
            train_reward=train_reward.astype('float32')
            train_done=train_done.astype('float32')

            #转换成张量
            train_state=torch.from_numpy(train_state)
            train_nextstate=torch.from_numpy(train_nextstate)
            train_reward=torch.from_numpy(train_reward)
            train_done = torch.from_numpy(train_done)
            #训练网络
            train_value=agent.value_net(train_state)
            #print(train_value.data)
            target_value=train_reward+0.9*agent.target_net(train_nextstate)*(1-train_done)
            loss=agent.critic(target_value,train_value)
            loss.backward()
            agent.optim_value.step()
            if num==10:
                for target_param, param in zip(agent.target_net.parameters(), agent.value_net.parameters()):  # 复制参数到目标网路targe_net
                    target_param.data.copy_(param.data)


        if done:
            #print(actions)
            break
        state=next_state
    return State,actions,rewards,next_State,dones


def train():
    total_reward=0
    count=0
    average_reward=0
    state=env.reset()
    while len(agent.rebuff)<=10000:
        action=env.action_space.sample()
        next_state,reward,done,_=env.step(action)
        agent.buff([state, next_state, reward, done])
        if done:
            state=env.reset()
        state=next_state


    while True:
        count+=1
        states,actions,rewards,next_State,dones=collect_data()
        agent.updata(states, actions,next_State,rewards,dones)
        rewards=np.array(rewards)
        total_reward=rewards.sum()
        average_reward=0.9*average_reward+0.1*total_reward
        print('回合%d  获得总奖励%d' %(count,total_reward))
        print('平均奖励%.4lf' %(average_reward))



agent=Agent()
train()












