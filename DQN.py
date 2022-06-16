import random
import math
import numpy as np
import gym
import torch
from torch import nn, optim
import pygame
#该游戏有连续的状态，两个动作
#构造Q网络

class Q_network(nn.Module):
    def __init__(self):
        super(Q_network, self).__init__()
        self.net=nn.Sequential(#定义容器，构造神经网络层
            nn.Linear(4,128),
            nn.Tanh(),
            nn.Linear(128,64),
            nn.ReLU(),
            nn.Linear(64,2),
            nn.ReLU()

        )
    def forward(self,state):
        return self.net(state)#直接调用属性，传入输入
policy_q=Q_network()
print(id(policy_q))
target_q=Q_network()
print(id(target_q))
class game():
    def __init__(self):
        self.buffer_len=10000
        self.repay_buffer=[]
        self.explore_rate=0.1
        self.policy_q=policy_q
        self.target_q=target_q
        self.env = gym.make('CartPole-v1')
        self.a=0.9
        self.criterion = nn.MSELoss()
        self.optim = optim.SGD(policy_q.parameters(), lr=0.01)
    def buffer(self, experiences):
        if len(self.repay_buffer)<self.buffer_len:
            self.repay_buffer.append(experiences)
        else:
            self.repay_buffer.pop(0)#删除第一个列表元素，括号内是索引
            self.repay_buffer.append(experiences)
    def __call__(self, *args, **kwargs):
        #探索环境，放到replay_buffer
        state=self.env.reset()
        flag=0
        action=0
        count=0
        num=0
        #复制网络参数
        for target_param, param in zip(self.target_q.parameters(), self.policy_q.parameters()):  # 复制参数到目标网路targe_net
            target_param.data.copy_(param.data)
        R=0
        average_R=0
        while True:
            if flag==1:
                self.env.render()
            #伊普西龙贪心策略
            if len(self.repay_buffer)>=self.buffer_len/5:
                if np.random.rand()>self.explore_rate:#,0——1的均匀分布，大于伊普西龙时，选择贪心
                    Q_values=self.policy_q(torch.Tensor(state))
                    action=torch.argmax(Q_values)
                    action=action.numpy()
                else:
                    action=self.env.action_space.sample()
            else:
                action=self.env.action_space.sample()
            self.explore_rate-=1e-6#伊普西龙随时间线性递减
            #与环境互动,得到的反馈放入buffer
            next_state,reward,done,_=self.env.step(action)
            self.buffer([state,action,reward,next_state,done])
            state=next_state
            R+=reward
            if done:
                average_R=0.95*average_R+0.05*R
                count+=1
                print('第%d回合的回报：%d' %(count,R))
                print('第%d回合的平均回报:%.4f' %(count,average_R))
                if average_R==400:
                    flag=1
                R=0
                state=self.env.reset()
            if len(self.repay_buffer)>=self.buffer_len/100:#开始训练网络
                self.optim.zero_grad()
                train_data=random.sample(self.repay_buffer,100)
                train_state = np.array([train_[0] for train_ in train_data])#numpy转为张量速度快
                train_action = np.array([[train_[1]] for train_ in train_data])
                train_reward = np.array([[train_[2]] for train_ in train_data])
                train_nextstate = np.array([train_[3] for train_ in train_data])
                train_done = torch.Tensor([[train_[4]] for train_ in train_data])#Tensor默认数据类型为float型
                #处理数据
                train_state=torch.from_numpy(train_state).float()#可以这样改变数据类型
                train_action=torch.from_numpy(train_action).float()
                train_reward=torch.from_numpy(train_reward).float()
                train_nextstate=torch.from_numpy(train_nextstate).float()
                #训练得到神经网络输出
                train_q=self.policy_q(train_state)
                train_q_action=torch.gather(train_q,1,train_action.long().data)#输入的都为张量，1表示横着进行索引，最后一个输入类型为int64即长整型
                train_q_nextstate=self.policy_q(train_nextstate)
                #选择最动作价值最大的
                train_q_nextstate_maxidex=torch.max(train_q_nextstate,dim=1,keepdim=True)[1]#【0】表示取出最大值的张量，【1】表示最大值索引的张量
                train_q_nextstate_max=self.target_q(train_nextstate)
                train_q_nextstate_max=torch.gather(train_q_nextstate_max,1,train_q_nextstate_maxidex.long())
                target_q_values=train_reward+self.a*train_q_nextstate_max*(1-train_done)
                loss=self.criterion(target_q_values,train_q_action)
                loss.backward()
                self.optim.step()
                num+=1
                if average_R>=490:
                    break
            if num==10:
                for target_param, param in zip(self.target_q.parameters(),self.policy_q.parameters()):  # 复制参数到目标网路targe_net
                    target_param.data.copy_(param.data)
                num=0
g=game()
g()
#开始玩游戏
R=0
average_R=0
state=g.env.reset()
while True:
    g.env.render()
    q_value=policy_q(torch.Tensor(state))
    action=torch.argmax(q_value)
    next_state,reward,done,_=g.env.step(action.numpy())
    state=next_state
    R+=reward

    if done:

        average_R=0.95*average_R+0.05*R
        print(average_R,R)
        R = 0
        state=g.env.reset()















