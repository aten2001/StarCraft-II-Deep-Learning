import torch
import torch.nn as nn
import numpy as np
import pandas as pd
'''
ResidualBlock and ResNet credit to https://github.com/yunjey/pytorch-tutorial/blob/master/tutorials/02-intermediate/deep_residual_network/main.py
'''

class RelationalNN(nn.Module):
    def __init__(self, im_size,  hidden_dim):
        super().__init__()
        out_channale = 7
        self.conv1 = nn.Conv2d(7, out_channale, stride=1, kernel_size=2)
        self.conv1 = nn.Conv2d(7, 1, stride=1, kernel_size=2)
    def forward(self, images):
        x = self.conv1(images)
        x = nn.ReLU(x)


# Residual block
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3,stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(in_channels, out_channels, kernel_size=3,stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample
        self.max_pool = nn.MaxPool2d(2, stride = 2)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        out = self.max_pool(out)
        return out

# ResNet
class ResNet(nn.Module):
    def __init__(self, in_channels, block, height, padding = 1, stride = 2, kernel = 4):
        super(ResNet, self).__init__()
        self.in_channels = in_channels
        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=4, stride=stride, padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self.make_layer(block, in_channels)
        self.max_pool = nn.MaxPool2d(2, stride = 2)

    def make_layer(self, block, out_channels, stride=1):
        downsample = None
        if (stride != 1) or (self.in_channels != out_channels):
            downsample = nn.Sequential(
                conv3x3(self.in_channels, out_channels),
                nn.BatchNorm2d(out_channels))
        return block(self.in_channels, out_channels, stride, downsample)

    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out)
        out = self.relu(out)
        out = self.layer1(out)
        out = self.max_pool(out)
        return out

class SpatialResNet(nn.Module):
    def __init__(self, minimap_shape=(7, 64, 64), screen_shape=(17, 64, 64)):
        self.minimapRes = ResNet(minimap_shape[0], ResidualBlock, minimap_shape[1])
        self.screenRes = ResNet(screen_shape[0], ResidualBlock, screen_shape[1])

    def forward(self, minimap, screen):
        minimap = self.minimapRes(minimap)
        screen = self.minimapRes(screen)
        out = torch.cat((minimap, screen), 0)
        return out



class MLP(nn.Module):
    def __init__(self, in_channels):
        self.mlp = torch.nn.Sequential(
          torch.nn.Linear(in_channels, 128),
          torch.nn.ReLU(),
          torch.nn.Linear(128, 64),
        )
    def forward(self, player, last_action):
        x = np.concatenate((player, last_action))
        out = self.mlp(x)
        return out





class QLearner:
    def __init__(self, actions, learning_rate=0.1, reward_decay=0.9, e_greedy=0.9):
        self.actions = actions
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon = e_greedy
        self.q_table = pd.DataFrame(columns=self.actions, dtype=np.float64)

    def choose_action(self, observation):
        self.check_if_state_exists(observation)
        if np.random.uniform() < self.epsilon:
            # Choose best action
            state_action = self.q_table.ix[observation, :]
            # Some actions have the same value
            state_action = state_action.reindex(np.random.permutation(state_action.index))
            action = state_action.idxmax()
        else:
            # choose random action
            action = np.random.choice(self.actions)

        return action

    def learn(self, s, a, r, s_):
        self.check_state_exist(s_)
        self.check_state_exist(s)

        q_predict = self.q_table.ix[s, a]
        q_target = r + self.gamma * self.q_table.ix[s_, :].max()

        # update
        self.q_table.ix[s, a] += self.lr * (q_target - q_predict)

    def check_if_state_exists(self, state):
        if state not in self.q_table.index:
            self.q_table = self.q_table.append(pd.Series([0] * len(self.actions), index=self.q_table.columns, name=state))
