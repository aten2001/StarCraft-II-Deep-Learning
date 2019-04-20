import math
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from ConvLSTM import ConvLSTM
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
        super().__init__()
        self.minimapRes = ResNet(minimap_shape[0], ResidualBlock, minimap_shape[1])
        self.screenRes = ResNet(screen_shape[0], ResidualBlock, screen_shape[1])

    def forward(self, minimap, screen):
        minimap = self.minimapRes(minimap)
        screen = self.minimapRes(screen)
        out = torch.cat((minimap, screen), 0)
        return out

class MemoryProcessing(nn.Module):
    def __init__(self, minimap_shape=(7, 64, 64), screen_shape=(17, 64, 64)):
        super().__init__()
        self.SpatialResNet = SpatialResNet()
        self.ConvLSTM = ConvLSTM(input_channels=24, hidden_channels=[96], kernel_size=3)

    def forward(self, minimap, screen):
        input3d = self.SpatialResNet(minimap, screen)
        state = self.ConvLSTM(input3d)
        return state



class MLP(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.mlp = torch.nn.Sequential(
          torch.nn.Linear(in_channels, 128),
          torch.nn.ReLU(),
          torch.nn.Linear(128, 64),
        )

    def forward(self, player, last_action):
        x = np.concatenate((player, last_action))
        out = self.mlp(x)
        return out


class MultiHeadAttention(nn.Module):

    def __init__(self, heads = 1, num_blocks = 1, d_model=32):
        super().__init__()

        self.h = heads
        self.memory = MemoryProcessing()
        self.d_model = d_model
        self.d_k = d_model // heads
        self.q_embedding = nn.Linear(24, d_model)
        self.k_embedding = nn.Linear(24, d_model)
        self.v_embedding = nn.Linear(24, d_model)
        self.output_mlp = nn.Sequential(
            nn.Linear(32, 384),
            nn.ReLU(),
            nn.Linear(384, 32)
        )

    def forward(self, minimap, screen):
        #batch_size = q.size(0)

        outputs3d = self.memory(minimap_shape=minimap.shape, screen_shape=screen.shape)[0]
        # Perform linear operation on slit into h heads, embedding_size
        # more detail http://jalammar.github.io/illustrated-transformer/

        k = copy.deepcopy(outputs3d)
        q = copy.deepcopy(outputs3d)
        v = copy.deepcopy(outputs3d)

        # split into heads, shape:  Sequential_length * head * d_k
        # in our case, should be shape (64, 1, 32)
        k = self.q_embedding(k).view(-1, self.h, self.d_k)
        q = self.q_embedding(q).view(-1, self.h, self.d_k)
        v = self.q_embedding(v).view(-1, self.h, self.d_k)

        # q, k be transposed into shape (h, sl, d_k)
        q = q.transpose(0, 1)
        k = k.transpose(0, 1)
        v= v.transpose(0, 1)

        # Calculate attention
        # scores shape (head, sl, d_k), in our case (1, 64, 32)
        scores = self.attention(q, k, v)

        # Concatenate heads and put through final linear layer
        # after concatenate have shape (sl, d_model)
        concat = scores.transpose(0, 1).contiguous().view(-1, self.d_model)
        output = self.out(concat)
        return output

    def attention(q, k, v, d_k = 32):
        # matmul should be q in shape of (h, sl, dk), k.transpose in shape of (h, dk, sl)
        # more details see https://medium.com/@kolloldas/building-the-mighty-transformer-for-sequence-tagging-in-pytorch-part-i-a1815655cd8
        scores = torch.matmul(q, k.transpose(1, 2)) / math.sqrt(d_k)
        scores = F.softmax(scores, dim=-1)
        # scores have shape (h, sl, sl), in our case (1, 64, 64), v in shape (h, sl, d_k)
        output = torch.matmul(scores, v)
        return output


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
