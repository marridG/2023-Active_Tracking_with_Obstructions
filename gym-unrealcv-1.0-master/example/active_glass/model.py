from __future__ import division
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from torch.autograd import Variable
from utils import norm_col_init, weights_init, weights_init_mlp, normal
import perception
import numpy as np
import math


def build_model(obs_space, action_space, args, device):
    model = A3C_Dueling(obs_space, action_space, args, device)
    model.train()
    return model


def wrap_action(self, action):
    action = np.squeeze(action)
    out = action * (self.action_high - self.action_low)/2 + (self.action_high + self.action_low)/2.0
    return out


def sample_action(continuous, mu_multi, sigma_multi, device, test=False):
    if continuous:
        mu = torch.clamp(mu_multi, -1.0, 1.0)
        sigma = F.softplus(sigma_multi) + 1e-5
        eps = torch.randn(mu.size())
        pi = np.array([math.pi])
        pi = torch.from_numpy(pi).float()
        eps = Variable(eps).to(device)
        pi = Variable(pi).to(device)
        action = (mu + sigma.sqrt() * eps).data
        act = Variable(action)
        prob = normal(act, mu, sigma, device)
        action = torch.clamp(action, -1.0, 1.0)
        entropy = 0.5 * ((sigma * 2 * pi.expand_as(sigma)).log() + 1)  # 0.5 * (log(2*pi*sigma) + 1
        log_prob = (prob + 1e-6).log()
        action_env = action.cpu().numpy()
    else:  # discrete
        logit = mu_multi
        prob = F.softmax(logit, dim=1)
        log_prob = F.log_softmax(logit, dim=1)
        entropy = -(log_prob * prob).sum(1, keepdim=True)
        if test:
            action = prob.max(1)[1].data
        else:
            action = prob.multinomial(1).data
            log_prob = log_prob.gather(1, Variable(action))
        action_env = np.squeeze(action.cpu().numpy())

    return action_env, entropy, log_prob


class ValueNet(nn.Module):
    def __init__(self, input_dim):
        super(ValueNet, self).__init__()
        self.critic_linear = nn.Linear(input_dim, 1)
        self.critic_linear.weight.data = norm_col_init(self.critic_linear.weight.data, 0.01)
        self.critic_linear.bias.data.fill_(0)

    def forward(self, x):
        value = self.critic_linear(x)
        return value


class PolicyNet(nn.Module):
    def __init__(self, input_dim, action_space, head_name, device):
        super(PolicyNet, self).__init__()
        self.head_name = head_name
        self.device = device
        if 'continuous' in head_name:
            num_outputs = action_space.shape[0]
            self.continuous = True
        else:
            num_outputs = action_space.n
            self.continuous = False

        self.actor_linear = nn.Linear(input_dim, num_outputs)
        if self.continuous:
            self.actor_linear2 = nn.Linear(input_dim, num_outputs)

        # init layers
        self.actor_linear.weight.data = norm_col_init(self.actor_linear.weight.data, 0.01)
        self.actor_linear.bias.data.fill_(0)
        if self.continuous:
            self.actor_linear2.weight.data = norm_col_init(self.actor_linear2.weight.data, 0.01)
            self.actor_linear2.bias.data.fill_(0)

    def forward(self, x, test=False):
        if self.continuous:
            mu = F.softsign(self.actor_linear(x))
            sigma = self.actor_linear2(x)
        else:
            mu = self.actor_linear(x)
            sigma = torch.ones_like(mu)

        action, entropy, log_prob = sample_action(self.continuous, mu, sigma, self.device, test)
        return action, entropy, log_prob


class A3C(torch.nn.Module):
    def __init__(self, obs_space, action_space, rnn_out=128, head_name='cnn_lstm',  stack_frames=1, sub_task=False, device=None):
        super(A3C, self).__init__()
        self.sub_task = sub_task
        self.head_name = head_name
        if 'cnn' in head_name:
            self.encoder = perception.CNN_simple(obs_space, stack_frames)
        if 'icml' in head_name:
            self.encoder = perception.ICML(obs_space, stack_frames)
        if 'maze' in head_name:
            self.encoder = perception.CNN_maze(obs_space, stack_frames)
        feature_dim = self.encoder.outdim

        if 'lstm' in head_name:
            self.lstm = nn.LSTMCell(feature_dim, rnn_out)
            self.lstm.bias_ih.data.fill_(0)
            self.lstm.bias_hh.data.fill_(0)
            feature_dim = rnn_out
        if 'gru' in head_name:
            self.lstm = nn.GRUCell(feature_dim, rnn_out)
            self.lstm.bias_ih.data.fill_(0)
            self.lstm.bias_hh.data.fill_(0)
            feature_dim = rnn_out

        #  create actor
        self.actor = PolicyNet(feature_dim, action_space, head_name, device)
        self.critic = ValueNet(feature_dim)

        self.apply(weights_init)
        self.train()

    def forward(self, inputs, test=False):
        x, (hx, cx) = inputs
        feature = self.encoder(x)
        if 'lstm' in self.head_name:
            hx, cx = self.lstm(feature, (hx, cx))
            feature = hx
        if 'gru' in self.head_name:
            hx = self.lstm(feature, hx)
            feature = hx
        value = self.critic(feature)
        action, entropy, log_prob = self.actor(feature, test)

        return value, action, entropy, log_prob, (hx, cx)


class TAT(torch.nn.Module):  # Tracker-aware Target
    def __init__(self, obs_space, action_space, rnn_out=128, head_name='cnn_lstm',  stack_frames=1, dim_action_tracker=-1, device=None):
        super(TAT, self).__init__()
        if dim_action_tracker > 0:
            self.sub_task = True
        else:
            self.sub_task = False
        self.head_name = head_name
        if 'cnn' in head_name:
            self.encoder = perception.CNN_simple(obs_space, stack_frames)
        if 'icml' in head_name:
            self.encoder = perception.ICML(obs_space, stack_frames)
        if 'maze' in head_name:
            self.encoder = perception.CNN_maze(obs_space, stack_frames)
        feature_dim = self.encoder.outdim

        if 'lstm' in head_name:
            self.lstm = nn.LSTMCell(feature_dim, rnn_out)
            self.lstm.bias_ih.data.fill_(0)
            self.lstm.bias_hh.data.fill_(0)
            feature_dim = rnn_out
        if 'gru' in head_name:
            self.lstm = nn.GRUCell(feature_dim, rnn_out)
            self.lstm.bias_ih.data.fill_(0)
            self.lstm.bias_hh.data.fill_(0)
            feature_dim = rnn_out

        #  create actor
        self.actor = PolicyNet(feature_dim, action_space, head_name, device)
        self.critic = ValueNet(feature_dim)

        self.fc_action_tracker = nn.Linear(dim_action_tracker, self.encoder.outdim)
        weights_init_mlp(self.fc_action_tracker)
        # create sub-task
        if self.sub_task:
            self.reward_aux = nn.Linear(feature_dim, 1)
            self.reward_aux.weight.data = norm_col_init(self.reward_aux.weight.data, 0.01)
            self.reward_aux.bias.data.fill_(0)

        self.apply(weights_init)
        self.train()

    def forward(self, inputs, test=False):
        x, (hx, cx), action_tracker = inputs
        feature = self.encoder(x)
        f_a_stu = self.fc_action_tracker(action_tracker)
        feature = feature + f_a_stu
        if 'lstm' in self.head_name:
            hx, cx = self.lstm(feature, (hx, cx))
            feature = hx
        if 'gru' in self.head_name:
            hx = self.lstm(feature, hx)
            feature = hx

        value = self.critic(feature)
        action, entropy, log_prob = self.actor(feature, test)

        R_pred = None
        if self.sub_task:
            R_pred = self.reward_aux(feature)

        return value, action, entropy, log_prob, (hx, cx), R_pred


class A3C_Dueling(torch.nn.Module):
    def __init__(self, obs_space, action_space, args, device=None):
        super(A3C_Dueling, self).__init__()
        self.num_agents = len(obs_space)
        obs_shapes = [obs_space[i].shape for i in range(self.num_agents)]   # [(1,80,80)]
        stack_frames = args.stack_frames
        rnn_out = args.rnn_out  # 128
        head_name = args.network
        self.single = args.single   # True
        self.device = device
        if 'continuous' in head_name:
            self.continuous = True
            self.action_dim_tracker = action_space[0].shape[0]
        else:
            self.continuous = False
            self.action_dim_tracker = action_space[0].n 
        self.player0 = A3C(obs_shapes[0], action_space[0], rnn_out, head_name, stack_frames, device=device)
        if not self.single:
            if 'tat' in head_name:
                self.tat = True
                self.player1 = TAT(obs_shapes[1], action_space[1], rnn_out,
                                   head_name, stack_frames*2, self.action_dim_tracker, device=device)
            else:
                self.tat = False
                self.player1 = A3C(obs_shapes[1], action_space[1], rnn_out, head_name, stack_frames, device=device)

    def forward(self, inputs, test=False):
        states, (hx, cx) = inputs

        # run tracker
        value0, action_0, entropy_0, log_prob_0, (hx_0, cx_0) = self.player0((states[0], (hx[:1], cx[:1])), test)

        if self.single or states.shape[0] == 1:
            return value0, [action_0], entropy_0, log_prob_0, (hx_0, cx_0), 0

        # run target
        R_pred = 0
        if self.tat:
            if self.continuous:  # onehot action
                action2target = torch.Tensor(action_0.squeeze())
            else:
                action2target = torch.zeros(self.action_dim_tracker)
                action2target[action_0] = 1
            state_target = torch.cat((states[0], states[1]), 0)
            value1, action_1, entropy_1, log_prob_1, (hx1, cx1), R_pred = self.player1(
                (state_target, (hx[1:], cx[1:]), action2target.to(self.device)), test)
        else:
            value1, action_1, entropy_1, log_prob_1, (hx1, cx1) = self.player1((states[1], (hx[1:], cx[1:])), test)
        entropies = torch.cat([entropy_0, entropy_1])
        log_probs = torch.cat([log_prob_0, log_prob_1])
        hx_out = torch.cat((hx_0, hx1))
        cx_out = torch.cat((cx_0, cx1))

        return torch.cat([value0, value1]), [action_0, action_1], entropies, log_probs, (hx_out, cx_out), R_pred


class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)


class ICMModel(nn.Module):
    def __init__(self, output_size, device):
        super(ICMModel, self).__init__()

        # self.input_size = input_size
        self.output_size = output_size  # action_space中action个数
        self.device = device

        feature_output = 7 * 7 * 64
        self.feature = nn.Sequential(
            nn.Conv2d(
                in_channels=2,
                out_channels=32,
                kernel_size=8,
                stride=4,
                padding=2),
            nn.LeakyReLU(),
            nn.Conv2d(
                in_channels=32,
                out_channels=64,
                kernel_size=4,
                stride=2),
            nn.LeakyReLU(),
            nn.Conv2d(
                in_channels=64,
                out_channels=64,
                kernel_size=3,
                stride=1),
            nn.LeakyReLU(),
            Flatten(),
            nn.Linear(feature_output, 512)
        ).to(device)

        self.inverse_net = nn.Sequential(
            nn.Linear(512 * 2, 512),
            nn.ReLU(),
            nn.Linear(512, output_size)
        ).to(device)

        self.residual = [nn.Sequential(
            nn.Linear(output_size + 512, 512),
            nn.LeakyReLU(),
            nn.Linear(512, 512),
        ).to(device)] * 8

        self.forward_net_1 = nn.Sequential(
            nn.Linear(output_size + 512, 512),
            nn.LeakyReLU()
        ).to(device)
        self.forward_net_2 = nn.Sequential(
            nn.Linear(output_size + 512, 512),
        ).to(device)

        for p in self.modules():
            if isinstance(p, nn.Conv2d):
                init.kaiming_uniform_(p.weight)
                p.bias.data.zero_()

            if isinstance(p, nn.Linear):
                init.kaiming_uniform_(p.weight, a=1.0)
                p.bias.data.zero_()

    def forward(self, inputs):
        """
        inputs = [state, next_state, action]
        state: 1 * 1 * 80 * 80
        next_state: 1 * 1 * 80 * 80
        action: 1 * output_size,其中每行为one-hot向量
        返回real_next_state_feature, pred_next_state_feature, pred_action
        real_next_state_feature: num_worker * 512
        pred_next_state_feature: num_worker * 512
        pred_action: num_worker * output_size
        """
        state, next_state, action = inputs

        # print(state.shape)
        channel_num = state.shape[-3]
        h = state.shape[-2]
        w = state.shape[-1]
        # _, _, channel_num, h, w = state.shape

        state = state.reshape(-1, channel_num, h, w)
        next_state = next_state.reshape(-1, channel_num, h, w)

        # test_feature = self.feature(torch.zeros(1, 1, 80, 80).to(self.device))
        encode_state = self.feature(state)  # num_worker * 512
        encode_next_state = self.feature(next_state)    # num_worker * 512
        # get pred action
        pred_action = torch.cat((encode_state, encode_next_state), 1)
        pred_action = self.inverse_net(pred_action) # num_worker * output_size
        # ---------------------

        # get pred next state
        pred_next_state_feature_orig = torch.cat((encode_state, action), 1)
        pred_next_state_feature_orig = self.forward_net_1(pred_next_state_feature_orig) # num_worker * 512

        # residual
        for i in range(4):
            pred_next_state_feature = self.residual[i * 2](torch.cat((pred_next_state_feature_orig, action), 1))
            pred_next_state_feature_orig = self.residual[i * 2 + 1](
                torch.cat((pred_next_state_feature, action), 1)) + pred_next_state_feature_orig

        pred_next_state_feature = self.forward_net_2(torch.cat((pred_next_state_feature_orig, action), 1))
        # num_worker * 512
        real_next_state_feature = encode_next_state
        return real_next_state_feature, pred_next_state_feature, pred_action

