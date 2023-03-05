from __future__ import division
import numpy as np
import torch
from torch.autograd import Variable
from torch.nn import L1Loss
from utils import ensure_shared_grads, RunningMeanStd, RewardScaling
from model import ICMModel
import torch.nn.functional as F
import torch.nn as nn


class Agent(object):
    def __init__(self, model, env, args, state, device, action_space_len):
        self.model = model
        self.env = env
        self.num_agents = len(env.observation_space)
        # print(self.env.observation_space)
        if 'continuous' in args.network:
            self.action_high = [env.action_space[i].high for i in range(self.num_agents)]
            self.action_low = [env.action_space[i].low for i in range(self.num_agents)]
            self.dim_action = env.action_space[0].shape[0]
        else:
            self.dim_action = 1

        self.eps_len = 0
        self.args = args
        self.values = []
        self.log_probs = []
        self.rewards = []
        self.entropies = []
        self.preds = []
        self.done = True
        self.info = None
        self.reward = 0
        self.device = device
        self.rnn_out = args.rnn_out
        self.num_steps = 0
        self.n_steps = 0
        self.state = state
        self.hxs = torch.zeros(self.num_agents, self.rnn_out).to(device)
        self.cxs = torch.zeros(self.num_agents, self.rnn_out).to(device)

        self.action_space_len = action_space_len    # action_space中action个数
        self.reward_rsc = RewardScaling(shape=(self.num_agents, 1))
        if args.intrinsic_reward:
            self.icm = ICMModel(device=device, output_size=self.action_space_len)
            self.eta = 0.01
            self.intrinsic_rewards = []
            self.actions = []
            self.states = []
            self.next_states = []

    def wrap_action(self, action, high, low):
        action = np.squeeze(action)
        out = action * (high - low)/2.0 + (high + low)/2.0
        return out

    def action_train(self):
        self.n_steps += 1
        value_multi, action_env_multi, entropy, log_prob, (self.hxs, self.cxs), R_pred = self.model(
            (Variable(self.state, requires_grad=True), (self.hxs, self.cxs)))

        if 'continuous' in self.args.network:
            action_env_multi = [self.wrap_action(action_env_multi[i], self.action_high[i], self.action_low[i])
                                for i in range(self.num_agents)]

        # model return action_env_multi, entropy, log_prob
        state_multi, reward_multi, self.done, self.info = self.env.step(action_env_multi)
        # state_multi: (1,1,1,80,80)

        # compute intrinsic reward
        if self.args.intrinsic_reward:
            intrinsic_reward = self.compute_intrinsic_reward(self.state, state_multi, action_env_multi)
            # print(f"action_train intrinsic_reward: {intrinsic_reward.detach().cpu()}")
            self.intrinsic_rewards.append(intrinsic_reward)
            self.actions.append(action_env_multi)
            self.states.append(self.state)
            self.next_states.append(torch.from_numpy(state_multi).float().to(self.device))

        # add to buffer
        self.reward_org = reward_multi.copy()
        self.reward = torch.tensor(reward_multi).float().to(self.device)
        self.state = torch.from_numpy(state_multi).float().to(self.device)
        self.eps_len += 1
        self.values.append(value_multi)

        self.entropies.append(entropy)
        self.log_probs.append(log_prob)
        self.rewards.append(self.reward.unsqueeze(1))
        self.preds.append(R_pred)
        return self

    def action_test(self):
        with torch.no_grad():
            value_multi, action_env_multi, entropy, log_prob, (self.hxs, self.cxs), R_pred = self.model(
                (Variable(self.state), (self.hxs, self.cxs)), True)

        if 'continuous' in self.args.network:
            action_env_multi = [self.wrap_action(action_env_multi[i], self.action_high[i], self.action_low[i])
                                for i in range(self.num_agents)]

        state_multi, self.reward, self.done, self.info = self.env.step(action_env_multi)
        if self.env.glass_detect_dir == 'eval':
            TRA_DIR = '/data3/songmingyu/glass/gym-unrealcv-1.0/example/glass/out.csv'
            from example.utils import io_util
            io_util.save_trajectory(self.info, TRA_DIR, 0)

        self.state = torch.from_numpy(state_multi).float().to(self.device)
        self.eps_len += 1
        return self

    def reset(self):
        self.state = torch.from_numpy(self.env.reset()).float().to(self.device)
        self.num_agents = self.state.shape[0]
        self.eps_len = 0
        self.reset_rnn_hiden()

    def clear_actions(self):
        self.values = []
        self.log_probs = []
        self.rewards = []
        self.entropies = []
        self.preds = []
        self.intrinsic_rewards = []
        self.actions = []
        self.states = []
        self.next_states = []
        return self

    def reset_rnn_hiden(self):
        self.cxs = torch.zeros(self.num_agents, self.rnn_out).to(self.device)
        self.hxs = torch.zeros(self.num_agents, self.rnn_out).to(self.device)
        self.cxs = Variable(self.cxs)
        self.hxs = Variable(self.hxs)

    def update_rnn_hiden(self):
        self.cxs = Variable(self.cxs.data)
        self.hxs = Variable(self.hxs.data)

    def optimize(self, params, optimizer, shared_model, training_mode, device_share):
        print('optimizing!')
        R = torch.zeros(self.num_agents, 1).to(self.device)
        if not self.done:
            # predict value
            state = self.state
            value_multi, _, _, _, _, _ = self.model(
                (Variable(state, requires_grad=True), (self.hxs, self.cxs)))
            for i in range(self.num_agents):
                R[i][0] = value_multi[i].data
        self.values.append(Variable(R).to(self.device))
        policy_loss = torch.zeros(self.num_agents, 1).to(self.device)
        value_loss = torch.zeros(self.num_agents, 1).to(self.device)
        pred_loss = torch.zeros(1, 1).to(self.device)
        entropies = torch.zeros(self.num_agents, self.dim_action).to(self.device)
        w_entropies = float(self.args.entropy)*torch.ones(self.num_agents, self.dim_action).to(self.device)
        if self.num_agents > 1:
            w_entropies[1:][:] = float(self.w_entropy_target)
        R = Variable(R, requires_grad=True).to(self.device)
        gae = torch.zeros(1, 1).to(self.device)
        l1_loss = L1Loss()

        # reward scaling
        
        self.reward_rsc.reset()
        # print(self.rewards[0].detach())
        # print(self.rewards[0].cpu().numpy())
        # test_reward = self.reward_rsc(self.rewards[0].cpu().numpy())
        # self.rewards[0] = torch.from_numpy(test_reward).to(self.device)
        # print(self.rewards[0].detach())
        rescale_rewards = [self.reward_rsc(x.detach().cpu().numpy()) for x in self.rewards]
        l = len(self.rewards)
        for i in range(l):
            self.rewards[i] = torch.from_numpy(rescale_rewards[i]).to(self.device)
        # self.rewards = [torch.from_numpy(r).to(self.deivce) for r in rescale_rewards] 
        
        advs1 = []
        advs2 = []
        for i in reversed(range(len(self.rewards))):
            if 'reward' in self.args.aux:
                pred_loss = pred_loss + l1_loss(self.preds[i][0], self.rewards[i][0])
            R = self.args.gamma * R + self.rewards[i]
            # if self.args.intrinsic_reward:
            #    R = R + self.intrinsic_rewards[i]
            advantage = R - self.values[i]
            advs1.append(advantage)
            advs2.append(advantage.detach().cpu().numpy())
        
        # adv normalization
        advs2 = np.array(advs2)
        mean = np.mean(advs2)
        std = np.mean(advs2) + 1e-8
        advs1 = [(advantage - mean) / std for advantage in advs1]
        

        for i in reversed(range(len(self.rewards))):
            value_loss = value_loss + 0.5 * advs1[i].pow(2)
            # Generalized Advantage Estimataion
            delta_t = self.rewards[i] + self.args.gamma * self.values[i + 1].data - self.values[i].data
            gae = gae * self.args.gamma * self.args.tau + delta_t
            policy_loss = policy_loss - \
                (self.log_probs[i] * Variable(gae)) - \
                (w_entropies * self.entropies[i]) # w_entropies=0.2
            entropies += self.entropies[i]

        self.model.zero_grad()
        loss_tracker = (policy_loss[0] + 0.5 * value_loss[0]).mean() ###
        if self.num_agents > 1:
            loss_target = (policy_loss[1] + 0.5 * value_loss[1]).mean()

        if training_mode == 0:  # train tracker
            loss = loss_tracker
        elif training_mode == 1:  # train target
            loss = loss_target
        else:
            loss = loss_tracker + loss_target
        if 'reward' in self.args.aux and training_mode != 0:
            loss += pred_loss.mean()

        if self.args.intrinsic_reward:
            true_actions = torch.LongTensor(np.array(self.actions)).reshape(-1).to(self.device)
            # print(len(true_actions), true_actions.size())
            action_onehot = torch.FloatTensor(len(self.actions), self.action_space_len).to(self.device)
            action_onehot.zero_()
            action_onehot.scatter_(1, true_actions.view(-1, 1), 1)

            states = torch.stack(self.states)
            next_states = torch.stack(self.next_states)
            real_next_state_feature, pred_next_state_feature, pred_action = self.icm(
                [states, next_states, action_onehot]
            )

            ce = nn.CrossEntropyLoss().to(self.device)
            forward_mse = nn.MSELoss().to(self.device)
            inverse_loss = ce(pred_action, true_actions)
            forward_loss = forward_mse(pred_next_state_feature, real_next_state_feature)
            loss = loss + inverse_loss + forward_loss
            print("use intrinsic reward to optimize")

        loss.backward()

        # torch.nn.utils.clip_grad_norm_(params, 50)
        ensure_shared_grads(self.model, shared_model, self.device, device_share)

        optimizer.step()
        self.clear_actions()
        print('optimizing finish!')
        return policy_loss, value_loss, entropies, pred_loss
    
    def compute_intrinsic_reward(self, state, next_state, action):
        """
        action: (1,2), [array(2)]
        next_state: (1,1,1,80,80), array()
        state: (1,1,1,80,80), tensor, dtype=float32
        """
        state = state.to(self.device)
        next_state = torch.FloatTensor(next_state).to(self.device)
        action = torch.LongTensor(np.array(action)).to(self.device)
        '''
        h = state.size(-2)  # 80
        w = state.size(-1)  # 80
        
        state = state.reshape(-1, 1, h, w)
        next_state = next_state.reshape(-1, 1, h, w)
        print(state.size())  # (1, 1, 80, 80)
        print(next_state.size())    # (1, 1, 80, 80)
        print(action.size())    # (1,)
        '''

        action_onehot = torch.FloatTensor(
            len(action), self.action_space_len).to(
            self.device)
        action_onehot.zero_()
        action_onehot.scatter_(1, action.view(-1, 1), 1)
        # print(action_onehot.size())

        real_next_state_feature, pred_next_state_feature, pred_action = self.icm(
            [state, next_state, action_onehot])
        intrinsic_reward = self.eta * F.mse_loss(real_next_state_feature, pred_next_state_feature, reduction='none').mean(-1)
        return intrinsic_reward
