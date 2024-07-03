import gymnasium as gym
import traceback
import argparse
import numpy as np
from time import time

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
import torch.optim.lr_scheduler as scheduler
from torch.utils.tensorboard import SummaryWriter

from utils import Logger, ReplayBuffer


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-episode", type=int, default=500)
    parser.add_argument("--test-episode", type=int, default=5)
    parser.add_argument("--stop", type=int, default=300, help="Earily stop score")

    parser.add_argument("--batch-size", type=int, default=64, help="Batch size for update")
    parser.add_argument("--update-step", type=int, default=10, help="Loop update steps")
    
    parser.add_argument("--gamma", type=float, default=0.9)
    parser.add_argument("--lmbda", type=float, default=0.95)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--seed", type=int, default=0)
    
    parser.add_argument("--show", type=int, default=0, help="0 for save gif, 1 for show windows")
    parser.add_argument("--last", type=int, default=0, help="0 for new, 1 for last time")

    parser.add_argument("--tag", type=str, default='CP_PPO')

    args = parser.parse_known_args()[0]
    return args


class Network(nn.Module):
    def __init__(self, in_dim, hid_dim, out_dim):
        super().__init__()

        self.fea_net = nn.Sequential(
            nn.Linear(in_dim, hid_dim),
            nn.ReLU(),
            nn.Linear(hid_dim, hid_dim),
            nn.ReLU(),
        )

        self.actor = nn.Sequential(
            nn.Linear(hid_dim, hid_dim),
            nn.ReLU(),
            nn.Linear(hid_dim, out_dim),
            nn.Softmax(dim=-1),
        )
        self.critic = nn.Sequential(
            nn.Linear(hid_dim, hid_dim),
            nn.ReLU(),
            nn.Linear(hid_dim, 1),
        )
        
    def forward(self, x, with_value=True):
        fea = self.fea_net(x)
        policy = Categorical(self.actor(fea))
        if with_value:
            value = self.critic(fea)
            return policy, value
        return policy


class Agent:
    """A2C Agent interacting with environment."""
    def __init__(
            self,
            env: gym.Env,
            buf: ReplayBuffer,
            gamma: float = 0.9,
            lmbda: float = 1,
            epsilon: float = 0.2,
            learning_rate: float = 3e-4,
            logging: Logger = None,
            device = None,
            seed = 0,
    ):
        self.random = np.random.RandomState(seed=seed)
        self._greedy = False

        action_num = env.action_space.n
        observation_dim = env.observation_space.shape[0]

        self.env = env
        self.buf = buf
        self.lmbda = lmbda
        self.gamma = gamma
        self.epsilon = epsilon

        self.device = device if device is not None else \
                      torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f'Load in {self.device}')

        self.net = Network(observation_dim, 64, action_num).to(self.device)
        self.optimizer = optim.Adam(self.net.parameters(), lr=learning_rate)
        self.scheduler = scheduler.StepLR(self.optimizer, step_size=100, gamma=0.8)

        self.logging = Logger('./log', 'ppo') if logging is None else logging

    def select(self, state: np.ndarray) -> np.ndarray:
        """Select an action from the input state."""
        policy, value = self.net(self._state_process(state))
        if self._greedy:
            action = policy.probs.argmax().detach()
        else:
            action = policy.sample().squeeze()
        return {
            'action': action.cpu().numpy(), 
            'log_prob': policy.log_prob(action), 
            'entropy': policy.entropy()}, \
            value

    def step(self, action: np.ndarray):
        """Take an action and return the response of the env."""
        next_state, reward, terminated, truncated, info = self.env.step(action)
        done = terminated or truncated
        return next_state, reward, done, info

    def update_model(self, update_step: int):
        exp = self.buf.sample(shuffle=False)
        states = exp['state'] # [B, H]
        actions = exp['action'] # [B]
        rewards = exp['reward'] # [B]
        next_states = exp['next_state'] # [B, H]
        dones = exp['done'] # [B]

        # [B] -> [B,1]
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(self.device)
        actions = torch.FloatTensor(actions).unsqueeze(1).to(self.device)

        policies, values = self.net(self._state_process(states))
        _, next_values = self.net(self._state_process(next_states))
        next_values = next_values.detach()

        log_probs_old = policies.log_prob(actions)
        targets = rewards + self.gamma * next_values * (1-dones)
        # detach
        log_probs_old, targets = log_probs_old.detach(), targets.detach()

        # cal the gae
        td_err = targets - values
        gae = torch.zeros_like(td_err)
        for t in reversed(range(len(td_err) - 1)):
            gae[t] = td_err[t] + self.gamma * self.lmbda * gae[t+1]
        gae = gae.detach()

        losses = []
        for _ in range(update_step):
            policies, values = self.net(self._state_process(states))
            log_probs = policies.log_prob(actions)
            entropy = policies.entropy()

            # pi / pi_old
            ratio = torch.exp(log_probs - log_probs_old)
            
            surr1 = ratio * gae
            surr2 = torch.clamp(ratio, 1-self.epsilon, 1+self.epsilon) * gae

            actor_loss  = -torch.min(surr1, surr2)
            critic_loss  = F.mse_loss(values, targets)
            loss = actor_loss + 0.5 * critic_loss - 0.02*entropy
            loss = loss.mean()

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            losses.append(loss.item())

        return np.mean(losses)

    def train(self, num_episode: int, update_step: int, stop_score=np.inf, show=True):
        """Train the agent."""
        self.greedy(False)
        losses, scores = [], []
        steps = 0

        for episode in range(1, num_episode+1):
            state, info = self.env.reset(seed=self.random.randint(1e9))

            score = num_action = 0
            done = False
            while not done:
                if show:
                    self.env.render()
                num_action += 1
                steps += 1

                action_info, value = self.select(state)
                action = action_info['action']
                next_state, reward, done, info = self.step(action)

                self.buf.store(
                    state=state,
                    action=action,
                    reward=reward, 
                    next_state=next_state, 
                    done=done)

                score += reward

                # collected a batch exp
                if self.buf.isfull():
                    loss = self.update_model(update_step)
                    self.logging.save_tensorboard('train-loss', loss, steps)
                    losses.append(loss)

                    self.buf.clear()

                state = next_state

                if done:
                    self.scheduler.step()
                    scores.append(score)
                    self.logging.save_tensorboard('train-score', score, episode)

                    if episode % 10 == 0:
                        self.logging.save_model(self.net)
            if episode % 10 == 0:
                print(f"[Train] The episode {episode} finished, " + \
                      f"got [{min(scores[-10:])}, {max(scores[-10:])}] score.")
            if np.mean(scores[-5:]) > stop_score:
                print(f'[Train] Get score {np.mean(scores[-5:])} (greater than {stop_score}), earily stop in Ep {episode}.')
                self.logging.save_model(self.net)
                break

    def test(self, num_episode_test, load_last = True, show=False):
        self.greedy(True)
        # load the last model to test
        if load_last:
            print(f'[Test] load last mode in {self.logging.date}')
            self.logging.load_model(self.net)

        frames, scores = [], []
        for episode in range(1, num_episode_test+1):
            state, info = self.env.reset(seed=self.random.randint(1e9))

            done = False
            score = 0
            while not done:
                if show:
                    self.env.render()
                else:
                    frames.append(self.env.render())
                action_info, _ = self.select(state)
                action = action_info['action']
                next_state, reward, done, info = self.step(action)
                score += reward
                state = next_state
            scores.append(score)
            print(f"[Test] The episode {episode} got {score} score.")
        if not show:
            self.logging.save_frames_as_gif(frames, info='')
        return np.mean(scores)

    def greedy(self, greedy_model: bool):
        self._greedy = greedy_model
        self.net.train(not greedy_model)

    def _state_process(self, state) -> torch.Tensor:
        if not isinstance(state, np.ndarray):
            return state
        
        # norm
        state = state / [2.4, 1, 0.21, 1]
        state = torch.FloatTensor(state)
        # only one state, reshape from [H] to [1, H]
        if len(state.shape) == 1:
            state.unsqueeze(0)
            
        return state.to(self.device)


def seed_torch(seed):
    torch.manual_seed(seed)
    if torch.backends.cudnn.enabled:
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.determinstic = True


if __name__ == '__main__':
    args = get_args()

    env = gym.make('CartPole-v1', render_mode='human' if args.show else 'rgb_array')

    if args.last:
        logging = Logger('log', args.tag, load_last=True)
    else:
        logging = Logger('log', args.tag, load_last=False)
        logging.save_argparse(args)
        logging.save_runtime()

    buf = ReplayBuffer(buf_size=args.batch_size)

    buf.seed(args.seed)
    seed_torch(args.seed)
    
    agent = Agent(env, buf, 
                  gamma=args.gamma, learning_rate=args.lr, lmbda=args.lmbda,
                  logging=logging, seed=args.seed)

    try:
        if not args.last:
            agent.train(args.train_episode, args.update_step, stop_score=args.stop, show=args.show)
        agent.test(args.test_episode, load_last=args.last, show=args.show)
    except:
        traceback.print_exc()
        env.close()