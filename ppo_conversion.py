
import torch
from torch import nn
import gymnasium as gym
from torchrl.envs import GymWrapper
import torch.nn.functional as F
import torch.optim as optim
from tensordict import TensorDict
from torchsummary import summary
from typing import Callable, Dict, List, Optional, Tuple, Type, Union
from datetime import datetime

class CustomActor(nn.Module):
    """Combined actor-critic network."""

    def __init__(self,
                 feature_dim: int,
                 last_layer_dim_pi: int = 6,
                 last_layer_dim_vf: int = 1,
                 ):
        """Initialize."""
        super().__init__()
        self.latent_dim_pi = last_layer_dim_pi
        self.latent_dim_vf = last_layer_dim_vf


        # Policy network
        self.pfully1 = nn.Linear(feature_dim, 20)
        self.pfully2 = nn.Linear(20, 20)
        self.pfully3 = nn.Linear(20, last_layer_dim_pi)
        self.pfully4 = nn.Linear(20, last_layer_dim_pi)

        # Value network
        self.qfully1 = nn.Linear(feature_dim, 20)
        self.qfully2 = nn.Linear(20, last_layer_dim_vf)

    def forward(self, inputs):
        return self.forward_actor(inputs), self.forward_critic(inputs)

    def forward_actor(self, inputs):
        a_f = F.relu(self.pfully1(inputs))
        a_f = F.relu(self.pfully2(a_f))
        a_m = self.pfully3(a_f)
        a_sd = self.pfully4(a_f)
        a_fsd = torch.clamp(a_sd, min=1e-2, max=1)
        dim = 0
        if len(torch.cat((a_m, a_fsd), dim=0).size()) > 1:
            dim = 1
        return torch.cat((a_m, a_fsd), dim=dim)

    def forward_critic(self, inputs):
        q = F.relu(self.qfully1(inputs))
        q = self.qfully2(q)
        return q

class auto_encoder(nn.Module):
    def __init__(self,
                 feature_dim: int,
                 hidden_layers: list
                 ):
        """Initialize."""
        super().__init__()
        self.first = nn.Linear(feature_dim, hidden_layers[0])
        self.layers = []
        for i in range(0, len(hidden_layers)-1):
            self.layers.append(nn.Linear(hidden_layers[i], hidden_layers[i+1]))
        self.layers.append(nn.Linear(hidden_layers[-1], feature_dim))

    def forward(self, inputs):
        a_f = F.relu(self.first(inputs))
        for i in self.layers:
            a_f = F.relu(i(a_f))
        return a_f




def evaluate(agent, batch_obs, batch_acts):
  # Calculate the log probabilities of batch actions using most
  # recent actor network.
  # This segment of code is similar to that in get_action()
  actor, V = agent.forward(batch_obs)
  mean, sd = actor.split(actor.size(1) // 2, dim=1)
  dist = torch.distributions.Normal(loc=mean, scale=sd)
  log_probs = dist.log_prob(batch_acts).to("cuda:0")
  nan_mask = torch.isnan(log_probs)
  # Return predicted values V and log probs log_probs
  return V, log_probs

def evaluate_auto(opt, auto_encoder, batch_obs, loss=nn.MSELoss()):
    opt.zero_grad()
    est = auto_encoder.forward(batch_obs)
    lossed = loss(est, batch_obs)
    print(torch.sum(lossed).item())
    lossed.backward()
    opt.step()


def run_ep(agent, env_e):
    done = False
    action_vals = []
    log_probs = []
    critic_vals = []
    rewards_ = []
    obs = []
    state = env_e.reset()["observation"].type(torch.FloatTensor)
    while not done:
        state = state.to("cuda:0")
        obs.append(state)
        actor, pred = agent.forward(state)
        critic_vals.append(pred)
        actions = actor[0:int(len(actor) / 2)]
        sd = actor[int(len(actor) / 2):]
        actions_dist = torch.distributions.Normal(loc=actions, scale=sd)
        actions = torch.clamp(actions_dist.sample(), -1, 1).to("cuda:0")
        action_vals.append(actions.clone().detach())
        log_probs.append(actions_dist.log_prob(actions).clone().detach())  # mod_action-
        outs = env_e.step(TensorDict({'action': actions}, batch_size=env.batch_size))["next"]
        done = outs["done"]
        state = outs["observation"].type(torch.cuda.FloatTensor)
        reward = outs["reward"].type(torch.cuda.FloatTensor)
        rewards_.append(reward)
    return log_probs,  critic_vals, rewards_, obs, action_vals


def compute_loss(rewards_ep, critic_estimates, log_probs, actions, obs, agent, critic_loss=nn.MSELoss()):
    V, curr_log_probs = evaluate(agent, obs, actions)
    advantage = ((rewards_ep-critic_estimates.clone().detach().to("cuda:0")) * log_probs)
    advantage_norm = (advantage - torch.mean(advantage)) / (torch.std(advantage) + 1e-10)
    div = curr_log_probs / log_probs
    surr_1 = div * advantage_norm
    surr_2 = torch.clamp(div, 1-0.2, 1+0.2).to("cuda:0")*advantage_norm

    c_loss = critic_loss(rewards_ep, V).to("cuda:0")
    s_dis = torch.mean(-torch.min(surr_1, surr_2).to("cuda:0")).to("cuda:0")
    return s_dis, c_loss

def f(inp):
    return torch.stack(tuple(inp), 0)


def run_batch(size, agent, env_, gamma_):
    u_log_probs, u_critic_vals, u_rewards, u_obs, u_action_vals = run_ep(agent, env_)
    rewards_ = f(u_rewards)
    obs = f(u_obs)
    critic_vals = f(u_critic_vals)
    log_probs = f(u_log_probs)
    action_vals = f(u_action_vals)
    raw_rewards =  f(u_rewards).clone().detach()
    for e in range(0, size):
        nlog_probs, ncritic_vals, nrewards, nobs, naction_vals = run_ep(agent, env_)
        raw_rewards = torch.cat((raw_rewards, f(nrewards).clone().detach()), 0)
        for t_i in range(len(nrewards)):
            for t in range(t_i + 1, len(nrewards)):
                nrewards[t_i] += nrewards[t] * (gamma_ ** (t_i - t))

        log_probs = torch.cat((log_probs, f(nlog_probs)), 0)
        critic_vals = torch.cat((critic_vals, f(ncritic_vals)), 0)
        rewards_ = torch.cat((rewards_, f(nrewards)), 0)
        obs = torch.cat((obs, f(nobs)), 0)
        action_vals = torch.cat((action_vals, f(naction_vals)), 0)
    return log_probs, critic_vals, rewards_, obs, action_vals, raw_rewards


def train(
        agent,
        env_,
        gamma_,
        batch_size_,
        update_actor=True,
        optimiser_=None,
        step_number=1,
        autos=None,
        auto_opts=None
):

    log_probs, critic_vals, rewards_, obs, action_vals, _ = run_batch(batch_size_, agent, env_, gamma_)
    if autos is not None and len(autos) == len(auto_opts):
        for i in range(0, len(autos)):
            obs_copy = obs.clone().detach()
            evaluate_auto(auto_opts[i], autos[i], obs_copy)

    for u in range(0, step_number):
        optimiser_.zero_grad()
        actor_losses, critic_losses = compute_loss(rewards_, critic_vals, log_probs, action_vals, obs, agent)
        if update_actor:
            actor_losses.backward()
        critic_losses.backward()
        torch.nn.utils.clip_grad_norm_(agent.parameters(), 0.5)
        optimiser_.step()


if __name__ == '__main__':
    torch.autograd.set_detect_anomaly(True)
    device = torch.device("cuda:0")
    torch.set_default_tensor_type("torch.cuda.FloatTensor")
    with torch.device(device):
        print(f"Using device: {device}")
        print_to = "results/PPO/from_scratch" + datetime.now().strftime("%m%d%H%M") + ".csv"
        writer = open(print_to, "w")
        writer.write("")
        writer.close()
        env = GymWrapper(gym.make('Walker2d-v4'), device=device)
        input_space = env.observation_space.shape[0]
        output_space = env.action_space.shape[0]
        num_processes = 1
        ac = CustomActor(input_space, output_space)
        ac = ac.to(device)
        ac.share_memory()

        for param in ac.parameters():
            param.requires_grad = True
        lr = 0.1

        summary(ac, (1, input_space))
        optimiser = optim.Adam(ac.parameters(), lr=lr)
        eps_to_train = 5000
        batch_size = 30
        gamma = 0.9
        base_weights = torch.mean(ac.pfully3.weight.clone().detach()).item()
        base_critic_weights = torch.mean(ac.qfully2.weight.clone().detach()).item()
        min_lr = 1e-5
        total_reward = 0

        enc_specs = [[17,17,17], [15,10,15], [15,5,15], [15,10,5,10,15], [15,15]]
        encs = []
        encs_opts = []
        for spec in enc_specs:
            auto = auto_encoder(input_space, spec)
            encs.append(auto)
            encs_opts.append(optim.Adam(auto.parameters(), lr=lr))

        for i in range(0, int(eps_to_train)):
            frac = (i - 1.0) / int(eps_to_train)
            new_lr = lr * (1.0 - frac)
            new_lr = max(new_lr, min_lr)
            optimiser.param_groups[0]["lr"] = new_lr

            print("Running Batch:", str(i))
            for rank in range(num_processes):
                train(ac, env, gamma, batch_size, True, optimiser, step_number=5, autos=encs, auto_opts=encs_opts)
            # Print gradients after backward pass
            new_weights = torch.mean(ac.pfully3.weight.clone().detach()).item()
            new_critic_weights = torch.mean(ac.qfully2.weight.clone().detach()).item()
            print("Actor shift: ", base_weights-new_weights, "Critic shift:", base_critic_weights-new_critic_weights)
            #eval_batch
            a, b, _, _, _, rewards = run_batch(2, ac, env, gamma)
            mean_reward = str(torch.mean(rewards).item()*len(rewards)/2)
            total_reward += float(mean_reward)
            print(mean_reward, "Running Avg: ", total_reward/(i+1), "Average episode length: ", len(rewards)/2)
            base_weights = new_weights
            base_critic_weights = new_critic_weights
            writer = open(print_to, "a")
            writer.write(mean_reward + ",")
            writer.close()
