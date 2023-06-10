from typing import Optional, Tuple
import torch
from torch.optim import Adam

from agent import GaussianPolicy
from replay import Batch
from temperature import AutotuningAlpha, ConstAlpha
from qfunction import QFunction


class SAC:
    def __init__(self, state_dim: int, action_dim: int, max_action: float, gamma: float, tau: float, learning_rate: float, temperature: Optional[float], q_hidden: int, pi_hidden: int, pi_nonlinearity: str) -> None:
        self.gamma = gamma
        self.tau = tau

        # initialize temperature alpha based on temperature parameter
        if temperature is None:
            self.alpha = AutotuningAlpha(action_dim, learning_rate=learning_rate)
            self.should_update_alpha = True
        else:
            self.alpha = ConstAlpha(temperature)
            self.should_update_alpha = False

        self.Q1 = QFunction(state_dim, action_dim, hidden_dim=q_hidden)
        self.Q1.train()
        self.Q1_optim = Adam(self.Q1.parameters(), lr=learning_rate)
        self.Q2 = QFunction(state_dim, action_dim, hidden_dim=q_hidden)
        self.Q2.train()
        self.Q2_optim = Adam(self.Q2.parameters(), lr=learning_rate)

        # define target Q networks and clone the parameters
        self.Q1_target = self.Q1.clone()
        self.Q2_target = self.Q2.clone()

        # define stochastic policy pi
        self.pi = GaussianPolicy(state_dim, action_dim, hidden_dim=pi_hidden, nonlinearity=pi_nonlinearity, max_action=max_action)
        self.pi.train()
        self.pi_optim = Adam(self.pi.parameters(), lr=learning_rate)

    # update Q by gradient step and return losses
    def update_Q(self, batch: Batch) -> Tuple[float, float]:
        # compute targets
        with torch.no_grad():
            # compute action and its log probability by sampling from the policy
            # it includes reparametrization trick and squashing as can be seen in agent.py
            a_, log_probs = self.pi.act(batch.s_)
            min_q = torch.minimum(self.Q1_target(batch.s_, a_), self.Q2_target(batch.s_, a_))
            target = batch.r + self.gamma * (1 - batch.d) * (min_q - self.alpha.get() * log_probs)


        self.Q1_optim.zero_grad()
        Q1_loss = ((self.Q1(batch.s, batch.a) - target).pow(2)).mean()
        Q1_loss.backward()
        self.Q1_optim.step()

        self.Q2_optim.zero_grad()
        Q2_loss = ((self.Q2(batch.s, batch.a) - target).pow(2)).mean()
        Q2_loss.backward()
        self.Q2_optim.step()

        return Q1_loss, Q2_loss

    def update_pi(self, batch: Batch) -> float:
        a, log_probs = self.pi.act(batch.s)

        for p, q in zip(self.Q1.parameters(), self.Q2.parameters()):
            p.requires_grad = False
            q.requires_grad = False
        min_q = torch.minimum(self.Q1(batch.s, a), self.Q2(batch.s, a))
        for p, q in zip(self.Q1.parameters(), self.Q2.parameters()):
            p.requires_grad = True
            q.requires_grad = True


        self.pi_optim.zero_grad()
        pi_loss = (self.alpha.get() * log_probs - min_q).mean()
        pi_loss.backward()
        self.pi_optim.step()

        return pi_loss

    def update_alpha(self, batch: Batch) -> None:
        if self.should_update_alpha:
            _, log_probs = self.pi.act(batch.s)
            self.alpha.update(log_probs)

    def update_targets(self) -> None:
        self.Q1_target.soft_update(self.Q1, tau=self.tau)
        self.Q2_target.soft_update(self.Q2, tau=self.tau)
