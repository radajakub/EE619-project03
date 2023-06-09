import torch
from torch import nn
import torch.nn.functional as F
from torch import optim

class ConstAlpha:
    def __init__(self, alpha: float) -> None:
        self.alpha = alpha

    def get(self) -> float:
        return self.alpha

    def update(self, *args) -> None:
        pass

class AutotuningAlpha:
    def __init__(self, action_dim: float, learning_rate: float) -> None:
        self.H = -action_dim
        self.log_alpha = torch.zeros(1, requires_grad=True)
        self.optimizer = optim.Adam([self.log_alpha], lr=learning_rate)
        self.alpha = self.log_alpha.exp().item()

    def get(self) -> float:
        return self.alpha

    def update(self, a_log_probs):
        alpha_loss = (self.log_alpha * (a_log_probs + self.H).detach()).mean()

        self.optimizer.zero_grad()
        alpha_loss.backward()
        self.optimizer.step()

        self.alpha = self.log_alpha.exp().item()
