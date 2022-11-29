import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchbnn as bnn
import torch_ard as nn_ard
from torch.optim import lr_scheduler
import numpy as np

from Common.Utils import weight_init
from Common.Buffer import Buffer


def _format(device, *inp):
    output = []
    for d in inp:
        if not isinstance(d, torch.Tensor):
            d = torch.tensor(d, device=device, dtype=torch.float32)
            d = d.unsqueeze(0)
        output.append(d)
    return output

class ModelNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, args,
                 net_type=None, hidden_dim=256):
        super(ModelNetwork, self).__init__()

        if net_type is None:
            self.net_type = args.net_type
        else:
            self.net_type = net_type

        self.state_dim = state_dim
        self.action_dim = action_dim

        self.n_history = args.n_history
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.batch_size = args.batch_size

        if self.net_type == "DNN":
            self.state_net = nn.Sequential(
                nn.Linear(self.state_dim*self.n_history, int(hidden_dim/2)),
                nn.Dropout(0.15),
                nn.ReLU()
            )
            self.action_net = nn.Sequential(
                nn.Linear(self.action_dim * 2, int(hidden_dim/2)),
                nn.Dropout(0.15),
                nn.ReLU()
            )
            self.next_state_net = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.Dropout(0.15),
                nn.ReLU(),
                nn.Linear(hidden_dim, self.state_dim)
            )

        if self.net_type == "BNN":
            self.state_net = nn.Sequential(
                bnn.BayesLinear(prior_mu=0, prior_sigma=0.1,
                                in_features=self.state_dim * self.n_history, out_features=int(hidden_dim / 2)),
                nn.Dropout(0.15),
                nn.ReLU()
            )
            self.action_net = nn.Sequential(
                bnn.BayesLinear(prior_mu=0, prior_sigma=0.1,
                                in_features=self.action_dim * 2, out_features=int(hidden_dim / 2)),
                nn.Dropout(0.15),
                nn.ReLU()
            )
            self.next_state_net = nn.Sequential(
                bnn.BayesLinear(prior_mu=0, prior_sigma=0.1, in_features=int(hidden_dim / 2), out_features=int(hidden_dim / 2)),
                nn.Dropout(0.15),
                nn.ReLU(),
                bnn.BayesLinear(prior_mu=0, prior_sigma=0.1, in_features=hidden_dim, out_features=self.state_dim)
            )

        self.mse_loss = nn.MSELoss()
        self.kl_loss = bnn.BKLLoss(reduction='mean', last_layer_only=False)
        self.kl_weight = args.model_kl_weight

        self.apply(weight_init)

    def forward(self, state, action):

        if type(state) is not torch.Tensor:
            state = torch.tensor(state, device=self.device, dtype=torch.float32)
            state = state.unsqueeze(0)
        if type(action) is not torch.Tensor:
            action = torch.tensor(action, device=self.device, dtype=torch.float32)
            action = action.unsqueeze(0)

        state = self.state_net(state)
        action = self.action_net(action)

        z = torch.cat([state, action], dim=-1)
        next_state = self.next_state_net(z)

        return next_state

    def trains(self):
        self.state_net.train()
        self.action_net.train()
        self.next_state_net.train()

    def evals(self):
        self.state_net.eval()
        self.action_net.eval()
        self.next_state_net.eval()

class PidNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, args, device):
        super(PidNetwork, self).__init__()

        self.n_history = args.n_history
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.device = device
        self.time_step = 1.

        self.P = nn.Sequential(
            nn.Linear(self.state_dim + self.action_dim, 64),
            # nn.Dropout(0.15),
            nn.ReLU(),
            nn.Linear(64, self.action_dim)
        )

        self.I = nn.Sequential(
            nn.Linear(self.state_dim + self.action_dim, 64),
            # nn.Dropout(0.15),
            nn.ReLU(),
            nn.Linear(64, self.action_dim),
            nn.ReLU()
        )

        self.D = nn.Sequential(
            nn.Linear(self.state_dim + self.action_dim, 64),
            # nn.Dropout(0.15),
            nn.ReLU(),
            nn.Linear(64, self.action_dim),
            nn.ReLU()
        )

        # self.sum = nn.Sequential(
        #     nn.Linear(action_dim*3, 64),
        #     # nn.Dropout(0.15),
        #     nn.ReLU(),
        #     nn.Linear(64, self.action_dim),
        #     nn.ReLU()
        # )

        self.apply(weight_init)

    def forward(self, state, action):

        out = _format(self.device, state, action)

        p_output = self.P(out)
        i_output = self.I(out)
        d_output = self.D(out)

        # error_gain = 1/(1 + torch.exp(-5*(torch.sqrt(torch.mean(error**2, dim=1, keepdim=True)) - 1)))

        return torch.cat([p_output, i_output, d_output], dim=-1)

    def trains(self):
        self.P.train()
        self.I.train()
        self.D.train()
        self.sum.update()

    def evals(self):
        self.P.eval()
        self.I.eval()
        self.D.eval()
        self.sum.eval()


class InverseModelNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, args, net_type=None, hidden_dim=256):
        super(InverseModelNetwork, self).__init__()

        if net_type is None:
            self.net_type = args.net_type
        else:
            self.net_type = net_type

        self.args = args
        self.state_dim = state_dim
        self.action_dim = action_dim

        self.n_history = args.n_history
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.hidden_dim = hidden_dim

        # # Regularization tech
        # self.ln = nn.LayerNorm(self.hidden_dim)
        # self.bn = nn.BatchNorm1d(self.hidden_dim)
        # self.do1 = nn.Dropout(0.15)

        # construct the structure of model network
        if self.net_type == "dnn":
            print("DNN created")

            self.network = nn.Sequential(
                nn.Linear(self.state_dim*2, self.hidden_dim),
                nn.ReLU(),
                nn.Linear(self.hidden_dim, self.hidden_dim),
                nn.ReLU(),
                nn.Linear(self.hidden_dim, self.action_dim)
            )

        if self.net_type == "bnn":
            print("BNN created")

            self.network = nn.Sequential(
                nn_ard.LinearARD(in_features= self.state_dim*2, out_features=self.hidden_dim),
                nn.ReLU(),
                nn_ard.LinearARD(in_features= self.hidden_dim, out_features=self.hidden_dim),
                nn.ReLU(),
                nn_ard.LinearARD(in_features=self.hidden_dim, out_features=self.action_dim)
            )

        self.apply(weight_init)

    def forward(self, state, next_state):
        # Tensorlizing
        out = _format(self.device, state, next_state)

        state = F.softsign(out[0])
        next_state = F.softsign(out[1])

        action = torch.tanh(self.network(torch.cat([state, next_state], dim=-1)))
        return action

    def trains(self):
        self.network.train()

    def evaluates(self):
        self.network.eval()

if __name__ == '__main__':
    pass
