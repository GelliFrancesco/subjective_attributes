import torch
import torch.nn as nn
import torch.distributions as tdist


class mlp(nn.Module):

    def __init__(self):
        super(mlp, self).__init__()

        self.fc1 = nn.Linear(1024, 1024)

        self.fc2 = nn.Linear(1024, 1)
        self.sigm = torch.nn.Sigmoid()

    def forward(self, data):
        return self.sigm(self.fc2(self.fc1(data)))


class neural_net(nn.Module):

    def __init__(self, num_attributes, aux_size):
        super(neural_net, self).__init__()

        self.fc1 = nn.Linear(4096, 4096)
        self.fc2 = nn.Linear(4096, 1024)
        self.l_relu = nn.LeakyReLU()

        # self.mlps = [mlp() for i in range(num_attributes)]
        self.mlps = nn.ModuleList([mlp() for _ in range(num_attributes)])

        normal_dist = tdist.Normal(0, 0.1)
        self.aux_stds = nn.Parameter(normal_dist.sample((aux_size, num_attributes)), requires_grad=True)
        # self.aux_stds = nn.ParameterList([nn.Parameter(normal_dist.sample((aux_size,)), requires_grad=True) for _ in range(num_attributes)])

    def forward(self, data):
        general = self.fc2(self.l_relu(self.fc1(data)))
        return [el(general) for el in self.mlps]
