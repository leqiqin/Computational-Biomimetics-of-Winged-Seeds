import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sim4opt2 import *
import pathlib, json
from datetime import datetime

class MLP(nn.Module):
    def __init__(self, n_input):
        super(MLP, self).__init__()
        self.layer1 = nn.Linear(n_input, 64)
        self.layer2 = nn.Linear(64, 64)
        # self.layer3 = nn.Linear(64, 64)
        # self.layer4 = nn.Linear(64, 64)
        self.layer5 = nn.Linear(64, 1)
        self.leaky_relu = nn.LeakyReLU(0.01)

    def forward(self, x):
        # x = (x - x.min()) / (x.max() - x.min())
        x = self.leaky_relu(self.layer1(x))
        x = self.leaky_relu(self.layer2(x))
        # x = self.leaky_relu(self.layer3(x))
        # x = self.leaky_relu(self.layer4(x))
        x = self.layer5(x)
        return x

    def reset_parameters(self):
        for layer in [self.layer1, self.layer2, self.layer5]:
            nn.init.kaiming_normal_(layer.weight, mode='fan_in', nonlinearity='leaky_relu')
            nn.init.constant_(layer.bias, 0)


def apply_input_constraint(input_tensor, max_sum=2.0, a=0, b=1):
    total_input_sum = input_tensor.sum()
    if total_input_sum > max_sum:
        scale_factor = max_sum / total_input_sum
        input_tensor.data *= scale_factor
    input_tensor.data = torch.clamp(input_tensor.data, min=a, max=b)


n_input = 11
surrogate = MLP(n_input).to(torchdeviceId)
input_tensor_tmp = torch.ones(1, n_input) * 0.2
input_tensor = input_tensor_tmp.to(torchdeviceId).requires_grad_(True)
print(input_tensor, " === initial pars")
optimizer_input = optim.Adam([input_tensor], lr=1e-2)
optimizer_surrogate = optim.Adam(surrogate.parameters(), lr=1e-2)

n = 7
sigma_0 = 0.1
# sigma_1 = 0.6
sigma_1 = 0.0

n_surrogate_opt = 100
n_input_opt = 3
max_sum = 3
a = 0.0
b = 0.5
data = []
weights_data = []
x_vel = 0.04

for step in range(100):
    loss = compute_loss_xfloating(input_tensor[0], sigma_1, x_vel, False)[0]
    print("step: ", step, "loss: ", loss, "par: ", input_tensor)
    data.append(loss)
    weights_data.append(np.array(input_tensor.clone().detach().cpu())[0])
    sigma_0 *= 0.99
    noisy_inputs = []
    f_outputs = []
    probs = []
    # surrogate.reset_parameters()
    while len(f_outputs) < n:
        noise = torch.randn(1, n_input).to(torchdeviceId) * sigma_0
        noisy_input = input_tensor + noise
        apply_input_constraint(noisy_input, max_sum, a, b)
        f_output, prob = compute_loss_xfloating(noisy_input[0], sigma_1, x_vel, False)
        if f_output < 999998:
            noisy_inputs.append(noisy_input)
            probs.append(prob)
            f_outputs.append(f_output)

    noisy_inputs = torch.cat(noisy_inputs, dim=0)
    probs = torch.tensor(probs).view(n, 1)
    f_outputs = torch.tensor(f_outputs).view(n, 1).to(torchdeviceId)
    for _ in range(n_surrogate_opt):
        optimizer_surrogate.zero_grad()
        model_outputs = surrogate(noisy_inputs)
        # print(model_outputs - f_outputs)
        loss_surrogate = ((model_outputs - f_outputs) ** 2).mean()
        # print(((model_outputs - f_outputs)**2).mean())
        loss_surrogate.backward(retain_graph=True)
        optimizer_surrogate.step()

    for _ in range(n_input_opt):
        optimizer_input.zero_grad()
        output = surrogate(input_tensor)
        loss_input = output.sum()
        loss_input.backward(retain_graph=True)
        optimizer_input.step()
        apply_input_constraint(input_tensor, max_sum, a, b)
    print(input_tensor, " === tmp pars")

print("Optimized input:")
print(input_tensor)
data_np = np.array(data)
weights_data_np = np.array(weights_data)

task_name = "expectation"
final_par = np.array(input_tensor.clone().detach().cpu())[0]
np.savetxt("bin/opt/par_{}.txt".format(task_name), final_par)
np.savetxt("bin/opt/loss_{}.txt".format(task_name), data_np)
print("save the result to bin/opt/loss_{}.txt and bin/opt/par_{}.txt".format(task_name, task_name))
