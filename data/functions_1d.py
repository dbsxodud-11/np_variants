import math

import torch


class SineFunc:
    def __init__(self, amplitude_min=-1.0, amplitude_max=1.0, shift_min=-math.pi, shift_max=math.pi):
        self.amplitude_min = amplitude_min
        self.amplitude_max = amplitude_max

        self.shift_min = shift_min
        self.shift_max = shift_max

    def sample_data(self, num_tasks, num_samples):
        amplitude = torch.rand(num_tasks, 1, 1) * (self.amplitude_max - self.amplitude_min) + self.amplitude_min
        shift = torch.rand(num_tasks, 1, 1) * (self.shift_max - self.shift_min) + self.shift_min

        x = torch.rand((num_tasks, num_samples, 1)) * math.pi * 2 - math.pi
        y = amplitude * torch.sin((x - shift))
        return x, y, (amplitude.flatten(), shift.flatten())