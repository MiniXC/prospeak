import torch
import torch.nn as nn
from models.diffusion_utils import StepEmbedding

class ConditionalDiffusion(nn.Module):
    def __init__(self, input_size, hidden_size, condition_size, num_speakers, num_layers=3, data_dropout_p=0.1):
        super(ConditionalDiffusion, self).__init__()
        self.condition_layer = nn.Linear(condition_size, hidden_size)
        self.speaker_embedding = nn.Embedding(num_speakers+1, hidden_size)
        self.speaker_unk_idx = num_speakers
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.layers = nn.ModuleList([nn.Linear(hidden_size, hidden_size) for _ in range(num_layers)])
        self.fc2 = nn.Linear(hidden_size, input_size)
        self.layer_norms = nn.ModuleList([nn.LayerNorm(hidden_size) for _ in range(num_layers+1)])
        self.condition_size = condition_size
        self.data_dropout_p = data_dropout_p
        self.step_embedding = StepEmbedding(10, hidden_size, hidden_size)

    def forward(self, x, condition, speaker, t):
        hidden = torch.relu(self.fc1(x))
        hidden = hidden + self.step_embedding(t).squeeze(1)
        if condition is not None and (self.data_dropout_p > torch.rand(1) or not self.training):
            hidden = hidden + torch.relu(self.condition_layer(condition))
        else:
            condition = torch.zeros((x.shape[0], self.condition_size))
            hidden = hidden + torch.relu(self.condition_layer(torch.zeros_like(condition).to(x.device)))
        if speaker is not None and (self.data_dropout_p > torch.rand(1) or not self.training):
            hidden = hidden + self.speaker_embedding(speaker)
        else:
            speaker = (torch.zeros((x.shape[0],)) + self.speaker_unk_idx).long().to(x.device)
            hidden = hidden + self.speaker_embedding(speaker)
        for i, layer in enumerate(self.layers):
            hidden = hidden + torch.relu(self.layer_norms[i](layer(hidden)))
        output = self.fc2(self.layer_norms[-1](hidden))
        return output