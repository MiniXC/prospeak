import torch
import torch.nn as nn
import torch.optim as optim

class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size, latent_size, num_layers=3):
        super(Encoder, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.layers = nn.ModuleList([nn.Linear(hidden_size, hidden_size) for _ in range(num_layers)])
        self.fc2_mean = nn.Linear(hidden_size, latent_size)
        self.fc2_logvar = nn.Linear(hidden_size, latent_size)

    def forward(self, x):
        # using relu as activation function and residual connections
        hidden_start = torch.relu(self.fc1(x))
        hidden = hidden_start
        for layer in self.layers:
            hidden = hidden_start + torch.relu(layer(hidden))
        mean = self.fc2_mean(hidden)
        logvar = self.fc2_logvar(hidden)
        return mean, logvar

class Decoder(nn.Module):
    def __init__(
            self,
            latent_size,
            hidden_size,
            output_size,
            num_speakers,
            condition_size,
            data_dropout_p,
            num_layers=3
        ):
        super(Decoder, self).__init__()
        self.condition_layer = nn.Linear(condition_size, hidden_size)
        self.speaker_embedding = nn.Embedding(num_speakers+1, hidden_size)
        self.speaker_unk_idx = num_speakers
        self.fc1 = nn.Linear(latent_size, hidden_size)
        self.layers = nn.ModuleList([nn.Linear(hidden_size, hidden_size) for _ in range(num_layers)])
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.data_dropout_p = data_dropout_p
        self.dropout = nn.Dropout(0.2)
        self.layer_norms = nn.ModuleList([nn.LayerNorm(hidden_size) for _ in range(num_layers+1)])

    def forward(self, z, condition, speaker):
        hidden_start = torch.relu(self.fc1(z))
        if condition is not None and (self.data_dropout_p > torch.rand(1) or not self.training):
            hidden_start = hidden_start + torch.relu(self.condition_layer(condition))
        else:
            hidden_start = hidden_start + torch.relu(self.condition_layer(torch.zeros_like(condition)))
        if speaker is not None and (self.data_dropout_p > torch.rand(1) or not self.training):
            hidden_start = hidden_start + self.speaker_embedding(speaker)
        else:
            hidden_start = hidden_start + self.speaker_embedding(torch.zeros_like(speaker)+self.speaker_unk_idx)
        hidden = hidden_start
        for i, layer in enumerate(self.layers):
            hidden = hidden_start + torch.relu(self.layer_norms[i](layer(hidden)))
        output = self.fc2(self.layer_norms[-1](hidden))
        return output

class ConditionalVAE(nn.Module):
    def __init__(
            self,
            input_size,
            hidden_size,
            latent_size,
            num_speakers,
            condition_size,
            data_dropout_p,
            num_enc_layers=3,
            num_dec_layers=3
        ):
        super(ConditionalVAE, self).__init__()
        self.encoder = Encoder(input_size, hidden_size, latent_size, num_enc_layers)
        self.decoder = Decoder(latent_size, hidden_size, input_size, num_speakers, condition_size, data_dropout_p, num_dec_layers)

    def reparameterize(self, mean, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mean + eps * std
        return z

    def forward(self, x, condition, speaker):
        mean, logvar = self.encoder(x)
        z = self.reparameterize(mean, logvar)
        output = self.decoder(z, condition, speaker)
        return output, mean, logvar
