import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import xavier_uniform_
from gym.spaces import Box, Discrete


class TransformerRL(nn.Module):
    def __init__(self, obs_space, action_space, d_model, nhead, num_layers, dropout):
        super(TransformerRL, self).__init__()
        # Positional encoding
        self.pos_encoder = nn.Linear(2, self.d_model)
        # Embedding layers for action
        self.d_model = d_model
        self.action_dim = None
        if isinstance(action_space, Box):
            self.action_dim = np.prod(action_space.shape).astype(int).item()
            self.action_embedding = nn.Linear(self.action_dim, self.d_model)
        elif isinstance(action_space, Discrete):
            self.action_dim = action_space.n
            self.action_embedding = nn.Embedding(self.action_dim, self.d_model)
        # Embedding layers for obs
        self.obs_space = np.prod(obs_space.shape).astype(int).item()
        self.obs_embedding = nn.Linear(self.obs_space, self.d_model)

        # BatchNorm layers
        self.obs_bn = nn.BatchNorm1d(self.d_model)
        self.action_bn = nn.BatchNorm1d(self.d_model)

        # Dropout layers
        self.dropout1 = nn.Dropout(p=dropout)
        self.dropout2 = nn.Dropout(p=dropout)

        # Transformer encoder layers
        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=2*self.d_model, nhead=nhead),
            num_layers=num_layers)

        self.transformer_decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(d_model=2*self.d_model, nhead=nhead),
            num_layers=num_layers)
        # Linear layer for decoding the next obs
        self.decoder_next_obs = nn.Linear(2 * self.d_model, self.obs_space)
        self.decoder_next_action = nn.Linear(2*self.d_model, self.action_dim)

    def forward(self, obs, action):
        # Embed the obs and action
        batch_size, seq_len, obs_dim = obs.shape
        batch_size, seq_len, action_dim = action.shape
        obs_embedded = self.obs_embedding(obs.view(-1, obs_dim))
        action_embedded = self.action_embedding(action.view(-1, action_dim))

        # Apply batch norm to the embeddings
        obs_embedded = self.obs_bn(obs_embedded).view(batch_size, seq_len, -1)
        action_embedded = self.action_bn(action_embedded).view(batch_size, seq_len, -1)

        # Concatenate the embeddings along the time dimension
        x = torch.cat([obs_embedded, action_embedded], dim=2)

        # Apply dropout
        x = self.dropout1(x)

        # Transformer encoder
        encoded_src = self.transformer_encoder(x)

        # Decoder input
        tgt = torch.zeros(batch_size, seq_len, 2*self.d_model, device=encoded_src.device)
        tgt[:, -1, :] = encoded_src[:, -1, :]

        # Apply dropout
        tgt = self.dropout2(tgt)

        # Transformer decoder
        tgt_embedded = self.transformer_decoder(tgt, encoded_src)

        # Decode next obs and action
        decoded_obs = self.decoder_next_obs(tgt_embedded)
        decoded_action = self.decoder_next_action(tgt_embedded)

        return decoded_obs, decoded_action

if __name__ == '__main__':
    import gym
    env = gym.make('CartPole-v1')
    obs_space = env.observation_space
    action_space = env.action_space
    model = TransformerRL(obs_space, action_space, d_model=64, nhead=4, num_layers=2, dropout=0.1)
    print(model)
    # F.gelu(approximate="tanh")
