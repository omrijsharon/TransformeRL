import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import xavier_uniform_
from gym.spaces import Box, Discrete


class TransformerRL(nn.Module):
    def __init__(self, state_space, action_space, d_model, nhead, num_layers, dropout):
        super(TransformerRL, self).__init__()
        # Embedding layers for action
        self.d_model = d_model
        self.action_dim = None
        if isinstance(action_space, Box):
            self.action_dim = np.prod(action_space.shape)
            self.action_embedding = nn.Linear(self.action_dim, self.d_model)
        elif isinstance(action_space, Discrete):
            self.action_dim = action_space.n
            self.action_embedding = nn.Embedding(self.action_dim, self.d_model)
        # Embedding layers for state
        self.state_dim = np.prod(state_space.shape)
        self.state_embedding = nn.Linear(self.state_dim, self.d_model)

        # BatchNorm layers
        self.state_bn = nn.BatchNorm1d(self.d_model)
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
        # Linear layer for decoding the next state
        self.decoder_next_state = nn.Linear(2*self.d_model, self.state_dim)
        self.decoder_next_action = nn.Linear(2*self.d_model, self.action_dim)

    def forward(self, state, action):
        # Embed the state and action
        batch_size, seq_len, state_dim = state.shape
        batch_size, seq_len, action_dim = action.shape
        state_embedded = self.state_embedding(state.reshape(-1, state_dim))
        action_embedded = self.action_embedding(action.reshape(-1, action_dim))

        # Apply batch norm to the embeddings
        state_embedded = self.state_bn(state_embedded).reshape(batch_size, seq_len, -1)
        action_embedded = self.action_bn(action_embedded).reshape(batch_size, seq_len, -1)

        # Concatenate the embeddings along the time dimension
        x = torch.cat([state_embedded, action_embedded], dim=2)

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

        # Decode next state and action
        decoded_state = self.decoder_next_state(tgt_embedded)
        decoded_action = self.decoder_next_action(tgt_embedded)

        return decoded_state, decoded_action

