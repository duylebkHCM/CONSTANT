import torch
import math
from torch import nn


class Word_Attention(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(Word_Attention, self).__init__()
        self.linear_query = nn.Linear(input_size, hidden_size)
        self.linear_key = nn.Linear(input_size, hidden_size)
        self.linear_value = nn.Linear(input_size, hidden_size)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        # x shape: (batch_size, seq_len, input_size)
        query = self.linear_query(x)
        key = self.linear_key(x)
        value = self.linear_value(x)

        # Calculate attention scores
        scores = query @ key.transpose(-2, -1)
        scores = self.softmax(scores)

        # Calculate weighted sum of the values
        word_embedding = scores @ value
        return word_embedding


class WordMLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size=None, drop_rate=0.0) -> None:
        super().__init__()
        output_size = output_size or input_size
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.act1 = nn.GELU()
        self.drop = nn.Dropout(drop_rate)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act1(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class CharEncBlock(nn.Module):
    def __init__(self, hidden_size, drop_rate=0.0) -> None:
        super().__init__()
        self.attn = Word_Attention(hidden_size, hidden_size)
        self.norm1 = nn.LayerNorm(hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size)
        self.mlp = WordMLP(hidden_size, int(hidden_size * 4), drop_rate=drop_rate)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class CharacterEncoder(nn.Module):
    def __init__(self, input_size, context_dim, max_seq_len, enc_depth=1, drop_rate=0.0):
        super(CharacterEncoder, self).__init__()
        self.embedding = nn.Embedding(input_size, context_dim)
        self.enc_block = nn.ModuleList([CharEncBlock(context_dim, drop_rate) for _ in range(enc_depth)])

        self.embedding_dim = context_dim
        self.max_seq_len = max_seq_len
        self.positional_encoding = self.get_positional_encoding()
        self.initialize_weights()

    def initialize_weights(self):
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            if isinstance(module, nn.Embedding):
                torch.nn.init.normal_(module.weight, 0, 0.02)

        self.apply(_basic_init)

    def forward(self, x):
        # x shape: (batch_size, seq_len)
        x = self.embedding(x)

        # Remove positional encoding for ablation study
        x += self.positional_encoding[: x.size(1), :].to(x.device)

        for blk in self.enc_block:
            x = blk(x)

        return x

    def get_positional_encoding(self):
        positional_encoding = torch.zeros(self.max_seq_len, self.embedding_dim)
        for pos in range(self.max_seq_len):
            for i in range(0, self.embedding_dim, 2):
                positional_encoding[pos, i] = math.sin(pos / (10000 ** (i / self.embedding_dim)))
                positional_encoding[pos, i + 1] = math.cos(pos / (10000 ** ((i + 1) / self.embedding_dim)))
        return positional_encoding
