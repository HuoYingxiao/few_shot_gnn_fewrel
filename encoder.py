import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class Encoder(nn.Module):
    def __init__(self, word_vec_mat, word2id, max_length=128, word_embedding_dim=50, pos_embedding_dim=5, hidden_size=230):
        nn.Module.__init__(self)

        self.word2id = word2id
        self.max_length = max_length
        self.word_embedding_dim = word_embedding_dim
        self.pos_embedding_dim = pos_embedding_dim

        # Word embedding
        word_vec_mat = torch.from_numpy(word_vec_mat)
        num = 400000
        self.word_embedding = nn.Embedding(num, self.word_embedding_dim,
                                           padding_idx=num-1)
        self.word_embedding.weight.data.copy_(word_vec_mat)

        # Position Embedding
        self.pos1_embedding = nn.Embedding(2 * max_length, pos_embedding_dim, padding_idx=0)
        self.pos2_embedding = nn.Embedding(2 * max_length, pos_embedding_dim, padding_idx=0)
        #encoder
        self.hidden_size = hidden_size
        self.embedding_dim = word_embedding_dim + pos_embedding_dim * 2
        self.conv = nn.Conv1d(self.embedding_dim, self.hidden_size, 3, padding=1)
        self.pool = nn.MaxPool1d(max_length)

        # For PCNN
        self.mask_embedding = nn.Embedding(4, 3)
        self.mask_embedding.weight.data.copy_(torch.FloatTensor([[1, 0, 0], [0, 1, 0], [0, 0, 1], [0, 0, 0]]))
        self.mask_embedding.weight.requires_grad = False
        self._minus = -100

    def forward(self, inputs):
        return inputs

    def embedding(self, inputs):
        word = inputs['word']
        pos1 = inputs['pos1']
        pos2 = inputs['pos2']

        max_index = 400000 - 2
        word = torch.clamp(word, 0, max_index)

        x = torch.cat([self.word_embedding(word),
                       self.pos1_embedding(pos1),
                       self.pos2_embedding(pos2)], 2)
        return x

    def encoding(self, inputs):
        x = self.conv(inputs.transpose(1, 2))
        x = F.relu(x)
        x = self.pool(x)
        return x.squeeze(2)

    def tokenize(self, raw_tokens, pos_head, pos_tail):
        # token -> index
        indexed_tokens = []
        for token in raw_tokens:
            token = token.lower()
            if token in self.word2id:
                indexed_tokens.append(self.word2id[token])
            else:
                indexed_tokens.append(self.word2id['[UNK]'])

        # padding
        while len(indexed_tokens) < self.max_length:
            indexed_tokens.append(self.word2id['[PAD]'])
        indexed_tokens = indexed_tokens[:self.max_length]

        # pos
        pos1 = np.zeros((self.max_length), dtype=np.int32)
        pos2 = np.zeros((self.max_length), dtype=np.int32)
        pos1_in_index = min(self.max_length, pos_head[0])
        pos2_in_index = min(self.max_length, pos_tail[0])
        for i in range(self.max_length):
            pos1[i] = i - pos1_in_index + self.max_length
            pos2[i] = i - pos2_in_index + self.max_length

        # mask
        mask = np.zeros((self.max_length), dtype=np.int32)
        mask[:len(indexed_tokens)] = 1

        return indexed_tokens, pos1, pos2, mask