import torch
from torch import optim, nn
import numpy as np
import json
import os
from GNN import GNN, train, eval
import data_loader
from data_loader import get_loader
from encoder import Encoder


def main():
    N_train = 5
    N_eval = 5
    K_shot = 1
    Q = 1
    max_length = 128

    batch_size = 4
    train_iter = 30000
    val_iter = 1000
    test_iter = 10000
    val_step = 3000
    ckpt = ''
    glove_mat = np.load('./pretrain/glove/glove.6B.50d_mat.npy')
    glove_word2id = json.load(open('./pretrain/glove/glove.6B.50d_word2id.json'))

    sentence_encoder = Encoder(glove_mat, glove_word2id, max_length)

    train_data_loader = get_loader('train_wiki', sentence_encoder, N=N_train, K=K_shot, Q=Q, batch_size=batch_size)
    val_data_loader = get_loader('val_wiki', sentence_encoder, N=N_train, K=K_shot, Q=Q, batch_size=batch_size)
    test_data_loader = get_loader('val_wiki', sentence_encoder, N=N_train, K=K_shot, Q=Q, batch_size=batch_size)

    # prefix = '-'.join(["GNN", "cnn", opt.train, opt.val, str(N), str(K)])

    model = GNN(sentence_encoder, N_train)

    if torch.cuda.is_available():
        model.cuda()

    train(model, train_data_loader, val_data_loader, batch_size, N_train, K_shot, Q)

    acc = eval(model, val_data_loader, batch_size, N_eval, K_shot, Q)
    print("RESULT: %.2f" % (acc * 100))


if __name__ == "__main__":
    main()
