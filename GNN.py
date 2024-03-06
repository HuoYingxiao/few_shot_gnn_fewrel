import os
import sys

import numpy as np

import gnn_iclr
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable


class GNN(nn.Module):

    def __init__(self, encoder, N, hidden_size=230):
        nn.Module.__init__(self)
        self.cost = nn.CrossEntropyLoss()
        self.sentence_encoder = encoder
        self.hidden_size = hidden_size
        self.node_dim = hidden_size + N
        self.gnn_obj = gnn_iclr.GNN_nl(N, self.node_dim, nf=96, J=1)

    def forward(self, support, query, N, K, NQ):
        '''
        support: Inputs of the support set.
        query: Inputs of the query set.
        N: Num of classes
        K: Num of instances for each class in the support set
        Q: Num of instances for each class in the query set
        '''
        #support = self.sentence_encoder(support)
        #query = self.sentence_encoder(query)
        support = self.sentence_encoder.embedding(support)
        support = self.sentence_encoder.encoding(support)
        query = self.sentence_encoder.embedding(query)
        query = self.sentence_encoder.encoding(query)

        support = support.view(-1, N, K, self.hidden_size)
        query = query.view(-1, NQ, self.hidden_size)

        B = support.size(0)
        D = self.hidden_size

        support = support.unsqueeze(1).expand(-1, NQ, -1, -1, -1).contiguous().view(-1, N * K, D)  # (B * NQ, N * K, D)
        query = query.view(-1, 1, D)  # (B * NQ, 1, D)
        labels = Variable(torch.zeros((B * NQ, 1 + N * K, N), dtype=torch.float)).cuda()
        for b in range(B * NQ):
            for i in range(N):
                for k in range(K):
                    labels[b][1 + i * K + k][i] = 1
        nodes = torch.cat([torch.cat([query, support], 1), labels], -1)  # (B * NQ, 1 + N * K, D + N)

        logits = self.gnn_obj(nodes)  # (B * NQ, N)
        _, pred = torch.max(logits, 1)
        return logits, pred

    def loss(self, logits, label):
        '''
        logits: Logits with the size (..., class_num)
        label: Label with whatever size.
        return: [Loss] (A single value)
        '''
        N = logits.size(-1)
        return self.cost(logits.view(-1, N), label.view(-1))

    def accuracy(self, pred, label):
        '''
        pred: Prediction results with whatever size
        label: Label with whatever size
        return: [Accuracy] (A single value)
        '''
        return torch.mean((pred.view(-1) == label.view(-1)).type(torch.FloatTensor))

    def load_model(self, ckpt):
        '''
        ckpt: Path of the checkpoint
        return: Checkpoint dict
        '''
        if os.path.isfile(ckpt):
            checkpoint = torch.load(ckpt)
            print("Successfully loaded checkpoint '%s'" % ckpt)
            return checkpoint
        else:
            raise Exception("No checkpoint found at '%s'" % ckpt)


def train(model, dataloader, dataloader_val, B, N=10, K=5, Q=5, na_rate=0, train_iter=30000, val_iter=1000,
          val_step=3000,
          lr=1e-3, lr_step_size=20000, weight_decay=1e-5, adv_dis_lr=1e-1, adv_enc_lr=1e-1):
    optimizer = optim.Adam(model.parameters(),
                           lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=lr_step_size)
    start_iter = 0

    model.train()
    best_acc = 0
    iter_loss = 0.0
    iter_loss_dis = 0.0
    iter_right = 0.0
    iter_right_dis = 0.0
    iter_sample = 0.0

    losses = []

    for it in range(start_iter, start_iter + train_iter):
        support, query, label, _ = next(dataloader)
        if torch.cuda.is_available():
            for k in support:
                support[k] = support[k].cuda()
            for k in query:
                query[k] = query[k].cuda()
            #support = {k: torch.from_numpy(v).cuda() for k, v in support.items()}
            #query = {k: torch.from_numpy(v).cuda() for k, v in query.items()}
            #label = torch.from_numpy(label).cuda()
            label = label.cuda()

        logits, pred = model(support, query,
                             N, K, Q * N + na_rate * Q)
        loss = model.loss(logits, label) / float(1)
        right = model.accuracy(pred, label)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10)

        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

        iter_loss += loss.data.item()
        iter_right += right.data.item()
        iter_sample += 1

        losses.append(loss.data.item())
        sys.stdout.write(
            'step: {0:4} | loss: {1:2.6f}, accuracy: {2:3.2f}%'.format(it + 1, iter_loss / iter_sample,
                                                                       100 * iter_right / iter_sample) + '\r')
        sys.stdout.flush()

        if (it + 1) % val_step == 0:
            print("")
            acc = eval(model, dataloader_val, B, N, K, Q, val_iter, na_rate=na_rate)
            model.train()
            # if acc > best_acc:
            #    print('Best checkpoint')
            if os.path.exists("trained_models"):
                torch.save({'state_dict': model.state_dict()}, "trained_models/gnn.pt")
            else:
                os.mkdir("trained_models")
                torch.save({'state_dict': model.state_dict()}, "trained_models/gnn.pt")
            #    best_acc = acc
            iter_loss = 0.
            iter_loss_dis = 0.
            iter_right = 0.
            iter_right_dis = 0.
            iter_sample = 0.
    print("\n####################\n")
    print("Finish training ")
    return losses


def eval(model, dataloader, B, N, K, Q, eval_iter, na_rate=0, ckpt=None):
    if ckpt is not None:
        if os.path.isfile(ckpt):
            checkpoint = torch.load(ckpt)
            print("Successfully loaded checkpoint '%s'" % ckpt)
            state_dict = checkpoint['state_dict']
            own_state = model.state_dict()
            for name, param in state_dict.items():
                if name not in own_state:
                    continue
                own_state[name].copy_(param)
    iter_right = 0.0
    iter_sample = 0.0
    all_classes = ['P177', 'P364', 'P2094', 'P361', 'P641', 'P59', 'P413', 'P206', 'P412', 'P155', 'P26', 'P410', 'P25',
                   'P463', 'P40', 'P921']
    confusion_mat = np.zeros([16, 16])

    with torch.no_grad():
        for it in range(eval_iter):
            support, query, label, class_names = next(dataloader)
            if torch.cuda.is_available():
                for k in support:
                    support[k] = support[k].cuda()
                for k in query:
                    query[k] = query[k].cuda()
                label = label.cuda()
            logits, pred = model(support, query, N, K, Q * N + Q * na_rate)

            count = 0
            for i in range(len(label.tolist())):
                p = pred.tolist()[i]
                l = label.tolist()[i]
                # print(int(count/25))
                classes = class_names[int(count / 25)]
                count += 1
                confusion_mat[all_classes.index(classes[l])][all_classes.index(classes[p])] += 1

            right = model.accuracy(pred, label)
            iter_right += right.data.item()
            iter_sample += 1

            sys.stdout.write(
                '[EVAL] step: {0:4} | accuracy: {1:3.2f}%'.format(it + 1, 100 * iter_right / iter_sample) + '\r')
            sys.stdout.flush()
        print("")
        print("confusion matrix: ")
        for row in confusion_mat:
            for element in row:
                print(f"{int(element)}", end=" ")  # Adjust 4d for different spacing
            print()
        # print(confusion_mat)

        precision = np.diag(confusion_mat) / np.sum(confusion_mat, axis=0)
        recall = np.diag(confusion_mat)/ np.sum(confusion_mat, axis=1)
        f1_scores = 2 * (precision * recall) / (precision + recall)

        print("")
        print(f"Precision: {precision}")
        print(f"Revall: {recall}")
        print(f"f1-score: {f1_scores}")
    return iter_right / iter_sample
