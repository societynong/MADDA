import torch
import numpy as np
import matplotlib.pyplot as plt
import fun
from plt_tsne import plt_tsne
import model
import losses.losses
LR = 1e-4
BATCH_SIZE = 100
EPOCH = 200
encoder_s = model.EmbeddingNet(256).cuda()
encoder_t = model.EmbeddingNet(256).cuda()

encoder_s.load_state_dict(torch.load('encoder_s_csi.pkl'))
encoder_t.load_state_dict(torch.load('encoder_t_csi.pkl'))

# loader_train_source,loader_val_source = fun.load_loaders('mnist',BATCH_SIZE)
# loader_train_target,loader_val_target = fun.load_loaders('usps',BATCH_SIZE)
loader_source = model.get_loader_csi('human1.pickle',BATCH_SIZE)
loader_target = model.get_loader_csi('human2.pickle',BATCH_SIZE)
for i in range(EPOCH):
    datazip = zip(loader_source,loader_target,)
    for step,((X_s,ys),(X_t,yt)) in enumerate(datazip):
        embs = encoder_s(X_s.cuda())
        embt = encoder_t(X_t.cuda())
        emb = np.concatenate((embs.cpu().data.numpy(),embt.cpu().data.numpy()),0)
        y = np.concatenate((ys.data.numpy(),yt.data.numpy()))
        plt_tsne(emb,y)



