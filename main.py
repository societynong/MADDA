import torch
import numpy as np
import matplotlib.pyplot as plt
import fun
from plt_tsne import plt_tsne
import model
import losses.losses
import test
LR = 1e-4
BATCH_SIZE = 64
EPOCH = 100
trained = True

encoder_s,opt_es = model.get_model('lenet',256)
encoder_t,opt_et = model.get_model('lenet',256)
encoder_s = encoder_s.cuda()
encoder_t = encoder_t.cuda()

# disc,opt_dis = model.get_model('disc',256)
# disc = disc.cuda()
disc = model.Disc()
disc = disc.cuda()
opt_dis = torch.optim.Adam(disc.parameters(), lr=1e-4, betas=(0.5, 0.9))


crit = losses.losses.triplet_loss#torch.nn.CrossEntropyLoss()

loader_train_source,loader_val_source = fun.load_loaders('mnist',BATCH_SIZE)
loader_train_target,loader_val_target = fun.load_loaders('usps',BATCH_SIZE)

if trained == True:
    encoder_s.load_state_dict(torch.load('encoder_s.pkl'))
    encoder_t.load_state_dict(torch.load('encoder_s.pkl'))
else :
    for e in range(50):
        fun.train_src(encoder_s,loader_train_source,opt_es)

    encoder_t.load_state_dict(encoder_s.state_dict())
    torch.save(encoder_s.state_dict(), 'encoder_s.pkl')
acc = fun.validate(encoder_s,encoder_s,loader_train_source,loader_val_source)

goal = 0.9

for i in range(EPOCH):
    # fun.train_disc(encoder_s,encoder_t,disc,loader_train_source,loader_train_target,opt_et,opt_dis,EPOCH = 3)
    fun.fit_disc(encoder_s,encoder_t,disc,loader_train_source,loader_train_target,opt_et,opt_dis,20)
    acc = fun.validate(encoder_s,encoder_t,loader_train_target,loader_val_target)
    # acc = test.validate(encoder_s, encoder_t, loader_train_source, loader_val_target)
    print("Source >>> Target with out centering acc:{:.4f}".format(acc))
    # fun.train_center(encoder_s,encoder_t,loader_train_source,loader_train_target,opt_et)
    acc = test.validate(encoder_s, encoder_t, loader_train_source, loader_val_target)
    print("Source >>> Target after centering acc:{:.4f}".format(acc))
    if acc > 0.9 :
        torch.save(encoder_t.state_dict(),'encoder_t.pkl')


