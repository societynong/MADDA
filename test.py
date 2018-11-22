import torch
from torch import optim
import numpy as np
from sklearn import preprocessing, neighbors
from addons import triplets_utils as tu
import fun

def validate(src_model, tgt_model, src_data_loader, tgt_data_loader):
    """Evaluation for target encoder by source classifier on target dataset."""
    # set eval state for Dropout and BN layers

    with torch.no_grad():
        
        X, y = fun.extract_embeddings(src_model, src_data_loader)

        Xtest, ytest = fun.extract_embeddings(tgt_model, tgt_data_loader)
        
        clf = neighbors.KNeighborsClassifier(n_neighbors=2)
        clf.fit(X, y)
        y_pred = clf.predict(Xtest)

        acc = (y_pred == ytest).mean()

        return acc


