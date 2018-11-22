
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from matplotlib import cm
import numpy as np

'''
plt_tsne

'''

def plt_tsne(data,labels,marker='o'):
    if(len(data.shape) > 2) :
        data = data.reshape(data.shape[0],-1)
    tsne = TSNE(perplexity=10, n_components=2, init='pca', n_iter=3000)

    low_dim_embs = tsne.fit_transform(data)
    plt.figure()
    plt.cla()
    plt.ion()
    X, Y = low_dim_embs[:, 0], low_dim_embs[:, 1]
    labels = labels - np.min(labels) + 1
    total = len(set(labels))
    num = 0
    for x, y, s in zip(X, Y, labels):
        c = cm.rainbow(int(255 * s / total))
        if num < labels.shape[0] / 2:
            marker = 'o'
        else:
            marker = 'x'
        # plt.text(x, y, s, backgroundcolor=c, fontsize=9)
        plt.scatter(x,y,c=c,marker=marker)
        num = num+1
    plt.xlim(X.min(), X.max())
    plt.ylim(Y.min(), Y.max())
    plt.title('Visualize last layer')
    # plt.show()
    # plt.pause(0.01)