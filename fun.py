import datasets
import numpy as np
import torch
def load_loaders(name,batch):
    train_loader = datasets.get_loader(name, "train",
                        batch_size=batch)
    if train_loader is None:
        print("no such dataset named ",name)
        return
    val_loader = datasets.get_loader(name, "val",
                        batch_size=batch)
    n_train = len(train_loader.dataset)
    n_test = len(val_loader.dataset)
    name = type(train_loader.dataset).__name__

    print("dataset ({}): train set: {} - val set: {}".format(name, n_train, n_test))
    return train_loader, val_loader

# def train_src(encoder_s,loader_train_source,opt_es,crit,EPOCH=200):
#     for i in range(EPOCH):
#         for step, (X, y) in enumerate(loader_train_source):
#             X = X.cuda()
#             y = y.cuda()
#             opt_es.zero_grad()
#             loss = crit(encoder_s,{"X":X,"y":y})
#             loss.backward()
#             opt_es.step()
#         print("Epoch {}/{}:Loss:{}".format(i + 1, EPOCH, loss.item()))



def train_disc(encoder_s,encoder_t,disc,loader_source,loader_target,opt_et,opt_dis,EPOCH = 3):
    encoder_t.train()
    disc.train()

    crit = torch.nn.CrossEntropyLoss()

    for epoch in range(EPOCH):
        data_zip = enumerate(zip(loader_source,loader_target))
        for step,((XS,_),(XT,_)) in data_zip:
            XS = XS.cuda()
            XT = XT.cuda()

            opt_dis.zero_grad()

            embs = encoder_s(XS)
            embt = encoder_t(XT)
            emb = torch.cat((embs,embt),0)

            pred_d = disc(emb.detach())

            label_src = torch.ones(embs.size(0)).long().cuda()
            label_tgt = torch.zeros(embt.size(0)).long().cuda()
            label = torch.cat((label_src,label_tgt),0)

            loss_disc = crit(pred_d,label)
            loss_disc.backward()

            opt_dis.step()

            pred_cls = torch.squeeze(pred_d.max(1)[1])
            acc = (pred_cls == label).float().mean()

            opt_dis.zero_grad()
            opt_et.zero_grad()

            embt = encoder_t(XT)

            pred_d = disc(embt)

            label_tgt = torch.ones(embt.size(0)).long().cuda()

            loss_tgt = crit(pred_d,label_tgt)

            loss_tgt.backward()

            opt_et.step()

            if(step%20 == 0):
                print("Epoch {}/{} loss_disc:{:.4f} loss_tgt:{:.4f} acc{:.4f}"
                      .format(epoch+1,EPOCH,loss_disc.item(),loss_tgt.item(),acc))


def fit_disc(src_model, tgt_model, disc,
             src_loader, tgt_loader,
             opt_tgt, opt_disc,
             epochs=200,
             ):
    tgt_model.train()
    disc.train()

    # setup criterion and opt
    criterion = torch.nn.CrossEntropyLoss()

    ####################
    # 2. train network #
    ####################

    for epoch in range(epochs):
        # zip source and target data pair

        data_zip = enumerate(zip(src_loader, tgt_loader))
        for step, ((images_src, _), (images_tgt, _)) in data_zip:
            ###########################
            # 2.1 train discriminator #
            ###########################

            # make images variable
            images_src = images_src.cuda()
            images_tgt = images_tgt.cuda()

            # zero gradients for opt
            opt_disc.zero_grad()

            # extract and concat features
            feat_src = src_model.extract_features(images_src)
            feat_tgt = tgt_model.extract_features(images_tgt)
            feat_concat = torch.cat((feat_src, feat_tgt), 0)

            # predict on discriminator
            pred_concat = disc(feat_concat.detach())

            # prepare real and fake label
            label_src = torch.ones(feat_src.size(0)).long()
            label_tgt = torch.zeros(feat_tgt.size(0)).long()
            label_concat = torch.cat((label_src, label_tgt), 0).cuda()

            # compute loss for disc
            loss_disc = criterion(pred_concat, label_concat)
            loss_disc.backward()

            # optimize disc
            opt_disc.step()

            pred_cls = torch.squeeze(pred_concat.max(1)[1])
            acc = (pred_cls == label_concat).float().mean()

            ############################
            # 2.2 train target encoder #
            ############################

            # zero gradients for opt
            opt_disc.zero_grad()
            opt_tgt.zero_grad()

            # extract and target features
            feat_tgt = tgt_model.extract_features(images_tgt)

            # predict on discriminator
            pred_tgt = disc(feat_tgt)

            # prepare fake labels
            label_tgt = torch.ones(feat_tgt.size(0)).long().cuda()

            # compute loss for target encoder
            loss_tgt = criterion(pred_tgt, label_tgt)
            loss_tgt.backward()

            # optimize target encoder
            opt_tgt.step()

            #######################
            # 2.3 print step info #
            #######################
        print("Epoch [{}/{}] - "
                      "d_loss={:.5f} g_loss={:.5f} acc={:.5f}"
                      .format(epoch + 1,
                              epochs,
                              loss_disc.item(),
                              loss_tgt.item(),
                              acc.item()))


def train_center(encoder_s,encoder_t,loader_source,loader_target,opt_et,EPOCH = 30):
    from sklearn.cluster import KMeans
    import losses.losses
    n_classes = 10

    encoder_s.train()
    encoder_t.train()

    embs,_ = extract_embeddings(encoder_s,loader_source)

    kmeans_s = KMeans(n_clusters = n_classes)
    kmeans_s.fit(embs)

    center_s = torch.FloatTensor(kmeans_s.cluster_centers_).cuda()

    for epoch in range(EPOCH):
        for step,(X,y) in enumerate(loader_target):
            X = X.cuda()
            y = y.cuda()

            opt_et.zero_grad()

            loss = losses.losses.center_loss(encoder_t,{"X":X,"y":y},encoder_s,
                                             center_s,None,kmeans_s,None)

            loss.backward()
            opt_et.step()


import losses.losses
def train_src(model, data_loader, opt):
  loss_sum = 0.
  for step, (images, labels) in enumerate(data_loader):
      # make images and labels variable
      images = images.cuda()
      labels = labels.squeeze_().cuda()

      # zero gradients for opt
      opt.zero_grad()

      # compute loss for critic
      loss = losses.losses.triplet_loss(model, {"X":images,"y":labels})

      loss_sum += loss.item()

      # optimize source classifier
      loss.backward()
      opt.step()

  return {"loss":loss_sum/step}

def extract_embeddings(model, dataloader):
    model.eval()
    n_samples = dataloader.batch_size * len(dataloader)
    embeddings = np.zeros((n_samples, model.n_outputs))
    labels = np.zeros(n_samples)
    k = 0

    for images, target in dataloader:
        with torch.no_grad():
            images = images.cuda()
            embeddings[k:k+len(images)] = model(images).data.cpu().numpy()
            labels[k:k+len(images)] = target.numpy()
            k += len(images)

    return embeddings, labels

def test_src(encoder_s,loader_train,loader_test):
    from sklearn.neighbors import KNeighborsClassifier
    knn = KNeighborsClassifier(n_neighbors=2)
    emds_train, y_train = extract_embeddings(encoder_s,loader_train)
    emds_test, y_test = extract_embeddings(encoder_s, loader_train)
    knn.fit(emds_train,y_train)
    y_pred = knn.predict(emds_test)
    acc = (y_pred == y_test).mean()
    print("test accuracy:{}".format(acc))


def validate(src_model, tgt_model, src_data_loader, tgt_data_loader):
    """Evaluation for target encoder by source classifier on target dataset."""
    # set eval state for Dropout and BN layers
    from sklearn.neighbors import KNeighborsClassifier
    with torch.no_grad():
        X, y = extract_embeddings(src_model, src_data_loader)

        Xtest, ytest = extract_embeddings(tgt_model, tgt_data_loader)

        clf = KNeighborsClassifier(n_neighbors=2)
        clf.fit(X, y)
        y_pred = clf.predict(Xtest)

        acc = (y_pred == ytest).mean()

        return acc