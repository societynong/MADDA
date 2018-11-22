import torch
import torch.nn as nn


def get_model(name, n_outputs):
    if name == "lenet":
        model = EmbeddingNet(n_outputs).cuda()
        opt = torch.optim.Adam(model.parameters(), lr=1e-4, betas=(0.5, 0.9))

        return model.cuda(), opt

    if name == "disc":
        model = Discriminator(
                                input_dims=500,
                                hidden_dims=500,
                                output_dims=2)
        opt = torch.optim.Adam(model.parameters(), lr=1e-4, betas=(0.5, 0.9))

        return model.cuda(), opt


class Discriminator(nn.Module):
    """Discriminator model for source domain."""

    def __init__(self, input_dims, hidden_dims, output_dims):
        """Init discriminator."""
        super(Discriminator, self).__init__()

        self.restored = False

        self.layer = nn.Sequential(
            nn.Linear(input_dims, hidden_dims),
            nn.ReLU(),
            nn.Linear(hidden_dims, hidden_dims),
            nn.ReLU(),
            nn.Linear(hidden_dims, output_dims)
        )

    def forward(self, input):
        """Forward the discriminator."""
        out = self.layer(input)
        return out


class EmbeddingNet(nn.Module):
    def __init__(self, n_outputs=128):
        super(EmbeddingNet, self).__init__()
        self.convnet = nn.Sequential(
            # 1st conv layer
            # input [3 x 30 x 30]
            # output [20 x 12 x 12]
            nn.Conv2d(1, 20, kernel_size=5),
            nn.MaxPool2d(kernel_size=2),
            nn.ReLU(),
            # 2nd conv layer
            # input [20 x 13 x 13]
            # output [50 x 4 x 4]
            nn.Conv2d(20, 50, kernel_size=5),
            #nn.Dropout2d(),
            nn.MaxPool2d(kernel_size=2),
            nn.ReLU()
        )
        self.n_classes = 10
        self.n_outputs = n_outputs
        self.fc = nn.Sequential(nn.Linear(50 * 4 * 4, 500),
                                nn.ReLU(),
                                # nn.Linear(512, 256),
                                # nn.ReLU(),
                                nn.Linear(500, self.n_outputs)
                                )

    def extract_features(self, x):
        output = self.convnet(x)
        output = output.view(output.size()[0], -1)
        output = self.fc[0](output)
        return output

    def forward(self, x):
        output = self.convnet(x)
        output = output.view(output.size()[0], -1)
        output = self.fc(output)
        return output

    def get_embedding(self, x):
        return self.forward(x)




class Encoder(torch.nn.Module):
    def __init__(self,n_outputs = 128):
        super(Encoder,self).__init__()
        self.conv1 = torch.nn.Sequential(
            torch.nn.Conv2d(1,20,kernel_size=5),
            torch.nn.MaxPool2d(kernel_size=2),
            torch.nn.ReLU(),

            torch.nn.Conv2d(20,50,kernel_size=5),
            torch.nn.MaxPool2d(kernel_size=2),
            torch.nn.ReLU(),
        )
        self.n_outputs = n_outputs
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(50*4*4,500),
            torch.nn.ReLU(),

            torch.nn.Linear(500,self.n_outputs)
        )

    def extract_features(self,x):
        output = self.conv1(x)
        output = output.view(output.size(0),-1)
        output = self.fc[0](output)
        return output

    def forward(self, x):
        output= self.conv1(x)
        output = output.view(output.size(0),-1)
        output = self.fc(output)
        return output

    def get_embedding(self,x):
        return self.forward(x)

class Classifier(torch.nn.Module):
    def __init__(self,n_class = 10):
        super(Classifier,self).__init__()
        self.fc1 = torch.nn.Linear(128,64)
        self.act = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(64,10)

    def forward(self, x):
        output = self.fc1(x)
        output = self.act(output)
        output = self.fc2(output)
        return output

class Disc(torch.nn.Module):
    def __init__(self,n_input = 500,n_hidden = 500,n_output = 2):
        super(Disc,self).__init__()
        self.fc1 = torch.nn.Linear(n_input,n_hidden)
        self.rl1 = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(n_hidden,n_hidden)
        self.rl2 = torch.nn.ReLU()
        self.fc3 = torch.nn.Linear(n_hidden,n_output)

    def forward(self, x):
        output = self.fc1(x)
        output = self.rl1(output)
        output = self.fc2(output)
        output = self.rl2(output)
        output = self.fc3(output)
        return output

import numpy as np
import pickle
import torch.utils.data as Data
def get_loader_csi(name,batch_size,n_classs,n_sample):
    with open(name,'rb') as f:
        dataset = pickle.load(f)
        X = dataset[0][:n_classs*n_sample]
        y = dataset[1][:n_classs*n_sample]
        X = torch.Tensor(X)
        y = torch.Tensor(y)
        dst = Data.TensorDataset(X,y)
        dtld = Data.DataLoader(dataset=dst,batch_size=batch_size,shuffle=True)
        return dtld


