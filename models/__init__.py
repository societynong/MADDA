from models import discriminator
from models import embedding
import torch


def get_model(name, n_outputs):
    if name == "lenet":
        model = embedding.EmbeddingNet(n_outputs).cuda()
        opt = torch.optim.Adam(model.parameters(), lr=1e-4, betas=(0.5, 0.9))

        return model.cuda(), opt

        
    if name == "disc":
        model = discriminator.Discriminator(input_dims=500,
                                      hidden_dims=500,
                                      output_dims=2)
        opt = torch.optim.Adam(model.parameters(), lr=1e-4, betas=(0.5, 0.9))
 

        return model.cuda(), opt


