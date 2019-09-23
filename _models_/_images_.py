import torch, torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


import torch, torch.nn as nn
import torch.nn.functional as F



class Reshape(nn.Module):
    def __init__(self, shape):
        nn.Module.__init__(self)
        self.shape=shape
    def forward(self,input):
        return input.view(self.shape)


class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.shape[0], -1)


class Classifier_mini(nn.Module):
    def __init__(self, inp_size=300):
        super(self.__class__, self).__init__()
        self.solver = nn.Sequential(
            nn.Linear(inp_size, 1024),
            nn.ReLU(), nn.Dropout(0.1), nn.BatchNorm1d(1024),
            nn.Linear(1024, 512),
            nn.ReLU(), nn.Dropout(0.1), nn.BatchNorm1d(512),
            nn.Linear(512, 150)
        )
        self.emb = EMBEDDING_LAYER()

    def forward(self, text_ix):
        h1 = self.emb(text_ix).sum(dim=1)
        res = self.solver(h1)
        return torch.tanh(res * 8)


D_K = 0.1


class GEN(nn.Module):
    def __init__(self):
        super(GEN, self).__init__()
        self.generator = nn.Sequential(nn.Linear(200, 180 * 4 * 4)
                                       , nn.LeakyReLU(0.2), nn.Dropout(D_K), nn.BatchNorm1d(180 * 4 * 4)
                                       , Reshape([-1, 180, 4, 4])

                                       , nn.ConvTranspose2d(180, 256, kernel_size=(3, 3))
                                       , nn.LeakyReLU(0.2), nn.BatchNorm2d(256), nn.Dropout(D_K)

                                       , nn.ConvTranspose2d(256, 512, kernel_size=(3, 3), stride=1)
                                       , nn.LeakyReLU(0.2), nn.BatchNorm2d(512), nn.Dropout(D_K)

                                       , nn.ConvTranspose2d(512, 256, kernel_size=(4, 4), stride=2, padding=0)
                                       , nn.LeakyReLU(0.2), nn.BatchNorm2d(256), nn.Dropout(D_K)

                                       , nn.ConvTranspose2d(256, 128, kernel_size=(3, 3), stride=1, padding=0)
                                       , nn.LeakyReLU(0.2), nn.BatchNorm2d(128), nn.Dropout(D_K)

                                       , nn.ConvTranspose2d(128, 100, kernel_size=(3, 3), stride=2, padding=0)
                                       , nn.LeakyReLU(0.2), nn.BatchNorm2d(100)

                                       , nn.ConvTranspose2d(100, 64, kernel_size=(4, 4), stride=2, padding=2)
                                       , nn.LeakyReLU(0.2), nn.BatchNorm2d(64)

                                       , nn.ConvTranspose2d(64, 32, kernel_size=(5, 5), padding=2)
                                       , nn.LeakyReLU(0.2)

                                       , nn.Conv2d(32, 3, kernel_size=(3, 3), padding=1)
                                       )
        self.emb = nn.Embedding(256, 100)

    def forward(self, noize, noize_int):
        noize_int_float = torch.tanh(self.emb(noize_int))
        noize = torch.tanh(noize)
        noize_main = torch.cat([noize_int_float, noize], dim=1)
        img = torch.sigmoid(self.generator(noize_main))
        return (img)


class image_generator_m(nn.Module):
    def __init__(self):
        super(self.__class__, self).__init__()
        self.txt = Classifier_mini()
        self.im = GEN()

    def forward(self, text_ix):
        h1 = nn.Dropout(0.1)(self.txt(text_ix))
        int_n, noise = sample_noise(batch_size=4)
        images = self.im(noise.cpu(), int_n.cpu())
        return images