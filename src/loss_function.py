import torch
import torch.nn as nn

from parse_config import config
from utils import get_similarity, get_contrast_loss, get_softmax_loss

class GE2ELoss(nn.Module):

    def __init__(self):
        super(GE2ELoss, self).__init__()

        if config.task == "tdsv":
            self.model_config = config.model.TD_SV_model
            self.train_config = config.train.TD_SV_train
        else:
            self.model_config = config.model.TI_SV_model
            self.train_config = config.train.TI_SV_train

        self.w = nn.Parameter(torch.tensor(10.0), requires_grad=True)
        self.b = nn.Parameter(torch.tensor(-5.0), requires_grad=True)

    def forward(self, embedding):
        '''
        :param embedding: shape -> [NxM, feature]
        :return:
        '''
        embedding = torch.reshape(embedding, (self.train_config.N, self.train_config.M, self.model_config.proj))
        similarity = self.w * get_similarity(embedding) + self.b # shape -> (N, M, N)

        if self.model_config.loss == "contrast":
            loss = get_contrast_loss(similarity)
        else:
            loss = get_softmax_loss(similarity)

        return loss


