import torch
import torch.nn as nn

from parse_config import config

class SpeechEmbedder(nn.Module):

    def __init__(self):
        super(SpeechEmbedder, self).__init__()

        if config.task == "tdsv":
            self.data_config = config.data.TD_SV_data
            self.model_config = config.model.TD_SV_model
        else:
            self.data_config = config.data.TI_SV_data
            self.model_config = config.model.TI_SV_model

        self.LSTM_stack = nn.LSTM(input_size=self.data_config.nmels, hidden_size=self.model_config.hidden, num_layers=self.model_config.num_layer, batch_first=True)

        for name, param in self.LSTM_stack.named_parameters():
          if 'bias' in name:
             nn.init.constant_(param, 0.0)
          elif 'weight' in name:
             nn.init.xavier_normal_(param)

        self.projection = nn.Linear(in_features=self.model_config.hidden, out_features=self.model_config.proj)
    
    def forward(self, utter):

        x, _ = self.LSTM_stack(utter)
        x = x[:, -1]
        x = self.projection(x)
        x = x / torch.norm(x, p=2, dim=1).unsqueeze(dim=1) # L2 normalization
        return x
