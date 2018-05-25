import torch
from torch import nn
import torch.nn.functional as F


class EntityDetection(nn.Module):
    def __init__(self, config):
        super(EntityDetection, self).__init__()
        self.config = config
        target_size = config.label
        self.embed = nn.Embedding(config.words_num, config.words_dim)
        if not config.train_embed:
            self.embed.weight.requires_grad = False

        if config.entity_detection_mode.upper() == 'LSTM':
            self.lstm = nn.LSTM(
                input_size=config.input_size,
                hidden_size=config.hidden_size,
                num_layers=config.num_layer,
                bidirectional=True,
                dropout=config.rnn_dropout
            )
        elif config.entity_detection_mode.upper() == 'GRU':
            self.gru = nn.GRU(input_size=config.input_size,
                                hidden_size=config.hidden_size,
                                num_layers=config.num_layer,
                                dropout=config.rnn_dropout,
                                bidirectional=True)

        self.dropout = nn.Dropout(p=config.rnn_fc_dropout)
        self.relu = nn.ReLU()
        # TODO: why need a dropout here?
        self.hidden2tag = nn.Sequential(
            nn.Linear(config.hidden_size * 2, config.hidden_size * 2),
            nn.BatchNorm1d(config.hidden_size * 2),
            self.relu,
            self.dropout,
            nn.Linear(config.hidden_size * 2, target_size)
        )

    def forward(self, x):
        # x = (sequence length, batch_size, dimension of embedding)
        text = x.text
        batch_size = text.size()[1]
        # print(type(text), text)
        x = self.embed(input=text)
        # h0 = c0 = (layer*direction, batch_size, hidden_dim)
        if self.config.entity_detection_mode.upper() == 'LSTM':
            # print(config.cuda)
            if self.config.cuda:
                h0 = torch.zeros(self.config.num_layer * 2, batch_size,
                                          self.config.hidden_size, dtype=torch.long).cuda()
                c0 = torch.zeros(self.config.num_layer * 2, batch_size,
                                          self.config.hidden_size, dtype=torch.long).cuda()
            else:
                h0 = torch.zeros(self.config.num_layer * 2, batch_size,
                                          self.config.hidden_size)
                c0 = torch.zeros(self.config.num_layer * 2, batch_size,
                                          self.config.hidden_size)
            # output = (sentence length, batch_size, hidden_size * num_direction)
            # ht = (layer*direction, batch, hidden_dim)
            # ct = (layer*direction, batch, hidden_dim)
            outputs, (ht, ct) = self.lstm(x, (h0, c0))
            # outputs: (seq_len, batch_size, hidden size * num_directions)
        elif self.config.entity_detection_mode.upper() == 'GRU':
            if self.config.cuda:
                h0 = torch.zeros(self.config.num_layer * 2, batch_size,
                                 self.config.hidden_size, dtype=torch.long).cuda()
            else:
                h0 = torch.zeros(self.config.num_layer * 2, batch_size,
                                 self.config.hidden_size)
            # output = (sentence length, batch_size, hidden_size * num_direction)
            # ht = (layer*direction, batch, hidden_dim)
            outputs, ht = self.gru(x, h0)
        else:
            print("Wrong Entity Prediction Mode")
            exit(1)
        tags = self.hidden2tag(outputs.view(-1, outputs.size(2)))
        # print(tags.shape)
        scores = F.log_softmax(tags, 1)
        return scores
