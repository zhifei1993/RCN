import torch
import torch.nn
from torch.nn import functional as F


class ComConV(torch.nn.Module):

    def __init__(self, data, ent_dim, rel_dim, **kwargs):
        super(ComConV, self).__init__()
        self.ent_dim = ent_dim
        self.rel_dim = rel_dim
        self.reshape_H = 2
        self.reshape_W = 200

        self.in_channels = kwargs["in_channels"]
        self.out_channels = kwargs["out_channels"]
        self.filt_height = kwargs["filt_height"]
        self.filt_width = kwargs["filt_width"]

        self.enti_embedding = torch.nn.Embedding(data.entities_num, ent_dim, padding_idx=0)
        self.rela_embedding = torch.nn.Embedding(data.relations_num, rel_dim, padding_idx=0)
        # self.ent_transfer = torch.nn.Embedding(data.entities_num, ent_dim, padding_idx=0)
        self.rel_transfer = torch.nn.Embedding(data.relations_num, rel_dim, padding_idx=0)

        filter_dim = self.in_channels * self.out_channels * self.filt_height * self.filt_width
        # self.fc1 = torch.nn.Linear(rel_dim, filter_dim)
        self.filter = torch.nn.Embedding(data.relations_num, filter_dim, padding_idx=0)

        self.input_drop = torch.nn.Dropout(kwargs["input_dropout"])
        self.hidden_drop = torch.nn.Dropout(kwargs["hidden_dropout"])
        self.feature_map_drop = torch.nn.Dropout2d(kwargs["feature_map_dropout"])

        self.loss = torch.nn.BCELoss()

        # self.conv = torch.nn.Conv2d(self.in_channels, self.out_channels, (self.filt_height, self.filt_width),
        #                             stride=1, padding=0, bias=True)

        self.bn0 = torch.nn.BatchNorm2d(self.in_channels)
        self.bn1 = torch.nn.BatchNorm2d(self.out_channels)
        self.bn2 = torch.nn.BatchNorm1d(ent_dim)

        self.register_parameter('bias', torch.nn.Parameter(torch.zeros(data.entities_num)))

        fc_length = (self.reshape_H - self.filt_height + 1) * (self.reshape_W - self.filt_width + 1) * self.out_channels
        self.fc = torch.nn.Linear(fc_length, ent_dim)

        self.avg_pool = torch.nn.AdaptiveAvgPool2d(1)
        self.senet = torch.nn.Sequential(
            torch.nn.Linear(self.out_channels, self.out_channels // 4),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(self.out_channels // 4, self.out_channels),
            torch.nn.Sigmoid()
        )

    def init(self):
        torch.nn.init.xavier_normal_(self.enti_embedding.weight.data)
        torch.nn.init.xavier_normal_(self.rela_embedding.weight.data)
        # torch.nn.init.xavier_normal_(self.ent_transfer.weight.data)
        torch.nn.init.xavier_normal_(self.rel_transfer.weight.data)
        torch.nn.init.xavier_normal_(self.filter.weight.data)

    def forward(self, entity_id, relation_id):
        batch_enti_embedding = self.enti_embedding(entity_id).reshape(-1, 1, self.ent_dim)
        batch_rela_embedding = self.rela_embedding(relation_id).reshape(-1, 1, self.rel_dim)
        # batch_ent_transfer = self.ent_transfer(entity_id).reshape(-1, 1, self.ent_dim)
        batch_rel_transfer = self.rel_transfer(relation_id).reshape(-1, 1, self.rel_dim)
        f = self.filter(relation_id)
        f = f.reshape(batch_enti_embedding.size(0) * self.in_channels * self.out_channels, 1, self.filt_height,
                      self.filt_width)

        # (b, 1, 200)
        e = batch_enti_embedding * batch_rel_transfer
        r = batch_rela_embedding * batch_enti_embedding * batch_rel_transfer

        # f = self.fc1(r)
        # f = f.reshape(batch_enti_embedding.size(0) * self.out_channels, 1, self.filt_height,
        #               self.filt_width)

        # (b, 1, 2, 200)
        x = torch.cat([e, r], 1).reshape(-1, 1, self.reshape_H, self.reshape_W)
        # x = e.reshape(-1, 1, self.reshape_H, self.reshape_W)
        x = self.bn0(x)
        x = self.input_drop(x)

        # (b, out, 2-filt_h+1, 200-filt_w+1)
        # x = self.conv(x)
        x = x.permute(1, 0, 2, 3)
        x = F.conv2d(x, f, groups=batch_enti_embedding.size(0))
        x = x.reshape(batch_enti_embedding.size(0), self.out_channels, self.reshape_H - self.filt_height + 1,
                      self.reshape_W - self.filt_width + 1)
        y = self.avg_pool(x).view(batch_enti_embedding.size(0), self.out_channels)
        y = self.senet(y).view(batch_enti_embedding.size(0), self.out_channels, 1, 1)
        x = x * y
        x = self.bn1(x)
        x = torch.relu(x)
        x = self.feature_map_drop(x)

        # (b, fc_length)→ (b, ent_dim)
        x = x.reshape(batch_enti_embedding.size(0), -1)
        x = self.fc(x)
        x = self.bn2(x)
        x = self.hidden_drop(x)

        x = torch.mm(x, self.enti_embedding.weight.transpose(1, 0))
        x += self.bias.expand_as(x)
        pred = torch.sigmoid(x)

        return pred
