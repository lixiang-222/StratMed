import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from dnc import DNC
from torch.nn.parameter import Parameter

# from layers import GraphConvolution

"""
Our model
"""


class BasicModel(nn.Module):
    def __init__(
            self,
            vocab_size,
            ddi_adj,
            ddi_mask_H,
            emb_dim=256,
            device=torch.device("cpu:0"),
    ):
        super(BasicModel, self).__init__()

        self.device = device

        # pre-embedding
        self.embeddings = nn.ModuleList(
            [nn.Embedding(vocab_size[i], emb_dim) for i in range(3)]
        )
        self.dropout = nn.Dropout(p=0.5)
        self.encoders = nn.ModuleList(
            [nn.GRU(emb_dim, emb_dim, batch_first=True) for _ in range(3)]
        )
        self.query = nn.Sequential(nn.ReLU(), nn.Linear(3 * emb_dim, vocab_size[2]))

        # graphs, bipartite matrix
        self.tensor_ddi_adj = torch.FloatTensor(ddi_adj).to(device)
        self.tensor_ddi_mask_H = torch.FloatTensor(ddi_mask_H).to(device)
        self.init_weights()

    def forward(self, patient):
        # patient health representation
        i1_seq = []
        i2_seq = []
        i3_seq = []

        def sum_embedding(embedding):
            return embedding.sum(dim=1).unsqueeze(dim=0)  # (1,1,dim)

        adm = patient[-1]
        i1 = sum_embedding(
            self.dropout(
                self.embeddings[0](
                    torch.LongTensor(adm[0]).unsqueeze(dim=0).to(self.device)
                )
            )
        )  # (1,1,dim)
        i2 = sum_embedding(
            self.dropout(
                self.embeddings[1](
                    torch.LongTensor(adm[1]).unsqueeze(dim=0).to(self.device)
                )
            )
        )

        if len(patient) <= 1:
            # i3 = sum_embedding(
            #     self.dropout(
            #         self.embeddings[2](
            #             torch.LongTensor([112]).unsqueeze(dim=0).to(self.device)  # 要改，pading不能和已有编码冲突
            #         )
            #     )
            # )
            i3 = torch.zeros((1, 1, 64)).to(self.device)
        else:
            adm = patient[-2]
            i3 = sum_embedding(
                self.dropout(
                    self.embeddings[2](
                        torch.LongTensor(adm[2]).unsqueeze(dim=0).to(self.device)
                    )
                )
            )
        # i3_seq.append(i3)
        # for adm in patient:
        #     if len(i3_seq) < len(patient) - 1:
        #         i3 = sum_embedding(
        #             self.dropout(
        #                 self.embeddings[2](
        #                     torch.LongTensor(adm[2]).unsqueeze(dim=0).to(self.device)
        #                 )
        #             )
        #         )
        #     else:
        #         i3 = sum_embedding(
        #             self.dropout(
        #                 self.embeddings[2](
        #                     torch.LongTensor([112]).unsqueeze(dim=0).to(self.device)  # 要改，pading不能和已有编码冲突
        #                 )
        #             )
        #         )
        #
        #     # i1_seq.append(i1)
        #     # i2_seq.append(i2)
        #     i3_seq.append(i3)

        # i1_seq = torch.cat(i1_seq, dim=1)  # (1,seq,dim)
        # i2_seq = torch.cat(i2_seq, dim=1)  # (1,seq,dim)
        # i3_seq = torch.cat(i3_seq, dim=1)  # (1,seq,dim)

        # o1, h1 = self.encoders[0](i1_seq)
        # o2, h2 = self.encoders[1](i2_seq)
        # o3, _ = self.encoders[2](i3_seq)

        patient_representations = torch.cat([i1, i2, i3], dim=-1).squeeze(
            dim=0
        )  # (seq, dim*2)
        result = self.query(patient_representations)[-1:, :]  # (seq, dim)

        neg_pred_prob = F.sigmoid(result)
        neg_pred_prob = neg_pred_prob.t() * neg_pred_prob  # (voc_size, voc_size)

        batch_neg = 0.0005 * neg_pred_prob.mul(self.tensor_ddi_adj).sum()

        return result, batch_neg

    def init_weights(self):
        """Initialize weights."""
        initrange = 0.1
        for item in self.embeddings:
            item.weight.data.uniform_(-initrange, initrange)
