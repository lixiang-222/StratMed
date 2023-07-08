import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.utils as utils
from sklearn.preprocessing import LabelEncoder
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv

"""用来代替nn.Embedding的图网络"""


class GraphNet(torch.nn.Module):
    def __init__(self, num_embeddings, device):
        super(GraphNet, self).__init__()
        self.device = device

        self.item_embedding = nn.Embedding(num_embeddings=num_embeddings, embedding_dim=64)

        self.conv1 = GCNConv(64, 128)
        self.lin1 = torch.nn.Linear(128, 64)
        self.relu = nn.ReLU()

    def forward(self, data):
        x = data.x.to(self.device)
        edge_index = data.edge_index.to(self.device)
        x = self.item_embedding(x).squeeze(1)  # n*1*64 特征编码后的结果

        x = F.relu(self.conv1(x, edge_index))  # n*128
        x = self.lin1(x)
        return x.sum(dim=0).unsqueeze(dim=0).unsqueeze(dim=0)


class ResidualLayer(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(ResidualLayer, self).__init__()
        self.linear1 = nn.Linear(input_dim, output_dim)
        self.linear2 = nn.Linear(output_dim, output_dim)

    def forward(self, x):
        out = self.linear1(x)
        out = torch.relu(out)
        out1 = self.linear2(out)
        return out1 + out  # 残差连接


class SafeDrugModel(nn.Module):
    def __init__(
            self,
            vocab_size,
            ddi_adj,
            ddi_mask_H,
            emb_dim=256,
            device=torch.device("cpu:0"),
            pretrained_embed=None
    ):
        super(SafeDrugModel, self).__init__()

        self.device = device

        # 构建疾病，程序，药物的embedding 用64维来构建  这里的+1是用来给padding用的
        self.embeddings = nn.ModuleList([nn.Embedding(vocab_size[i] + 1, emb_dim) for i in range(3)])
        """用于ehr和ddi构图"""
        self.graph_embeddings = nn.ModuleList([GraphNet(vocab_size[i], device) for i in range(3)])
        # 从用图谱计算好的药物embedding中映射过来，只是看看效果
        self.med_fix_embedding_layer = nn.Linear(400, emb_dim)

        self.dropout = nn.Dropout(p=0.5)

        # 诊断、程序、药物都用GRU
        self.encoders = nn.ModuleList(
            [nn.GRU(emb_dim, emb_dim, batch_first=True) for _ in range(3)]
        )

        # 拼接之后直接走线性层输出
        self.result_linear = nn.Sequential(
            nn.ReLU(),
            ResidualLayer(2 * emb_dim, vocab_size[2])
        )

        # 原文中的东西，不敢删
        self.tensor_ddi_adj = torch.FloatTensor(ddi_adj).to(device)
        self.tensor_ddi_mask_H = torch.FloatTensor(ddi_mask_H).to(device)

    def forward(self, input):
        # patient health representation
        # i0_seq = []
        # i1_seq = []
        # i2_seq = []

        def sum_embedding(embedding):
            return embedding.sum(dim=1).unsqueeze(dim=0)  # (1,1,dim)

        """对诊断和程序不用历史信息直接映射，对药物历史信息用gru表示"""
        adm = input[-1]
        """用nn.Embedding直接做嵌入"""
        x = torch.LongTensor(sorted(adm[0]))
        session_id = LabelEncoder().fit_transform(adm[0])  # 适用于GAT的独特方法
        edge_index = torch.combinations(torch.LongTensor(session_id)).t().contiguous()
        data = Data(x=x.unsqueeze(1), edge_index=utils.to_undirected(edge_index))  # 这里是有向边，我要给他弄成无向的
        i0 = self.graph_embeddings[0](data)

        x = torch.LongTensor(sorted(adm[1]))
        session_id = LabelEncoder().fit_transform(adm[1])  # 适用于GAT的独特方法
        edge_index = torch.combinations(torch.LongTensor(session_id)).t().contiguous()
        data = Data(x=x.unsqueeze(1), edge_index=utils.to_undirected(edge_index))  # 这里是有向边，我要给他弄成无向的
        i1 = self.graph_embeddings[1](data)

        # """使用药物padding的方法"""
        # for adm in input:
        #     if len(i2_seq) < len(input) - 1:
        #         """方法四：用ehr和ddi图构图，然后再图上训练药物"""
        #
        #         x = torch.LongTensor(sorted(adm[2]))
        #         session_id = LabelEncoder().fit_transform(adm[2])  # 适用于GAT的独特方法
        #         edge_index = torch.combinations(torch.LongTensor(session_id)).t().contiguous()
        #         data = Data(x=x.unsqueeze(1), edge_index=utils.to_undirected(edge_index))  # 这里是有向边，我要给他弄成无向的
        #         i2 = self.graph_embeddings[2](data)
        #         # ddi_file = "../data/output/ddi_A_final.pkl"
        #     else:
        #         """方法二：用一个新的嵌入来代替"""
        #         i2 = sum_embedding(
        #             self.dropout(
        #                 self.embeddings[2](
        #                     torch.LongTensor([112]).unsqueeze(dim=0).to(self.device)
        #                 )
        #             )
        #         )
        #
        #     i2_seq.append(i2)
        # i2_seq = torch.cat(i2_seq, dim=1)  # (1,seq,dim)
        # o2, _ = self.encoders[2](i2_seq)

        patient_representations = torch.cat([i0, i1], dim=-1).squeeze(  # 拼接输出(1,3*64)
            dim=0
        )  # (seq, dim*3)
        # patient_representations = torch.cat([i0, i1, o2[:, -1:, :]], dim=-1).squeeze(  # 拼接输出(1,3*64)
        #     dim=0
        # )  # (seq, dim*3)
        """"""

        """自己改写的匹配方法"""
        # 方法四：直接把患者的表示放进来，通过几个线性层来看
        result = self.result_linear(patient_representations)

        """"""

        neg_pred_prob = F.sigmoid(result)
        neg_pred_prob = neg_pred_prob.t() * neg_pred_prob  # (voc_size, voc_size)

        batch_neg = 0.0005 * neg_pred_prob.mul(self.tensor_ddi_adj).sum()

        return result, batch_neg
