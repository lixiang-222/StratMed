import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.nn import GCNConv
from torch_geometric.typing import OptTensor, Adj

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
        self.emb_dim = emb_dim

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
            i3 = torch.zeros((1, 1, self.emb_dim)).to(self.device)
        else:
            adm = patient[-2]
            i3 = sum_embedding(
                self.dropout(
                    self.embeddings[2](
                        torch.LongTensor(adm[2]).unsqueeze(dim=0).to(self.device)
                    )
                )
            )

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


class GCN_SW(GCNConv):
    # pycharm让我加的，我没看懂
    def edge_update(self) -> torch.Tensor:
        pass

    def __init__(self, in_channels, out_channels, edge_types, normalize=False, bias=True):
        super(GCN_SW, self).__init__(in_channels, out_channels, normalize, bias)
        self.edge_types = edge_types

        self.edge_embedding = nn.Embedding(edge_types, 1)
        self.ddi_weight = nn.Parameter(torch.tensor(0.1))
        self.init_weight()

    def forward(self, x: Tensor, edge_index: Adj, edge_weight: OptTensor = None,
                ddi_weight: OptTensor = None) -> Tensor:
        x = self.lin(x)

        edge_weight = self.edge_embedding(edge_weight)
        ddi_weight = ddi_weight.unsqueeze(1) * self.ddi_weight
        edge_weight = edge_weight - ddi_weight

        out = self.propagate(edge_index, x=x, edge_weight=edge_weight, size=None)

        if self.bias is not None:
            out = out + self.bias

        # print(self.edge_embedding.weight.data)
        return out

    def init_weight(self):
        weight = 1 / self.edge_types
        for i in range(self.edge_types):
            self.edge_embedding.weight.data[i].copy_(weight * i)


class GCN_MF(GCNConv):
    # pycharm让我加的，我没看懂
    def edge_update(self) -> Tensor:
        pass

    def __init__(self, in_channels, out_channels, edge_types, normalize=False, bias=True):
        super(GCN_MF, self).__init__(in_channels, out_channels, normalize, bias)
        self.edge_types = edge_types
        self.edge_embedding = nn.Embedding(edge_types, in_channels)
        self.init_weight()

    def forward(self, x: Tensor, edge_index: Adj, edge_weight: OptTensor = None) -> Tensor:
        x = self.lin(x)

        edge_weight = self.edge_embedding(edge_weight)

        out = self.propagate(edge_index, x=x, edge_weight=edge_weight, size=None)


        if self.bias is not None:
            out = out + self.bias

        # print(self.edge_embedding.weight.data)
        return out

    def message(self, x_j: Tensor, edge_weight: OptTensor) -> Tensor:
        result = edge_weight * x_j
        return result

    def init_weight(self):
        weight = 0.8 / self.edge_types
        for i in range(self.edge_types):
            self.edge_embedding.weight.data[i].copy_(weight * i)


class GraphNet(nn.Module):
    def __init__(self, voc_size, emb_dim, pretrained_embedding, ehr_adj, ddi_adj, bucket_layer, device):
        super(GraphNet, self).__init__()
        self.device = device
        self.voc_size = voc_size
        self.pretrained_embedding = pretrained_embedding
        self.emb_dim = emb_dim
        self.ehr_adj = ehr_adj
        self.ddi_adj = ddi_adj

        # 嵌入数量+1表示一个padding位置
        self.item_embedding = nn.Embedding(voc_size[0] + voc_size[1] + voc_size[2], emb_dim)

        # 三个分着的同构图
        self.conv_med_med = GCN_SW(emb_dim, emb_dim, bucket_layer[2])

        # 异构图训练
        self.conv_hetero_diag_med = GCN_MF(emb_dim, emb_dim, bucket_layer[0])
        self.conv_hetero_pro_med = GCN_MF(emb_dim, emb_dim, bucket_layer[1])

        # 用来对抗过拟合
        self.dropout = nn.Dropout(p=0.5)

        # 用来链接残差
        self.residual_weight_diag = nn.Parameter(torch.tensor(1.0))
        self.residual_weight_proc = nn.Parameter(torch.tensor(1.0))
        self.residual_weight_med = nn.Parameter(torch.tensor(1.0))

        # 对各种Embedding的初始化
        self.init_weights()

        self.lin = nn.Linear(emb_dim, emb_dim)
        self.lin2 = nn.Linear(emb_dim,emb_dim)

    def forward(self, adm):
        """改变编号"""
        adm, new_adm, nodes = self.graph_id_merge(adm, self.voc_size)

        """同类型节点交互"""
        # 生成实体的嵌入（特征）
        x = torch.LongTensor(sorted(nodes)).to(self.device)
        x = self.item_embedding(x)

        # 如果是最后一次就诊，则不考虑药物）
        if len(new_adm[2]) == 0:
            # 生成实体的嵌入（特征）
            x1 = self.lin(x)

        else:
            # 生成边
            edges = self.build_isomorphic_edges(new_adm[2])
            edge_index = self.node2complete_graph2coo(nodes, edges)

            # 生成边的权重
            edge_weight = self.edges2weight(edges)
            ddi_weight = self.ddi2weight(edges)

            # 开始训练
            x1 = self.conv_med_med(x, edge_index, edge_weight, ddi_weight)
            x1 /= len(adm[2])
        x1 = F.relu(x1)  # 里面是按照诊断，程序，药物，顺序排列的节点的结果(n*64)

        """异构节点交互，不应该包含同构节点的交互"""
        # 构建诊断-药物的异构图
        edges = self.build_heterogeneous_edges(new_adm, 2, 0)
        if len(edges) != 0:
            edge_index = self.node2complete_graph2coo(nodes, edges)  # 利用图节点和边变成pyg需要的coo模式
            edge_weight = self.edges2weight(edges)
            x2 = self.conv_hetero_diag_med(x1, edge_index, edge_weight)  # n*64
            x2 /= len(adm[2])
            x2 = F.relu(x2)
        else:
            x2 = torch.zeros([len(nodes), self.emb_dim], dtype=torch.float).to(self.device)

        # 构建程序-药物的异构图
        edges = self.build_heterogeneous_edges(new_adm, 2, 1)
        if len(edges) != 0:
            edge_index = self.node2complete_graph2coo(nodes, edges)  # 利用图节点和边变成pyg需要的coo模式
            edge_weight = self.edges2weight(edges)
            x3 = self.conv_hetero_pro_med(x1, edge_index, edge_weight)  # n*64
            x3 /= len(adm[2])
            x3 = F.relu(x3)
        else:
            x3 = torch.zeros([len(nodes), self.emb_dim], dtype=torch.float).to(self.device)

        x_out = x1 + x2 + x3
        # x_out = self.lin2(x_out)
        x_out = self.dropout(x_out)  # 不知道加在这里对不对

        """训练完毕，准备输出"""
        i1, i2, i3 = self.residual_output(x_out, adm)

        # print("0号诊断",self.item_embedding.weight.data[0][0:3])
        # print("0号程序",self.item_embedding.weight.data[1958][0:3])
        # m = 107
        # if m in adm[2]:
        #     print(m, "号药物嵌入", "现在：", self.item_embedding.weight.data[3388 + m][0:3],
        #           "预训练：", self.pretrained_embedding[2].weight.data[m][0:3])
        # print("padding:", self.pad_embedding.weight.data[0][0:3])

        return i1, i2, i3

    def ddi2weight(self, edges):
        ddi_weight = []
        for edge in edges:
            ddi_weight.append(self.ddi_adj[edge[0] - self.voc_size[0] - self.voc_size[1]][
                                  edge[1] - self.voc_size[0] - self.voc_size[1]])
        ddi_weight = torch.LongTensor(ddi_weight).to(self.device)
        return ddi_weight

    def graph_id_merge(self, adm, voc_size):
        """将一次就诊（会话）中的所有节点输入之后将节点编号合在一个体系内"""
        new_adm = [[] for _ in range(3)]
        for diag in adm[0]:
            new_adm[0].append(diag)
        for pro in adm[1]:
            new_adm[1].append(pro + voc_size[0])
        for med in adm[2]:
            new_adm[2].append(med + voc_size[0] + voc_size[1])

        adm = [sorted(adm[0]), sorted(adm[1]), sorted(adm[2])]
        new_adm = [sorted(new_adm[0]), sorted(new_adm[1]), sorted(new_adm[2])]
        nodes = sorted(new_adm[0] + new_adm[1] + new_adm[2])

        return adm, new_adm, nodes

    def node2complete_graph2coo(self, nodes, edges):
        """
        将节点原来的编号转换为pyg的格式

        输入是原编号下的节点和边
        nodes:[0,4,2,3]
        edges:[[0,4],[4,2],[2,3]]

        输出是用于pyg的coo格式的边
        edge_index:tensor((0,3),(3,1),(1,2))
        """
        # 重新排列数字，按照pyg的格式输输出
        if len(edges) == 0:
            edge_index = torch.combinations(torch.LongTensor([0])).t().contiguous()
        else:
            voc = {}
            for i, node in enumerate(nodes):
                voc[node] = i
            edge_index = []
            for edge in edges:
                edge_index.append((voc[edge[0]], voc[edge[1]]))
            edge_index = torch.LongTensor(edge_index).t().contiguous()  # 列表变成coo形式

        edge_index = edge_index.to(self.device)

        return edge_index

    def build_isomorphic_edges(self, nodes):
        """通过一串节点构建一个完全图（同构）"""
        edges = []
        for node1 in nodes:
            for node2 in nodes:
                if node1 != node2:
                    edges.append([node1, node2])

        return edges

    def build_heterogeneous_edges(self, new_adm, type_a, type_b):
        """在两种节点之间建立异构的单向图（同种节点无连接）"""

        edges = []
        for node1 in new_adm[type_a]:
            for node2 in new_adm[type_b]:
                edges.append([node1, node2])

        return edges

    def edges2weight(self, edges):
        edge_weight = []
        for j in range(len(edges)):
            weight = self.ehr_adj[edges[j][0]][edges[j][1]]
            edge_weight.append(weight)
        edge_weight = torch.LongTensor(edge_weight).to(self.device)
        return edge_weight

    def residual_output(self, x, adm):
        """统计预训练的代码和训练之后的代码，后面加在一起做残差"""

        embedding_pretrained = [torch.zeros([1, self.emb_dim], dtype=torch.float).to(self.device) for _ in range(3)]
        embedding_trained = [torch.zeros([1, self.emb_dim], dtype=torch.float).to(self.device) for _ in range(3)]

        i = 0
        for diag in adm[0]:
            embedding_pretrained[0] += self.pretrained_embedding[0].weight.data[diag]
            embedding_trained[0] += x[i]
            i += 1
        for pro in adm[1]:
            embedding_pretrained[1] += self.pretrained_embedding[1].weight.data[pro]
            embedding_trained[1] += x[i]
            i += 1

        if len(adm[2]) == 0:  # 加上这个模块，padding的是一个嵌入，不加这个模块，padding的是一个全0嵌入
            embedding_trained[2] = torch.zeros((1, self.emb_dim)).to(self.device)
        for med in adm[2]:
            embedding_pretrained[2] += self.pretrained_embedding[2].weight.data[med]
            embedding_trained[2] += x[i]
            i += 1

        # 做一个简单的残差

        i1 = embedding_pretrained[0] + embedding_trained[0]
        i2 = embedding_pretrained[1] + embedding_trained[1]
        i3 = embedding_pretrained[2] + embedding_trained[2]

        return i1.unsqueeze(0), i2.unsqueeze(0), i3.unsqueeze(0)

    def init_weights(self):
        """初始化权重"""
        # 项目嵌入的初始化（用预训练的值来初始化）
        self.item_embedding.weight.data[:self.voc_size[0]].copy_(self.pretrained_embedding[0].weight.data)
        self.item_embedding.weight.data[self.voc_size[0]:self.voc_size[0] + self.voc_size[1]].copy_(
            self.pretrained_embedding[1].weight.data)
        self.item_embedding.weight.data[
        self.voc_size[0] + self.voc_size[1]:self.voc_size[0] + self.voc_size[1] + self.voc_size[2]].copy_(
            self.pretrained_embedding[2].weight.data)


class AdvancedModel(nn.Module):
    def __init__(
            self,
            vocab_size,
            ddi_adj,
            ddi_mask_H,
            ehr_adj,
            bucket_layer,
            emb_dim=256,
            device=torch.device("cpu:0"),
            pretrained_embeddings=None
    ):
        super(AdvancedModel, self).__init__()

        self.device = device

        self.graph_embeddings = GraphNet(vocab_size, emb_dim, pretrained_embeddings, ehr_adj, ddi_adj, bucket_layer,
                                         device)

        self.encoders = nn.ModuleList(
            [nn.GRU(emb_dim, emb_dim, batch_first=True) for _ in range(3)]
        )
        self.query = nn.Sequential(nn.ReLU(), nn.Linear(3 * emb_dim, vocab_size[2]))

        # graphs, bipartite matrix
        self.tensor_ddi_adj = torch.FloatTensor(ddi_adj).to(device)
        self.tensor_ddi_mask_H = torch.FloatTensor(ddi_mask_H).to(device)

    def forward(self, patient):
        # patient health representation
        i1_seq = []
        i2_seq = []
        i3_seq = []

        for adm_num, adm in enumerate(patient):
            if adm_num == 0:  # 是第一次来
                adm_new = adm[:]
                adm_new[2] = []
                i1, i2, i3 = self.graph_embeddings(adm_new)
            else:  # 不是第一次来
                adm_new = adm[:]
                adm_new[2] = patient[adm_num - 1][2][:]
                i1, i2, i3 = self.graph_embeddings(adm_new)

            i1_seq.append(i1)
            i2_seq.append(i2)
            i3_seq.append(i3)

        i1_seq = torch.cat(i1_seq, dim=1)  # (1,seq,dim)
        i2_seq = torch.cat(i2_seq, dim=1)  # (1,seq,dim)
        i3_seq = torch.cat(i3_seq, dim=1)  # (1,seq,dim)

        o1, _ = self.encoders[0](i1_seq)
        o2, _ = self.encoders[1](i2_seq)
        o3, _ = self.encoders[2](i3_seq)

        patient_representations = torch.cat([o1, o2, o3], dim=-1).squeeze(0)  # (seq, dim*3)
        result = self.query(patient_representations)[-1:, :]  # (1, 药物向量)

        neg_pred_prob = F.sigmoid(result)
        neg_pred_prob = neg_pred_prob.t() * neg_pred_prob  # (voc_size, voc_size)

        batch_neg = 0.0005 * neg_pred_prob.mul(self.tensor_ddi_adj).sum()

        return result, batch_neg
