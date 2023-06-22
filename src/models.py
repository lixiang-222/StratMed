import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GATConv
from torch_geometric.utils import k_hop_subgraph

"""用来代替nn.Embedding的图网络"""


class MyGraphNet(torch.nn.Module):
    def __init__(self, item_number, embed_dim, device):
        super(MyGraphNet, self).__init__()
        self.device = device
        self.item_embedding = torch.nn.Embedding(num_embeddings=item_number, embedding_dim=embed_dim)

        self.conv1 = GATConv(embed_dim, 128)
        # self.pool1 = TopKPooling(128, ratio=0.8)
        self.conv2 = GATConv(128, 128)
        # self.pool2 = TopKPooling(128, ratio=0.8)
        self.conv3 = GATConv(128, 128)
        # self.pool3 = TopKPooling(128, ratio=0.8)
        self.dropout = nn.Dropout(p=0.3)

        self.lin1 = torch.nn.Linear(128, embed_dim)

        self.act1 = torch.nn.ReLU()

    def forward(self, data, target_item=None):
        x, edge_index, batch = data.x, data.edge_index, data.batch  # x:n*1,其中每个图里点的个数是不同的
        x = x.to(self.device)
        edge_index = edge_index.to(self.device)
        x = self.item_embedding(x)  # n*1*64 特征编码后的结果
        x = x.squeeze(1)  # n*64

        x1 = F.relu(self.conv1(x, edge_index))  # n*128
        x2 = F.relu(self.conv2(x1, edge_index))  # n*256
        x3 = F.relu(self.conv3(x2, edge_index))  # n*256

        x = x1 + x2 + x3

        x = self.lin1(x)
        x = self.act1(x)
        x = F.dropout(x, p=0.5, training=self.training)  # 这里的x是一个(1,64)的嵌入了，表示的是一个图

        x = x[target_item]

        x = x.sum(dim=0).unsqueeze(dim=0)

        return x


class SafeDrugModel(nn.Module):
    def __init__(
            self,
            vocab_size,
            ddi_adj,
            ddi_mask_H,
            emb_dim=256,
            device=torch.device("cpu:0")
    ):
        super(SafeDrugModel, self).__init__()

        self.device = device

        # 自己构建的疾病图嵌入(应该没用了）
        self.diagnose_graph_emb = MyGraphNet(1958, emb_dim, device=device)

        # 自己构建的药物图嵌入
        self.big_kg = MyGraphNet(14648, emb_dim, device=device)

        # 构建疾病，程序，药物的embedding 用64维来构建  这里的+1是用来给padding用的
        self.embeddings = nn.ModuleList([nn.Embedding(vocab_size[i] + 1, emb_dim) for i in range(3)])

        # 构建5种基本信息的embedding 用64维构建
        self.basic_embeddings = nn.ModuleList([nn.Embedding(vocab_size[j], emb_dim) for j in range(3, 8)])

        # 从用图谱计算好的药物embedding中映射过来，只是看看效果
        self.med_fix_embedding_layer = nn.Linear(400, emb_dim)

        self.dropout = nn.Dropout(p=0.5)

        # 诊断、程序、药物都用GRU
        self.encoders = nn.ModuleList(
            [nn.GRU(emb_dim, emb_dim, batch_first=True) for _ in range(3)]
        )

        # 给基本信息整合用的线性层
        self.basic_linear = nn.Linear(5 * emb_dim, emb_dim)

        # 患者表示最后的表示
        self.query = nn.Sequential(nn.ReLU(), nn.Linear(2 * emb_dim, emb_dim))
        # self.query = nn.Sequential(nn.ReLU(), nn.Linear(emb_dim, emb_dim))

        self.result_mlp = nn.Linear(2 * emb_dim, 112)

        # 拼接之后直接走线性层输出
        self.result_linear = nn.Sequential(
            nn.Linear(3 * emb_dim, 2 * emb_dim),
            nn.ReLU(),
            nn.Linear(2 * emb_dim, 112)
        )

        # 原文中的东西，不敢删
        self.tensor_ddi_adj = torch.FloatTensor(ddi_adj).to(device)
        self.tensor_ddi_mask_H = torch.FloatTensor(ddi_mask_H).to(device)
        # self.init_weights()

    def forward(self, input, med_kg_voc, kg_edge):
        # patient health representation
        # i1_seq = []
        # i2_seq = []
        i3_seq = []

        def sum_embedding(embedding):
            return embedding.sum(dim=1).unsqueeze(dim=0)  # (1,1,dim)

        # def basic_information_process(input):  # 可以先做线性变换
        #     # 5个信息构成5个(1,1,64)的表示
        #     b1, b2, b3, b4, b5 = \
        #         [self.basic_embeddings[i](torch.LongTensor([input[0][3][i]]).unsqueeze(dim=0).to(self.device)) for i
        #          in range(5)]
        #
        #     # 5个表示合成一个(1,5*64)
        #     b_seq = torch.cat([b1, b2, b3, b4, b5], dim=-1).squeeze(  # 拼接输出(n,4*64) n表示有几次的就诊记录
        #         dim=0
        #     )
        #     # （1,5,64） 过一个线性层，不要直接加起来
        #     b_seq = self.basic_linear(b_seq)
        #     b = self.dropout(b_seq)
        #
        #     # 因为后面要和疾病/程序拼接，这里把基本数据扩充到和就诊次数一样的维度（用于timesNet的时候不要这个操作）
        #     b = b.repeat(1, len(input), 1)
        #     return b

        """自己加的，加入从知识图谱学习到的药物表示"""
        # med_fix_embedding = np.load("../data/output/med_embedding.npy")
        # med_fix_embedding = torch.tensor(med_fix_embedding).to(self.device)
        # med_fix_embedding = self.med_fix_embedding_layer(med_fix_embedding)  # 出来之后是还有个（1，112，64）的向量

        """原文中的需要gru的代码"""
        # for adm in input:
        #
        #     """对诊断的图嵌入"""
        #     # sess_item_id = LabelEncoder().fit_transform(adm[0])
        #     # node_features = sorted(adm[0])
        #     # x = torch.LongTensor(node_features).unsqueeze(1)  # 后面要做嵌入的变形
        #     #
        #     # # 将所有的节点都连起来
        #     # source_nodes = []
        #     # for item in sess_item_id:
        #     #     source_nodes += [item] * (len(sess_item_id) - 1)
        #     #
        #     # target_nodes = []
        #     # for item in sess_item_id:
        #     #     a = list(sess_item_id)
        #     #     a.remove(item)
        #     #     target_nodes += a
        #     #
        #     # edge_index = torch.tensor([source_nodes, target_nodes], dtype=torch.long)
        #     #
        #     # y = torch.FloatTensor([1])
        #     #
        #     # data = Data(x=x, edge_index=edge_index, y=y)
        #     #
        #     # i1 = self.diagnose_graph_emb(data)
        #     # i1 = i1.unsqueeze(dim=0)
        #     """"""
        #
        #     i1 = sum_embedding(
        #         self.dropout(
        #             self.embeddings[0](
        #                 torch.LongTensor(adm[0]).unsqueeze(dim=0).to(self.device)
        #             )
        #         )
        #     )  # (1,1,dim)
        #     i2 = sum_embedding(
        #         self.dropout(
        #             self.embeddings[1](
        #                 torch.LongTensor(adm[1]).unsqueeze(dim=0).to(self.device)
        #             )
        #         )
        #     )
        #
        #     """作弊试试"""
        #     # i3= sum_embedding(
        #     #     self.dropout(
        #     #         self.embeddings[2](
        #     #             torch.LongTensor(adm[2]).unsqueeze(dim=0).to(self.device)
        #     #         )
        #     #     )
        #     # )
        #     # i3_seq.append(i3)
        #     """"""
        #
        #     i1_seq.append(i1)
        #     i2_seq.append(i2)
        #
        #     """后来改的利用图谱加入的嵌入"""
        #     # if len(i3_seq) < len(input) - 1:
        #     #     i3 = [0 for _ in range(64)]
        #     #     i3 = torch.tensor(i3).to(self.device)
        #     #     # 把所有涉及到的嵌入挨个加进来
        #     #     for item in adm[2]:
        #     #         i3 = i3 + med_fix_embedding[item]
        #     #     i3 = i3.view(1, 1, 64)
        #     # else:
        #     #     i3 = sum_embedding(
        #     #         self.dropout(
        #     #             self.embeddings[2](
        #     #                 torch.LongTensor([112]).unsqueeze(dim=0).to(self.device)  # 要改，pading不能和已有编码冲突
        #     #             )
        #     #         )
        #     #     )
        #
        #     """对药物图谱的嵌入"""
        #     if len(i3_seq) < len(input) - 1:
        #         # 1.先将112的药物编号映射到整个图谱之中
        #         med_seq = []  # 代表了本次就诊中药物在知识图谱中的编号
        #         for med in adm[2]:
        #             med_seq.append(med_kg_voc[med])
        #
        #         # 2.将整个图和标志都输入到图神经网络之中
        #         node_features = sorted(list(range(14648)))
        #         x = torch.LongTensor(node_features).unsqueeze(1)  # 后面要做嵌入的变形
        #
        #         edge_index = torch.tensor(kg_edge, dtype=torch.long)
        #
        #         data = Data(x=x, edge_index=edge_index)
        #
        #         i3 = self.med_graph_emb(data, target_item=med_seq)
        #         i3 = i3.unsqueeze(dim=0)
        #     else:
        #         i3 = sum_embedding(
        #             self.dropout(
        #                 self.embeddings[2](
        #                     torch.LongTensor([112]).unsqueeze(dim=0).to(self.device)  # 要改，pading不能和已有编码冲突
        #                 )
        #             )
        #         )
        #     #
        #     #     """正常利用embedding层做的嵌入"""
        #     #     if len(i3_seq) < len(input) - 1:
        #     #         i3 = sum_embedding(
        #     #             self.dropout(
        #     #                 self.embeddings[2](
        #     #                     torch.LongTensor(adm[2]).unsqueeze(dim=0).to(self.device)
        #     #                 )
        #     #             )
        #     #         )
        #     #     else:
        #     #         i3 = sum_embedding(
        #     #             self.dropout(
        #     #                 self.embeddings[2](
        #     #                     torch.LongTensor([112]).unsqueeze(dim=0).to(self.device)  # 要改，pading不能和已有编码冲突
        #     #                 )
        #     #             )
        #     #         )
        #     #
        #     i3_seq.append(i3)
        #
        # i1_seq = torch.cat(i1_seq, dim=1)  # (1,seq,dim)
        # i2_seq = torch.cat(i2_seq, dim=1)  # (1,seq,dim)
        # i3_seq = torch.cat(i3_seq, dim=1)  # (1,seq,dim)
        #
        # # 让两个64维的表示，分别通过GRU，生成output和hidden（hidden没用）
        # o1, _ = self.encoders[0](i1_seq)
        # o2, _ = self.encoders[1](i2_seq)
        # o3, _ = self.encoders[2](i3_seq)

        """对诊断和程序不用历史信息直接映射，对药物历史信息用gru表示"""
        adm = input[-1]
        i1 = sum_embedding(
            self.dropout(
                self.embeddings[0](
                    torch.LongTensor(adm[0]).unsqueeze(dim=0).to(self.device)
                )
            )
        )
        i2 = sum_embedding(
            self.dropout(
                self.embeddings[0](
                    torch.LongTensor(adm[1]).unsqueeze(dim=0).to(self.device)
                )
            )
        )

        for adm in input:
            if len(i3_seq) < len(input) - 1:
                """方法一：正常利用embedding层做的嵌入"""
                # i3 = sum_embedding(
                #     #             self.dropout(
                #     #                 self.embeddings[2](
                #     #                     torch.LongTensor(adm[2]).unsqueeze(dim=0).to(self.device)
                #     #                 )
                #     #             )
                #     #         )

                """方法二：对药物图谱的嵌入"""
                # 把边变成tensor形式
                edge_index = torch.tensor(kg_edge, dtype=torch.long)

                # 代表了本次就诊中药物在知识图谱中的编号
                med_seq = []
                for med in adm[2]:
                    med_seq.append(med_kg_voc[med])

                # geometric中自带的函数k跳子图，返回的
                # subset是子图中的节点（图谱编码，从小到大排好顺序的）
                # sub_edge_index是子图中的边（按照子图定义的新编码）
                # mapping是最开始的几个节点（med_seq）在子图新编码中的对照
                # edge_mask没有用到，也不知道是什么
                sub_set, sub_edge_index, mapping, edge_mask = k_hop_subgraph(med_seq, 2, edge_index, relabel_nodes=True)

                # 构建data，输入到大图中
                data = Data(x=sub_set, edge_index=sub_edge_index)
                i3 = self.big_kg(data, target_item=mapping)

                i3 = i3.unsqueeze(dim=0)
            else:
                i3 = sum_embedding(
                    self.dropout(
                        self.embeddings[2](
                            torch.LongTensor([112]).unsqueeze(dim=0).to(self.device)  # 要改，pading不能和已有编码冲突
                        )
                    )
                )

            i3_seq.append(i3)
        i3_seq = torch.cat(i3_seq, dim=1)  # (1,seq,dim)
        o3, _ = self.encoders[2](i3_seq)


        """"""

        """自己写的集成基本数据的方法，在用rnn的时候把基本数据扩充到和就诊次数一样的维度，不用rnn时候不用扩充"""
        # basic = basic_information_process(input)

        """这里做了改动，加入了新的b和o3"""
        patient_representations = torch.cat([i1, i2, i3], dim=-1).squeeze(  # 拼接输出(1,3*64)
            dim=0
        )  # (seq, dim*3)

        # # 原文中用的计算患者表示query的方式
        # query = self.query(patient_representations)[-1:, :]  # (seq, dim)

        # # MPNN embedding
        # MPNN_match = F.sigmoid(torch.mm(query, self.MPNN_emb.t()))
        # MPNN_att = self.MPNN_layernorm(MPNN_match + self.MPNN_output(MPNN_match))
        #
        # # local embedding
        # bipartite_emb = self.bipartite_output(
        #     F.sigmoid(self.bipartite_transform(query)), self.tensor_ddi_mask_H.t()
        # )

        # 原文方法：用到的局部信息和全局信息的点击融合
        # result = torch.mul(bipartite_emb, MPNN_att)

        """自己改写的匹配方法"""
        # 方法一：利用患者表示和112个药物表示分别内积，生成的是（1，112）的一个矩阵，代表患者和每个药物的匹配值，无法收敛！！
        # result = torch.matmul(query, med_fix_embedding.t())

        # 方法二：把患者表示复制成112个，把112个药物表示跟他横着拼接起来，穿过一个线性层
        # query = query.repeat(1, 112, 1)  # (1,64)->(1,112,64)
        # query = query.view(1, 112 * 64)  # (1,112,64)->(1,112*64)

        # med_embedding = med_fix_embedding.view(1, 112 * 64)  # (112,64)->(1,112*64)
        # med_embedding = med_fix_embedding.unsqueeze(dim=0)  # (112,64)->(1,112,64)

        # 方法三：不利用知识图谱的固定嵌入，直接利用前面训练的药物嵌入和这里的药物嵌入进行拼凑
        # med_embedding = self.embeddings[2](torch.LongTensor(list(range(112))).to(self.device))
        #
        # med_embedding = med_embedding.view(1, 112 * 64)
        # result = self.result_mlp(torch.cat([query, med_embedding], dim=-1))  # (1,112,64*2)->(1,112,1)
        # result = result.squeeze(dim=-1)
        # 应该是竖着拼接，然后是（128-1）线性层

        # 方法四：直接把患者的表示放进来，通过几个线性层来看
        result = self.result_linear(patient_representations)[-1:, :]
        # result = torch.cat([query, o3[:, -1, :]], dim=-1)
        # result = self.result_mlp(result)[-1:, :]

        """"""

        neg_pred_prob = F.sigmoid(result)
        neg_pred_prob = neg_pred_prob.t() * neg_pred_prob  # (voc_size, voc_size)

        batch_neg = 0.0005 * neg_pred_prob.mul(self.tensor_ddi_adj).sum()

        return result, batch_neg
