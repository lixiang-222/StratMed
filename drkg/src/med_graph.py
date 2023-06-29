"""这个文件的工作是将 药物-化合物-疾病 这一组关系单独提取构图"""
import dill
import numpy as np
from tqdm import tqdm


# indexing file and final record
class Voc(object):
    def __init__(self):
        self.idx2word = {}
        self.word2idx = {}

    def add_sentence(self, sentence):
        for word in sentence:
            if word not in self.word2idx:
                self.idx2word[len(self.word2idx)] = word
                self.word2idx[word] = len(self.word2idx)

    def add_word(self, word):
        if word not in self.word2idx:
            self.idx2word[len(self.word2idx)] = word
            self.word2idx[word] = len(self.word2idx)



# 1.把所有的相关的信息整合到两个列表中
relation = []
with open("drkg.tsv", 'r') as f:
    for line in f:
        if "Compound:Atc" in line:
            relation.append(line.split('\t'))
        elif "Compound:Disease" in line:
            relation.append(line.split('\t'))

source_text = []
target_text = []
for line in relation:
    source_text.append(line[0])
    target_text.append(line[2].rstrip('\n'))

# 将所有ATC4的标准列成一个表格
atc = []
for item in target_text:  # 只在target中有药物的存在
    if "Atc" in item:
        if len(item) >= 9:
            atc1 = item[0:9]
            if atc1 in atc:
                continue
            else:
                atc.append(atc1)

# 删除达不到ATC4的编码的边
for i, target in enumerate(target_text):
    if "Atc" in target:
        for item in atc:
            if item in target:
                target_text[i] = item
                break
        else:
            target_text[i] = '-'
            source_text[i] = '-'

source_text1 = []
target_text1 = []
for source, target in zip(source_text, target_text):
    if source != '-':
        source_text1.append(source)
        target_text1.append(target)
edge = [source_text1, target_text1]

# 2.重新定义新的实体编号，构建邻接矩阵
kg_voc = Voc()
for item in source_text:
    kg_voc.add_word(item)

for item in target_text:
    kg_voc.add_word(item)

# 把上面写的图谱映射的东西放进文档中
with open("new_entites.txt", "w+") as f:
    for item in kg_voc.word2idx.items():
        f.write(item[0] + '\t' + str(item[1]) + '\n')

dill.dump(
    obj={"new_voc": kg_voc, "new_edge": edge},
    file=open("newfile", "wb"),
)

"""3.读文件，构建邻接矩阵，重新弄两条数组表示边的关系"""
f = dill.load(open("newfile", "rb"))
kg_voc = f["new_voc"]
edge = f["new_edge"]

# 构建邻接矩阵
graph = np.zeros((len(kg_voc.word2idx), len(kg_voc.word2idx)))

for source, target in zip(edge[0], edge[1]):
    graph[kg_voc.word2idx[source]][kg_voc.word2idx[target]] += 1

sources = []
targets = []
for i in range(len(kg_voc.word2idx)):
    for j in range(len(kg_voc.word2idx)):
        if graph[i][j] > 0:
            sources.append(i)
            targets.append(j)

sources1 = []
targets1 = []
for source, target in zip(sources, targets):
    sources1.append(kg_voc.idx2word[source])
    targets1.append(kg_voc.idx2word[target])

edge = [sources, targets]  # 用数字编码的边
relation = [sources1, targets1]  # 用文字编码的边

edge = dill.load(open("../output/disease_atc_edge.pkl", "rb"))

edge2 = []
edge2.append(edge[0] + edge[1])
edge2.append(edge[1] + edge[0])

# 生成数字编码构成的pkl
with open("../output/disease_atc_edge.pkl", "wb") as f:
    dill.dump(edge2, f)

# 生成文字编码构成的txt
# with open("new_relation.txt", "w+") as f:
#     for source, target in zip(relation[0], relation[1]):
#         f.write(source + '\t' + target + '\n')

"""4.让112种药物映射过来"""
voc = dill.load(open("voc_final.pkl", "rb"))

# 构建列表形式的药物映射
med_big_list = []
for item in voc["med_voc"].idx2word.items():
    med = [item[0], kg_voc.word2idx['Atc::' + item[1]], item[1]]
    med_big_list.append(med)
# with open('med_mapping_list.pkl', 'wb') as f:
#     dill.dump(med_big_voc, f)
with open("med_mapping_list.txt", "w+") as f:
    for line in med_big_list:
        f.write(str(line[0])+'\t'+str(line[1])+'\t'+line[2]+'\n')

# 构建字典形式的药物映射
# med_big_voc = {}
# for item in voc["med_voc"].idx2word.items():
#     med_big_voc[item[0]] =  kg_voc.word2idx['Atc::' + item[1]]
#
# with open('med_mapping.pkl', 'wb') as f:
#     dill.dump(med_big_voc, f)
