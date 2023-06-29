"""此文件集合了所有用于图谱映射的方法"""

import dill
import numpy as np
from tqdm import tqdm


# SafeDrug里面编写的一种字典的类
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


"""0.读取所有文件"""
# 0.1读取图谱中的实体
kg_entity = []
with open("../input/entities.tsv", 'r') as f:
    for line in f:
        kg_entity.append(line.rstrip('\n').split('\t'))

# 0.2读取数据集中的诊断，药物的信息
data_voc = dill.load(open("../input/voc_final.pkl", "rb"))

# 0.3读取图谱实体之间的关系
relation = []
with open("../input/drkg.tsv", 'r') as f:
    for line in f:
        if "Compound:Atc" in line:
            relation.append(line.rstrip('\n').split('\t'))
        elif "Compound:Disease" in line:
            relation.append(line.rstrip('\n').split('\t'))

"""1.从大图谱中抽取【药物-化合物-疾病】之间的关系，构成新的小图谱"""

print("---1.开始构建小图谱---")

# 1.1把大图谱中的相关的信息合并到一个表中
source_text = []
target_text = []
for line in relation:
    source_text.append(line[0])
    target_text.append(line[2])

# 1.2将所有达不到ATC4标准关系的删除，同时只保留atc4级别的编码
source_text1 = []
target_text1 = []
for source, target in zip(source_text, target_text):  # 在关系中只有target有药物
    if "Atc" in target:  # 药物的部分需要筛选
        if len(target) >= 9:
            atc4 = target[0:9]
            source_text1.append(source)
            target_text1.append(atc4)
    else:  # 不是药物的部分不用筛选
        source_text1.append(source)
        target_text1.append(target)
edges = [source_text1, target_text1]

# 1.3得到新图谱编号（小图谱）和实体名称之间的映射关系
# kg_entities_voc = dill.load(open("../output_pkl/new_kg_entities.pkl", 'rb'))
kg_entities_voc = Voc()

for item in edges[0]:
    kg_entities_voc.add_word(item)

for item in edges[1]:
    kg_entities_voc.add_word(item)

# 1.4得到新图谱编号（小图谱）和旧图谱（大图谱）之间的映射关系

# newKG_oldKG_map_voc = dill.load(open("../output_pkl/newKG_oldKG_map.pkl",'rb'))
newKG_oldKG_map_voc = Voc()
for item in tqdm(kg_entities_voc.idx2word.items()):
    for line in kg_entity:
        if item[1] == line[0]:
            newKG_oldKG_map_voc.add_word(line[1])
            break
    else:
        print("写映射的时候出bug了，这次整个编码都是错的")

newKG_oldKG_map_list = []
for item in newKG_oldKG_map_voc.idx2word.items():
    entity = [item[0], kg_entities_voc.idx2word[item[0]], item[1]]
    newKG_oldKG_map_list.append(entity)

# 写文件
dill.dump(newKG_oldKG_map_voc, file=open("../output_pkl/newKG_oldKG_map.pkl", "wb"))

dill.dump(kg_entities_voc, file=open("../output_pkl/new_kg_entities.pkl", "wb"))

with open("../output_txt/new_kg_entities.txt", "w+") as f:
    for item in kg_entities_voc.word2idx.items():
        f.write(item[0] + '\t' + str(item[1]) + '\n')

with open("../output_txt/newKG_oldKG_map.txt", "w+") as f:
    for line in newKG_oldKG_map_list:
        f.write(str(line[0]) + '\t' + str(line[1]) + '\t' + str(line[2]) + '\n')

# 1.5之前因为节点合并，导致存在相同的两条边，这里将不同的边给合并 （构建邻接矩阵的方法）
graph = np.zeros((len(kg_entities_voc.word2idx), len(kg_entities_voc.word2idx)))

for source, target in zip(edges[0], edges[1]):
    graph[kg_entities_voc.word2idx[source]][kg_entities_voc.word2idx[target]] += 1

# 生成的是经过处理的边
nonzero_indices = np.nonzero(graph)
sources = nonzero_indices[0]
targets = nonzero_indices[1]
print(len(sources))
print(len(targets))

sources1 = []
targets1 = []
for source, target in zip(sources, targets):
    sources1.append(kg_entities_voc.idx2word[source])
    targets1.append(kg_entities_voc.idx2word[target])
print(len(sources1))
print(len(targets1))

edges1 = [sources, targets]  # 用数字编码的边，单向边
edges2 = [edges1[0] + edges1[1], edges1[1] + edges1[0]]  # edges2做的是双向边

relation = [sources1, targets1]  # 用文字编码的边

# 写文件

with open("../output_txt/new_kg_edges.txt", "w+") as f:
    for source, target in zip(relation[0], relation[1]):
        f.write(str(source) + '\t' + str(target) + '\n')

dill.dump(edges2, file=open("../output_pkl/new_kg_edges.pkl", "wb"))

print("---1.小图谱构建完成---")

"""2.从数据集的atc到图谱中atc的映射"""

print("---2.开始转化数据集中的ATC和图谱中的ATC---")
kg_voc = dill.load(open("../output_pkl/new_kg_entities.pkl", "rb"))  # 省时间用的

# 2.1构建药物在图谱中的映射的列表
med_map_list = []
for item in data_voc["med_voc"].idx2word.items():
    med = [item[0], kg_voc.word2idx['Atc::' + item[1]], item[1]]
    med_map_list.append(med)

# 2.2构建药物在图谱中映射的字典
med_map_voc = Voc()
for med in med_map_list:
    med_map_voc.add_word(med[1])

# 2.3把写好的东西写入文件
with open("../output_txt/med_map.txt", "w+") as f:
    for line in med_map_list:
        f.write(str(line[0]) + '\t' + str(line[1]) + '\t' + line[2] + '\n')

dill.dump(med_map_voc, file=open("../output_pkl/med_map.pkl", "wb"))

print("---2.转化数据集中的ATC和图谱中的ATC完成---")

"""3.从数据集中的ICD9到图谱中MESH,DO,OMIM的"""
print("---3.开始转化数据集中的ICD9和图谱中的DO/MESH/OMIM---")

# 3.0提取所有信息
diseases = []
all_entities = []
with open("../output_txt/new_kg_entities.txt", "r+") as f:
    for line in f:
        all_entities.append(line.replace("Disease::", "").rstrip('\n').split('\t'))
        if "Disease::" in line:
            diseases.append(line.replace("Disease::", "").rstrip('\n').split('\t'))

# 3.1在列表中对将omim疾病转换为do疾病，转换不了的用-表示
total = 0
for disease in diseases:
    if "OMIM" in disease[0]:
        total += 1
print("OMIM总数 = ", total)

success = 0
with open("../input/OMIMinDO.tsv", "r+") as f:
    for line in f:
        do2omim = line.rstrip('\n').split('\t')
        for disease in diseases:
            if (do2omim[2] in disease[0]) or (disease[0] in do2omim[2]):  # 值得思考应该是谁in谁，目前懒得想，设置成两个都可以
                disease[0] = do2omim[0]
                success += 1
print("OMIM-DOID转化成功数量 = ", success)

total = 0
for disease in diseases:
    if "OMIM" in disease[0]:
        total += 1
        disease[0] = "-"
print("OMIM-DOID转化失败数量 = ", total)

# 3.2在列表中对将mesh疾病转换为do疾病，转换不了的用-表示
total = 0
for disease in diseases:
    if "MESH" in disease[0]:
        total += 1
print("MESH总数 = ", total)

success = 0
with open("../input/MESHinDO.tsv", "r+") as f:
    for line in f:
        do2mesh = line.rstrip('\n').split('\t')
        for disease in diseases:
            if (do2mesh[2] in disease[0]) or (disease[0] in do2mesh[2]):  # 值得思考应该是谁in谁，目前懒得想，设置成两个都可以
                disease[0] = do2mesh[0]
                success += 1
print("MESH-DO转化成功数量 = ", success)

total = 0
for disease in diseases:
    if "MESH" in disease[0]:
        total += 1
        disease[0] = "-"
print("MESH-DO转化失败数量 = ", total)

# 3.3在列表中将do映射成icd10，转换不了的用-表示
total = 0
for disease in diseases:
    if "DOID" in disease[0]:
        total += 1
print("DO总数 = ", total)

success = 0
with open("../input/ICD10inDO.tsv", "r+") as f:
    for line in f:
        do2icd10 = line.rstrip('\n').split('\t')
        for disease in diseases:
            if (do2icd10[0] in disease[0]) or (disease[0] in do2icd10[0]):  # 值得思考应该是谁in谁，目前懒得想，设置成两个都可以
                disease[0] = do2icd10[2].replace(".", '')
                success += 1
print("DO-ICD10转化成功数量 = ", success)

total = 0
for disease in diseases:
    if "DOID" in disease[0]:
        total += 1
        disease[0] = "-"
print("DO-ICD10转化失败数量 = ", total)

# 3.4在列表中将icd10cm映射成icd9cm，转换不了的用-表示
total = 0
for disease in diseases:
    if "ICD10" in disease[0]:
        total += 1
print("ICD10总数 = ", total)

success = 0
with open("../input/icd9toicd10cmgem.csv", "r+") as f:
    for line in f:
        icd9_2_icd10 = line.rstrip('\n').split(',')
        for disease in diseases:
            if (icd9_2_icd10[1] in disease[0].replace("ICD10CM:", '')) or (
                    disease[0].replace("ICD10CM:", '') in icd9_2_icd10[1]):  # 值得思考应该是谁in谁，目前懒得想，设置成两个都可以
                disease[0] = icd9_2_icd10[0]
                success += 1
print("ICD10-ICD9转化成功数量 = ", success)

total = 0
for disease in diseases:
    if "ICD10" in disease[0]:
        total += 1
        disease[0] = "-"
print("ICD10-ICD9转化失败数量 = ", total)

# 3.5整理数据，生成映射
new_disease = []
for disease in diseases:
    if disease[0] != '-':
        new_disease.append(disease)

# 3.6数据集-图映映射
diag_voc = data_voc["diag_voc"]

for disease in new_disease:
    disease[0] = disease[0].replace('"', '')

# 构建列表形式的疾病映射
diagnose_map_list = []
success = 0
for diag in diag_voc.idx2word.items():
    diagnose = [diag[0], diag[1]]
    for disease in new_disease:
        if diag[1] == disease[0]:
            diagnose.append(int(disease[1]))
            diagnose.append(all_entities[diagnose[-1]][0])
            success += 1
            diagnose_map_list.append(diagnose)
            break

# 构建字典形式的疾病映射
diagnose_map_voc = Voc()
for diagnose in diagnose_map_list:
    diagnose_map_voc.word2idx[diagnose[2]] = diagnose[0]
    diagnose_map_voc.idx2word[diagnose[0]] = diagnose[2]

# 3.7生成文件

# 数据集-图谱映射
with open("../output_txt/diseases_map.txt", "w+") as f:
    for diagnose in diagnose_map_list:
        f.write(str(diagnose[0]) + '\t' + str(diagnose[1]) + '\t'
                + str(diagnose[2]) + '\t' + str(diagnose[3]) + '\n')

dill.dump(diagnose_map_voc, open("../output_pkl/diseases_map.pkl", "wb"))
print("---3.转化数据集中的ICD9和图谱中的DO/MESH/OMIM完毕---")

"""4.按照新图谱的结果，把所有图中的节点transE的结果映射过来做预处理（400维的向量）"""
print("---4.开始生成新子图的TransE---")

# 4.0读取一些必要文件
kg_map = dill.load(open("../output_pkl/newKG_oldKG_map.pkl", "rb"))
entity_emb = np.load('../embed/DRKG_TransE_l2_entity.npy')

print(entity_emb[0])
# 4.1整理新的实体嵌入
new_entity_emb = []
for i in range(len(kg_map.idx2word)):
    new_entity_emb.append(entity_emb[int(kg_map.idx2word[i])])

# 4.2写入文件
np.save("../output_pkl/new_DRKG_TransE_entity.npy", new_entity_emb)

print("---4.完成生成新子图的TransE---")
