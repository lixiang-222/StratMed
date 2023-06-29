import dill
from collections import defaultdict
import numpy as np
from statistics import mean

entity = []
with open("../input/entities.tsv", 'r') as f:
    for line in f:
        entity.append(line.split('\t'))
voc = dill.load(open("../input/voc_final.pkl", "rb"))

# 1.读取到所有的项目
new_dict = defaultdict(list)
for line in entity:
    # print(line)
    str1 = "Atc::"
    for item in voc["med_voc"].word2idx.items():
        str2 = item[0]
        if str1 + str2 in line[0]:
            new_dict[str2].append(line[1].rstrip('\n'))

for item in new_dict.items():
    print(item)

# 2.找到项目映射
all_embedding = np.load("../embed/DRKG_TransE_l2_entity.npy")


def sum_embedding(embedding):
    return embedding.sum(dim=1).unsqueeze(dim=0)  # (1,1,dim)


med_embedding = []
for item in new_dict.items():
    # print(item)
    temp_embedding = []
    for num in item[1]:
        temp_embedding.append(all_embedding[(int(num))])
    # temp_embedding.sum()
    temp_embedding = list(map(mean, zip(*temp_embedding)))
    med_embedding.append(temp_embedding)

for item in med_embedding:
    print(item)
# print(med_embedding)
#
# # np.save("med_embedding.npy", med_embedding)
