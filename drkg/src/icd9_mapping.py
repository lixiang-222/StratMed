"""这个文件的工作是将 drkg中的疾病(mesh+do)映射到icd9编码上"""
import dill

"""0.在所有实体中提取疾病的信息"""
f = dill.load(open("newfile", "rb"))
kg_voc = f["new_voc"]

diseases = []
with open("new_entites.txt", "r+") as f:
    for line in f:
        if "Disease::" in line:
            diseases.append(line.replace("Disease::", "").rstrip('\n').split('\t'))

"""1.在列表中对将omim疾病转换为do疾病，转换不了的用-表示"""
total = 0
for disease in diseases:
    if "OMIM" in disease[0]:
        total += 1
print("\nomim_total = ", total)

success = 0
with open("OMIMinDO.tsv", "r+") as f:
    for line in f:
        do2omim = line.rstrip('\n').split('\t')
        for disease in diseases:
            if (do2omim[2] in disease[0]) or (disease[0] in do2omim[2]):  # 值得思考应该是谁in谁，目前懒得想，设置成两个都可以
                disease[0] = do2omim[0]
                success += 1
print("omim2doid_success = ", success)

total = 0
for disease in diseases:
    if "OMIM" in disease[0]:
        total += 1
        disease[0] = "-"
print("omim_remain = ", total)

"""2.在列表中对将mesh疾病转换为do疾病，转换不了的用-表示"""
total = 0
for disease in diseases:
    if "MESH" in disease[0]:
        total += 1
print("mesh_total = ", total)

success = 0
with open("MESHinDO.tsv", "r+") as f:
    for line in f:
        do2mesh = line.rstrip('\n').split('\t')
        for disease in diseases:
            if (do2mesh[2] in disease[0]) or (disease[0] in do2mesh[2]):  # 值得思考应该是谁in谁，目前懒得想，设置成两个都可以
                disease[0] = do2mesh[0]
                success += 1
print("mesh2doid_success = ", success)

total = 0
for disease in diseases:
    if "MESH" in disease[0]:
        total += 1
        disease[0] = "-"
print("mesh_remain = ", total)

"""3.在列表中将do映射成icd10，转换不了的用-表示"""
total = 0
for disease in diseases:
    if "DOID" in disease[0]:
        total += 1
print("doid_total = ", total)

success = 0
with open("ICD10inDO.tsv", "r+") as f:
    for line in f:
        do2icd10 = line.rstrip('\n').split('\t')
        for disease in diseases:
            if (do2icd10[0] in disease[0]) or (disease[0] in do2icd10[0]):  # 值得思考应该是谁in谁，目前懒得想，设置成两个都可以
                disease[0] = do2icd10[2].replace(".", '')
                success += 1
print("doid2icd10_success = ", success)

total = 0
for disease in diseases:
    if "DOID" in disease[0]:
        total += 1
        disease[0] = "-"
print("do_remain = ", total)

"""4.在列表中将icd10cm映射成icd9cm，转换不了的用-表示"""
total = 0
for disease in diseases:
    if "ICD10" in disease[0]:
        total += 1
print("ICD10_total = ", total)

success = 0
with open("icd9toicd10cmgem.csv", "r+") as f:
    for line in f:
        icd9_2_icd10 = line.rstrip('\n').split(',')
        for disease in diseases:
            if (icd9_2_icd10[1] in disease[0].replace("ICD10CM:", '')) or (
                    disease[0].replace("ICD10CM:", '') in icd9_2_icd10[1]):  # 值得思考应该是谁in谁，目前懒得想，设置成两个都可以
                disease[0] = icd9_2_icd10[0]
                success += 1
print("icd10-ICD9_success = ", success)

total = 0
for disease in diseases:
    if "ICD10" in disease[0]:
        total += 1
        disease[0] = "-"
print("ICD10_remain = ", total)

"""5.整理数据，输出数据"""
diseases1 = []
for disease in diseases:
    if disease[0] != '-':
        diseases1.append(disease)

diseases = diseases1
with open("diseases_entities.txt", "w+") as f:
    for disease in diseases:
        f.write(disease[0] + '\t' + str(disease[1]) + '\n')

dill.dump(diseases, open("../output/diseases_entities.pkl", "wb"))

voc = dill.load(open("../input/voc_final.pkl", "rb"))
diag_voc = voc["diag_voc"]
diseases = dill.load(open("../output/diseases_entities.pkl", "rb"))

for disease in diseases:
    disease[0] = disease[0].replace('"','')

diagnose_map = []
success = 0
for diag in diag_voc.idx2word.items():
    diagnose = [diag[0], diag[1]]
    for disease in diseases:
        if diag[1] == disease[0]:
            diagnose.append(int(disease[1]))
            success += 1
            diagnose_map.append(diagnose)
            break

# 构建字典形式的疾病映射
diagnose_big_voc = {}
for diagnose in diagnose_map:
    diagnose_big_voc[diagnose[0]] = diagnose[2]

with open('../output/diagnose_map.pkl', 'wb') as f:
    dill.dump(diagnose_big_voc, f)




# dill.dump(diagnose_map, open("../output/diagnose_map.pkl", "wb"))

# with open("../output/diagnose_map.txt", "w+") as f:
#     for diagnose in diagnose_map:
#         f.write(str(diagnose[0]) + '\t' + diagnose[1] + '\t' + str(diagnose[2]) + '\n')
# print(1)
