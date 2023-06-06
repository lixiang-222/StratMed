from xml.dom.pulldom import ErrorHandler
import pandas as pd
import dill
import numpy as np
from collections import defaultdict
from rdkit import Chem
from rdkit.Chem import BRICS


##### process medications #####
# load med data
def med_process(med_file):
    med_pd = pd.read_csv(med_file, dtype={"NDC": "category"})

    # 留下SUBJECT_ID(患者编号),HADM_ID(病案号),ICUSTAY_ID(ICU病案号),STARTDATE(处方开始的日期),DRUG(药物名字),NDC(药物编码)

    # med_pd.drop(columns=['ROW_ID','DRUG_TYPE','DRUG_NAME_POE','DRUG_NAME_GENERIC',
    #                     'FORMULARY_DRUG_CD','PROD_STRENGTH','DOSE_VAL_RX',
    #                     'DOSE_UNIT_RX','FORM_VAL_DISP','FORM_UNIT_DISP', 'GSN', 'FORM_UNIT_DISP',
    #                     'ROUTE','ENDDATE','DRUG'], axis=1, inplace=True)
    med_pd.drop(
        columns=[
            "ROW_ID",
            "DRUG_TYPE",
            "DRUG_NAME_POE",
            "DRUG_NAME_GENERIC",
            "FORMULARY_DRUG_CD",
            "PROD_STRENGTH",
            "DOSE_VAL_RX",
            "DOSE_UNIT_RX",
            "FORM_VAL_DISP",
            "FORM_UNIT_DISP",
            "GSN",
            "FORM_UNIT_DISP",
            "ROUTE",
            "ENDDATE",
        ],
        axis=1,
        inplace=True,
    )
    # 普通数据处理（数据清洗，包含去除掉NDC编码为0，用pad填充空缺值，把编码转成数字，时间变成标准的格式等）还有一些看不懂的数据处理
    med_pd.drop(index=med_pd[med_pd["NDC"] == "0"].index, axis=0, inplace=True)
    med_pd.fillna(method="pad", inplace=True)
    med_pd.dropna(inplace=True)
    med_pd.drop_duplicates(inplace=True)
    med_pd["ICUSTAY_ID"] = med_pd["ICUSTAY_ID"].astype("int64")
    med_pd["STARTDATE"] = pd.to_datetime(
        med_pd["STARTDATE"], format="%Y-%m-%d %H:%M:%S"
    )
    med_pd.sort_values(
        by=["SUBJECT_ID", "HADM_ID", "ICUSTAY_ID", "STARTDATE"], inplace=True
    )
    med_pd = med_pd.reset_index(drop=True)

    med_pd = med_pd.drop(columns=["ICUSTAY_ID"])
    med_pd = med_pd.drop_duplicates()
    med_pd = med_pd.reset_index(drop=True)
    # 最后的药物表中，只留下患者编号，案例编号，开始时间，药物名称，药物NDC编码，同时按照上述顺序排序
    return med_pd


# ATC3-to-drugname
def ATC3toDrug(med_pd):
    atc3toDrugDict = {}
    for atc3, drugname in med_pd[["ATC3", "DRUG"]].values:
        if atc3 in atc3toDrugDict:
            atc3toDrugDict[atc3].add(drugname)
        else:
            atc3toDrugDict[atc3] = set(drugname)

    return atc3toDrugDict


def atc3toSMILES(ATC3toDrugDict, druginfo):
    drug2smiles = {}
    atc3tosmiles = {}
    for drugname, smiles in druginfo[["name", "moldb_smiles"]].values:
        if type(smiles) == type("a"):
            drug2smiles[drugname] = smiles
    for atc3, drug in ATC3toDrugDict.items():
        temp = []
        for d in drug:
            try:
                temp.append(drug2smiles[d])
            except:
                pass
        if len(temp) > 0:
            atc3tosmiles[atc3] = temp[:3]

    return atc3tosmiles


# medication mapping
def codeMapping2atc4(med_pd):
    with open(ndc2RXCUI_file, "r") as f:
        ndc2RXCUI = eval(f.read())
    med_pd["RXCUI"] = med_pd["NDC"].map(ndc2RXCUI)
    med_pd.dropna(inplace=True)

    RXCUI2atc4 = pd.read_csv(RXCUI2atc4_file)
    RXCUI2atc4 = RXCUI2atc4.drop(columns=["YEAR", "MONTH", "NDC"])
    RXCUI2atc4.drop_duplicates(subset=["RXCUI"], inplace=True)
    med_pd.drop(index=med_pd[med_pd["RXCUI"].isin([""])].index, axis=0, inplace=True)

    med_pd["RXCUI"] = med_pd["RXCUI"].astype("int64")
    med_pd = med_pd.reset_index(drop=True)
    med_pd = med_pd.merge(RXCUI2atc4, on=["RXCUI"])
    med_pd.drop(columns=["NDC", "RXCUI"], inplace=True)
    med_pd["ATC4"] = med_pd["ATC4"].map(lambda x: x[:4])
    med_pd = med_pd.rename(columns={"ATC4": "ATC3"})
    med_pd = med_pd.drop_duplicates()
    med_pd = med_pd.reset_index(drop=True)
    return med_pd


# visit >= 2
# 找出来有一个患者存在两次看病的人
def process_visit_lg2(med_pd):
    a = (
        med_pd[["SUBJECT_ID", "HADM_ID"]]
        .groupby(by="SUBJECT_ID")["HADM_ID"]
        .unique()
        .reset_index()
    )
    a["HADM_ID_Len"] = a["HADM_ID"].map(lambda x: len(x))
    a = a[a["HADM_ID_Len"] > 1]
    # a = a[a["HADM_ID_Len"] > 2]  # 试试考虑序列长度从3起步的
    return a


# most common medications
def filter_300_most_med(med_pd):
    med_count = (
        med_pd.groupby(by=["ATC3"])
        .size()
        .reset_index()
        .rename(columns={0: "count"})
        .sort_values(by=["count"], ascending=False)
        .reset_index(drop=True)
    )
    med_pd = med_pd[med_pd["ATC3"].isin(med_count.loc[:299, "ATC3"])]

    return med_pd.reset_index(drop=True)


##### process diagnosis #####
def diag_process(diag_file):
    diag_pd = pd.read_csv(diag_file)
    diag_pd.dropna(inplace=True)
    diag_pd.drop(columns=["SEQ_NUM", "ROW_ID"], inplace=True)
    diag_pd.drop_duplicates(inplace=True)
    diag_pd.sort_values(by=["SUBJECT_ID", "HADM_ID"], inplace=True)
    diag_pd = diag_pd.reset_index(drop=True)

    def filter_2000_most_diag(diag_pd):
        diag_count = (
            diag_pd.groupby(by=["ICD9_CODE"])
            .size()
            .reset_index()
            .rename(columns={0: "count"})
            .sort_values(by=["count"], ascending=False)
            .reset_index(drop=True)
        )
        diag_pd = diag_pd[diag_pd["ICD9_CODE"].isin(diag_count.loc[:1999, "ICD9_CODE"])]

        return diag_pd.reset_index(drop=True)

    diag_pd = filter_2000_most_diag(diag_pd)

    return diag_pd


##### process procedure #####
def procedure_process(procedure_file):
    pro_pd = pd.read_csv(procedure_file, dtype={"ICD9_CODE": "category"})
    pro_pd.drop(columns=["ROW_ID"], inplace=True)
    pro_pd.drop_duplicates(inplace=True)
    pro_pd.sort_values(by=["SUBJECT_ID", "HADM_ID", "SEQ_NUM"], inplace=True)
    pro_pd.drop(columns=["SEQ_NUM"], inplace=True)
    pro_pd.drop_duplicates(inplace=True)
    pro_pd.reset_index(drop=True, inplace=True)

    return pro_pd


# 自己写的对患者基本信息的处理，输入是两个文件，输出是患者id和对应的基本信息
def basic_info_process(adm_file, patient_file):
    # adm表，提取种族，婚姻状况，就诊时间
    adm_pd = pd.read_csv(adm_file, dtype={"SUBJECT_ID": int, "MARITAL_STATUS": "category", "ETHNICITY": "category"})
    adm_pd.drop(columns=["ROW_ID", "HADM_ID", "DISCHTIME", "DEATHTIME", "ADMISSION_TYPE",
                         "ADMISSION_LOCATION", "DISCHARGE_LOCATION", "INSURANCE", "LANGUAGE", "RELIGION",
                         "EDREGTIME", "EDOUTTIME", "DIAGNOSIS", "HOSPITAL_EXPIRE_FLAG",
                         "HAS_CHARTEVENTS_DATA"], inplace=True)
    adm_pd.drop_duplicates(inplace=True)

    # patient表，提取性别，出生日期，死亡标签
    patient_pd = pd.read_csv(patient_file, dtype={"SUBJECT_ID": int, "MARITAL_STATUS": "category"})
    patient_pd.drop(columns=["ROW_ID", "DOD", "DOD_HOSP", "DOD_SSN"],
                    inplace=True)

    # 整合两个表
    basic_info_pd = pd.merge(adm_pd, patient_pd, on="SUBJECT_ID")
    basic_info_pd.sort_values(by=["SUBJECT_ID"], inplace=True)

    # 用出生日期和就诊日期计算患者就诊时候的年龄
    basic_info_pd['birthday'] = pd.to_datetime(basic_info_pd['DOB']).dt.round("D").dt.year
    basic_info_pd['admitday'] = pd.to_datetime(basic_info_pd['ADMITTIME']).dt.round("D").dt.year
    basic_info_pd['AGE'] = basic_info_pd.admitday - basic_info_pd.birthday
    basic_info_pd.drop(columns=['DOB', 'ADMITTIME', 'birthday', 'admitday'], inplace=True)
    basic_info_pd.reset_index(drop=True, inplace=True)

    # 洗数据 婚姻状态nan变成UNKNOWN，年龄里300+的异常数据 变成不考虑0岁的平均值53，性别男的是1，女的是0
    basic_info_pd.MARITAL_STATUS.fillna('UNKNOWN (DEFAULT)', inplace=True)
    basic_info_pd.AGE.replace([i for i in range(300, 312)], 53, inplace=True)
    # basic_info_pd.GENDER.replace('M', 1, inplace=True)
    # basic_info_pd.GENDER.replace('F', 0, inplace=True)

    # 每个患者的信息只保留一次（把多次的就诊记录只取第一次）
    basic_info_pd.drop_duplicates(subset='SUBJECT_ID', keep='first', inplace=True)
    basic_info_pd.reset_index(drop=True, inplace=True)

    # 返回的是种族，婚姻状况，性别，年龄，死亡 共5个固定基本信息
    return basic_info_pd


def filter_1000_most_pro(pro_pd):
    pro_count = (
        pro_pd.groupby(by=["ICD9_CODE"])
        .size()
        .reset_index()
        .rename(columns={0: "count"})
        .sort_values(by=["count"], ascending=False)
        .reset_index(drop=True)
    )
    pro_pd = pro_pd[pro_pd["ICD9_CODE"].isin(pro_count.loc[:1000, "ICD9_CODE"])]

    return pro_pd.reset_index(drop=True)


###### combine four tables #####
def combine_process(med_pd, diag_pd, pro_pd, basic_info_pd):
    med_pd_key = med_pd[["SUBJECT_ID", "HADM_ID"]].drop_duplicates()
    diag_pd_key = diag_pd[["SUBJECT_ID", "HADM_ID"]].drop_duplicates()
    pro_pd_key = pro_pd[["SUBJECT_ID", "HADM_ID"]].drop_duplicates()

    combined_key = med_pd_key.merge(
        diag_pd_key, on=["SUBJECT_ID", "HADM_ID"], how="inner"
    )
    combined_key = combined_key.merge(
        pro_pd_key, on=["SUBJECT_ID", "HADM_ID"], how="inner"
    )

    diag_pd = diag_pd.merge(combined_key, on=["SUBJECT_ID", "HADM_ID"], how="inner")
    med_pd = med_pd.merge(combined_key, on=["SUBJECT_ID", "HADM_ID"], how="inner")
    pro_pd = pro_pd.merge(combined_key, on=["SUBJECT_ID", "HADM_ID"], how="inner")
    # basic_info_pd = basic_info_pd.merge(combined_key, on=["SUBJECT_ID"], how="inner")

    # flatten and merge
    diag_pd = (
        diag_pd.groupby(by=["SUBJECT_ID", "HADM_ID"])["ICD9_CODE"]
        .unique()
        .reset_index()
    )
    med_pd = med_pd.groupby(by=["SUBJECT_ID", "HADM_ID"])["ATC3"].unique().reset_index()
    pro_pd = (
        pro_pd.groupby(by=["SUBJECT_ID", "HADM_ID"])["ICD9_CODE"]
        .unique()
        .reset_index()
        .rename(columns={"ICD9_CODE": "PRO_CODE"})
    )
    med_pd["ATC3"] = med_pd["ATC3"].map(lambda x: list(x))
    pro_pd["PRO_CODE"] = pro_pd["PRO_CODE"].map(lambda x: list(x))
    data = diag_pd.merge(med_pd, on=["SUBJECT_ID", "HADM_ID"], how="inner")
    data = data.merge(pro_pd, on=["SUBJECT_ID", "HADM_ID"], how="inner")
    #     data['ICD9_CODE_Len'] = data['ICD9_CODE'].map(lambda x: len(x))
    data["ATC3_num"] = data["ATC3"].map(lambda x: len(x))

    data = data.merge(basic_info_pd, on=["SUBJECT_ID"], how="inner")
    return data


def statistics(data: object) -> object:
    print("#patients ", data["SUBJECT_ID"].unique().shape)
    print("#clinical events ", len(data))

    diag = data["ICD9_CODE"].values
    med = data["ATC3"].values
    pro = data["PRO_CODE"].values

    unique_diag = set([j for i in diag for j in list(i)])
    unique_med = set([j for i in med for j in list(i)])
    unique_pro = set([j for i in pro for j in list(i)])

    print("#diagnosis ", len(unique_diag))
    print("#med ", len(unique_med))
    print("#procedure", len(unique_pro))

    (
        avg_diag,
        avg_med,
        avg_pro,
        max_diag,
        max_med,
        max_pro,
        cnt,
        max_visit,
        avg_visit,
    ) = [0 for i in range(9)]

    for subject_id in data["SUBJECT_ID"].unique():
        item_data = data[data["SUBJECT_ID"] == subject_id]
        x, y, z = [], [], []
        visit_cnt = 0
        for index, row in item_data.iterrows():
            visit_cnt += 1
            cnt += 1
            x.extend(list(row["ICD9_CODE"]))
            y.extend(list(row["ATC3"]))
            z.extend(list(row["PRO_CODE"]))
        x, y, z = set(x), set(y), set(z)
        avg_diag += len(x)
        avg_med += len(y)
        avg_pro += len(z)
        avg_visit += visit_cnt
        if len(x) > max_diag:
            max_diag = len(x)
        if len(y) > max_med:
            max_med = len(y)
        if len(z) > max_pro:
            max_pro = len(z)
        if visit_cnt > max_visit:
            max_visit = visit_cnt

    print("#avg of diagnoses ", avg_diag / cnt)
    print("#avg of medicines ", avg_med / cnt)
    print("#avg of procedures ", avg_pro / cnt)
    print("#avg of vists ", avg_visit / len(data["SUBJECT_ID"].unique()))

    print("#max of diagnoses ", max_diag)
    print("#max of medicines ", max_med)
    print("#max of procedures ", max_pro)
    print("#max of visit ", max_visit)


##### indexing file and final record
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


# create voc set
def create_str_token_mapping(df):
    diag_voc = Voc()
    med_voc = Voc()
    pro_voc = Voc()

    for index, row in df.iterrows():
        diag_voc.add_sentence(row["ICD9_CODE"])
        med_voc.add_sentence(row["ATC3"])
        pro_voc.add_sentence(row["PRO_CODE"])

    dill.dump(
        obj={"diag_voc": diag_voc, "med_voc": med_voc, "pro_voc": pro_voc},
        file=open(vocabulary_file, "wb"),
    )
    return diag_voc, med_voc, pro_voc


"""自己写的处理基本信息的字典"""


# create basic voc set
def create_basic_token_mapping(df):
    marital_voc = Voc()
    ethnicity_voc = Voc()
    gender_voc = Voc()
    expire_voc = Voc()
    age_voc = Voc()

    for index, row in df.iterrows():
        marital_voc.add_word(row["MARITAL_STATUS"])
        ethnicity_voc.add_word(row["ETHNICITY"])
        gender_voc.add_word(row["GENDER"])
        expire_voc.add_word(row["EXPIRE_FLAG"])
        age_voc.add_word(row["AGE"])

    dill.dump(
        obj={"marital_voc": marital_voc, "ethnicity_voc": ethnicity_voc, "gender_voc": gender_voc,
             "expire_voc": expire_voc, "age_voc": age_voc},
        file=open(basic_info_vocabulary_file, "wb"),
    )
    return marital_voc, ethnicity_voc, gender_voc, expire_voc, age_voc


""""""


# create final records
def create_patient_record(df, diag_voc, med_voc, pro_voc, marital_voc, ethnicity_voc, gender_voc, expire_voc, age_voc):
    records = []  # (patient, code_kind:4, codes)  code_kind:diag, proc, med, basic_info
    for subject_id in df["SUBJECT_ID"].unique():
        item_df = df[df["SUBJECT_ID"] == subject_id]
        patient = []
        for index, row in item_df.iterrows():
            admission = []
            admission.append([diag_voc.word2idx[i] for i in row["ICD9_CODE"]])
            admission.append([pro_voc.word2idx[i] for i in row["PRO_CODE"]])
            admission.append([med_voc.word2idx[i] for i in row["ATC3"]])

            basic_info = []
            basic_info.append(marital_voc.word2idx[row["MARITAL_STATUS"]])
            basic_info.append(ethnicity_voc.word2idx[row["ETHNICITY"]])
            basic_info.append(gender_voc.word2idx[row["GENDER"]])
            basic_info.append(expire_voc.word2idx[row["EXPIRE_FLAG"]])
            basic_info.append(age_voc.word2idx[row["AGE"]])
            admission.append(basic_info)

            patient.append(admission)
        records.append(patient)
    dill.dump(obj=records, file=open(ehr_sequence_file, "wb"))
    return records


# get ddi matrix
def get_ddi_matrix(records, med_voc, ddi_file):
    TOPK = 40  # topk drug-drug interaction
    cid2atc_dic = defaultdict(set)
    med_voc_size = len(med_voc.idx2word)
    med_unique_word = [med_voc.idx2word[i] for i in range(med_voc_size)]
    atc3_atc4_dic = defaultdict(set)
    for item in med_unique_word:
        atc3_atc4_dic[item[:4]].add(item)

    with open(cid2atc6_file, "r") as f:
        for line in f:
            line_ls = line[:-1].split(",")
            cid = line_ls[0]
            atcs = line_ls[1:]
            for atc in atcs:
                if len(atc3_atc4_dic[atc[:4]]) != 0:
                    cid2atc_dic[cid].add(atc[:4])

    # ddi load
    ddi_df = pd.read_csv(ddi_file)
    # fliter sever side effect
    ddi_most_pd = (
        ddi_df.groupby(by=["Polypharmacy Side Effect", "Side Effect Name"])
        .size()
        .reset_index()
        .rename(columns={0: "count"})
        .sort_values(by=["count"], ascending=False)
        .reset_index(drop=True)
    )
    ddi_most_pd = ddi_most_pd.iloc[-TOPK:, :]
    # ddi_most_pd = pd.DataFrame(columns=['Side Effect Name'], data=['as','asd','as'])
    fliter_ddi_df = ddi_df.merge(
        ddi_most_pd[["Side Effect Name"]], how="inner", on=["Side Effect Name"]
    )
    ddi_df = (
        fliter_ddi_df[["STITCH 1", "STITCH 2"]].drop_duplicates().reset_index(drop=True)
    )

    # weighted ehr adj
    ehr_adj = np.zeros((med_voc_size, med_voc_size))
    for patient in records:
        for adm in patient:
            med_set = adm[2]
            for i, med_i in enumerate(med_set):
                for j, med_j in enumerate(med_set):
                    if j <= i:
                        continue
                    ehr_adj[med_i, med_j] = 1
                    ehr_adj[med_j, med_i] = 1
    dill.dump(ehr_adj, open(ehr_adjacency_file, "wb"))

    # ddi adj
    ddi_adj = np.zeros((med_voc_size, med_voc_size))
    for index, row in ddi_df.iterrows():
        # ddi
        cid1 = row["STITCH 1"]
        cid2 = row["STITCH 2"]

        # cid -> atc_level3
        for atc_i in cid2atc_dic[cid1]:
            for atc_j in cid2atc_dic[cid2]:

                # atc_level3 -> atc_level4
                for i in atc3_atc4_dic[atc_i]:
                    for j in atc3_atc4_dic[atc_j]:
                        if med_voc.word2idx[i] != med_voc.word2idx[j]:
                            ddi_adj[med_voc.word2idx[i], med_voc.word2idx[j]] = 1
                            ddi_adj[med_voc.word2idx[j], med_voc.word2idx[i]] = 1
    dill.dump(ddi_adj, open(ddi_adjacency_file, "wb"))

    return ddi_adj


def get_ddi_mask(atc42SMLES, med_voc):
    # ATC3_List[22] = {0}
    # ATC3_List[25] = {0}
    # ATC3_List[27] = {0}
    fraction = []
    for k, v in med_voc.idx2word.items():  # k是idx，v是药物atc编码
        tempF = set()
        for SMILES in atc42SMLES[v]:
            try:
                m = BRICS.BRICSDecompose(Chem.MolFromSmiles(SMILES))
                for frac in m:
                    tempF.add(frac)
            except:
                pass
        fraction.append(tempF)
    fracSet = []
    for i in fraction:
        fracSet += i
    fracSet = list(set(fracSet))  # set of all segments

    ddi_matrix = np.zeros((len(med_voc.idx2word), len(fracSet)))
    for i, fracList in enumerate(fraction):
        for frac in fracList:
            ddi_matrix[i, fracSet.index(frac)] = 1
    return ddi_matrix


if __name__ == "__main__":
    # files can be downloaded from https://mimic.physionet.org/gettingstarted/dbsetup/
    # please change into your own MIMIC folder
    # med_file = "/srv/local/data/physionet.org/files/mimiciii/1.4/PRESCRIPTIONS.csv"
    # diag_file = "/srv/local/data/physionet.org/files/mimiciii/1.4/DIAGNOSES_ICD.csv"
    # procedure_file = (
    #     "/srv/local/data/physionet.org/files/mimiciii/1.4/PROCEDURES_ICD.csv"
    # )

    med_file = "./input/PRESCRIPTIONS.csv"
    diag_file = "./input/DIAGNOSES_ICD.csv"
    procedure_file = "./input/PROCEDURES_ICD.csv"

    # 下面是自己写的input
    adm_file = "./input/ADMISSIONS.csv"
    patient_file = "./input/PATIENTS.csv"

    # input auxiliary files
    med_structure_file = "./output/atc32SMILES.pkl"
    RXCUI2atc4_file = "./input/RXCUI2atc4.csv"
    cid2atc6_file = "./input/drug-atc.csv"
    ndc2RXCUI_file = "./input/ndc2RXCUI.txt"
    ddi_file = "./input/drug-DDI.csv"
    drugbankinfo = "./input/drugbank_drugs_info.csv"

    # output files
    ddi_adjacency_file = "./output/ddi_A_final.pkl"
    ehr_adjacency_file = "./output/ehr_adj_final.pkl"
    ehr_sequence_file = "./output/records_final.pkl"
    vocabulary_file = "./output/voc_final.pkl"
    basic_info_vocabulary_file = "./output/basic_info_voc_final.pkl"
    ddi_mask_H_file = "./output/ddi_mask_H.pkl"
    atc3toSMILES_file = "./output/atc3toSMILES.pkl"

    # for med
    # 把数据读取进来
    med_pd = med_process(med_file)
    # 找出来超过两条记录的患者
    med_pd_lg2 = process_visit_lg2(med_pd).reset_index(drop=True)
    # 后面数据不全 我看不出来什么意思
    med_pd = med_pd.merge(
        med_pd_lg2[["SUBJECT_ID"]], on="SUBJECT_ID", how="inner"
    ).reset_index(drop=True)
    # 把ndc编码换成act3编码，表中按照药物名称排序
    med_pd = codeMapping2atc4(med_pd)
    # 只考虑频率高于300的药物（实际上就是全部考虑了）
    med_pd = filter_300_most_med(med_pd)

    # med to SMILES mapping
    atc3toDrug = ATC3toDrug(med_pd)
    druginfo = pd.read_csv(drugbankinfo)
    atc3toSMILES = atc3toSMILES(atc3toDrug, druginfo)
    dill.dump(atc3toSMILES, open(atc3toSMILES_file, "wb"))
    med_pd = med_pd[med_pd.ATC3.isin(atc3toSMILES.keys())]
    print("complete medication processing")

    # for diagnosis
    diag_pd = diag_process(diag_file)
    # 患者编号，病例编号，疾病ICD9编码，只保留频率排在前2000的疾病（一共约7000种病）
    print("complete diagnosis processing")

    # for procedure
    pro_pd = procedure_process(procedure_file)
    # pro_pd = filter_1000_most_pro(pro_pd)
    # 患者编号，病历编号，治疗过程的ICD9编号，以患者编号，病例编号，治疗顺序排序
    print("complete procedure processing")

    # 这里是自己写的，整理基本的信息
    basic_info_pd = basic_info_process(adm_file, patient_file)
    print("complete basic information processing")

    # combine
    data = combine_process(med_pd, diag_pd, pro_pd, basic_info_pd)
    statistics(data)
    print("complete combining")

    # create vocab
    diag_voc, med_voc, pro_voc = create_str_token_mapping(data)
    print("obtain medical voc")

    """自己写的处理基本信息字典"""
    # create basic vocab
    marital_voc, ethnicity_voc, gender_voc, expire_voc, age_voc = create_basic_token_mapping(data)
    print("obtain basic information voc")

    # create ehr sequence data
    records = create_patient_record(data, diag_voc, med_voc, pro_voc, marital_voc, ethnicity_voc, gender_voc,
                                    expire_voc, age_voc)
    print("obtain ehr sequence data")

    # create ddi adj matrix
    ddi_adj = get_ddi_matrix(records, med_voc, ddi_file)
    print("obtain ddi adj matrix")

    # get ddi_mask_H
    ddi_mask_H = get_ddi_mask(atc3toSMILES, med_voc)
    dill.dump(ddi_mask_H, open(ddi_mask_H_file, "wb"))
