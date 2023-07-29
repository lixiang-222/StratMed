import dill

data_path = "../data/output/records_final.pkl"
voc_path = "../data/output/voc_final.pkl"

ddi_adj_path = "../data/output/ddi_A_final.pkl"
ddi_mask_path = "../data/output/ddi_mask_H.pkl"
molecule_path = "../data/output/atc3toSMILES.pkl"

ddi_adj = dill.load(open(ddi_adj_path, "rb"))
ddi_mask_H = dill.load(open(ddi_mask_path, "rb"))
data = dill.load(open(data_path, "rb"))
molecule = dill.load(open(molecule_path, "rb"))

voc = dill.load(open(voc_path, "rb"))

print("a")