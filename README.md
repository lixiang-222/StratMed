# StratMed

## Folder Specification

> - data/: we use the same data set and processing methods as safedrug2021
>> - procesing.py: data preprocessing file.  
>> - Input/ (extracted from external resources)
>>> - PRESCRIPTIONS.csv: the prescription file from MIMIC-III raw dataset  
>>> - DIAGNOSES_ICD.csv: the diagnosis file from MIMIC-III raw dataset  
>>> - PROCEDURES_ICD.csv: the procedure file from MIMIC-III raw dataset  
>>> - RXCUI2atc4.csv: this is a NDC-RXCUI-ATC4 mapping file, and we only need the RXCUI to ATC4 mapping. This file is obtained from https://github.com/sjy1203/GAMENet, where the name is called ndc2atc_level4.csv.  
>>> - drug-atc.csv: this is a CID-ATC file, which gives the mapping from CID code to detailed ATC code (we will use the prefix of the ATC code latter for aggregation). This file is obtained from https://github.com/sjy1203/GAMENet.  
>>> - ndc2RXCUI.txt: NDC to RXCUI mapping file. This file is obtained from https://github.com/sjy1203/GAMENet, where the name is called ndc2rxnorm_mapping.csv.  
>>> - drugbank_drugs_info.csv: drug information table downloaded from drugbank here https://drive.google.com/file/d/1EzIlVeiIR6LFtrBnhzAth4fJt6H_ljxk/view?usp=sharing, which is used to map drug name to drug SMILES string.  
>>> - drug-DDI.csv: this a large file, containing the drug DDI information, coded by CID. The file could be downloaded from https://drive.google.com/file/d/1mnPc0O0ztz0fkv3HF-dpmBb8PLWsEoDz/view?usp=sharing  

>> - Output/
>>> - atc3toSMILES.pkl: drug ID (we use ATC-3 level code to represent drug ID) to drug SMILES string dict
>>> - ddi_A_final.pkl: ddi adjacency matrix
>>> - ddi_matrix_H.pkl: H mask structure (This file is created by ddi_mask_H.py)
>>> - ehr_adj_final.pkl: used in GAMENet baseline (if two drugs appear in one set, then they are connected)
>>> - ehr_adj.pkl: Used to store various relationship layers after relationship stratification
>>> - records_final.pkl: The final diagnosis-procedure-medication EHR records of each patient, used for train/val/test split (NOTE: we only provide the first 100 entries as examples here. We cannot distribute the whole MIMIC-III data https://physionet.org/content/mimiciii/1.4/, then please download the dataset by yourself and use our processing code to obtain the full records.).
>>> - voc_final.pkl: diag/prod/med index to code dictionary

> - src/: source code and training results
>> - trained/: The trained model is used for testing 
>>> - trained_model.pt
>> - pretrained/: The pre-trained model is used for testing 
>>> - pretrained_model.pt
>> - results/: Training Results and Parameters
>>> - (empty now)
>> - main.py: main framework and process of the model
>> - model.py: model details and implementation
>> - util.py: small functions and toolkits


## Step 1: Package Dependency  
- Create an environment and install necessary packages
```angular2html
conda create -n StratMed python=3.9
conda activate StratMed
pip3 install torch torchvision torchaudio
pip3 install scikit-learn
pip3 install dill
pip3 install dnc
pip3 install pytorch_geometric
pip3 install numpy = 1.22.3
```
- If you still need another package, please proceed with the installation as described above

## Step 2: Data Processing  
- Due to medical resource protection policies, we do not have the right to share relevant datasets. And if you want to reprocess the data yourself, please download the dataset from the official website and store it in a folder according to the format in the folder specification
- MIMIC-III: https://physionet.org/content/mimiciii/1.4/ (The application process is relatively complex, which may take about a week)
- DDI file: https://drive.google.com/file/d/1mnPc0O0ztz0fkv3HF-dpmBb8PLWsEoDz/view?usp=sharing
- run processing.py
```angular2html
cd data
python processing.py
```

## Step 3: Run the code  
```angular2html
cd src
python main.py
```

## Citation & Acknowledgement
We are grateful to everyone who contributed to this project. The article has been published in the journal Knowledge-Based System. You are welcome to quote our work or provide valuable guidance. Related papers：https://www.sciencedirect.com/science/article/pii/S0950705123009887

If the code and the paper are useful for you, it is appreciable to cite our paper:
```bash
@article{li2024stratmed,
  title={StratMed: Relevance stratification between biomedical entities for sparsity on medication recommendation},
  author={Li, Xiang and Liang, Shunpan and Hou, Yulei and Ma, Tengfei},
  journal={Knowledge-Based Systems},
  volume={284},
  pages={111239},
  year={2024},
  publisher={Elsevier}
}
```
