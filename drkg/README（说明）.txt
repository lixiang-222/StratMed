embed：原来的自带的
	DRKG_TransE_l2_entity.npy 	对整个图做TransE，生成的所有实体的表示
	DRKG_TransE_l2_relation.npy 	对整个图做TransE，生成的所有关系的表示
	别的文件不了解

input：从各个地方来的原文件，代码里在他们之上做更改整合
	drkg.tsv 				DRKG原版知识图谱 内容是 实体A-关系-实体B
	entities.tsv 			DRKG原版实体编号
	entity2src.tsv 			介绍各个实体的来源（我没用到过）
	icd9toicd10cmgem.csv	ICD9到ICD10的对照表
	ICD10inDO.tsv 			DO到ICD10的对照表
	MESHinDO.tsv			DO到MESH的对照表
	OMIMinDO.tsv			DO到OMIM的对照表
	relation_glossary.tsv	介绍各种边
	relations.tsv			DRKG原版边的编号
	voc_final.pkl			SafeDrug里面用到的数据集，包含了诊断，程序，药物的所有种类

output_pkl：用dill存储的输出，给计算机用
	diseases_map.pkl	字典，数据集中编号-新图谱中编号
	med_map.pkl			字典，数据集中编号-新图谱中编号
	new_kg_edges.pkl	列表，新图谱中边的信息
	new_kg_entities.pkl	字典，新图谱中编号-实体名称
	newKG_oldKG_map.pkl	字典，新图谱中编号-旧图谱中编号

output_txt：用txt存储的输出，给人看
	diseases_map.txt	数据集中编号，ICD9编码，新图谱中编号，实体名称
	med_map.txt			数据集中编号，新图谱中编号，ATC4编码
	new_kg_edges.txt	实体名称-实体名称
	new_kg_entities.txt	实体名称，新图谱中编号
	newKG_oldKG_map.txt	新图谱中编码，实体名称，旧图谱中编码

src：代码
	preprocess.py		input-output的代码



