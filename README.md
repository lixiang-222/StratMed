# Drug-recommendations
基于知识图谱的药物组合推荐，总体框架改编来源safedrug

1.首先把data/processing.py运行一下生成output里面的文件（已经运行过了不用运行了）

2.运行src/SafeDrug.py


目前的跟SafeDrug不一样的地方是：

1.患者表示-加入患者基本信息

2.患者表示-加入以往的吃的药的信息

3.药物表示-用drkg中的知识图谱的药物表示，用atc3编码对齐本项目的药物

4.匹配推荐-用112个64维度的药物表示和1个64维度的患者表示分别做内积，生成一个112长度的列表，做最终的结果，目前这里有无法收敛，解决ing
