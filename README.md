# Drug-recommendations
基于知识图谱的药物组合推荐，总体框架改编来源safedrug

使用方法：

1.配置环境lx.yml

2.运行src/SafeDrug.py


目前已经应用的可以提升效果的方法：

1.加入以往的吃的药的信息

2.把原来作者的关于分子理论的创新点全部删除，只保留一个线性层

3.将诊断和程序的历史信息消除，只保留当次的结果

目前的改动：
1.精简框架，不再使用GRU拟合diagnose和procedure的历史记录，但是对medication的GRU仍然保留（实验证明效果是提升的）

2.药物表示的部分不再仅使用nn.Embedding了，利用drkg图谱来训练药物的嵌入。这次加入了drkg中的部分图谱，主要截取了atc-compound-disease的部分，其中atc可以和本数据集的药物编码对应上，但是disease没法和诊断对应上，同时compound只是一个中间媒介，没有任何意义

3.将大部分原文中用来做对比算法的代码删除了
