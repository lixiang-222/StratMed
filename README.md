# Drug-recommendations
自己写的

使用方法：

1.配置环境lx.yml

2.运行src/SafeDrug.py

目前的改动：

1.精简框架，不再使用GRU拟合diagnose和procedure的历史记录，但是对medication的GRU仍然保留（实验证明效果是提升的）

2.药物表示的部分不再仅使用nn.Embedding了，利用drkg图谱来训练药物的嵌入。这次加入了drkg中的部分图谱，主要截取了atc-compound-disease的部分，其中atc可以和本数据集的药物编码对应上，但是disease没法和诊断对应上，同时compound只是一个中间媒介，没有任何意义

3.将大部分原文中用来做对比算法的代码删除了
