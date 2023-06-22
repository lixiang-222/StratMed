# Drug-recommendations

在MIMIC-III数据集上的药物推荐，框架改变来自SafeDrug2021

目前最高分0.5311  SOTA得分（MoleRec2023):0.5305

使用方法：

1.配置环境lx.yml

2.运行src/SafeDrug.py

相对上一版本的改动：

1.精简框架，不再使用GRU拟合diagnose和procedure的历史记录，但是对medication的GRU仍然保留（实验证明效果是提升的）；

2.药物表示的部分不再仅使用nn.Embedding了，利用drkg图谱来训练药物的嵌入。这次加入了drkg中的部分图谱，主要截取了atc-compound-disease的部分，其中atc可以和本数据集的药物编码对应上，但是disease没法和诊断对应上，同时compound只是一个中间媒介，没有任何意义；

3.将大部分原文中用来做对比算法的代码删除了；

目前已有提升的操作：

1.把原作者对于分子理论的部分全部去掉，在患者表示之后直接过MLP输出；

2.在患者表示中加入药物历史表示；

3.不要考虑患者历史诊断和程序的信息，只使用当次的效果反而更好；
