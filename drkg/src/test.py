# edge = dill.load(open("disease_atc_edge.pkl", "rb"))
edge = [[1, 2, 3], [4, 5, 6]]

# 将两个列表拼接
edge2 = []
edge2.append(edge[0] + edge[1])
edge2.append(edge[1] + edge[0])

# 打印拼接结果
print("拼接后的列表：", edge2)
