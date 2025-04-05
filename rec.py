from sentence_transformers import SentenceTransformer
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
# from sklearn.preprocessing import MinMaxScale

# 加载预训练模型（中文推荐'paraphrase-multilingual-MiniLM-L12-v2'）
model = SentenceTransformer('all-MiniLM-L6-v2') 
df = pd.read_csv('user_data/users.csv')
user_id = 1
user_embeddings = df.loc[df['id'] == user_id, 'user_embedding'][0].apply(
            lambda x: np.frombuffer(eval(x), dtype=np.float32)  # 移除非必要的bytes()
        )
# 读取 CSV 文件
df = pd.read_csv('arxiv_data/arxiv_cv_papers.csv')
df.info()

# 假设df是你的DataFrame对象，获取第一列（文章名字列）
article_names = df.iloc[:, 1]
article_names = article_names.tolist()

# 生成标题向量
title_embeddings = model.encode(article_names)  # 每个标题得到384维向量

# print(title_embeddings.shape)
title_embedding_mean = title_embeddings.mean(axis=0)
# print(title_embedding_mean.shape)

np.save('user_data/ori_recal_embedding.npy', title_embedding_mean)

# 召回

K = 1000  # 假设这里的k为5，你可以根据实际情况修改
# 计算每个title embedding与title embedding mean的余弦相似度

similarities = cosine_similarity(title_embeddings, title_embedding_mean.reshape(1, -1)).flatten()

# 获取相似度最大的k个index
top_K_indices = np.argsort(similarities)[-K:][::-1]
# 获取df中这1000个topk indices的样本
top_K_samples = df.iloc[top_K_indices]
# 保存为CSV文件
top_K_samples.to_csv('arxiv_data/arxiv_recal_samples.csv', index=False)

# 排序

article_summary = top_K_samples.iloc[:, 5]
article_summary = article_summary.tolist()

summary_embeddings = model.encode(article_summary)  # 每个摘要得到384维向量
# 将 summary_embeddings 转换为 DataFrame
summary_embeddings_df = pd.DataFrame(summary_embeddings)

# 在第 8 列插入 summary_embeddings_df
top_K_samples.insert(7, 'embeddings', summary_embeddings_df.values.tolist())

top_K_samples.info()

user_embeddings = np.load('user_data/ori_recal_embedding.npy')
# summary_embeddings_mean = summary_embeddings.mean(axis=0)
# 保存summary_embeddings_mean为npy文件
np.save('user_data/ori_rank_embedding.npy', user_embeddings)

sim_summary = cosine_similarity(summary_embeddings, user_embeddings.reshape(1, -1)).flatten()

k = 100
# 重排

# viewed = [410, 192, 615]
viewed = []
# 获取相似度最大的k个index
top_k_indices = np.argsort(sim_summary)[-k:][::-1]
# 可以使用布尔索引来替代列表推导式，以实现更优雅的矩阵操作。
# 首先创建一个布尔数组，标记viewed中的元素为False
mask = np.isin(np.arange(len(sim_summary)), viewed, invert=True)
# 然后结合argsort和布尔索引来筛选出不在viewed中的top_k_indices
top_k_indices = np.argsort(sim_summary[mask])[-k:][::-1]
# 由于使用了布尔索引，需要映射回原始索引
top_k_indices = np.arange(len(sim_summary))[mask][top_k_indices]
top_k_indices = [idx for idx in top_k_indices if idx not in viewed]


# 获取df中这1000个topk indices的样本
top_k_samples = top_K_samples.iloc[top_k_indices]

print(top_k_indices)


# In[31]:


print(top_k_samples)


# In[32]:


import pandas as pd

# 假设top_k_samples是一个列表或者DataFrame
# 如果top_k_samples是列表，将其转换为DataFrame
if not isinstance(top_k_samples, pd.DataFrame):
    top_k_samples = pd.DataFrame(top_k_samples)

# 保存为CSV文件
top_k_samples.to_csv('arxiv_data/arxiv_samples.csv', index=False)


# In[ ]:




