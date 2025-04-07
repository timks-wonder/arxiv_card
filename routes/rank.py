import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from time import time
import os

def rank_papers(
    user_id: int = 1,
    k: int = 100,
    recall_dir: str = 'user_data',
    user_path: str = 'user_data/users.csv'
):
    """执行论文排序的函数
    
    Args:
        user_id: 用户ID，默认1
        k: 排序保留数量，默认100
        recall_dir: 召回结果目录，默认user_data
        user_path: 用户数据路径
    """
    t0 = time()

    user_df = pd.read_csv(user_path)
    recall_path = f'{recall_dir}/user_{user_id}/arxiv_recall_samples.csv'
    recall_df = pd.read_csv(recall_path)
    
    user_emb_bytes = eval(user_df.loc[user_df['id'] == user_id, 'user_embedding'].values[0])
    user_embeddings = np.frombuffer(user_emb_bytes, dtype=np.float32)

    summary_embeddings = recall_df['summary_embeddings'].apply(lambda x: np.frombuffer(eval(x)[0], dtype=np.float32))
    summary_embeddings = np.array(summary_embeddings.tolist())

    
    # 计算相似度
    sim_scores = cosine_similarity(
        summary_embeddings,
        user_embeddings.reshape(1, -1)
    ).flatten()
    
    # # 过滤已读论文
    # viewed = []
    # available_mask = np.isin(np.arange(len(sim_scores)), viewed, invert=True)
    top_indices = np.argsort(sim_scores)[-k:][::-1]
    # top_indices = np.arange(len(sim_scores))[available_mask][top_indices]
    # top_indices = [idx for idx in top_indices if idx not in viewed]
    
    # 保存结果
    output_dir = f'{recall_dir}/user_{user_id}'
    os.makedirs(output_dir, exist_ok=True)
    ranked_df = recall_df.iloc[top_indices]
    ranked_df.to_csv(f'{output_dir}/arxiv_rank_samples.csv', index=False)
    
    print(f'排序用时：{time()-t0:.2f}s')
    return ranked_df

# if __name__ == "__main__":
#     rank_papers()