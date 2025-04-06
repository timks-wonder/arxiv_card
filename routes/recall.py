import os
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from time import time

def recall_with_defaults(
    user_id: int = 1,
    K: int = 1000,
    user_path: str = 'user_data/users.csv',
    paper_path: str = 'arxiv_data/arxiv_cv_papers.csv',
    output_dir: str = 'user_data'
):
    """执行召回过程的函数
    
    Args:
        user_id: 用户ID，默认为1
        K: 召回数量，默认1000
        user_path: 用户数据路径
        paper_path: 论文数据路径
        output_dir: 输出目录
    """
    t0 = time()
    viewed_path = f'{output_dir}/user_{user_id}/user_viewed.csv'
    if not os.path.exists(viewed_path):
        viewed_df = pd.DataFrame(columns=['read_article_id'])
        viewed_df.to_csv(viewed_path, index=False)
    else:
        viewed_df = pd.read_csv(viewed_path)
    
    viewed_article_ids = viewed_df['read_article_id'].tolist()
    print(viewed_article_ids)
    
    # 处理用户embedding
    user_df = pd.read_csv(user_path)
    # 读取论文数据
    paper_df = pd.read_csv(paper_path)
    
    user_emb_bytes = eval(user_df.loc[user_df['id'] == user_id, 'user_embedding'].values[0])
    user_embeddings = np.frombuffer(user_emb_bytes, dtype=np.float32)

    title_embeddings = paper_df['title_embeddings'].apply(lambda x: np.frombuffer(eval(x)[0], dtype=np.float32))
    title_embeddings = np.array(title_embeddings.tolist())

    # 计算相似度
    similarities = cosine_similarity(
        title_embeddings, 
        user_embeddings.reshape(1, -1)
    ).flatten()

    # 过滤已读论文
    available_mask = np.isin(np.arange(len(similarities)), viewed_article_ids, invert=True)
    top_K_indices = np.argsort(similarities[available_mask])[-K:][::-1]
    top_K_indices = np.arange(len(similarities))[available_mask][top_K_indices]
    top_K_indices = [idx-1 for idx in top_K_indices if idx not in viewed_article_ids]
    top_K_samples = paper_df.iloc[top_K_indices]
    
    # 保存结果
    save_dir = f'{output_dir}/user_{user_id}'
    os.makedirs(save_dir, exist_ok=True)
    top_K_samples.to_csv(f'{save_dir}/arxiv_recal_samples.csv', index=False)
    
    print(f'召回用时：{time()-t0:.2f}s')
    return top_K_samples

# if __name__ == "__main__":
    # recall_with_defaults()