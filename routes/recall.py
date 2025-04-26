import os
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from time import time

def recall_with_defaults(
    user_id: int = 1,
    K: int = 1000,
    start_date: str = None,
    end_date: str = None,
    category: str = 'CV',
    user_path: str = 'user_data/users.csv',
    output_dir: str = 'user_data'
):
    """执行召回过程的函数
    
    Args:
        user_id: 用户ID，默认为1
        K: 召回数量，默认1000
        start_date: 开始日期(YYYY-MM-DD)，可选
        end_date: 结束日期(YYYY-MM-DD)，可选
        category: 论文类别，默认为'CV'
        user_path: 用户数据路径
        output_dir: 输出目录
    """
    # 读取类别映射文件
    t0 = time()
    cls_df = pd.read_csv('arxiv_data/cls.csv')
    paper_path = f"arxiv_data/{cls_df.loc[cls_df['category'] == category, 'filename'].values[0]}"
    
    save_dir = f'{output_dir}/user_{user_id}'
    os.makedirs(save_dir, exist_ok=True)
    viewed_path = f'{output_dir}/user_{user_id}/user_viewed.csv'
    if not os.path.exists(viewed_path):
        viewed_df = pd.DataFrame(columns=['read_article_id'])
        viewed_df.to_csv(viewed_path, index=False)
    else:
        viewed_df = pd.read_csv(viewed_path)
    
    viewed_article_ids = viewed_df['read_article_id'].tolist()
    
    # 处理用户embedding
    user_df = pd.read_csv(user_path)
    # 读取论文数据
    paper_df = pd.read_csv(paper_path)
    
    # 添加日期筛选
    if start_date or end_date:
        paper_df['published'] = pd.to_datetime(paper_df['published'])
        if start_date:
            paper_df = paper_df[paper_df['published'] >= pd.to_datetime(start_date)]
        if end_date:
            paper_df = paper_df[paper_df['published'] <= pd.to_datetime(end_date)]
    
    # 过滤已浏览过的论文
    paper_df = paper_df[~paper_df['id'].isin(viewed_article_ids)]
    
    user_emb_bytes = eval(user_df.loc[user_df['id'] == user_id, 'user_embedding'].values[0])
    user_embeddings = np.frombuffer(user_emb_bytes, dtype=np.float32)

    title_embeddings = paper_df['title_embeddings'].apply(lambda x: np.frombuffer(eval(x)[0], dtype=np.float32))
    title_embeddings = np.array(title_embeddings.tolist())

    # 计算相似度
    similarities = cosine_similarity(
        title_embeddings, 
        user_embeddings.reshape(1, -1)
    ).flatten()

    # 获取相似度最高的K篇论文
    top_K_indices = np.argsort(similarities)[-K:][::-1]
    top_K_samples = paper_df.iloc[top_K_indices]
    
    top_K_samples.to_csv(f'{save_dir}/arxiv_recall_samples.csv', index=False)
    
    print(f'召回用时：{time()-t0:.2f}s')
    return top_K_samples

# if __name__ == "__main__":
    # recall_with_defaults()