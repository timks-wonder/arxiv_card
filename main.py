import arxiv
import pandas as pd
from datetime import datetime
from sentence_transformers import SentenceTransformer

import os

import os

cv_papers_path = 'arxiv_data/arxiv_cv_papers_1.csv'

if os.path.exists(cv_papers_path):
    df = pd.read_csv(cv_papers_path)
    # 获取CSV文件中文章的最大id
    max_id = df['id'].max() if not df.empty else 0
    # 获取最新日期
    latest_date = pd.to_datetime(df['published']).max()
    # 获取最新日期的后一天
    next_latest_date = latest_date + pd.Timedelta(days=1)
else:
    max_id = 0
    latest_date = pd.to_datetime('2025-01-01')
    next_latest_date = latest_date

today = datetime.now().strftime('%Y%m%d')
latest_date_str = next_latest_date.strftime('%Y%m%d')
if next_latest_date > datetime.now():
    today = next_latest_date

# 更新搜索查询
query = f"cat:cs.CV AND submittedDate:[{latest_date_str} TO {today}]"
print(f'爬取范围：{query}')

# 配置搜索参数
search = arxiv.Search(
    query=query,
    max_results=10000,
    sort_by=arxiv.SortCriterion.SubmittedDate
)

# 获取数据并结构化
data = []
from tqdm import tqdm
from arxiv import UnexpectedEmptyPageError
import numpy as np

try:
    for idx, result in enumerate(tqdm(search.results(), 
                                   desc='爬取进度',
                                   unit='篇',
                                   ncols=100,
                                   colour='green'), start=max_id+1):
        try:
            entry = {
                "id": idx,
                "title": result.title,
                "authors": ", ".join([a.name for a in result.authors]),
                "categories": ", ".join(result.categories),
                "published": result.published.strftime('%Y-%m-%d'),
                "summary": result.summary.replace('\n', ' ').strip(),
                "url": result.entry_id
            }
            data.append(entry)
        except UnexpectedEmptyPageError:
            print(f"\n在获取第{idx}篇论文时遇到空页面")
            continue
        except Exception as e:
            print(f"\n处理第{idx}篇论文时出错: {str(e)}")
            continue

except Exception as e:
    print(f"\n爬取过程中发生错误: {str(e)}")

model = SentenceTransformer('all-MiniLM-L6-v2') 
# 转换为DataFrame并保存
if data:
    df = pd.DataFrame(data)

    # 假设df是你的DataFrame对象，获取第一列（文章名字列）
    article_names = df.iloc[:, 1]
    article_names = article_names.tolist()
    article_summary = df.iloc[:, 5]
    article_summary = article_summary.tolist()

    # 生成标题向量
    title_embeddings = model.encode(article_names)  # 每个标题得到384维向量
    summary_embeddings = model.encode(article_summary)  # 每个摘要得到384维向量
    # 将标题向量和摘要向量转换为二进制格式
    title_embeddings_binary = [np.array(embedding).tobytes() for embedding in title_embeddings]
    summary_embeddings_binary = [np.array(embedding).tobytes() for embedding in summary_embeddings]

    title_embeddings_df = pd.DataFrame(title_embeddings_binary)
    summary_embeddings_df = pd.DataFrame(summary_embeddings_binary)

    df.insert(7, 'title_embeddings', title_embeddings_df.values.tolist())
    df.insert(8, 'summary_embeddings', summary_embeddings_df.values.tolist())

    # 检查文件是否存在
    if os.path.exists(cv_papers_path):
        # 读取现有的CSV文件
        existing_df = pd.read_csv(cv_papers_path)
        # 追加新数据
        combined_df = pd.concat([existing_df, df], ignore_index=True)
        # 保存合并后的数据
        combined_df.to_csv(cv_papers_path, index=False)
    else:
        # 如果文件不存在，直接保存新数据
        df.to_csv(cv_papers_path, index=False)
    
    print(f"成功保存{len(df)}条论文数据")
else:
    print("未找到相关论文数据")