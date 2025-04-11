from flask import Blueprint, jsonify, session, render_template  # 添加render_template
import pandas as pd
from pathlib import Path
from flask import request
import numpy as np
from numpy.linalg import norm
import pandas as pd
from datetime import datetime
from pathlib import Path

from .rank import rank_papers
from .recall import recall_with_defaults

bp = Blueprint('paper', __name__, url_prefix='/api')

# # CSV文件路径
# DATA_PATH = Path(__file__).parent.parent / 'arxiv_data/arxiv_samples.csv'

@bp.route('/papers')
def get_papers():
    try:
        user_id = (request.args.get('user_id'))
        recall_with_defaults(user_id=int(user_id), K=1000)
        rank_papers(user_id=int(user_id), k=100)
        data_path = Path(__file__).parent.parent / f'user_data/user_{user_id}/arxiv_recall_samples.csv'
        df = pd.read_csv(data_path)
        # if len(filtered_df) < 90:
        #     pass
        
        papers = [{
            'id': row.id,
            'title': row.title,
            'authors': row.authors,
            'categories': row.categories,
            'published': row.published,
            'summary': row.summary,
            'url': row.url,
            'summary_embeddings': np.frombuffer(eval(row.summary_embeddings)[0], dtype=np.float32).tolist()
        } for _, row in df.head(10).iterrows()]
        return jsonify(papers)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

def behave_log(data, is_like=True):
    log = {
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'user_id': data['user_id'],
        'paper_id': data['paper_id'],
        'action_type': 'like' if is_like else 'dislike'
    }
    
    LOG_PATH = Path(__file__).parent.parent / 'user_data/behavior_log.csv'
    
    try:
        if LOG_PATH.exists():
            df = pd.read_csv(LOG_PATH)
        else:
            df = pd.DataFrame(columns=['timestamp', 'user_id', 'paper_id', 'action_type'])
        
        # 替换过时的append方法
        new_entry = pd.DataFrame([log])
        df = pd.concat([df, new_entry], ignore_index=True)
        
        df.to_csv(LOG_PATH, index=False)
        return True
    except Exception as e:
        print(f"记录行为日志失败: {str(e)}")
        return False

@bp.route('/viewed', methods=['GET'])
def add_viewed_record():
    user_id = request.args.get('user_id')
    paper_id = request.args.get('paper_id')
    """添加论文浏览记录到用户文件"""
    viewed_path = f'user_data/user_{user_id}/user_viewed.csv'
    viewed_df = pd.read_csv(viewed_path)
    
    # 添加新记录
    new_entry = pd.DataFrame({'read_article_id': [paper_id]})
    viewed_df = pd.concat([viewed_df, new_entry], ignore_index=True)
    viewed_df.to_csv(viewed_path, index=False)
    return jsonify({'status': 'success'})

@bp.route('/like', methods=['POST'])
def handle_like():
    data = request.get_json()
    user_id = data['user_id']
    # paper_id = data['paper_id']
    
    # # 添加浏览记录
    # add_viewed_record(user_id, paper_id)
    
    from .auth import get_users_df, save_users_df
    df = get_users_df()
    user = df[df['id'] == user_id].iloc[0]
        
    # 获取当前embedding
    user_emb = user['user_embedding']
        
    # 处理embedding更新
    paper_emb = np.array(eval(data['summary_embeddings']))
    updated_emb = user_emb * 0.9 + paper_emb * 0.1
    user_emb_normalized = updated_emb / norm(updated_emb, 2)

    df.loc[df['id'] == user_id, 'user_embedding'][0] = user_emb_normalized
    
    save_users_df(df)
    
    behave_log(data, is_like=True)
    return jsonify({'status': 'success'})

@bp.route('/dislike', methods=['POST'])
def handle_dislike():
    data = request.get_json()
    user_id = data['user_id']
    # paper_id = data['paper_id']
    
    # # 添加浏览记录
    # add_viewed_record(user_id, paper_id)
    
    from .auth import get_users_df, save_users_df
    df = get_users_df()
    user = df[df['id'] == user_id].iloc[0]
        
    # 获取当前embedding
    user_emb = user['user_embedding']
        
    # 处理embedding更新
    paper_emb = np.array(eval(data['summary_embeddings']))
    updated_emb = user_emb * 1.01 - paper_emb * 0.01
    user_emb_normalized = updated_emb / norm(updated_emb, 2)

    df.loc[df['id'] == user_id, 'user_embedding'][0] = user_emb_normalized
    save_users_df(df)
        
    behave_log(data, is_like=False)
    return jsonify({'status': 'success'})


@bp.route('/browse')
def browse():
    return render_template('browse.html', username=session['username'])


