from unicodedata import category
from flask import Blueprint, jsonify, session, render_template  # 添加render_template
import pandas as pd
from pathlib import Path
from flask import request
import numpy as np
from numpy.linalg import norm
import pandas as pd
from datetime import datetime
from pathlib import Path
import os

from .rank import rank_papers
from .recall import recall_with_defaults
import json

bp = Blueprint('paper', __name__, url_prefix='/api')

from tencentcloud.common import credential
from tencentcloud.common.profile.client_profile import ClientProfile
from tencentcloud.common.profile.http_profile import HttpProfile
from tencentcloud.tmt.v20180321 import tmt_client, models
from tencentcloud.common.exception.tencent_cloud_sdk_exception import TencentCloudSDKException

@bp.route('/papers')
def get_papers():
    try:
        user_id = (request.args.get('user_id'))
        start_date = request.args.get('start_date', None)  # 默认为None
        end_date = request.args.get('end_date', None)      # 默认为None
        category = request.args.get('category', None)      # 默认为None

        recall_with_defaults(user_id=int(user_id), K=100, start_date=start_date, end_date=end_date, category=category)
        rank_papers(user_id=int(user_id), k=3)
        data_path = Path(__file__).parent.parent / f'user_data/user_{user_id}/arxiv_recall_samples.csv'
        df = pd.read_csv(data_path)
        
        papers = [{
            'id': row.id,
            'title': row.title,
            'authors': row.authors,
            'categories': row.categories,
            'published': row.published,
            'summary': row.summary,
            'url': row.url,
            'liked': row.liked,
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
    user_id = int(data['user_id'])
    paper_id = data['paper_id']
    category = data['category']
    
    # 1. 更新论文点赞数
    # 读取类别映射文件
    cls_path = Path(__file__).parent.parent / 'arxiv_data/cls.csv'
    cls_df = pd.read_csv(cls_path)
    filename = cls_df[cls_df['category'] == category]['filename'].iloc[0]
    
    # 读取对应类别的论文文件
    papers_path = Path(__file__).parent.parent / f'arxiv_data/{filename}'
    papers_df = pd.read_csv(papers_path)
    
    # 找到对应论文并更新liked
    papers_df.loc[papers_df['id'] == paper_id, 'liked'] += 1
    papers_df.to_csv(papers_path, index=False)
    
    from .auth import get_users_df, save_users_df
    df = get_users_df()
    user = df[df['id'] == user_id].iloc[0]
        
    # 获取当前embedding
    user_emb = user['user_embedding']
        
    # 处理embedding更新
    paper_emb = np.array(data['summary_embeddings'])
    updated_emb = user_emb * 0.9 + paper_emb * 0.1
    user_emb_normalized = updated_emb / norm(updated_emb, 2)

    df.loc[df['id'] == user_id, 'user_embedding'][0] = user_emb_normalized
    
    save_users_df(df)
    
    behave_log(data, is_like=True)
    return jsonify({'status': 'success'})

@bp.route('/dislike', methods=['POST'])
def handle_dislike():
    data = request.get_json()
    user_id = int(data['user_id'])
    # paper_id = data['paper_id']
    
    # # 添加浏览记录
    # add_viewed_record(user_id, paper_id)
    
    from .auth import get_users_df, save_users_df
    df = get_users_df()
    user = df[df['id'] == user_id].iloc[0]
        
    # 获取当前embedding
    user_emb = user['user_embedding']
        
    # 处理embedding更新
    paper_emb = np.array(data['summary_embeddings'])
    updated_emb = user_emb * 1.01 - paper_emb * 0.01
    user_emb_normalized = updated_emb / norm(updated_emb, 2)

    df.loc[df['id'] == user_id, 'user_embedding'][0] = user_emb_normalized
    save_users_df(df)
        
    behave_log(data, is_like=False)
    return jsonify({'status': 'success'})

@bp.route('/browse')
def browse():
    return render_template('browse.html', username=session['username'])

TENCENT_REGION = "ap-shanghai"

@bp.route('/translate', methods=['POST'])
def translate_text():
    try:
        data = request.get_json()
        text = data.get('text')
        from_code = data.get('from', 'en')
        to_code = data.get('to', 'zh')
        
        user_id = data.get('user_id')  # 从请求体中获取user_id
        config_path = Path(__file__).parent.parent / f'user_data/user_{user_id}/tencent_config.csv'
        
        if not config_path.exists():
            return jsonify({
                'original': text,
                'translated': '',
                'from': from_code,
                'to': to_code
            })
            
        df = pd.read_csv(config_path)
        secret_id = df.iloc[0]['secret_id']
        secret_key = df.iloc[0]['secret_key']
        
        if not secret_id or not secret_key:
            return jsonify({
                'original': text,
                'translated': '',
                'from': from_code,
                'to': to_code
            })
        
        # 初始化腾讯云客户端
        cred = credential.Credential(secret_id, secret_key)
        http_profile = HttpProfile()
        http_profile.endpoint = "tmt.tencentcloudapi.com"
        
        client_profile = ClientProfile()
        client_profile.httpProfile = http_profile
        client = tmt_client.TmtClient(cred, TENCENT_REGION, client_profile)
        # 构建请求参数
        params = {
            "SourceText": text,
            "Source": from_code,
            "Target": to_code,
            "ProjectId": 0
        }
        req = models.TextTranslateRequest()
        req.from_json_string(json.dumps(params))
        
        # 调用API
        resp = client.TextTranslate(req)
        return jsonify({
            'original': text,
            'translated': resp.TargetText,
            'from': from_code,
            'to': to_code
        })
    except TencentCloudSDKException as e:
        return jsonify({'error': f"腾讯云翻译错误: {str(e)}"}), 500
    except Exception as e:
        return jsonify({'error': f"翻译服务异常: {str(e)}"}), 500

@bp.route('/comments', methods=['POST'])
def add_comment():
    try:
        data = request.get_json()
        user_id = data['user_id']
        paper_id = data['paper_id']
        content = data['content']
        
        # 评论存储路径
        comment_path = Path(__file__).parent.parent / 'user_data/comments.csv'
        
        # 创建文件如果不存在
        if not comment_path.exists():
            pd.DataFrame(columns=['id', 'user_id', 'paper_id', 'content', 'timestamp']).to_csv(comment_path, index=False)
            
        df = pd.read_csv(comment_path)
        
        # 生成新评论ID
        new_id = df['id'].max() + 1 if not df.empty and 'id' in df.columns else 1
        
        # 添加新评论
        new_comment = {
            'id': new_id,
            'user_id': user_id,
            'paper_id': paper_id,
            'content': content,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        df = pd.concat([df, pd.DataFrame([new_comment])], ignore_index=True)
        df.to_csv(comment_path, index=False)
        
        return jsonify({'status': 'success', 'comment': new_comment})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@bp.route('/papers/<paper_id>/comments')
def get_comments(paper_id):
    try:
        comment_path = Path(__file__).parent.parent / 'user_data/comments.csv'
        if not comment_path.exists():
            return jsonify([])
            
        df = pd.read_csv(comment_path)
        # 添加类型转换
        df['paper_id'] = df['paper_id'].astype(str)  # 新增代码
        paper_comments = df[df['paper_id'] == paper_id].to_dict('records')
        if not paper_comments:
            return jsonify([])
        # 关联用户信息
        users_df = pd.read_csv(Path(__file__).parent.parent / 'user_data/users.csv')
        users_df['id'] = users_df['id'].astype(int)  # 新增代码
        for comment in paper_comments:
            user = users_df[users_df['id'] == int(comment['user_id'])].iloc[0]
            comment['username'] = user['username']
        return jsonify(paper_comments)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

def get_comment_by_id(comment_id):
    """根据评论ID获取评论信息"""
    comment_path = Path(__file__).parent.parent / 'user_data/comments.csv'
    if not comment_path.exists():
        return None
        
    df = pd.read_csv(comment_path)
    comment = df[df['id'] == comment_id].to_dict('records')
    if not comment:
        return None
        
    return comment[0]

def delete_comment_from_db(comment_id):
    """从数据库删除评论"""
    comment_path = Path(__file__).parent.parent / 'user_data/comments.csv'
    if not comment_path.exists():
        return False
        
    df = pd.read_csv(comment_path)
    # 确保有id列
    if 'id' not in df.columns:
        return False
        
    # 删除指定评论
    df = df[df['id'] != comment_id]
    df.to_csv(comment_path, index=False)
    return True

@bp.route('/comments/<int:comment_id>', methods=['DELETE'])
def delete_comment(comment_id):
    try:
        data = request.get_json()
        if not data or 'user_id' not in data:
            return jsonify({'error': '未提供用户ID'}), 401
            
        current_user_id = int(data['user_id'])
        
        # 获取评论
        comment = get_comment_by_id(comment_id)
        if not comment:
            return jsonify({'error': '评论不存在'}), 404
            
        # 检查评论是否属于当前用户
        if int(comment['user_id']) != current_user_id:
            return jsonify({'error': '无权删除他人评论'}), 403
            
        # 执行删除操作
        if delete_comment_from_db(comment_id):
            return jsonify({'success': True, 'message': '评论删除成功'})
        else:
            return jsonify({'error': '删除评论失败'}), 500
            
    except Exception as e:
        return jsonify({'error': f'服务器错误: {str(e)}'}), 500