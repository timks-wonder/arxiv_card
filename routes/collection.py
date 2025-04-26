from flask import Blueprint, jsonify, render_template, session
import pandas as pd
from flask import request
import pandas as pd
import os
from .auth import get_users_df  # 新增导入

bp = Blueprint('collection', __name__, url_prefix='/api')

@bp.route('/collection', methods=['POST'])
def handle_collection():
    data = request.json
    user_id = data['user_id']
    paper = data['paper']
    action = data['action']

    collection_path = f'user_data/user_{user_id}/collection.csv'
    
    # 获取用户配置的收藏数量限制
    users_df = get_users_df()
    user = users_df[users_df['id'] == int(user_id)].iloc[0]
    max_collections = user['num_collections']  # 从用户数据中获取

    try:
        if action == 'collect':
            # 检查文件是否存在
            if not os.path.exists(collection_path):
                # 创建新文件并写入表头
                pd.DataFrame(columns=['paper_id', 'authors', 'title', 'categories', 'published', 'summary', 'url']).to_csv(collection_path, index=False)
            
            # 读取现有收藏
            collection_df = pd.read_csv(collection_path)
            
            # 检查是否已收藏
            if paper['url'] not in collection_df['url'].values:
                # 添加新收藏
                new_entry = pd.DataFrame({
                    'paper_id': [paper['id']],
                    'authors': [paper['authors']],
                    'title': [paper['title']],
                    'categories': [paper['categories']],
                    'published': [paper['published']],
                    'summary': [paper['summary']],
                    'url': [paper['url']]
                })
                collection_df = pd.concat([collection_df, new_entry], ignore_index=True)
                
                if len(collection_df) > max_collections:
                    collection_df = collection_df.tail(max_collections)  # 保留最新的20条记录
                
                collection_df.to_csv(collection_path, index=False)

                return jsonify({'status': 'success', 'message': 'Paper collected'})
            else:
                return jsonify({'status': 'info', 'message': 'Paper already collected'})
                
        elif action == 'uncollect':
            if os.path.exists(collection_path):
                collection_df = pd.read_csv(collection_path)
                # 移除收藏
                collection_df = collection_df[collection_df['url'] != paper['url']]
                collection_df.to_csv(collection_path, index=False)
                return jsonify({'status': 'success', 'message': 'Paper uncollected'})
            return jsonify({'status': 'info', 'message': 'No collection file found'})
            
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

@bp.route('/get_collection', methods=['GET'])
def get_collection():
    user_id = request.args.get('user_id')
    collection_path = f'user_data/user_{user_id}/collection.csv'

    if not os.path.exists(collection_path):
        return jsonify([])  # 返回空数组而不是错误信息

    try:
        collection_df = pd.read_csv(collection_path)
        
        papers = [{
            'id': row.paper_id,
            'authors': row.authors,
            'title': row.title,
            'categories': row.categories,
            'published': row.published,
            'summary': row.summary,
            'url': row.url
        } for _, row in collection_df.iterrows()]
        
        return jsonify(papers)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

