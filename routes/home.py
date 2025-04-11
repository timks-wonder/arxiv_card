from flask import Blueprint, request, redirect, url_for, session, jsonify
import pandas as pd
import os
import csv

home_bp = Blueprint('home', __name__)

@home_bp.route('/update_interests', methods=['POST'])
def update_interests():
    if 'user_id' not in session:
        return redirect(url_for('auth.login'))
    
    user_id = session['user_id']
    keywords = request.get_json().get('keywords', '')
    
    user_dir = os.path.join('user_data', f'user_{user_id}')
    os.makedirs(user_dir, exist_ok=True)
    
    csv_path = os.path.join(user_dir, 'user_inters.csv')
    
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
    else:
        df = pd.DataFrame(columns=['key_words'])
        
    new_entry = pd.DataFrame([{'key_words': keywords}])
    df = pd.concat([df, new_entry], ignore_index=True)
    df.to_csv(csv_path, index=False)
    
    return jsonify({'success': True})

@home_bp.route('/get_interests', methods=['GET'])
def get_interests():
    if 'user_id' not in session:
        return jsonify({'success': False})
    
    user_id = session['user_id']
    csv_path = os.path.join('user_data', f'user_{user_id}', 'user_inters.csv')
    
    if not os.path.exists(csv_path):
        df = pd.DataFrame(columns=['key_words'])
        df.to_csv(csv_path, index=False)
        return jsonify({'success': False, 'keywords': []})
    else:
        df = pd.read_csv(csv_path)
        keywords = df['key_words'].tolist()
        # 将 keywords 转为列表，处理可能的空值
        keywords = [kw for kw in keywords if pd.notna(kw)]
    
    return jsonify({'success': True, 'keywords': keywords})


@home_bp.route('/delete_interest', methods=['POST'])
def delete_interest():
    
    user_id = session['user_id']
    keyword_to_delete = request.get_json().get('keyword', '')
    
    user_dir = os.path.join('user_data', f'user_{user_id}')
    csv_path = os.path.join(user_dir, 'user_inters.csv')
    
    df = pd.read_csv(csv_path)
    if keyword_to_delete not in df['key_words'].values:
        return jsonify({'success': False, 'message': '关键词不存在'})
    
    # 删除匹配的关键词
    df = df[df['key_words'] != keyword_to_delete]
    df.to_csv(csv_path, index=False)
    
    return jsonify({'success': True, 'message': '关键词已删除'})