from flask import Blueprint, redirect, url_for, render_template, request, jsonify, session
import pandas as pd
from pathlib import Path
import os

bp = Blueprint('home', __name__, url_prefix='/home')

@bp.route('/gohome', methods=['GET'])
def go_home():
    return render_template('home.html', 
                         username=session['username'],
                         base_url='127.0.0.1',  # 添加模板变量
                         port=5001)

@bp.route('/save_tencent_config', methods=['POST'])
def save_tencent_config():
    data = request.get_json()
    user_id = session.get('user_id')
    
    # 保存到用户目录下的配置文件
    config_path = Path(__file__).parent.parent / f'user_data/user_{user_id}/tencent_config.csv'
    config_path.parent.mkdir(exist_ok=True)
    
    df = pd.DataFrame([{
        'secret_id': data['secret_id'],
        'secret_key': data['secret_key']
    }])
    
    df.to_csv(config_path, index=False)
    return jsonify({'success': True})

@bp.route('/get_tencent_config')
def get_tencent_config():
    user_id = session.get('user_id')
    config_path = Path(__file__).parent.parent / f'user_data/user_{user_id}/tencent_config.csv'
    
    if config_path.exists():
        df = pd.read_csv(config_path)
        return jsonify({
            'secret_id': df.iloc[0]['secret_id'],
            'secret_key': df.iloc[0]['secret_key']
        })
    return jsonify({'secret_id': '', 'secret_key': ''})

@bp.route('/upload_bg', methods=['POST'])
def upload_bg():
    if 'bg_file' not in request.files:
        return jsonify(success=False, message='未选择文件')
    
    file = request.files['bg_file']
    if file.filename == '':
        return jsonify(success=False, message='无效的文件')
    
    user_id = session.get('user_id')
    if not user_id:
        return jsonify(success=False, message='未登录')
    
    upload_dir = os.path.join('user_data', f'user_{user_id}')
    os.makedirs(upload_dir, exist_ok=True)
    
    file.save(os.path.join(upload_dir, 'bg.jpg'))
    return jsonify(success=True)