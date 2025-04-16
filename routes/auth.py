from flask import Blueprint, request, redirect, url_for
from werkzeug.security import generate_password_hash, check_password_hash
import numpy as np
from flask import render_template, session  # 确保已导入session
import pandas as pd
from pathlib import Path

bp = Blueprint('auth', __name__, url_prefix='/auth')

# 用户数据路径
USERS_PATH = Path(__file__).parent.parent / 'user_data/users.csv'

def get_users_df():
    if USERS_PATH.exists():
        df = pd.read_csv(USERS_PATH)
        # 转换二进制字段（修复：添加dtype参数）
        df['user_embedding'] = df['user_embedding'].apply(
            lambda x: np.frombuffer(eval(x), dtype=np.float32)  # 移除非必要的bytes()
        )
        df['interes_embedding'] = df['interes_embedding'].apply(
            lambda x: np.frombuffer(eval(x), dtype=np.float32)  # 直接使用eval结果
        )
        return df
    else:
        return pd.DataFrame(columns=['id', 'username', 'password', 'user_embedding', 'interes_embedding'])

def save_users_df(df):
    df = df.copy()
    df['user_embedding'] = df['user_embedding'].apply(
        lambda x: str(x.tobytes())
    )
    df['interes_embedding'] = df['interes_embedding'].apply(
        lambda x: str(x.tobytes())
    )
    df.to_csv(USERS_PATH, index=False)

@bp.route('/register', methods=['POST'])
def register():
    username = request.form['username']
    password = generate_password_hash(request.form['password'])
    
    # 加载初始embedding（添加类型转换）
    rank_emb = np.load('user_data/ori_rank_embedding.npy').astype(np.float32)
    recall_emb = np.load('user_data/ori_recall_embedding.npy').astype(np.float32)

    try:
        df = get_users_df()
        if username in df['username'].values:
            return '用户名已存在'
        
        new_id = df['id'].max() + 1 if not df.empty else 1
        new_user = pd.DataFrame([{
            'id': new_id,
            'username': username,
            'password': password,
            'user_embedding': rank_emb,
            'interes_embedding': recall_emb
        }])
        
        save_users_df(pd.concat([df, new_user], ignore_index=True))
        return redirect(url_for('auth.login'))
    except Exception as e:
        return f'注册失败: {str(e)}'

@bp.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'GET':
        return render_template('login_.html')
    
    try:
        username = request.form.get('username', '')
        password = request.form.get('password', '')
        
        if not username or not password:
            raise ValueError("用户名和密码不能为空")
            
        df = get_users_df()
        user = df[df['username'] == username]
        
        if not user.empty and check_password_hash(user.iloc[0]['password'], password):
            # 将numpy.int64转换为Python原生int
            session['user_id'] = int(user.iloc[0]['id'])
            session['username'] = username
            return redirect('/')
            
        return '无效的用户名或密码'
    
    except Exception as e:
        print(f"登录错误: {str(e)}")
        return f"请求参数错误: {str(e)}", 400

@bp.route('/logout', methods=['POST'])
def logout():
    session.clear()
    return redirect(url_for('auth.login'))