from flask import Flask, render_template  # 添加render_template
from flask_cors import CORS
from routes.auth import bp as auth_bp
from routes.paper import bp as paper_bp
from routes.home import bp as home_bp
from routes.search import bp as search_bp
from routes.collection import bp as collection_bp
from flask import session, redirect, url_for
from flask import send_from_directory
from flask import Flask, request, redirect

app = Flask(__name__)
CORS(app)
app.secret_key = '9da7004e9ecfae743243975254718e2b813961925ccb834f'
app.register_blueprint(auth_bp)
app.register_blueprint(paper_bp)
app.register_blueprint(home_bp)
app.register_blueprint(search_bp)
app.register_blueprint(collection_bp)

def login_required(f):
    def wrapper(*args, **kwargs):
        if 'user_id' not in session:
            return redirect(url_for('auth.login'))
        return f(*args, **kwargs)
    return wrapper

@app.route('/')
@login_required
def index():
    print('username:', session.get('username'))
    return render_template('browse.html', 
                         username=session['username'],
                         base_url=base_url,
                         port=port)  # 新增port参数


@app.route('/search')
def search_page():
    return render_template('search.html', 
                         username=session['username'],
                         base_url=base_url,
                         port=port)

@app.route('/to_collection', methods=['GET'])
def to_collection():
    return render_template('collection.html', username=session['username'], base_url=base_url, port=port)

@app.route('/user_data/user_<int:user_id>/<path:filename>')
def user_data(user_id, filename):
    return send_from_directory(f'user_data/user_{user_id}', filename)

@app.before_request
def before_request():
    if not request.is_secure:
        url = request.url.replace('http://', 'https://', 1)
        return redirect(url, code=301)

if __name__ == '__main__':
    base_url = '127.0.0.1'
    port = 5001  # 将端口定义为变量
    app.run(host=base_url, port=port)