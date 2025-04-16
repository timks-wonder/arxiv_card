from flask import Flask, render_template  # 添加render_template
from flask_cors import CORS
from routes.auth import bp as auth_bp
from routes.paper import bp as paper_bp
from routes.home import home_bp as home_bp
from flask import session, redirect, url_for

app = Flask(__name__)
CORS(app)
app.secret_key = '9da7004e9ecfae743243975254718e2b813961925ccb834f'
app.register_blueprint(auth_bp)
app.register_blueprint(paper_bp)
app.register_blueprint(home_bp)

def login_required(f):
    def wrapper(*args, **kwargs):
        if 'user_id' not in session:
            return redirect(url_for('auth.login'))
        return f(*args, **kwargs)
    return wrapper

@app.route('/')
@login_required  # 添加装饰器
def index():
    print('username:', session.get('username'))
    return render_template('browser.html', username=session['username'])

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001)