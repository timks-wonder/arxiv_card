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
    return render_template('browse.html', 
                         username=session['username'],
                         base_url=base_url,
                         port=port)  # 新增port参数

@app.route('/proxy/pdf')
def proxy_pdf():
    import requests
    from flask import request, Response
    
    pdf_url = request.args.get('url')
    response = requests.get(pdf_url, stream=True)
    
    return Response(
        response.iter_content(chunk_size=1024),
        content_type=response.headers['Content-Type'],
        headers={
            'Content-Disposition': response.headers.get('Content-Disposition', 'inline'),
            'Access-Control-Allow-Origin': '*'
        }
    )

if __name__ == '__main__':
    base_url = '127.0.0.1'
    port = 5001  # 将端口定义为变量
    app.run(host=base_url, port=port)