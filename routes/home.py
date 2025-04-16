from flask import Blueprint, redirect, url_for, render_template

home_bp = Blueprint('home', __name__, url_prefix='/home')

@home_bp.route('/gohome', methods=['GET'])
def go_home():
    return render_template('home.html')