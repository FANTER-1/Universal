                                                        config.py
                                                        python

import os

class Config:
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'you-will-never-guess'
    SQLALCHEMY_DATABASE_URI = os.environ.get('DATABASE_URL') or 'sqlite:///app.db'
    SQLALCHEMY_TRACK_MODIFICATIONS = False
                                                        run.py
                                                        python

from app import create_app

app = create_app()

if __name__ == "__main__":
    app.run(debug=True)
                                                        app/__init__.py
                                                        python

from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from flask_migrate import Migrate
from flask_login import LoginManager
from config import Config

db = SQLAlchemy()
migrate = Migrate()
login = LoginManager()
login.login_view = 'auth.login'

def create_app(config_class=Config):
    app = Flask(__name__)
    app.config.from_object(config_class)

    db.init_app(app)
    migrate.init_app(app, db)
    login.init_app(app)

    from app.routes.main import bp as main_bp
    app.register_blueprint(main_bp)

    from app.routes.auth import bp as auth_bp
    app.register_blueprint(auth_bp, url_prefix='/auth')

    return app

from app import models
                                                            app/models.py
                                                            python

from app import db, login
from flask_login import UserMixin
from werkzeug.security import generate_password_hash, check_password_hash

class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(64), index=True, unique=True)
    email = db.Column(db.String(120), index=True, unique=True)
    password_hash = db.Column(db.String(128))

    def set_password(self, password):
        self.password_hash = generate_password_hash(password)

    def check_password(self, password):
        return check_password_hash(self.password_hash, password)

@login.user_loader
def load_user(id):
    return User.query.get(int(id))
                                                    app/forms.py
                                                    python

from flask_wtf import FlaskForm
from wtforms import StringField, PasswordField, SubmitField
from wtforms.validators import DataRequired, Email, EqualTo

class LoginForm(FlaskForm):
    username = StringField('Username', validators=[DataRequired()])
    password = PasswordField('Password', validators=[DataRequired()])
    submit = SubmitField('Sign In')

class RegistrationForm(FlaskForm):
    username = StringField('Username', validators=[DataRequired()])
    email = StringField('Email', validators=[DataRequired(), Email()])
    password = PasswordField('Password', validators=[DataRequired()])
    password2 = PasswordField('Repeat Password', validators=[DataRequired(), EqualTo('password')])
    submit = SubmitField('Register')
                                                        app/routes/main.py
                                                        python

from flask import Blueprint, render_template

bp = Blueprint('main', __name__)

@bp.route('/')
def index():
    return render_template('index.html')

@bp.route('/about')
def about():
    return render_template('about.html')
                                                            app/routes/auth.py
                                                            python

from flask import Blueprint, render_template, redirect, url_for, flash
from flask_login import current_user, login_user, logout_user
from app.models import User
from app.forms import LoginForm, RegistrationForm
from app import db

bp = Blueprint('auth', __name__)

@bp.route('/login', methods=['GET', 'POST'])
def login():
    if current_user.is_authenticated:
        return redirect(url_for('main.index'))
    form = LoginForm()
    if form.validate_on_submit():
        user = User.query.filter_by(username=form.username.data).first()
        if user is None or not user.check_password(form.password.data):
            flash('Invalid username or password')
            return redirect(url_for('auth.login'))
        login_user(user)
        return redirect(url_for('main.index'))
    return render_template('login.html', title='Sign In', form=form)

@bp.route('/logout')
def logout():
    logout_user()
    return redirect(url_for('main.index'))

@bp.route('/register', methods=['GET', 'POST'])
def register():
    if current_user.is_authenticated:
        return redirect(url_for('main.index'))
    form = RegistrationForm()
    if form.validate_on_submit():
        user = User(username=form.username.data, email=form.email.data)
        user.set_password(form.password.data)
        db.session.add(user)
        db.session.commit()
        flash('Congratulations, you are now a registered user!')
        return redirect(url_for('auth.login'))
    return render_template('register.html', title='Register', form=form)
  
                                                      app/templates/base.html
                                                      html

<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>{% block title %}My Web App{% endblock %}</title>
    <link rel="stylesheet" href="{{ url_for('static', filename='style.css') }}">
</head>
<body>
    <nav>
        <a href="{{ url_for('main.index') }}">Home</a>
        <a href="{{ url_for('main.about') }}">About</a>
        {% if current_user.is_anonymous %}
        <a href="{{ url_for('auth.login') }}">Login</a>
        <a href="{{ url_for('auth.register') }}">Register</a>
        {% else %}
        <a href="{{ url_for('auth.logout') }}">Logout</a>
        {% endif %}
    </nav>
    <div class="content">
        {% block content %}{% endblock %}
    </div>
</body>
</html>

                                                            app/templates/index.html
                                                            html

{% extends "base.html" %}

{% block title %}Home{% endblock %}

{% block content %}
<h1>Welcome to My Web App</h1>
{% endblock %}

                                                                app/templates/about.html
                                                                html

{% extends "base.html" %}

{% block title %}About{% endblock %}

{% block content %}
<h1>About Us</h1>
<p>This is a simple Flask web application.</p>
{% endblock %}

                                                                app/templates/login.html
                                                                html

{% extends "base.html" %}

{% block title %}Login{% endblock %}

{% block content %}
<h1>Sign In</h1>
<form action="" method="post" novalidate>
    {{ form.hidden_tag() }}
    <p>
        {{ form.username.label }}<br>
        {{ form.username(size=32) }}<br>
        {% for error in form.username.errors %}
        <span style="color: red;">[{{ error }}]</span>
        {% endfor %}
    </p>
    <p>
        {{ form.password.label }}<br>
        {{ form.password(size=32) }}<br>
        {% for error in form.password.errors %}
        <span style="color: red;">[{{ error }}]</span>
        {% endfor %}
    </p>
    <p>{{ form.submit() }}</p>
</form>
{% endblock %}

                                        app/templates/register.html
                                        html

{% extends "base.html" %}

{% block title %}Register{% endblock %}

{% block content %}
<h1>Register</h1>
<form action="" method="post">
    {{ form.hidden_tag() }}
    <p>
        {{ form.username.label }}<br>
        {{ form.username(size=32) }}<br>
        {% for error in form.username.errors %}
        <span style="color: red;">[{{ error }}]</span>
        {% endfor %}
    </p>
    <p>
        {{ form.email.label }}<br>
        {{ form.email(size=64) }}<br>
        {% for error in form.email.errors %}
        <span style="color: red;">[{{ error }}]</span>
        {% endfor %}
    </p>
    <p>
        {{ form.password.label }}<br>
        {{ form.password(size=32) }}<br>
        {% for error in form.password.errors %}
        <span style="color: red;">[{{ error }}]</span>
        {% endfor %}
    </p>
    <p>
        {{ form.password2.label }}<br>
        {{ form.password2(size=32) }}<br>
        {% for error in form.password2.errors %}
        <span style="color: red;">[{{ error }}]</span>
        {% endfor %}
    </p>
    <p>{{ form.submit() }}</p>
</form>
{% endblock %}

                                            app/static/style.css
                                            css

body {
    font-family: Arial, sans-serif;
}

nav {
    background-color: #f8f9fa;
    padding: 10px;
}

nav a {
    margin-right: 10px;
    text-decoration: none;
    color: #007bff;
}

nav a:hover {
    text-decoration: underline;
}

.content {
    padding: 20px;
}
                        requirements.txt

Flask
Flask-SQLAlchemy
Flask-Migrate
Flask-Login
Flask-WTF
