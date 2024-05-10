from functools import wraps


from flask import render_template, url_for, flash, redirect, request, session
from werkzeug.security import check_password_hash,generate_password_hash

from finetune.models import User, Translation
from finetune.forms import RegistrationForm,LoginForm
from finetune import app, db

def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'login' not in session:
            flash("Please log in or sign up before annotating data.",'danger')
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function

@app.route("/about/en")
def about():
    show_language_selection = True
    return render_template('about_english.html', title='About - English',show_language_selection=show_language_selection)

@app.route("/about/fr")
def about_fr():
    show_language_selection = True
    return render_template('about_french.html', title='About - French',show_language_selection=show_language_selection)

@app.route("/about/mfe")
def about_mfe():
    show_language_selection = True
    return render_template('about_mauritian-creole.html', title='About - Spanish',show_language_selection=show_language_selection)

@app.route("/")
@app.route("/home")
def home():
    # session.clear()
    return render_template('layout.html',session=session)

@app.route("/register",methods=['GET', 'POST'])
def register():
    form = RegistrationForm()
    if form.validate_on_submit():
        session['login'] = True
        session['guest'] = False
        session['username'] = form.username.data

        hashed_password = generate_password_hash(form.password.data)
        user = User(username=form.username.data, email=form.email.data, password=hashed_password)
        db.session.add(user)
        db.session.commit()
        flash(f'Account created for {form.username.data}!','success')
        return redirect(url_for('home'))
    return render_template('register.html',title='Register',form=form)

@app.route("/login",methods=['GET', 'POST'])
def login():
    form = LoginForm()
    if form.validate_on_submit():
        user = User.query.filter_by(username=form.username.data).first()
        if user and check_password_hash(user.password, form.password.data):
            session['login'] = True
            session['guest'] = False
            session['username'] = form.username.data
            flash('You have been logged in!','success')
            return redirect(url_for('home'))
        else:
            flash('Login Unsuccessful. Please change username and password','danger')
    return render_template('login.html',title='Login',form=form)

@app.route("/guest_login",methods=['GET', 'POST'])
def guest_login():
    session['login'] = True
    session['username'] = 'Guest'
    # session['guest_user'] = {'username': 'Guest'}
    flash('You have been logged in as guest!','success')
    return redirect(url_for('home'))

@app.route("/annotate",methods=['GET', 'POST'])
@login_required
def annotate():
    translation = Translation.query.filter_by(rating='-1').first()

    if request.method == 'POST':
        session['test'] = 'T'
        translation.rating = request.form['rating']
        translation.suggested_text = request.form['suggested_text']
        db.session.commit()
        return redirect(url_for('annotate'))
    return render_template('annotate.html',title='Annotate',translation=translation)

@app.route("/logout")
def logout():
    session.pop('login', None)
    if 'username' in session:
        session.pop('username',None)
    flash('You have been logged out!', 'success')
    return redirect(url_for('home'))