from flask import Flask, render_template, request,redirect, url_for,session
from flask_sqlalchemy import SQLAlchemy
import bcrypt
import pickle
import numpy as np


model = pickle.load(open('knn_model.pkl', 'rb'))
model2 = pickle.load(open('knn_model2.pkl', 'rb'))

app = Flask(__name__)

# Configure primary database and secondary feedback database
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///database.db'
app.config['SQLALCHEMY_BINDS'] = {
    'feedback': 'sqlite:///feedback.db'  # Use 'feedback' as the bind key
}
app.secret_key = 'secret_key'
db = SQLAlchemy(app)

# Define User model for the primary database
class User(db.Model):
    __bind_key__ = None  # Use the primary database
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    email = db.Column(db.String(100), unique=True)
    password = db.Column(db.String(100))

    def __init__(self, email, password, name):
        self.name = name
        self.email = email
        self.password = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')

    def check_password(self, password):
        return bcrypt.checkpw(password.encode('utf-8'), self.password.encode('utf-8'))

# Define Feedback model for the feedback database
class Feedback(db.Model):
    __bind_key__ = 'feedback'  # Use the feedback database
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    email = db.Column(db.String(100), unique=False)
    message = db.Column(db.String(100))

    def __init__(self, name, email, message):
        self.name = name
        self.email = email
        self.message = message

# Create the tables for the primary and feedback databases
with app.app_context():
    # db.drop_all()  # Drop all tables
    db.create_all()

@app.route('/')
def man():
    return render_template('Login.html')

@app.route('/Home')
def home():
    return render_template('Batsman.html')

@app.route('/Admin')
def admin():
    if session['email']:
        all_users = User.query.all()
        feedback_list = Feedback.query.all()
        return render_template('Admin.html', all_users=all_users,feedback_list=feedback_list)


@app.route('/Chose')
def chose():
    return render_template('Chose.html')

@app.route('/Bowler')
def bowler():
    return render_template('Bowler.html')

@app.route('/index')
def index():
    return render_template('index.html')

@app.route('/index2')
def index2():
    return render_template('Admin.html')

@app.route('/Contactus')
def contactus():
    success_message = request.args.get('message')
    return render_template('Contactus.html',success_message=success_message)

@app.route('/About')
def about():
    return render_template('About.html')

@app.route('/Admin_Login')
def Adminlogin():
    return render_template('Admin_login.html')

@app.route('/register')
def Register():
    return render_template('Register.html')

@app.route('/Login', methods=['POST','GET'])
def Login():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']

        user = User.query.filter_by(email=email).first()
        
        if user and user.check_password(password):
            session['email'] = user.email
            return redirect(url_for('index'))
        else:
            return redirect(url_for('man',error="Ivalid credentials"))
    else:
        return redirect(url_for('man',error="Ivalid credentials"))
    
@app.route('/AdminLogin', methods=['POST','GET'])
def AdminLogin():
    email = request.form['email']
    password = request.form['password']
    # Example of validating the email and password 
    if email == 'Admin@login.com' and password == 'AdminLogin':
        # If email and password are correct, redirect to the home page
        return redirect(url_for('admin'))
    else:
        return redirect(url_for('Adminlogin',error="Ivalid credentials"))

#New user registration function
@app.route('/register',methods=['GET','POST'])
def register():
        # handle request
    name = request.form['name']
    email = request.form['email']
    password = request.form['password']

    if email == "Admin@login.com":
        # Return HTML with a script to display an alert
        alert_script = "<script>alert('You cannot use \"Admin@login.com\" as your email. Please enter a different email.'); window.history.back();</script>"
        return alert_script
    try:
        new_user = User(name=name, email=email, password=password)
        db.session.add(new_user)
        db.session.commit()
            # Redirect to the 'man' page after successful registration
        return redirect(url_for('man'))

    except Exception as e:
        # If an error occurs, rollback the session and display an alert
        db.session.rollback()
        error_message = f"An error occurred: {str(e)}. Please try again."
        return f"""<script>
                alert('{error_message}');
                window.history.back();
            </script>"""


#feed back system function 
@app.route('/feedback',methods=['GET','POST'])
def feedback():
    if request.method == 'POST':
        # handle request
        name = request.form['name']
        email = request.form['email']
        message = request.form['message']

        new_user = Feedback(name=name,email=email,message=message)
        db.session.add(new_user)
        db.session.commit()
        return redirect(url_for('contactus',message='Your data has been recorded successfully.'))
    else:
        return "Unsupported request method."

@app.route('/delete_user/<int:user_id>', methods=['POST'])
def delete_user(user_id):
    # Fetch the user by ID
    user_to_delete = User.query.get_or_404(user_id)
    
    # Delete the user from the database
    db.session.delete(user_to_delete)
    db.session.commit()
    
    # Redirect back to the dashboard or Admin.html page
    return redirect(url_for('admin'))

@app.route('/delete_feedback/<int:feedback_id>', methods=['POST'])
def delete_feedback(feedback_id):
    # Get the feedback entry by ID
    feedback_to_delete = Feedback.query.get(feedback_id)
    
    if feedback_to_delete:
        # Remove the feedback entry from the database
        db.session.delete(feedback_to_delete)
        db.session.commit()
    
    # Redirect back to the same page or the feedback list page
    return redirect(url_for('admin'))


#predicting bowler training   
@app.route('/predict2', methods=['POST'])
def prediction2():
    data0 = float(request.form['Overs'])
    data1 = float(request.form['Runs'])
    data2 = float(request.form['Wickets'])
    data3 = float(request.form['Economy'])
    data4 = float(request.form['Average'])
    data5 = float(request.form['Bowler_stat'])
    data6 = float(request.form['4s'])
    data7 = float(request.form['6s'])
    data8 = float(request.form['Dots'])
    arr = np.array([[data0,data1, data2, data3, data4,data5,data6,data7,data8]])
    pred = model2.predict(arr)
    prediction = pred[0]
    if prediction == "Technical Skills":
      return render_template('bowlers/Module1.html') 
    if prediction == "Tactics and Strategy":
       return render_template('bowlers/Module2.html')
    if prediction == "Precision and Consistency":
       return render_template('bowlers/Module3.html')
    if prediction == "Wicket-taking and Attack":
        return render_template('bowlers/Module4.html')
    return render_template('Test.html', prediction=pred[0])

#predicting batsman training 
@app.route('/predict', methods=['POST'])
def prediction():
    data0 = float(request.form['Runs'])
    data1 = float(request.form['Ball_Faced'])
    data2 = float(request.form['Average'])
    data3 = float(request.form['Strike-rate'])
    data4 = float(request.form['Highest_Score'])
    data5 = float(request.form['4s'])
    data6 = float(request.form['6s'])
    data7 = float(request.form['50s'])
    data8 = float(request.form['100s'])
    arr = np.array([[data0,data1, data2, data3, data4,data5,data6,data7,data8]])
    pred = model.predict(arr)
    prediction = pred[0]
    if prediction == "module 6":
      return render_template('batsmen/Module6.html') 
    if prediction == "module 4":
       return render_template('batsmen/Module4.html')
    if prediction == "module 1,3 and 4":
       return render_template('batsmen/Module1,3and4.html')
    if prediction == "module 1 and 2":
        return render_template('batsmen/Module1and2.html')
    if prediction == "module 3 and 4":
       return render_template('batsmen/Module3and4.html')
    if prediction == "module 1,2 and 4":
        return render_template('batsmen/Module1,2and4.html')
    if prediction == 'module 1,2 and 3':
       return render_template('batsmen/Module1,2and3.html')
    return render_template('batsmen/Module1,2and3.html', prediction=pred[0])

if __name__ == "__main__":
    app.run(debug=True)