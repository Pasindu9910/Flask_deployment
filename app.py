from flask import Flask, render_template, request,redirect, url_for
import pickle
import numpy as np

model = pickle.load(open('knn_model.pkl', 'rb'))
model2 = pickle.load(open('knn_model2.pkl', 'rb'))

app = Flask(__name__)


@app.route('/')
def man():
    return render_template('Login.html')

@app.route('/Home')
def home():
    return render_template('Home.html')

@app.route('/Chose')
def chose():
    return render_template('Chose.html')

@app.route('/Bowler')
def bowler():
    return render_template('Bowler.html')

@app.route('/index')
def index():
    return render_template('index.html')

@app.route('/Contactus')
def contactus():
    return render_template('Contactus.html')

@app.route('/About')
def about():
    return render_template('About.html')

@app.route('/Login', methods=['POST','GET'])
def Login():
    email = request.form['email']
    password = request.form['password']

    # Example of validating the email and password 
    if email == 'example@example.com' and password == 'password':
        # If email and password are correct, redirect to the home page
        return redirect(url_for('index'))
    if email == 'example@example.com' and password == 'password':
        # If email and password are correct, redirect to the home page
        return redirect(url_for('index'))
    if email == 'example@example.com' and password == 'password':
        # If email and password are correct, redirect to the home page
        return redirect(url_for('index'))
    else:
        # If email and password are incorrect, redirect back to the login page
        return redirect(url_for('man'))
    
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