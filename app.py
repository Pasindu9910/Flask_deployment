from flask import Flask, render_template, request,redirect, url_for,session,jsonify 
from flask_sqlalchemy import SQLAlchemy
import json
import random
import bcrypt
import pickle
import numpy as np

import sklearn
print("scikit-learn version:", sklearn.__version__)

with open('training_model.pkl', 'rb') as file:
    loaded_model, loaded_le = pickle.load(file)
with open('training_model1.pkl', 'rb') as file:
    loaded_model1, loaded_le1 = pickle.load(file)
model3 = pickle.load(open('baller.pkl', 'rb'))
model4 = pickle.load(open('batsman.pkl', 'rb'))
model6 = pickle.load(open('winning.pkl', 'rb'))

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
    return render_template('Loogin.html')

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

@app.route('/Chose2')
def chose2():
    return render_template('Chose2.html')

@app.route('/Bowler')
def bowler():
    return render_template('Bowler.html')

@app.route('/NewBowler')
def newbowler():
    return render_template('NewBowler/ballerT.html')

@app.route('/NewBatsman')
def newbatsman():
    return render_template('NewBats/batsmanT.html')


@app.route('/PlayerCombination')
def Playercombination():
    return render_template('PlayerCombination.html')

@app.route('/combinebatsmen')
def combinebatsmen():
    return render_template('batsmencombine.html')

@app.route('/combinebowler')
def combinebowler():
    return render_template('combinebowler.html')

@app.route('/NewWinningPrediction')
def winner():
    return render_template('WinningPred.html')

@app.route('/index')
def index():
    return render_template('Index.html')

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

        if user is None:
            alert_script = '<script>alert("Your email is not registered."); window.history.back();</script>'
            return alert_script
        if user and user.check_password(password):
            session['email'] = user.email
            return redirect(url_for('index'))
        else:
            alert_script = f'<script>alert("Incorrect password. Please try again."); window.location.href = "{url_for("man")}";</script>'
            return alert_script


@app.route('/AdminLogin', methods=['POST','GET'])
def AdminLogin():
    email = request.form['email']
    password = request.form['password']
    # Example of validating the email and password 
    if email == 'Admin@login.com' and password == 'AdminLogin':
        # If email and password are correct, redirect to the home page
        return redirect(url_for('admin'))
    else:
        alert_script = f'<script>alert("Incorrect credentials try again!!"); window.location.href = "{url_for("Adminlogin")}";</script>'
        return alert_script


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
        alert_script = "<script>alert('Something went wrong'); window.history.back();</script>"
        return alert_script

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
    data0 = float(request.form['Innings'])
    data1 = float(request.form['Overs'])
    data2 = float(request.form['Runs'])
    data3 = float(request.form['Wickets'])
    data4 = float(data2/data1) if data1 != 0 else 0
    data5 = float(data2/data3) if data1 != 0 else 0
    data6 = float(data1*6/data3) if data1 != 0 else 0
    data7 = float(request.form['4wickets'])
    data8 = float(request.form['5wickets'])
    data9 = float(request.form['4s'])
    data10 = float(request.form['6s'])
    data11 = float(request.form['Dots'])
    arr = np.array([[data0,data1, data2, data3, data4,data5,data6,data7,data8,data9,data10,data11]])
    new_predictions = loaded_model1.predict(arr)
    new_predictions_decoded = loaded_le1.inverse_transform(new_predictions)
    prediction = new_predictions_decoded[0]
    if prediction == "Technical Skills":
      return render_template('bowlers/module1.html') 
    if prediction == "Tactics and Strategy":
       return render_template('bowlers/module2.html')
    if prediction == "Precision and Consistency":
       return render_template('bowlers/module3.html')
    if prediction == "Wicket-taking and Attack":
       return render_template('bowlers/module4.html')
    

#predicting batsman training 
@app.route('/predict', methods=['POST'])
def prediction():
    data0 = float(request.form['Runs'])
    data1 = float(request.form['Ball_Faced'])
    data2 = float(request.form['Average'])
    data3 = float(data0/data1*100) if data1 != 0 else 0
    data4 = float(request.form['Highest_Score'])
    data5 = float(request.form['4s'])
    data6 = float(request.form['6s'])
    data7 = float(request.form['50s'])
    data8 = float(request.form['100s'])
    arr = np.array([[data0,data1, data2, data3, data4,data5,data6,data7,data8]])
    new_predictions = loaded_model.predict(arr)
    new_predictions_decoded = loaded_le.inverse_transform(new_predictions)
    prediction = new_predictions_decoded[0]
    if prediction == "Advanced Scoring Strategies":
      return render_template('batsmen/Advanced Scoring Strategies.html') 
    if prediction == "Power Hitting Skills":
       return render_template('batsmen/Power Hitting Skills.html')
    if prediction == "Consistent Batting Techniques":
       return render_template('batsmen/Consistent Batting Techniques.html')
    if prediction == "Experience and Control":
        return render_template('batsmen/Experience and Control.html')
    if prediction == "Fundamentals and Basics":
       return render_template('batsmen/Fundamentals and Basics.html')
    if prediction == "General Training":
       return render_template('batsmen/General Training.html')


@app.route('/predict3', methods=['POST'])
def ballerT():
    data1 = float(request.form['region'])
    data2 = float(request.form['matches'])
    data3 = float(request.form['innings'])
    data4 = float(request.form['balls'])
    data5 = float(request.form['runs'])
    data6 = float(request.form['wickets'])
    data7 = float(request.form['econ'])
    data8 = float(request.form['sr'])
    data9 = float(request.form['4s'])
    data10 = float(request.form['5s'])
    data11 = float(request.form['hw'])
    data12 = float(request.form['rg'])
    name = request.form['name']
    arr = np.array([[data1, data2, data3, data4,data5,data6,data7,data8,data9,data10,data11,data12]])
    pred = model3.predict(arr)
    formatted_pred = round(pred[0], 2)
    if 'predictions' not in session:
        session['predictions'] = []
    
    session['predictions'].append({'name': name, 'prediction': formatted_pred})
    session.modified = True
    
    return redirect(url_for('show_predictions1'))

@app.route('/clear_predictions1', methods=['POST'])
def clear_predictions1():
    session.pop('predictions', None)
    return redirect(url_for('show_predictions1'))

@app.route('/show_predictions1', methods=['GET'])
def show_predictions1():
    predictions = session.get('predictions', [])
    return render_template('NewBowler/ballerData.html', predictions=predictions)

@app.route('/predict4', methods=['POST'])
def batsmanT():
    data1 = float(request.form['region'])
    data2 = float(request.form['matches'])
    data3 = float(request.form['innings'])
    data4 = float(request.form['no'])
    data5 = float(request.form['bf'])
    data6 = float(request.form['runs'])
    data7 = float(request.form['hs'])
    data8 = float(request.form['sr'])
    data9 = float(request.form['cn'])
    data10 = float(request.form['hcn'])
    data11 = float(request.form['dk'])
    name = request.form['name']
    arr = np.array([[data1, data2, data3, data4,data5,data6,data7,data8,data9,data10,data11]])
    pred = model4.predict(arr)
    formatted_pred = round(pred[0], 2)
    if 'predictions' not in session:
        session['predictions'] = []
    
    session['predictions'].append({'name': name, 'prediction': formatted_pred})
    session.modified = True
    
    return redirect(url_for('show_predictions'))

@app.route('/clear_predictions', methods=['POST'])
def clear_predictions():
    session.pop('predictions', None)
    return redirect(url_for('show_predictions'))

@app.route('/show_predictions', methods=['GET'])
def show_predictions():
    predictions = session.get('predictions', [])
    return render_template('NewBats/batsmanData.html', predictions=predictions)


########################################################################################


@app.route('/predict6', methods=['POST'])
def prediction3():
    data0 = float(request.form['team1'])
    data1 = float(request.form['team2'])
    data2 = float(request.form['team1overs'])
    data3 = float(request.form['team1wickets'])
    data4 = float(request.form['team1runs'])

    arr = np.array([[data0, data1, data2, data3, data4]])

    pred = model6.predict(arr)

    return render_template('Winnerpredresult.html', data0=data0, data1=data1, prediction=pred[0])



@app.route('/batsmanss', methods=['POST'])
def batsmanss():
    data = []
    for i in range(1, 6):
        player = {
            'name': request.form.get(f'name{i}'),
            'innings': float(request.form.get(f'innings{i}')),
            'not_outs': float(request.form.get(f'not_outs{i}')),
            'runs': float(request.form.get(f'runs{i}')),
            'strike_rate': float(request.form.get(f'sr{i}')),
            'HighestScore': float(request.form.get(f'HighestScore{i}')),
            'hundreds': float(request.form.get(f'100s{i}')),
            'fifties': float(request.form.get(f'50s{i}')),
            'zeros': float(request.form.get(f'0s{i}')),
            'average': float(request.form.get(f'average{i}'))
        }
        data.append(player)

    # Save data to a JSON file
    with open('player_stats.json', 'w') as f:
        json.dump(data, f, indent=4)

        # Load batsmen data from JSON file
    with open('player_stats.json', 'r') as f:
        batsmen = json.load(f)

    # Parameters
    NUM_BATSMEN = 3
    POPULATION_SIZE = 10
    NUM_GENERATIONS = 50
    MUTATION_RATE = 0.1

    def fitness(chromosome, batsmen):
        # Define the fitness function based on multiple metrics
        total_runs = sum(batsmen[i]['runs'] for i in chromosome)
        total_highest_score = sum(batsmen[i]['HighestScore'] for i in chromosome)
        total_strike_rate = sum(batsmen[i]['strike_rate'] for i in chromosome) / NUM_BATSMEN
        total_100s = sum(batsmen[i]['hundreds'] for i in chromosome)
        total_50s = sum(batsmen[i]['fifties'] for i in chromosome)
        total_0s = sum(batsmen[i]['zeros'] for i in chromosome)
        total_average = sum(batsmen[i]['average'] for i in chromosome) / NUM_BATSMEN

        # Example weighting for fitness calculation
        fitness_score = (total_runs * 1.5) + (total_highest_score) + (total_strike_rate * 2) + (total_100s * 3) + (total_50s * 1.5) - (total_0s * 2) + (total_average * 2)

        return fitness_score

    def create_chromosome(batsmen):
        return random.sample(range(len(batsmen)), NUM_BATSMEN)

    def crossover(parent1, parent2):
        crossover_point = random.randint(1, NUM_BATSMEN - 1)
        child1 = parent1[:crossover_point] + [i for i in parent2 if i not in parent1[:crossover_point]]
        child2 = parent2[:crossover_point] + [i for i in parent1 if i not in parent2[:crossover_point]]
        return child1, child2

    def mutate(chromosome, batsmen):
        if random.random() < MUTATION_RATE:
            available_batsmen = [i for i in range(len(batsmen)) if i not in chromosome]
            if available_batsmen:
                idx = random.randint(0, NUM_BATSMEN - 1)
                new_batsman = random.choice(available_batsmen)
                chromosome[idx] = new_batsman

    # Initialize population
    population = [create_chromosome(batsmen) for _ in range(POPULATION_SIZE)]

    # Main GA loop
    for generation in range(NUM_GENERATIONS):
        # Evaluate fitness
        population = sorted(population, key=lambda x: fitness(x, batsmen), reverse=True)

        # Selection
        selected = population[:POPULATION_SIZE // 2]

        # Crossover
        offspring = []
        for i in range(0, len(selected), 2):
            if i + 1 < len(selected):
                child1, child2 = crossover(selected[i], selected[i + 1])
                offspring.append(child1)
                offspring.append(child2)

        # Mutation
        for individual in offspring:
            mutate(individual, batsmen)

        # Create new population
        population = selected + offspring

    # Best solution
    best_combination = max(population, key=lambda x: fitness(x, batsmen))
    best_fitness = fitness(best_combination, batsmen)

    # Convert IDs to names
    best_combination_names = [batsmen[i]['name'] for i in best_combination]

    # Redirect to summary page with results
    return redirect(url_for('algorithm_summary', best_combination_names=best_combination_names, best_fitness=best_fitness))

@app.route('/algorithm_summary')
def algorithm_summary():
    best_combination_names = request.args.getlist('best_combination_names')
    best_fitness = request.args.get('best_fitness')

    return render_template('algorithm_result.html', best_combination_names=best_combination_names, best_fitness=best_fitness)


@app.route('/bowlerss', methods=['POST'])
def bowlerss():
    data = []
    for i in range(1, 6):
        player = {
            'name': request.form.get(f'name{i}'),
            'innings': float(request.form.get(f'innings{i}')),
            'wickets': float(request.form.get(f'wickets{i}')),
            'economy': float(request.form.get(f'economy{i}')),
            'strike_rate': float(request.form.get(f'sr{i}')),
            '4s': float(request.form.get(f'4s{i}')),
            '5s': float(request.form.get(f'5s{i}')),
            'hw': float(request.form.get(f'hw{i}')),
            'average': float(request.form.get(f'average{i}'))
        }
        data.append(player)

    # Save data to a JSON file
    with open('player_stats1.json', 'w') as f:
        json.dump(data, f, indent=4)
    with open('player_stats1.json', 'r') as f:
        bowlers = json.load(f)

    # Parameters
    NUM_BOWLERS = 2
    POPULATION_SIZE = 10
    NUM_GENERATIONS = 50
    MUTATION_RATE = 0.1

    def fitness(chromosome,bowlers):
    # Define the fitness function based on multiple metrics
        total_wickets = sum(bowlers[i]['wickets'] for i in chromosome)
        total_economy = sum(bowlers[i]['economy'] for i in chromosome) / NUM_BOWLERS
        total_sr = sum(bowlers[i]['strike_rate'] for i in chromosome) / NUM_BOWLERS
        total_4w = sum(bowlers[i]['4s'] for i in chromosome)
        total_5w = sum(bowlers[i]['5s'] for i in chromosome)
        total_hw = sum(bowlers[i]['hw'] for i in chromosome)
        total_average = sum(bowlers[i]['average'] for i in chromosome) / NUM_BOWLERS

        # Example weighting for fitness calculation
        fitness_score = (total_wickets * 2) - (total_economy) - (total_sr / 10) + (total_4w * 1.5) + (total_5w * 2) + (
        total_hw) - (total_average / 10)
    
        return fitness_score
    
    def create_chromosome(bowlers):
        return random.sample(range(len(bowlers)), NUM_BOWLERS)
    
    def crossover(parent1, parent2):
        crossover_point = random.randint(1, NUM_BOWLERS - 1)
        child1 = parent1[:crossover_point] + [i for i in parent2 if i not in parent1[:crossover_point]]
        child2 = parent2[:crossover_point] + [i for i in parent1 if i not in parent2[:crossover_point]]
        return child1, child2
    
    def mutate(chromosome,bowlers):
        if random.random() < MUTATION_RATE:
            idx = random.randint(0, NUM_BOWLERS - 1)
            available_bowlers = [i for i in range(len(bowlers)) if i not in chromosome]
            if available_bowlers:  # Ensure there are bowlers available to select
                new_bowler = random.choice(available_bowlers)
                chromosome[idx] = new_bowler
    # Initialize population
    population = [create_chromosome(bowlers) for _ in range(POPULATION_SIZE)]

    # Main GA loop
    for generation in range(NUM_GENERATIONS):
        # Evaluate fitness
        population = sorted(population, key=lambda chromosome: fitness(chromosome, bowlers), reverse=True)

        # Selection
        selected = population[:POPULATION_SIZE // 2]

        # Crossover
        offspring = []
        for i in range(0, len(selected), 2):
            if i + 1 < len(selected):
                child1, child2 = crossover(selected[i], selected[i + 1])
                offspring.append(child1)
                offspring.append(child2)
        # Mutation
        for individual in offspring:
            mutate(individual,bowlers)

        # Create new population
        population = selected + offspring
    # Best solution
    best_combination = max(population, key=lambda chromosome: fitness(chromosome, bowlers))
    best_fitness = fitness(best_combination,bowlers)

    best_combination_names = [bowlers[i]['name'] for i in best_combination]

    # Redirect to summary page with results
    return redirect(url_for('algorithm_summary1', best_combination_names=best_combination_names, best_fitness=best_fitness))

@app.route('/algorithm_summary1')
def algorithm_summary1():
    best_combination_names = request.args.getlist('best_combination_names')
    best_fitness = request.args.get('best_fitness')

    return render_template('algorithm_result1.html', best_combination_names=best_combination_names, best_fitness=best_fitness)

if __name__ == "__main__":
    app.run(debug=True)