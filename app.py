from flask import Flask, redirect, url_for, render_template, request
import pandas as pd
import numpy as np
import sqlite3
from flask import Flask, render_template, request, redirect, url_for, session, g

import os
from datetime import timedelta

from keras.layers import Dropout
from keras.layers import LSTM
from keras.layers import Dense
from keras.models import Sequential
from sklearn.preprocessing import MinMaxScaler


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


class DataStore():
    sc = None
    regressor = None


data = DataStore()


def train_model(stock):

    dataset_train = pd.read_csv(stock+'.csv')
    training_set = dataset_train.iloc[:, 1:2].values

# Feature Scaling
    sc = MinMaxScaler(feature_range=(0, 1))
    training_set_scaled = sc.fit_transform(training_set)

    X_train = []
    y_train = []
    for i in range(10, 4568):

        X_train.append(training_set_scaled[i-10:i, 0])
        y_train.append(training_set_scaled[i, 0])
        # count=count+1
    X_train, y_train = np.array(X_train), np.array(y_train)


# Reshaping
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))


# Part 2 - Building the RNN

# Initialising the RNN
    regressor = Sequential()

# Adding the first LSTM layer and some Dropout regularisation
    regressor.add(LSTM(units=50, return_sequences=True,
                  input_shape=(X_train.shape[1], 1)))
    regressor.add(Dropout(0.2))

# Adding a second LSTM layer and some Dropout regularisation
    regressor.add(LSTM(units=50, return_sequences=True))
    regressor.add(Dropout(0.2))

# Adding a third LSTM layer and some Dropout regularisation
    regressor.add(LSTM(units=50, return_sequences=True))
    regressor.add(Dropout(0.2))

# Adding a fourth LSTM layer and some Dropout regularisation
    regressor.add(LSTM(units=50))
    regressor.add(Dropout(0.2))

# Adding the output layer
    regressor.add(Dense(units=1))

# Compiling the RNN
    regressor.compile(optimizer='adam', loss='mean_squared_error')
# Fitting the RNN to the Training set

    data.sc = sc
    data.regressor = regressor

    try:
        # regressor.load_weights("stock_pred_weights.h5")
        regressor.load_weights(stock+".h5")
    except Exception as e:
        print(e)
        regressor.fit(X_train, y_train, epochs=30, batch_size=32)
        regressor.save_weights(stock+".h5")


print("Model loaded...")
###############################################################################################

# Regressor Model ended....!


def predictOpen(openPrice):
    try:
        X_test1 = [float(item) for item in openPrice.split(",")]
        open_Xtest = X_test1
        X_test1 = np.array(X_test1)

        X_test1 = X_test1.reshape(-1, 1)
        X_test1 = data.sc.transform(X_test1)

        X_test1 = np.reshape(X_test1, (1, 10, 1))
        predicted_stock_price = data.regressor.predict(X_test1)
        predicted_stock_price = data.sc.inverse_transform(
            predicted_stock_price)
        predVal = predicted_stock_price[0][0]
        return round(predVal, 2), open_Xtest
    except:
        return "Something Went Wrong! Please check the input Values. It must be 10 open prices seperated by comma.", ['0']


app = Flask(__name__)
app.secret_key = os.urandom(24)


@app.route('/')
def Demo():
    return render_template('start.html')


@app.route('/register')
def register():
    return render_template('register.html')


@app.route("/savedetails", methods=["POST", "GET"])
def saveDetails():
    msg = "msg"
    if request.method == "POST":
        try:
            new_user = request.form["username"]
            new_pwd = request.form["password"]
            email = request.form["email"]
            phone = request.form["phone"]
            with sqlite3.connect("project_db.db") as con:
                cur = con.cursor()

                # con.execute('CREATE TABLE students (username TEXT, password TEXT, Email TEXT, phone TEXT)')
                # print("Table created successfully")
                # con.close()
                cur.execute("INSERT into student_data (username,password,email,phone) values (?,?,?,?)",
                            (new_user, new_pwd, email, phone))
                con.commit()
                msg = "Student successfully Added now you can login with your credentials"
        except:
            con.rollback()
            msg = "Student already exist"
        finally:
            return render_template("pass.html", a=msg)
            con.close()


@app.route('/valid_login', methods=['POST', 'GET'])
def valid_login():

    msg1 = ""
    r = ""
    if request.method == "POST":
        session.pop = ('user', None)

        username = request.form['username']

        pwd = request.form['password']

        con = sqlite3.connect("project_db.db")
        cur = con.cursor()

        cur.execute("SELECT * FROM student_data WHERE username ='" +
                    username+"' and password ='"+pwd+"'")
        r = cur.fetchall()
        print(r)

        # return render_template('pass.html', n=r, c=username, d=pwd)
        # print(r)
        try:
            if (username == r[0][0] and pwd == r[0][1]):
                session['user'] = username

                # msg1 = "Logged in successfully"
                return redirect(url_for('home'))
        except:
            msg1 = "Wrong username or password "
            return render_template('login.html', info=msg1)

    return render_template('login.html')


@app.route('/home')
def home():
    if g.user:
        return render_template('home.html')
    return redirect(url_for('login'))


@app.route('/home', methods=['POST', 'GET'])
def home_fun():
    if g.user:
        if request.method == 'POST':
            stock = request.form['sk']
            print(stock)

            train_model(stock)
            openPrice = request.form["openPrice"]
            predVal, open_Xtest = predictOpen(openPrice)
            return render_template('home.html', predVal=predVal, openPrice=openPrice)
        return render_template('home.html', predVal=None)
    return redirect(url_for('login'))


@app.before_request
def before_request():
    g.user = None

    if 'user' in session:
        g.user = session['user']


@app.before_first_request  # runs before FIRST request (only once)
def make_session_permanent():
    session.permanent = True
    app.permanent_session_lifetime = timedelta(minutes=5)


@app.route('/login')
def login():
    return render_template('login.html')


@app.route('/logout')
def logout():
    session.pop = ('user', None)
    session.clear()
    return render_template('login.html')


if __name__ == '__main__':

    app.run(port=504, debug=True)


# 97.56,97.85,98.41,99.09,99.21,97.80,95.33,95.10,96.12,93.53 amazon

# 95.45,94.74,94.43,94.49,95.37,94.85,93.00,91.70,91.92,89.44 Google

# 359.16,349.50,357.55,356.63,355.00,347.90,342.85,337.50,331.23,319.30 Netflix

# 778.81,788.36,786.08,795.26,806.4,807.86,805,807.14,807.48,807.08

# 806.4,807.86,805,807.14,807.48,807.08,805.81,805.12,806.91,807.25

# 329.8,322.0,328.3,313.,310.5,314.4,311.9,314.8,312.1,319.3

# 294.16,291.91,292.07,287.68,284.92,284.32,287.95,290.41,291.38,291.34

# 302.44,303.18,304.87,304.87,302.81,304.11,304.63,305.32,300.28,301.36
