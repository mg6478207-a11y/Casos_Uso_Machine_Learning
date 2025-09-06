from flask import Flask
from flask import render_template

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/Salud')
def Salud():
    return render_template('Salud.html')

@app.route('/Ciberseguridad')
def Ciberseguridad():
    return render_template('Ciberseguridad.html')

@app.route('/Retail')
def Retail():
    return render_template('Retail.html')

@app.route('/Transporte')
def Transporte():
    return render_template('Transporte.html')

if __name__ == '__main__':
    app.run(debug=True)
