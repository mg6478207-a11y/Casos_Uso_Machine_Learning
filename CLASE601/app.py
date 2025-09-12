from flask import Flask, render_template, request, send_file
from LRModel import generar_grafico

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

# Casos de uso
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

# Regresión Lineal - Conceptos
@app.route('/LRConceptos')
def LRConceptos():
    return render_template('LRConceptos.html')



@app.route("/LRPractico", methods=["GET", "POST"])
def lr_practico():
    peso_estimado = None
    volumen = None
    densidad = None

    if request.method == "POST":
        volumen = float(request.form["volumen"])
        densidad = float(request.form["densidad"])
        peso_estimado = volumen * densidad  # o tu modelo real

    return render_template("LRPractico.html",
                           peso_estimado=peso_estimado,
                           volumen=volumen,
                           densidad=densidad)

@app.route("/grafico")
def grafico():
    # Recuperar valores pasados desde el formulario
    from flask import request
    volumen = request.args.get("volumen", type=float)
    densidad = request.args.get("densidad", type=float)
    peso = request.args.get("peso", type=float)

    img = generar_grafico(volumen, densidad, peso)
    return send_file(img, mimetype="image/png")

if __name__ == '__main__':
    app.run(debug=True)
