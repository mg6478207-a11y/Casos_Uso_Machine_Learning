from flask import Flask, render_template, request, send_file
from LRModel import generar_grafico
import regresionLogistica as RL
import clasificacion as CLF  # Nuevo script

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

#Regresion Logistica
@app.route('/ConceptosRL')
def ConceptosRL():
    return render_template('ConceptosRL.html')

# Regresión Logística - Práctico
# Se ejecuta una sola vez al iniciar la app
_data_RL = RL.load_data(RL.DATA_PATH)
X_train_RL, X_test_RL, y_train_RL, y_test_RL = RL.split_xy(_data_RL)
_model_RL, _scaler_RL = RL.train_model(X_train_RL, y_train_RL)
_accuracy_RL, _report_html_RL, _cm_RL, _cm_img_path_RL = RL.evaluate(_model_RL, _scaler_RL, X_test_RL, y_test_RL)
_informe_RL = RL.informe_breve()

# ========== NUEVO: ruta del práctico de Regresión Logística ==========
@app.route('/PracticoRL', methods=['GET', 'POST'])
def PracticoRL():
    pred_label = None
    pred_prob = None
    threshold = request.form.get('threshold', default=0.5, type=float)

    form_vals = {
        'CapitalInicial': request.form.get('CapitalInicial'),
        'Experiencia': request.form.get('Experiencia'),
        'NumSocios': request.form.get('NumSocios'),
        'AniosOperacion': request.form.get('AniosOperacion'),
    }

    if request.method == 'POST' and all(form_vals.values()):
        features = {
            'CapitalInicial': float(form_vals['CapitalInicial']),
            'Experiencia': form_vals['Experiencia'],   # 'Baja' | 'Media' | 'Alta'
            'NumSocios': float(form_vals['NumSocios']),
            'AniosOperacion': float(form_vals['AniosOperacion'])
        }
        pred_label, pred_prob = RL.predict_label(_model_RL, _scaler_RL, features, threshold)
        pred_prob = round(pred_prob, 4)

    return render_template(
        'PracticoRL.html',
        accuracy_pct=round(_accuracy_RL * 100, 2),
        report_html=_report_html_RL,
        cm_img_path=_cm_img_path_RL,
        pred_label=pred_label,
        pred_prob=pred_prob,
        threshold=threshold,
        form_vals=form_vals,
        informe=_informe_RL
    )
#-------------------------------------------------------------------------------
#Menu tipos de algoritmos de clasificacion
@app.route('/AlgConceptos')
def AlgConceptosL():
    return render_template('AlgConceptos.html')

# Entrenamiento inicial al cargar la app
_data_CLF = CLF.load_data(CLF.DATA_PATH)
X_train_CLF, X_test_CLF, y_train_CLF, y_test_CLF = CLF.split_xy(_data_CLF)
_model_CLF = CLF.train_model(X_train_CLF, y_train_CLF)
_accuracy_CLF, _report_html_CLF, _cm_CLF, _cm_img_path_CLF = CLF.evaluate(_model_CLF, X_test_CLF, y_test_CLF)
_informe_CLF = CLF.informe_breve()


@app.route('/AlgPractico', methods=['GET', 'POST'])
def AlgPractico():
    pred_label = None
    pred_prob = None
    threshold = request.form.get('threshold', default=0.5, type=float)

    form_vals = {
        'CapitalInicial': request.form.get('CapitalInicial'),
        'Experiencia': request.form.get('Experiencia'),
        'NumSocios': request.form.get('NumSocios'),
        'AniosOperacion': request.form.get('AniosOperacion'),
    }

    if request.method == 'POST' and all(form_vals.values()):
        features = {
            'CapitalInicial': float(form_vals['CapitalInicial']),
            'Experiencia': int(form_vals['Experiencia']),
            'NumSocios': float(form_vals['NumSocios']),
            'AniosOperacion': float(form_vals['AniosOperacion'])
        }
        pred_label, pred_prob = CLF.predict_label(_model_CLF, features, threshold)
        pred_prob = float(pred_prob)

    return render_template(
        'AlgPractico.html',
        accuracy_pct=round(_accuracy_CLF * 100, 2),
        report_html=_report_html_CLF,
        cm_img_path=_cm_img_path_CLF,
        pred_label=pred_label,
        pred_prob=pred_prob,
        threshold=threshold,
        form_vals=form_vals,
        informe=_informe_CLF
    )

if __name__ == '__main__':
    app.run(debug=True)
