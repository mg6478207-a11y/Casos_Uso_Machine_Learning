import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import io

def generar_grafico(volumen=None, densidad=None, peso=None):
    # Datos de ejemplo (puedes poner tu dataset real)
    vol = np.random.rand(20) * 100
    dens = np.random.rand(20) * 5
    pes = vol * dens

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    # Puntos de entrenamiento
    ax.scatter(vol, dens, pes, c="red", marker="o", label="Datos")

    # Si el usuario ingresó valores, los dibujamos también
    if volumen is not None and densidad is not None and peso is not None:
        # Punto predicho
        ax.scatter([volumen], [densidad], [peso],
                   c="blue", s=100, label="Predicción")

        # Línea vertical al plano XY
        ax.plot([volumen, volumen], [densidad, densidad], [0, peso],
                color="blue", linestyle="--", linewidth=2, label="Proyección vertical")

    # Etiquetas
    ax.set_xlabel("Volumen (cm³)")
    ax.set_ylabel("Densidad (g/cm³)")
    ax.set_zlabel("Peso (g)")
    ax.legend()

    img = io.BytesIO()
    plt.savefig(img, format="png")
    img.seek(0)
    return img
