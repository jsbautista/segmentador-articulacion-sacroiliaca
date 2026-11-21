
"""Visualización de segmentaciones y errores en TEST."""

import csv

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch

from matplotlib.colors import ListedColormap

from sij_prep.io import repo_root
from sij_medsam2.datos import (
    casos_seleccionados,
    leer_veredictos,
    stir_preprocesada,
)
from sij_medsam2.pistas import cortes_evaluacion, cortes_pista
from sij_medsam2.registro import mascara_experta

from sij_unet.evaluar import ROI, cargar_modelo, predecir
from sij_unet.particiones import TEST, verificar_particiones


COLORES_MASCARA = ListedColormap([
    "#000000",  # Fondo, no se dibuja
    "#ef6c59",  # Ilíaco 1
    "#38b87c",  # Sacro
    "#4e92e8",  # Ilíaco 3
])

COLORES_ERROR = ListedColormap([
    "#ff4545",  # Falso positivo
    "#18c6eb",  # Falso negativo
    "#ffb833",  # Confusión entre estructuras
])

COLORES_CAMBIOS = ListedColormap([
    "#22c55e",  # Error de MedSAM2 corregido por U-Net
    "#f04f9b",  # Error introducido por U-Net
    "#facc15",  # Ambos se equivocan
])


def dibujar(ax, imagen, mascara, titulo, colores, fondo):
    """Superpone una máscara sobre el corte STIR."""

    ax.imshow(imagen, cmap="gray", vmin=0, vmax=255)

    if mascara is not None:
        visible = np.ma.masked_where(mascara == fondo, mascara)

        ax.imshow(
            visible,
            cmap=colores,
            vmin=0,
            vmax=colores.N - 1,
            alpha=0.65,
            interpolation="nearest",
        )

    ax.set_title(titulo, fontsize=11)
    ax.axis("off")


def mapa_errores(prediccion, referencia):
    """Distingue falsos positivos, falsos negativos y clases confundidas."""

    errores = np.full(referencia.shape, -1, dtype=np.int8)

    falsos_positivos = (prediccion != 0) & (referencia == 0)
    falsos_negativos = (prediccion == 0) & (referencia != 0)

    confusiones = (
        (prediccion != 0)
        & (referencia != 0)
        & (prediccion != referencia)
    )

    errores[falsos_positivos] = 0
    errores[falsos_negativos] = 1
    errores[confusiones] = 2

    return errores


def mapa_cambios(medsam, unet, referencia):
    """Identifica errores corregidos e introducidos respecto a MedSAM2."""

    error_medsam = medsam != referencia
    error_unet = unet != referencia

    corregidos = error_medsam & ~error_unet
    introducidos = ~error_medsam & error_unet
    ambos_fallan = error_medsam & error_unet

    cambios = np.full(referencia.shape, -1, dtype=np.int8)

    cambios[corregidos] = 0
    cambios[introducidos] = 1
    cambios[ambos_fallan] = 2

    conteos = {
        "corregidos": int(np.count_nonzero(corregidos)),
        "introducidos": int(np.count_nonzero(introducidos)),
        "ambos_fallan": int(np.count_nonzero(ambos_fallan)),
    }

    return cambios, conteos


def crear_figura(n, corte, imagen, referencia, predicciones, carpeta):
    """Genera una figura comparativa por corte."""

    medsam = predicciones["MedSAM2"]
    unet45 = predicciones["U-Net 45"]
    unet179 = predicciones["U-Net 179"]

    cambios45, conteos45 = mapa_cambios(
        medsam, unet45, referencia
    )

    cambios179, conteos179 = mapa_cambios(
        medsam, unet179, referencia
    )

    fig, axes = plt.subplots(2, 5, figsize=(19, 8.5))

    fig.suptitle(
        f"STIR_{n} | Corte {corte} | Comparación en TEST",
        fontsize=16,
        fontweight="bold",
    )

    # Primera fila: segmentaciones.
    dibujar(
        axes[0, 0], imagen, None,
        "Imagen STIR", COLORES_MASCARA, 0
    )

    dibujar(
        axes[0, 1], imagen, referencia,
        "Máscara experta", COLORES_MASCARA, 0
    )

    dibujar(
        axes[0, 2], imagen, medsam,
        "MedSAM2", COLORES_MASCARA, 0
    )

    dibujar(
        axes[0, 3], imagen, unet45,
        "U-Net 45", COLORES_MASCARA, 0
    )

    dibujar(
        axes[0, 4], imagen, unet179,
        "U-Net 179", COLORES_MASCARA, 0
    )

    # Segunda fila: errores frente a la máscara experta.
    for columna, nombre in enumerate(
        ("MedSAM2", "U-Net 45", "U-Net 179")
    ):
        errores = mapa_errores(
            predicciones[nombre], referencia
        )

        dibujar(
            axes[1, columna],
            imagen,
            errores,
            f"Errores: {nombre}",
            COLORES_ERROR,
            -1,
        )

    # Comparación de cada U-Net frente a MedSAM2.
    dibujar(
        axes[1, 3],
        imagen,
        cambios45,
        "Cambios: MedSAM2 vs U-Net 45",
        COLORES_CAMBIOS,
        -1,
    )

    dibujar(
        axes[1, 4],
        imagen,
        cambios179,
        "Cambios: MedSAM2 vs U-Net 179",
        COLORES_CAMBIOS,
        -1,
    )

    # Leyendas.
    fig.text(
        0.5, 0.105,
        "ESTRUCTURAS:  coral = ilíaco 1"
        "    |    verde = sacro    |    azul = ilíaco 3",
        ha="center",
        fontsize=10,
    )

    fig.text(
        0.5, 0.075,
        "ERRORES:  rojo = falso positivo"
        "    |    cian = falso negativo"
        "    |    naranja = estructura confundida",
        ha="center",
        fontsize=10,
    )

    fig.text(
        0.5, 0.045,
        "CAMBIOS:  verde = error corregido"
        "    |    rosa = error introducido"
        "    |    amarillo = ambos fallan",
        ha="center",
        fontsize=10,
    )

    fig.tight_layout(rect=(0, 0.13, 1, 0.94))

    ruta = carpeta / f"STIR_{n}_corte_{corte:02d}.png"
    fig.savefig(ruta, dpi=160, bbox_inches="tight")
    plt.close(fig)

    return conteos45, conteos179


def visualizar():
    verificar_particiones()

    dispositivo = torch.device(
        "mps" if torch.backends.mps.is_available() else "cpu"
    )

    modelos = {
        "U-Net 45": cargar_modelo(45, dispositivo),
        "U-Net 179": cargar_modelo(179, dispositivo),
    }

    metadatos = {
        caso["n"]: caso
        for caso in casos_seleccionados("kaggle", leer_veredictos())
    }

    carpeta = (
        repo_root()
        / "resultados/unet/final/test/visualizaciones"
    )
    carpeta.mkdir(parents=True, exist_ok=True)

    y0, y1, x0, x1 = ROI
    filas = []

    for n in TEST:
        caso = metadatos[n]

        pistas = cortes_pista(caso["nucleo"], "k1")
        cortes = cortes_evaluacion(caso["nucleo"], pistas)

        imagenes = stir_preprocesada(n, "kaggle")
        experta = mascara_experta(n)

        ruta_medsam = (
            repo_root()
            / f"data/processed/medsam2/kaggle/STIR_{n}_k1.npz"
        )

        if not ruta_medsam.is_file():
            raise FileNotFoundError(
                f"No se encontró MedSAM2: {ruta_medsam}"
            )

        with np.load(ruta_medsam) as archivo:
            medsam = archivo["pred"]

        if (
            imagenes.shape != experta.shape
            or imagenes.shape != medsam.shape
        ):
            raise ValueError(
                f"Las dimensiones no coinciden en STIR_{n}."
            )

        print(f"\nSTIR_{n}: {len(cortes)} cortes")

        for corte in cortes:
            imagen_completa = imagenes[corte]

            imagen = imagen_completa[y0:y1, x0:x1]
            referencia = experta[corte, y0:y1, x0:x1]

            predicciones = {
                "MedSAM2": medsam[corte, y0:y1, x0:x1]
            }

            for nombre, modelo in modelos.items():
                predicciones[nombre] = predecir(
                    modelo, imagen_completa, dispositivo
                )

            conteos45, conteos179 = crear_figura(
                n,
                corte,
                imagen,
                referencia,
                predicciones,
                carpeta,
            )

            for nombre, conteos in (
                ("U-Net 45", conteos45),
                ("U-Net 179", conteos179),
            ):
                filas.append({
                    "caso": n,
                    "corte": corte,
                    "modelo": nombre,
                    **conteos,
                    "balance": (
                        conteos["corregidos"]
                        - conteos["introducidos"]
                    ),
                })

            print(f"  Corte {corte}: imagen guardada")

    # Guardar el análisis de correcciones.
    ruta_csv = carpeta / "correcciones.csv"

    with ruta_csv.open(
        "w", newline="", encoding="utf-8"
    ) as archivo:
        escritor = csv.DictWriter(
            archivo,
            fieldnames=list(filas[0]),
        )
        escritor.writeheader()
        escritor.writerows(filas)

    print("\nRESUMEN DE CORRECCIONES FRENTE A MEDSAM2")

    for nombre in modelos:
        datos = [f for f in filas if f["modelo"] == nombre]

        corregidos = sum(f["corregidos"] for f in datos)
        introducidos = sum(f["introducidos"] for f in datos)

        print(
            f"{nombre}: "
            f"{corregidos} píxeles corregidos | "
            f"{introducidos} errores introducidos | "
            f"balance {corregidos - introducidos:+d}"
        )

    print(f"\nVisualizaciones guardadas en: {carpeta}")


if __name__ == "__main__":
    visualizar()
