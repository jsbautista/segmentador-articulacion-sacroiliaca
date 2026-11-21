
"""Evaluación en TEST: MedSAM2 y U-Net de 45 y 179 cortes."""

import csv
from collections import defaultdict

import numpy as np
import torch

from sij_prep.io import repo_root
from sij_medsam2.datos import casos_seleccionados, leer_veredictos, stir_preprocesada
from sij_medsam2.pistas import cortes_evaluacion, cortes_pista
from sij_medsam2.registro import mascara_experta
from sij_unet.modelo import UNet
from sij_unet.particiones import TEST, verificar_particiones


ROI = (96, 320, 40, 360)
CLASES = {1: "iliaco_1", 2: "sacro", 3: "iliaco_3"}
MODELOS = ("MedSAM2", "U-Net 45", "U-Net 179")


def cargar_modelo(cortes, dispositivo):
    ruta = repo_root() / f"resultados/unet/final/{cortes}_cortes/mejor_modelo.pt"

    if not ruta.is_file():
        raise FileNotFoundError(f"No se encontró el checkpoint: {ruta}")

    guardado = torch.load(ruta, map_location="cpu", weights_only=True)

    if tuple(guardado.get("roi", ROI)) != ROI:
        raise ValueError(f"La ROI del modelo de {cortes} cortes no coincide.")

    if guardado.get("tipo_pista", "k1") != "k1":
        raise ValueError("Este evaluador requiere modelos entrenados con k1.")

    modelo = UNet().to(dispositivo)
    modelo.load_state_dict(guardado["modelo"])
    modelo.eval()

    print(
        f"U-Net {cortes}: época {guardado['epoca']} "
        f"| Dice VAL {guardado['val_dice']:.4f}"
    )

    return modelo


def predecir(modelo, imagen, dispositivo):
    y0, y1, x0, x1 = ROI

    recorte = imagen[y0:y1, x0:x1].astype(np.float32) / 255.0
    entrada = torch.from_numpy(recorte[None, None]).to(dispositivo)

    with torch.inference_mode():
        salida = modelo(entrada).argmax(dim=1)

    return salida[0].cpu().numpy().astype(np.uint8)


def conteos(prediccion, referencia, clase):
    pred = prediccion == clase
    ref = referencia == clase

    return (
        int(np.count_nonzero(pred & ref)),
        int(np.count_nonzero(pred & ~ref)),
        int(np.count_nonzero(~pred & ref)),
    )


def puntajes(tp, fp, fn):
    total = tp + fp + fn

    if total == 0:
        return float("nan"), float("nan")

    dice = 2 * tp / (2 * tp + fp + fn)
    iou = tp / total

    return dice, iou


def promedio(valores):
    validos = [v for v in valores if np.isfinite(v)]
    return float(np.mean(validos)) if validos else float("nan")


def guardar_csv(ruta, filas):
    with ruta.open("w", newline="", encoding="utf-8") as archivo:
        escritor = csv.DictWriter(archivo, fieldnames=list(filas[0]))
        escritor.writeheader()
        escritor.writerows(filas)


def evaluar():
    verificar_particiones()

    dispositivo = torch.device(
        "mps" if torch.backends.mps.is_available() else "cpu"
    )

    print(f"Dispositivo: {dispositivo}")

    redes = {
        "U-Net 45": cargar_modelo(45, dispositivo),
        "U-Net 179": cargar_modelo(179, dispositivo),
    }

    metadatos = {
        caso["n"]: caso
        for caso in casos_seleccionados("kaggle", leer_veredictos())
    }

    y0, y1, x0, x1 = ROI

    filas_corte = []
    acumulados = defaultdict(lambda: [0, 0, 0])
    cobertura = defaultdict(lambda: [0, 0])

    for n in TEST:
        caso = metadatos[n]
        paciente = caso["paciente"]

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
                f"Falta predicción MedSAM2: {ruta_medsam}. "
                "Necesitamos la inferencia k1 guardada, no reentrenar MedSAM2."
            )

        with np.load(ruta_medsam) as archivo:
            medsam = archivo["pred"]

        if imagenes.shape != experta.shape or imagenes.shape != medsam.shape:
            raise ValueError(f"Dimensiones incompatibles en STIR_{n}.")

        if not cortes:
            raise ValueError(f"STIR_{n} no tiene cortes evaluables.")

        print(f"STIR_{n} | paciente {paciente} | cortes: {cortes}")

        for corte in cortes:
            imagen = imagenes[corte]
            ref_completa = experta[corte]
            referencia = ref_completa[y0:y1, x0:x1]

            predicciones = {
                "MedSAM2": medsam[corte, y0:y1, x0:x1]
            }

            for nombre, red in redes.items():
                predicciones[nombre] = predecir(
                    red, imagen, dispositivo
                )

            for clase, estructura in CLASES.items():
                cobertura[clase][0] += int(
                    np.count_nonzero(referencia == clase)
                )

                cobertura[clase][1] += int(
                    np.count_nonzero(ref_completa == clase)
                )

                for nombre in MODELOS:
                    tp, fp, fn = conteos(
                        predicciones[nombre],
                        referencia,
                        clase,
                    )

                    dice, iou = puntajes(tp, fp, fn)

                    filas_corte.append(dict(
                        caso=n,
                        paciente=paciente,
                        corte=corte,
                        modelo=nombre,
                        estructura=estructura,
                        dice=dice,
                        iou=iou,
                        tp=tp,
                        fp=fp,
                        fn=fn,
                    ))

                    datos = acumulados[n, paciente, nombre, clase]
                    datos[0] += tp
                    datos[1] += fp
                    datos[2] += fn

    # Métricas agregadas por caso y estructura.
    filas_caso = []

    for (n, paciente, nombre, clase), (tp, fp, fn) in sorted(
        acumulados.items()
    ):
        dice, iou = puntajes(tp, fp, fn)

        filas_caso.append(dict(
            caso=n,
            paciente=paciente,
            modelo=nombre,
            estructura=CLASES[clase],
            dice=dice,
            iou=iou,
            tp=tp,
            fp=fp,
            fn=fn,
        ))

    # Resumen: media entre los tres casos de TEST.
    filas_resumen = []

    print("\nRESULTADOS TEST (promedio de los 3 casos)")

    for nombre in MODELOS:
        dices = []
        ious = []

        for estructura in CLASES.values():
            filas = [
                fila for fila in filas_caso
                if fila["modelo"] == nombre
                and fila["estructura"] == estructura
            ]

            dice = promedio([fila["dice"] for fila in filas])
            iou = promedio([fila["iou"] for fila in filas])

            filas_resumen.append(dict(
                modelo=nombre,
                estructura=estructura,
                dice=dice,
                iou=iou,
                fp=sum(fila["fp"] for fila in filas),
                fn=sum(fila["fn"] for fila in filas),
                casos=len(filas),
            ))

            dices.append(dice)
            ious.append(iou)

        print(f"{nombre:10s} | Dice promedio: {promedio(dices):.4f}")

        for fila in filas_resumen[-3:]:
            print(
                f"  {fila['estructura']:9s}: "
                f"Dice {fila['dice']:.4f} | "
                f"IoU {fila['iou']:.4f}"
            )

        filas_resumen.append(dict(
            modelo=nombre,
            estructura="promedio",
            dice=promedio(dices),
            iou=promedio(ious),
            fp=sum(fila["fp"] for fila in filas_resumen[-3:]),
            fn=sum(fila["fn"] for fila in filas_resumen[-3:]),
            casos=len(TEST),
        ))

    # Comprobar cuánto hueso experto queda dentro de la ROI.
    print("\nCobertura de la ROI respecto a la máscara experta en TEST:")

    for clase, (dentro, total) in cobertura.items():
        valor = 100 * dentro / total if total else float("nan")

        print(f"  {CLASES[clase]}: {valor:.2f}%")

        if np.isfinite(valor) and valor < 95:
            print("  AVISO: la ROI excluye más del 5% de esa estructura.")

    # Guardar resultados.
    carpeta = repo_root() / "resultados/unet/final/test"
    carpeta.mkdir(parents=True, exist_ok=True)

    guardar_csv(carpeta / "por_corte.csv", filas_corte)
    guardar_csv(carpeta / "por_caso.csv", filas_caso)
    guardar_csv(carpeta / "resumen.csv", filas_resumen)

    print(f"\nResultados guardados en: {carpeta}")


if __name__ == "__main__":
    evaluar()
