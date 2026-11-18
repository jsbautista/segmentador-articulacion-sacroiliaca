
"""Entrenamiento U-Net con 45 y 179 cortes pseudoetiquetados."""

import argparse
import csv

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from sij_prep.io import repo_root
from .datos import ROI, preparar_datos
from .modelo import UNet
from .particiones import TRAIN, VAL, verificar_particiones


def perdida_combinada(logits, mascaras):
    """Entropía cruzada ponderada y Dice sin fondo."""
    pesos = logits.new_tensor([0.2, 3.0, 1.0, 3.0])

    ce = F.cross_entropy(
        logits,
        mascaras,
        weight=pesos,
    )

    probabilidades = torch.softmax(
        logits, dim=1
    )[:, 1:]

    objetivos = F.one_hot(
        mascaras.long(),
        num_classes=4,
    ).permute(0, 3, 1, 2).float()[:, 1:]

    dimensiones = (0, 2, 3)

    interseccion = (
        probabilidades * objetivos
    ).sum(dim=dimensiones)

    denominador = (
        probabilidades.sum(dim=dimensiones)
        + objetivos.sum(dim=dimensiones)
    )

    dice = (
        2 * interseccion + 1.0
    ) / (denominador + 1.0)

    presentes = objetivos.sum(dim=dimensiones) > 0

    if presentes.any():
        perdida_dice = 1 - dice[presentes].mean()
    else:
        perdida_dice = logits.sum() * 0.0

    return 0.5 * ce + 0.5 * perdida_dice


def ejecutar_epoca(modelo, cargador, optimizador=None):
    """Ejecuta entrenamiento o validación de una época."""
    entrenando = optimizador is not None
    modelo.train(entrenando)

    perdida_total = 0.0
    muestras = 0

    intersecciones = np.zeros(3, dtype=np.float64)
    denominadores = np.zeros(3, dtype=np.float64)

    for imagenes, mascaras in cargador:
        if entrenando:
            optimizador.zero_grad()

        with torch.set_grad_enabled(entrenando):
            logits = modelo(imagenes)
            perdida = perdida_combinada(logits, mascaras)

            if entrenando:
                perdida.backward()
                optimizador.step()

        cantidad = imagenes.size(0)
        perdida_total += perdida.item() * cantidad
        muestras += cantidad

        predicciones = logits.detach().argmax(dim=1)

        for clase in (1, 2, 3):
            pred = predicciones == clase
            real = mascaras == clase

            intersecciones[clase - 1] += (
                pred & real
            ).sum().item()

            denominadores[clase - 1] += (
                pred.sum().item() + real.sum().item()
            )

    dices = {}

    for clase in (1, 2, 3):
        i = clase - 1

        if denominadores[i] > 0:
            dices[clase] = float(
                2 * intersecciones[i] / denominadores[i]
            )
        else:
            dices[clase] = float("nan")

    valores = [
        valor
        for valor in dices.values()
        if np.isfinite(valor)
    ]

    dice_medio = (
        float(np.mean(valores))
        if valores
        else float("nan")
    )

    return perdida_total / muestras, dices, dice_medio


def revisar_cobertura(nombre, cobertura, minimo=0.95):
    """Comprueba cuánto hueso conserva el recorte."""
    print(f"\nCobertura de ROI en {nombre}:")

    for clase, fraccion in cobertura.items():
        if not np.isfinite(fraccion):
            print(f"  Clase {clase}: sin píxeles.")
            continue

        print(
            f"  Clase {clase}: "
            f"{100 * fraccion:.2f}% conservado"
        )

        if fraccion < minimo:
            raise ValueError(
                f"ROI {ROI} insuficiente para "
                f"clase {clase} en {nombre}."
            )


def entrenar(cortes=45, epocas=30):
    """Entrena una U-Net desde cero con 45 o 179 cortes."""
    if cortes not in (45, 179):
        raise ValueError("Los cortes deben ser 45 o 179.")

    if epocas < 1:
        raise ValueError("Las épocas deben ser positivas.")

    torch.manual_seed(42)
    np.random.seed(42)
    torch.set_num_threads(2)

    verificar_particiones()

    train, cobertura_train = preparar_datos(
        TRAIN, cortes
    )

    # La validación es idéntica en ambos experimentos.
    val, cobertura_val = preparar_datos(
        VAL, 45
    )

    if len(train) != cortes:
        raise ValueError(
            f"Se esperaban {cortes} cortes TRAIN, "
            f"pero se obtuvieron {len(train)}."
        )

    print(f"\nEntrenamiento con {cortes} cortes")
    print(f"Cortes TRAIN: {len(train)}")
    print(f"Cortes VAL: {len(val)}")

    revisar_cobertura("TRAIN", cobertura_train)
    revisar_cobertura("VAL", cobertura_val)

    cargador_train = DataLoader(
        train,
        batch_size=1,
        shuffle=True,
        num_workers=0,
    )

    cargador_val = DataLoader(
        val,
        batch_size=1,
        shuffle=False,
        num_workers=0,
    )

    modelo = UNet()

    optimizador = torch.optim.Adam(
        modelo.parameters(),
        lr=0.001,
    )

    salida = (
        repo_root()
        / "resultados/unet/final"
        / f"{cortes}_cortes"
    )
    salida.mkdir(parents=True, exist_ok=True)

    mejor_dice = -1.0
    mejor_epoca = None
    historial = []

    for epoca in range(1, epocas + 1):
        train_loss, _, _ = ejecutar_epoca(
            modelo,
            cargador_train,
            optimizador,
        )

        val_loss, dices, dice_medio = ejecutar_epoca(
            modelo,
            cargador_val,
        )

        print(
            f"Época {epoca}/{epocas} | "
            f"Train loss: {train_loss:.4f} | "
            f"Val loss: {val_loss:.4f} | "
            f"Val Dice: {dice_medio:.4f}",
            flush=True,
        )

        print(
            f"  Ilíaco 1: {dices[1]:.4f} | "
            f"Sacro: {dices[2]:.4f} | "
            f"Ilíaco 3: {dices[3]:.4f}",
            flush=True,
        )

        historial.append({
            "epoca": epoca,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_dice": dice_medio,
            "dice_iliaco_1": dices[1],
            "dice_sacro": dices[2],
            "dice_iliaco_3": dices[3],
        })

        if (
            np.isfinite(dice_medio)
            and dice_medio > mejor_dice
        ):
            mejor_dice = dice_medio
            mejor_epoca = epoca

            torch.save(
                {
                    "modelo": modelo.state_dict(),
                    "epoca": epoca,
                    "val_dice": dice_medio,
                    "val_dice_por_clase": dices,
                    "cortes_entrenamiento": cortes,
                    "cortes_validacion": len(val),
                    "tipo_pista": "k1",
                    "roi": ROI,
                    "train_casos": list(TRAIN),
                    "val_casos": list(VAL),
                },
                salida / "mejor_modelo.pt",
            )

            print("  Mejor modelo guardado", flush=True)

    with (
        salida / "historial.csv"
    ).open("w", newline="", encoding="utf-8") as archivo:
        escritor = csv.DictWriter(
            archivo,
            fieldnames=historial[0].keys(),
        )
        escritor.writeheader()
        escritor.writerows(historial)

    print("\nEntrenamiento finalizado.")
    print(f"Cortes utilizados: {cortes}")
    print(f"Mejor época: {mejor_epoca}")
    print(f"Mejor Dice VAL: {mejor_dice:.4f}")
    print(f"Resultados: {salida}")


def main():
    parser = argparse.ArgumentParser(
        description=__doc__
    )

    parser.add_argument(
        "--cortes",
        choices=("45", "179", "ambos"),
        default="ambos",
    )

    parser.add_argument(
        "--epocas",
        type=int,
        default=30,
    )

    argumentos = parser.parse_args()

    cantidades = (
        (45, 179)
        if argumentos.cortes == "ambos"
        else (int(argumentos.cortes),)
    )

    for cantidad in cantidades:
        entrenar(cantidad, argumentos.epocas)


if __name__ == "__main__":
    main()
