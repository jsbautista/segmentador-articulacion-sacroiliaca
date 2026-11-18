
"""Imágenes STIR y pseudoetiquetas MedSAM2 para entrenar U-Net."""

import numpy as np
import torch
from torch.utils.data import TensorDataset

from sij_medsam2.datos import (
    casos_seleccionados,
    leer_veredictos,
    stir_preprocesada,
)
from sij_medsam2.pistas import cortes_evaluacion, cortes_pista
from sij_prep.io import repo_root


ROI = (96, 320, 40, 360)


def preparar_datos(casos: list[int], cantidad_cortes: int):
    """Prepara las imágenes y máscaras de los experimentos 45 o 179."""
    if cantidad_cortes not in (45, 179):
        raise ValueError("La cantidad de cortes debe ser 45 o 179.")

    metadatos = {
        caso["n"]: caso
        for caso in casos_seleccionados(
            "kaggle", leer_veredictos()
        )
    }

    raiz = repo_root() / "data/processed/medsam2/kaggle"
    y0, y1, x0, x1 = ROI

    imagenes = []
    etiquetas = []

    originales = np.zeros(4, dtype=np.int64)
    retenidos = np.zeros(4, dtype=np.int64)

    for n in casos:
        if n not in metadatos:
            raise ValueError(f"Caso STIR_{n} no seleccionado.")

        ruta = raiz / f"STIR_{n}_k1.npz"

        if not ruta.is_file():
            raise FileNotFoundError(ruta)

        volumen = stir_preprocesada(n, "kaggle")

        with np.load(ruta) as archivo:
            pseudo = archivo["pred"]

        if volumen.shape != pseudo.shape or volumen.ndim != 3:
            raise ValueError(f"Dimensiones incorrectas en STIR_{n}.")

        if not np.isin(pseudo, (0, 1, 2, 3)).all():
            raise ValueError(f"Etiquetas inválidas en STIR_{n}.")

        if volumen.shape[1] < y1 or volumen.shape[2] < x1:
            raise ValueError(f"ROI fuera de STIR_{n}.")

        nucleo = sorted(metadatos[n]["nucleo"])
        pistas = cortes_pista(nucleo, "k1")

        if cantidad_cortes == 45:
            # Solo cortes del núcleo sin pistas.
            indices = cortes_evaluacion(nucleo, pistas)
        else:
            # Todos los cortes del volumen sin pistas.
            indices = [
                corte
                for corte in range(volumen.shape[0])
                if corte not in pistas
            ]

        if not indices or set(indices) & set(pistas):
            raise ValueError(f"Cortes inválidos en STIR_{n}.")

        for corte in indices:
            mascara = pseudo[corte]
            mascara_roi = mascara[y0:y1, x0:x1]

            imagenes.append(
                volumen[corte, y0:y1, x0:x1].astype(
                    np.float32
                ) / 255.0
            )

            etiquetas.append(
                mascara_roi.astype(np.int64)
            )

            originales += np.bincount(
                mascara.ravel(),
                minlength=4,
            )

            retenidos += np.bincount(
                mascara_roi.ravel(),
                minlength=4,
            )

    if not imagenes:
        raise ValueError("No se encontraron imágenes.")

    cobertura = {
        clase: (
            float(retenidos[clase] / originales[clase])
            if originales[clase]
            else float("nan")
        )
        for clase in (1, 2, 3)
    }

    dataset = TensorDataset(
        torch.from_numpy(
            np.stack(imagenes)[:, None, :, :]
        ),
        torch.from_numpy(
            np.stack(etiquetas)
        ),
    )

    return dataset, cobertura
