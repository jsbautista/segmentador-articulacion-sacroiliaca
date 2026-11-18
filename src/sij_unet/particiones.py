
from sij_medsam2.datos import casos_seleccionados, leer_veredictos

TRAIN = [4, 7, 17, 18, 20, 23, 24, 25, 27, 41]
VAL = [28, 31, 32]
TEST = [33, 36, 40]


def verificar_particiones():
    casos = casos_seleccionados("kaggle", leer_veredictos())
    pacientes = {c["n"]: c["paciente"] for c in casos}

    grupos = {
        "train": TRAIN,
        "val": VAL,
        "test": TEST,
    }

    for nombre, numeros in grupos.items():
        print(
            nombre,
            "| casos:", len(numeros),
            "| pacientes:", len({pacientes[n] for n in numeros}),
        )

    conjuntos = {
        nombre: {pacientes[n] for n in numeros}
        for nombre, numeros in grupos.items()
    }

    assert not (conjuntos["train"] & conjuntos["val"])
    assert not (conjuntos["train"] & conjuntos["test"])
    assert not (conjuntos["val"] & conjuntos["test"])

    assert set(TRAIN + VAL + TEST) == set(pacientes)

    print("Particiones correctas: sin pacientes compartidos.")


if __name__ == "__main__":
    verificar_particiones()
