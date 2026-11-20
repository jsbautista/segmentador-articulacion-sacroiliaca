"""Datos del estudio: configuración, carga, validación, métricas y exportación.

Este módulo no importa Streamlit ni TensorFlow, por lo que se puede probar de
forma aislada. NiBabel solo se importa cuando hay que leer o escribir NIfTI.
"""
from __future__ import annotations

import csv
import gzip
import hashlib
import io
import json
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import yaml

# --------------------------------------------------------------------------- #
# Configuración
# --------------------------------------------------------------------------- #

DEFAULT_CONFIG: dict[str, Any] = {
    "data": {"slice_axis": 2, "max_voxels": 150_000_000},
    "display": {"transpose": True, "flip_vertical": True},
    "raw": {
        "width": 400,
        "height": 400,
        "dtype": "int16",
        "byte_order": "little"
    },
    "model": {
        "path": "models/unet_si.keras",
        "name": "U-Net SI-Assist",
        "confirmed": False,
        "mode": "2d",
        "input_size": None,
        "channels": 1,
        "batch_size": 8,
        "swap_hw": False,
        "normalization": {"type": "percentile", "percentiles": [1, 99]},
        "crop": {"enabled": False},
        "output": {"type": "auto", "threshold": 0.5},
        "classes": {1: "Región segmentada"},
    },
}


class ConfigError(ValueError):
    """config.yaml no es válido."""


class StudyError(ValueError):
    """Problema con el archivo del estudio; el mensaje está pensado para el usuario."""


def _merge(base: dict, override: dict | None) -> dict:
    out = deepcopy(base)
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _merge(out[key], value)
        else:
            out[key] = value
    return out


def load_config(path: str | Path) -> dict[str, Any]:
    """Lee config.yaml y lo combina con los valores por defecto."""
    path = Path(path)
    user: dict = {}
    if path.exists():
        try:
            user = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        except yaml.YAMLError as exc:
            raise ConfigError(f"config.yaml no es un YAML válido: {exc}") from exc
    cfg = _merge(DEFAULT_CONFIG, user)
    # Las clases se reemplazan, no se mezclan con las de por defecto.
    if "classes" in (user.get("model") or {}):
        cfg["model"]["classes"] = user["model"]["classes"] or {}
    try:
        cfg["model"]["classes"] = {int(k): str(v) for k, v in cfg["model"]["classes"].items()}
    except (TypeError, ValueError) as exc:
        raise ConfigError("`model.classes` debe mapear números enteros a nombres.") from exc
    if cfg["data"]["slice_axis"] not in (0, 1, 2):
        raise ConfigError("`data.slice_axis` debe ser 0, 1 o 2.")
    return cfg


# --------------------------------------------------------------------------- #
# Volumen y lectura
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class Volume:
    """Volumen MRI ya validado, en el espacio original del archivo."""

    data: np.ndarray                      # float32, forma (X, Y, Z)
    affine: np.ndarray                    # 4x4
    header: Any                           # cabecera NIfTI original o None (.npy)
    name: str
    sha256: str
    spacing: tuple[float, float, float]
    spacing_known: bool
    axcodes: tuple[str, ...] | None
    display_range: tuple[float, float]    # p1-p99, para ventana de visualización
    n_nonfinite: int
    source: str = "nifti"  # "nifti" | "npy" | "raw"

    @property
    def shape(self) -> tuple[int, int, int]:
        return tuple(int(s) for s in self.data.shape)  # type: ignore[return-value]


def study_stem(name: str) -> str:
    low = name.lower()
    for ext in (".nii.gz", ".nii", ".npy", ".raw"):
        if low.endswith(ext):
            return name[: -len(ext)]
    return Path(name).stem


def _read_nifti(raw: bytes, max_bytes: int):
    import nibabel as nib

    try:
        if raw[:2] == b"\x1f\x8b":  # gzip por firma, sin fiarse de la extensión
            with gzip.GzipFile(fileobj=io.BytesIO(raw)) as fh:
                raw = fh.read(max_bytes + 1)
            if len(raw) > max_bytes:
                raise StudyError("El archivo descomprimido es demasiado grande para cargarlo.")
        img = nib.Nifti1Image.from_bytes(raw)
        arr = img.get_fdata(dtype=np.float32)
    except StudyError:
        raise
    except Exception as exc:  # noqa: BLE001 - cualquier fallo de lectura es del archivo
        raise StudyError(
            "No se pudo leer el archivo como NIfTI-1. Verifica que no esté dañado "
            f"ni sea otro formato ({type(exc).__name__}: {exc})."
        ) from exc
    zooms = tuple(float(z) for z in img.header.get_zooms()[:3])
    axcodes = tuple(nib.aff2axcodes(img.affine))
    return arr, np.asarray(img.affine, dtype=np.float64), img.header, zooms, axcodes


def _read_npy(raw: bytes):
    try:
        arr = np.load(io.BytesIO(raw), allow_pickle=False)
    except Exception as exc:  # noqa: BLE001
        raise StudyError(
            "El archivo no es un .npy válido. Si contiene objetos de Python (pickle) no se carga por seguridad; "
            "guarda el volumen como un arreglo numérico con np.save()."
        ) from exc
    return arr, np.eye(4), None, (1.0, 1.0, 1.0), None

RAW_DTYPES = (
    "uint8", "int8",
    "uint16", "int16",
    "uint32", "int32",
    "float32", "float64"
)


def raw_slice_count(n_bytes: int, params: dict) -> tuple[int, int]:
    """Devuelve cortes completos y bytes sobrantes."""
    per_slice = (
        int(params["width"])
        * int(params["height"])
        * np.dtype(params["dtype"]).itemsize
    )

    return divmod(n_bytes, per_slice)


def _raw_array(raw: bytes, params: dict) -> np.ndarray:
    """
    Lee un archivo RAW sin cabecera.

    Devuelve un arreglo (ancho, alto, cortes).
    """
    try:
        w = int(params["width"])
        h = int(params["height"])

        dtype = np.dtype(params["dtype"]).newbyteorder(
            ">" if params.get("byte_order") == "big" else "<"
        )

    except (KeyError, TypeError, ValueError) as exc:
        raise StudyError(
            "Parámetros .raw incompletos: se requieren ancho, alto y tipo de dato."
        ) from exc

    if w < 1 or h < 1 or params["dtype"] not in RAW_DTYPES:
        raise StudyError("Parámetros .raw no válidos.")

    n, extra = raw_slice_count(len(raw), params)

    if extra or n == 0:
        raise StudyError(
            f"El tamaño del archivo ({len(raw):,} bytes) "
            f"no corresponde a cortes de {w}×{h} "
            f"con tipo {params['dtype']}: sobran {extra:,} bytes. "
            "Revisa ancho, alto y tipo de dato."
        )

    return (
        np.frombuffer(raw, dtype=dtype)
        .reshape(n, h, w)
        .transpose(2, 1, 0)
    )


def _read_raw(raw: bytes, params: dict):
    arr = _raw_array(raw, params).astype(np.float32)

    sp = params.get("spacing")

    zooms = (
        tuple(float(v) for v in sp)
        if sp else (1.0, 1.0, 1.0)
    )

    affine = np.diag([*zooms, 1.0])

    return arr, affine, None, zooms, None


def mask_to_raw(mask: np.ndarray) -> bytes:
    """
    Convierte la máscara a RAW uint8,
    conservando la disposición de los cortes.
    """
    return np.ascontiguousarray(
        np.asarray(mask, dtype=np.uint8).transpose(2, 1, 0)
    ).tobytes()

def _validate(arr: np.ndarray, max_voxels: int) -> tuple[np.ndarray, int]:
    original_shape = arr.shape
    if not (np.issubdtype(arr.dtype, np.number) or arr.dtype == bool):
        raise StudyError(f"El archivo contiene datos no numéricos ({arr.dtype}).")
    if np.iscomplexobj(arr):
        raise StudyError("Los volúmenes con valores complejos no están soportados.")
    arr = np.squeeze(arr)
    if arr.ndim != 3:
        raise StudyError(
            f"Se requiere un volumen tridimensional; el archivo tiene forma {original_shape}."
        )
    if min(arr.shape) < 8:
        raise StudyError(f"El volumen es demasiado pequeño ({arr.shape}).")
    if arr.size > max_voxels:
        raise StudyError(
            f"El volumen tiene {arr.size:,} vóxeles; el máximo configurado es {max_voxels:,}."
        )
    arr = np.asarray(arr, dtype=np.float32)
    bad = ~np.isfinite(arr)
    n_bad = int(bad.sum())
    if n_bad:
        arr = np.where(bad, 0.0, arr).astype(np.float32)
    if float(arr.max()) == float(arr.min()):
        raise StudyError("El volumen tiene intensidad constante; no hay imagen que mostrar.")
    return arr, n_bad


def load_volume(raw: bytes, name: str, max_voxels: int = 150_000_000, raw_params: dict | None = None) -> Volume:
    """Lee y valida un volumen NIfTI (.nii/.nii.gz) o NumPy (.npy)."""
    low = name.lower()
    source = "nifti"
    if low.endswith((".nii", ".nii.gz")):
        arr, affine, header, zooms, axcodes = _read_nifti(raw, max_bytes=8 * max_voxels)
        spacing_known = True
    elif low.endswith(".npy"):
        arr, affine, header, zooms, axcodes = _read_npy(raw)
        spacing_known, source = False, "npy"

    elif low.endswith(".raw"):
        if not raw_params:
            raise StudyError(
                "Los archivos .raw no tienen cabecera: "
                "indica ancho, alto y tipo de dato."
            )
        arr, affine, header, zooms, axcodes = _read_raw(
            raw,
            raw_params
        )

        spacing_known = bool(raw_params.get("spacing"))
        source = "raw"
    else:
        raise StudyError("Formato no soportado. Usa .nii, .nii.gz, .npy o .raw.")

    arr, n_bad = _validate(arr, max_voxels)
    if len(zooms) != 3 or not all(np.isfinite(zooms)) or min(zooms) <= 0:
        zooms, spacing_known = (1.0, 1.0, 1.0), False

    lo, hi = np.percentile(arr[::2, ::2, ::2], [1, 99])
    if hi <= lo:
        lo, hi = float(arr.min()), float(arr.max())
    return Volume(
        data=arr,
        affine=affine,
        header=header,
        name=name,
        sha256=hashlib.sha256(raw).hexdigest(),
        spacing=zooms,  # type: ignore[arg-type]
        spacing_known=spacing_known,
        axcodes=axcodes,
        display_range=(float(lo), float(hi)),
        n_nonfinite=n_bad,
        source=source,
    )


def load_reference(raw: bytes, name: str, volume: Volume, max_voxels: int = 150_000_000, raw_params: dict | None = None):
    """Lee una máscara de referencia y comprueba que sea comparable con el volumen.

    Devuelve (máscara uint8, lista de avisos). No remuestrea: si la forma no
    coincide, lanza StudyError.
    """
    low = name.lower()
    if low.endswith((".nii", ".nii.gz")):
        arr, affine, _, _, _ = _read_nifti(raw, max_bytes=8 * max_voxels)
    elif low.endswith(".npy"):
        arr, affine, _, _, _ = _read_npy(raw)
    elif low.endswith(".raw"):
        if not raw_params:
            raise StudyError(
                "Los archivos .raw no tienen cabecera: "
                "indica ancho, alto y tipo de dato."
            )

        arr, affine, _, _, _ = _read_raw(raw, raw_params)
    else:
        raise StudyError(
            "Formato no soportado. Usa .nii, .nii.gz, .npy o .raw."
        )
    arr = np.squeeze(arr)
    if arr.ndim != 3:
        raise StudyError(f"La máscara de referencia debe ser 3D; forma recibida: {arr.shape}.")
    if tuple(arr.shape) != volume.shape:
        raise StudyError(
            f"La máscara de referencia tiene forma {tuple(arr.shape)} y el volumen {volume.shape}. "
            "Deben coincidir; la aplicación no remuestrea referencias."
        )
    if not np.allclose(arr, np.rint(arr)) or arr.min() < 0 or arr.max() > 255:
        raise StudyError("El archivo no parece una máscara de etiquetas enteras entre 0 y 255.")
    warnings: list[str] = []
    if name.lower().endswith((".nii", ".nii.gz")) and not np.allclose(affine, volume.affine, atol=1e-3):
        warnings.append(
            "La matriz espacial (affine) de la referencia difiere de la del volumen; "
            "se comparan por posición de vóxel."
        )
    return np.rint(arr).astype(np.uint8), warnings


# --------------------------------------------------------------------------- #
# Métricas
# --------------------------------------------------------------------------- #

def compute_stats(mask: np.ndarray, classes: dict[int, str], spacing: tuple[float, float, float],
                  slice_axis: int, spacing_known: bool = True) -> dict[str, Any]:
    """Conteos reales sobre la máscara. Todo son tipos nativos de Python (JSON)."""
    m = np.moveaxis(mask, slice_axis, 0)
    n = int(m.shape[0])
    in_plane = [a for a in range(3) if a != slice_axis]
    pixel_area = float(spacing[in_plane[0]] * spacing[in_plane[1]]) if spacing_known else None
    voxel_vol = float(np.prod(spacing)) if spacing_known else None

    per_slice = {int(lab): (m == lab).sum(axis=(1, 2)) for lab in sorted(classes)}
    total_seg = int((mask > 0).sum())
    per_class = []
    for lab, counts in per_slice.items():
        vox = int(counts.sum())
        per_class.append({
            "label": lab,
            "name": classes[lab],
            "voxels": vox,
            "percent_of_segmented": (100.0 * vox / total_seg) if total_seg else 0.0,
            "volume_mm3": vox * voxel_vol if voxel_vol else None,
            "slices_with_class": int((counts > 0).sum()),
        })
    return {
        "n_slices": n,
        "slice_axis": int(slice_axis),
        "slices_with_segmentation": int((m > 0).any(axis=(1, 2)).sum()),
        "total_voxels_segmented": total_seg,
        "total_voxels": int(mask.size),
        "pixel_area_mm2": pixel_area,
        "voxel_volume_mm3": voxel_vol,
        "spacing_known": bool(spacing_known),
        "per_class": per_class,
        "per_slice": {lab: counts.astype(int).tolist() for lab, counts in per_slice.items()},
    }


def dice_binary(a: np.ndarray, b: np.ndarray) -> float | None:
    """Dice entre dos máscaras booleanas. None si ambas están vacías (indefinido)."""
    a, b = np.asarray(a, dtype=bool), np.asarray(b, dtype=bool)
    den = int(a.sum()) + int(b.sum())
    return None if den == 0 else float(2.0 * np.logical_and(a, b).sum() / den)


def dice_report(pred: np.ndarray, ref: np.ndarray, classes: dict[int, str]) -> dict[str, Any]:
    """Dice global (binario, todo lo >0) y por clase presente en la referencia."""
    per_class = {}
    for lab in sorted(set(classes) | (set(np.unique(ref).tolist()) - {0})):
        per_class[int(lab)] = dice_binary(pred == lab, ref == lab)
    return {"global_binary": dice_binary(pred > 0, ref > 0), "per_class": per_class}


# --------------------------------------------------------------------------- #
# Exportación
# --------------------------------------------------------------------------- #

def mask_to_nifti_gz(mask: np.ndarray, volume: Volume, description: str) -> bytes:
    """Máscara uint8 como .nii.gz con el affine y la cabecera del volumen original.

    `Nifti1Image.to_bytes()` devuelve bytes sin comprimir; se comprimen aquí para que
    el contenido coincida con la extensión .nii.gz.
    """
    import nibabel as nib

    if tuple(mask.shape) != volume.shape:
        raise ValueError("La máscara no tiene la misma forma que el volumen.")
    if volume.header is not None:
        header = volume.header.copy()

    else:
        header = nib.Nifti1Header()

        if volume.spacing_known:
            header.set_zooms(volume.spacing)
    header.set_data_dtype(np.uint8)
    img = nib.Nifti1Image(np.asarray(mask, dtype=np.uint8), volume.affine, header)
    img.header.set_slope_inter(1, 0)
    img.header["descrip"] = description[:79].encode("ascii", "replace")
    return gzip.compress(img.to_bytes(), compresslevel=6)


def per_slice_csv(stats: dict[str, Any], classes: dict[int, str]) -> bytes:
    """CSV por corte (cortes numerados desde 1), con BOM para que Excel lo abra bien."""
    labels = sorted(int(k) for k in stats["per_slice"])
    header = ["corte"] + [f"pixeles_{classes[l]}" for l in labels]
    if stats["pixel_area_mm2"] is not None:
        header += [f"area_mm2_{classes[l]}" for l in labels]
    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(header)
    for i in range(stats["n_slices"]):
        counts = [stats["per_slice"][l][i] for l in labels]
        row = [i + 1] + counts
        if stats["pixel_area_mm2"] is not None:
            row += [round(c * stats["pixel_area_mm2"], 3) for c in counts]
        writer.writerow(row)
    return ("\ufeff" + buf.getvalue()).encode("utf-8")


def summary_json(meta: dict[str, Any], stats: dict[str, Any], volume: Volume,
                 dice: dict[str, Any] | None = None) -> bytes:
    """Resumen del estudio y de la segmentación en JSON (sin las series por corte)."""
    payload = {
        "generado_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "aviso": ("Herramienta académica. No es un dispositivo médico ni está validada "
                  "clínicamente."),
        "estudio": {
            "archivo": volume.name,
            "formato_origen": volume.source,
            "sha256": volume.sha256,
            "dimensiones": list(volume.shape),
            "espaciado_mm": list(volume.spacing) if volume.spacing_known else None,
            "orientacion": "".join(volume.axcodes) if volume.axcodes else None,
        },
        "segmentacion": meta,
        "resultados": {k: v for k, v in stats.items() if k != "per_slice"},
        "metricas_vs_referencia": dice,
    }
    return json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8")
