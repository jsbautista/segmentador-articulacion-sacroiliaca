"""Inferencia: contrato entre la interfaz y el modelo.

La interfaz solo conoce `InferenceService.predict(volume, progress) -> InferenceResult`.
Para conectar la U-Net real basta con ajustar `config.yaml` y, si el pipeline del
equipo lo requiere, completar los puntos marcados con `INTEGRACIÓN` en `UNetService`.

No importa Streamlit. TensorFlow/Keras solo se importa al cargar el modelo.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Any, Callable

import numpy as np
from scipy.ndimage import zoom

from .data import Volume, compute_stats

ProgressFn = Callable[[float, str], None]


class InferenceError(RuntimeError):
    """Fallo de inferencia con mensaje apto para el usuario."""


# --------------------------------------------------------------------------- #
# Contrato
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class ModelInfo:
    name: str
    kind: str                 # "demo" | "unet"
    framework: str
    description: str
    config_confirmed: bool    # ¿el equipo confirmó el pipeline (orientación, normalización...)?
    fingerprint: str          # cambia si cambian el modelo o su configuración

    @property
    def is_demo(self) -> bool:
        return self.kind == "demo"


@dataclass(frozen=True)
class InferenceResult:
    """Lo mínimo que el front necesita para visualizar, medir y exportar."""

    mask: np.ndarray                 # uint8, misma forma que el volumen, espacio original
    classes: dict[int, str]          # etiqueta (>0) -> nombre
    shape: tuple[int, int, int]
    affine: np.ndarray
    spacing: tuple[float, float, float]
    volume_sha256: str
    model: ModelInfo
    slice_axis: int
    elapsed_s: float
    created_utc: str
    stats: dict[str, Any]

    def to_metadata(self) -> dict[str, Any]:
        return {
            "modelo": {
                "nombre": self.model.name,
                "tipo": self.model.kind,
                "es_demostracion": self.model.is_demo,
                "framework": self.model.framework,
                "descripcion": self.model.description,
                "configuracion_confirmada": self.model.config_confirmed,
                "huella": self.model.fingerprint,
            },
            "clases": {str(k): v for k, v in self.classes.items()},
            "eje_de_corte": self.slice_axis,
            "tiempo_s": round(self.elapsed_s, 2),
            "creado_utc": self.created_utc,
            "volumen_sha256": self.volume_sha256,
        }


class InferenceService(ABC):
    """Interfaz que implementan DemoService y UNetService."""

    info: ModelInfo
    slice_axis: int = 2

    @abstractmethod
    def _predict_mask(self, volume: Volume, progress: ProgressFn) -> tuple[np.ndarray, dict[int, str]]:
        """Devuelve (máscara con la forma del volumen, {etiqueta: nombre})."""

    def predict(self, volume: Volume, progress: ProgressFn | None = None) -> InferenceResult:
        """Ejecuta el modelo y verifica el contrato antes de entregar el resultado."""
        t0 = perf_counter()
        mask, classes = self._predict_mask(volume, progress or (lambda f, m: None))
        mask = np.asarray(mask)
        if tuple(mask.shape) != volume.shape:
            raise InferenceError(
                f"La máscara tiene forma {tuple(mask.shape)} y el volumen {volume.shape}; "
                "debe estar en el espacio original."
            )
        if not np.issubdtype(mask.dtype, np.integer) and mask.dtype != bool:
            raise InferenceError(f"La máscara debe contener etiquetas enteras (recibido {mask.dtype}).")
        if mask.size and (mask.min() < 0 or mask.max() > 255):
            raise InferenceError("Las etiquetas de la máscara deben estar entre 0 y 255.")
        mask = mask.astype(np.uint8)
        undeclared = set(np.unique(mask).tolist()) - {0} - set(classes)
        if undeclared:
            raise InferenceError(f"La máscara contiene etiquetas sin nombre en `classes`: {sorted(undeclared)}.")
        stats = compute_stats(mask, classes, volume.spacing, self.slice_axis, volume.spacing_known)
        return InferenceResult(
            mask=mask, classes=dict(classes), shape=volume.shape, affine=volume.affine,
            spacing=volume.spacing, volume_sha256=volume.sha256, model=self.info,
            slice_axis=self.slice_axis, elapsed_s=perf_counter() - t0,
            created_utc=datetime.now(timezone.utc).isoformat(timespec="seconds"), stats=stats,
        )


# --------------------------------------------------------------------------- #
# Modo demostración (NO es una U-Net)
# --------------------------------------------------------------------------- #

class DemoService(InferenceService):
    """Máscara sintética por umbrales de intensidad.

    Existe solo para probar la interfaz sin modelo. No segmenta estructuras
    anatómicas y la interfaz lo señala en todo momento.
    """

    def __init__(self, slice_axis: int = 2):
        self.slice_axis = slice_axis
        self.info = ModelInfo(
            name="DEMOSTRACIÓN (umbral de intensidad)", kind="demo", framework="numpy",
            description="Máscara sintética por umbral de intensidad. No es una U-Net ni una segmentación anatómica.",
            config_confirmed=True, fingerprint="demo-v1",
        )

    def _predict_mask(self, volume, progress):
        lo, hi = volume.display_range
        mask = np.zeros(volume.shape, dtype=np.uint8)
        mask[volume.data >= lo + 0.60 * (hi - lo)] = 1
        mask[volume.data >= lo + 0.85 * (hi - lo)] = 2
        progress(1.0, "Demostración lista")
        return mask, {1: "Demo · intensidad alta", 2: "Demo · intensidad muy alta"}


# --------------------------------------------------------------------------- #
# U-Net real
# --------------------------------------------------------------------------- #

def resolve_model_path(cfg: dict, root: Path) -> Path:
    raw = os.environ.get("Sacroiliaca_ASSIST_MODEL") or cfg["model"]["path"]
    path = Path(raw)
    return path if path.is_absolute() else root / path


def unet_availability(cfg: dict, root: Path) -> tuple[bool, str]:
    """¿Se puede ejecutar la U-Net real? Devuelve (sí/no, motivo o ruta)."""
    fw = str(cfg["model"].get("framework", "torch")).lower()
    if fw not in ("keras", "torch"):
        return False, f"`model.framework` debe ser `keras` o `torch` (actual: `{fw}`)."
    path = resolve_model_path(cfg, root)
    if not path.exists():
        return False, (f"No se encontró el modelo en `{path}`. Copia el archivo entrenado a esa "
                       "ruta o ajusta `model.path` en config.yaml.")
    mods = ("torch",) if fw == "torch" else ("tensorflow", "keras")
    if all(importlib.util.find_spec(m) is None for m in mods):
        return False, f"No está instalado {' ni '.join(mods)} (pip install {mods[0]})."
    return True, str(path)

STIR_PARAMS = {
    "mask_fraction": 0.05, "spacing_xyz_mm": (0.6, 0.6, 3.0), "shrink": 4,
    "iterations": (50, 50, 50), "low_pct": 0.5, "high_pct": 99.5,
}


def _sij_stir(vol_xyz: np.ndarray) -> np.ndarray:
    """Réplica del preprocesado del equipo: máscara corporal -> N4 -> percentiles 0.5-99.5 -> uint8 -> /255."""
    try:
        import SimpleITK as sitk  # type: ignore
    except ImportError as exc:
        raise InferenceError("Falta SimpleITK para el preprocesado STIR (pip install SimpleITK).") from exc
    p = STIR_PARAMS
    d = np.ascontiguousarray(np.transpose(vol_xyz, (2, 1, 0)))              # (D, H, W)

    raw = d > p["mask_fraction"] * np.percentile(d, 99)                     # máscara corporal
    filled = np.zeros(raw.shape, dtype=np.uint8)
    for k in range(raw.shape[0]):
        filled[k] = sitk.GetArrayFromImage(sitk.BinaryFillhole(sitk.GetImageFromArray(raw[k].astype(np.uint8))))
    cc = sitk.RelabelComponent(sitk.ConnectedComponent(sitk.GetImageFromArray(filled)), sortByObjectSize=True)
    mask = sitk.GetArrayFromImage(cc) == 1
    if not mask.any():
        raise InferenceError("No se encontró cuerpo en el volumen (máscara corporal vacía).")

    def to_img(a):
        im = sitk.GetImageFromArray(a)
        im.SetSpacing(tuple(float(v) for v in p["spacing_xyz_mm"]))
        return im

    d32 = d.astype(np.float32)
    img = to_img(d32 + 1.0)
    msk = sitk.Cast(to_img(mask.astype(np.uint8)), sitk.sitkUInt8)
    f = [int(p["shrink"]), int(p["shrink"]), 1]
    n4 = sitk.N4BiasFieldCorrectionImageFilter()
    n4.SetMaximumNumberOfIterations(list(p["iterations"]))
    n4.Execute(sitk.Shrink(img, f), sitk.Shrink(msk, f))
    bias = np.exp(sitk.GetArrayFromImage(n4.GetLogBiasFieldAsImage(img)).astype(np.float32))
    corr = ((d32 + 1.0) / bias - 1.0).astype(np.float32)

    lo, hi = np.percentile(corr[mask].astype(np.float64), [p["low_pct"], p["high_pct"]])
    if hi <= lo:
        raise InferenceError(f"Rango de normalización degenerado (lo={lo}, hi={hi}).")
    u8 = np.rint((np.clip(corr.astype(np.float64), lo, hi) - lo) / (hi - lo) * 255.0).astype(np.uint8)
    return np.ascontiguousarray(np.transpose(u8.astype(np.float32) / 255.0, (2, 1, 0)))

def normalize(vol: np.ndarray, cfg: dict) -> np.ndarray:
    """Normalización por volumen según `model.normalization` (INTEGRACIÓN: confirmar con el equipo)."""
    kind = (cfg.get("type") or "none").lower()
    if kind == "sij_stir":      # .raw STIR crudo: réplica del preprocesado del equipo
        return _sij_stir(vol)
    vol = vol.astype(np.float32, copy=False)
    if kind == "scale255":      # volumen ya preprocesado en uint8 (0-255)
        return (vol / 255.0).astype(np.float32)
    if kind == "none":
        return vol
    if kind == "zscore":
        std = float(vol.std())
        return np.zeros_like(vol) if std == 0 else ((vol - vol.mean()) / std).astype(np.float32)
    if kind == "minmax":
        lo, hi = float(vol.min()), float(vol.max())
    elif kind == "percentile":
        lo, hi = (float(v) for v in np.percentile(vol, cfg.get("percentiles", [1, 99])))
    else:
        raise InferenceError(f"Normalización desconocida en config.yaml: `{kind}`.")
    if hi <= lo:
        return np.zeros_like(vol)
    return ((np.clip(vol, lo, hi) - lo) / (hi - lo)).astype(np.float32)


def _resize(img: np.ndarray, size: tuple[int, int], order: int) -> np.ndarray:
    """Remuestreo alineado por centros de píxel (grid_mode), sin desplazar la imagen."""
    if tuple(img.shape) == tuple(size):
        return img
    out = zoom(img, (size[0] / img.shape[0], size[1] / img.shape[1]),
               order=order, grid_mode=True, mode="nearest")
    if tuple(out.shape) != tuple(size):
        raise InferenceError(f"Remuestreo inesperado: {out.shape} en lugar de {tuple(size)}.")
    return out


def _to_labels(y: np.ndarray, out_cfg: dict) -> tuple[np.ndarray, int]:
    """Salida del modelo (B,H,W,K) -> etiquetas (B,H,W) y nº de clases foreground."""
    if y.ndim != 4:
        raise InferenceError(f"Salida inesperada del modelo: forma {y.shape}; se esperaba (lote, alto, ancho, canales).")
    k = y.shape[-1]
    kind = (out_cfg.get("type") or "auto").lower()
    if kind == "auto":
        kind = "sigmoid" if k == 1 else "softmax"
    if kind == "logits":  # salida cruda de una red PyTorch, sin activación final
        if k == 1:
            prob = 1.0 / (1.0 + np.exp(-y[..., 0].astype(np.float64)))
            return (prob >= float(out_cfg.get("threshold", 0.5))).astype(np.uint8), 1
        # INTEGRACIÓN: se asume canal 0 = fondo. Confirmar con el equipo.
        return np.argmax(y, axis=-1).astype(np.uint8), k - 1
    if kind == "sigmoid":
        if k != 1:
            raise InferenceError(f"`output.type: sigmoid` requiere 1 canal; el modelo devuelve {k}.")
        return (y[..., 0] >= float(out_cfg.get("threshold", 0.5))).astype(np.uint8), 1
    if kind == "softmax":
        # INTEGRACIÓN: se asume canal 0 = fondo. Confirmar con el equipo.
        return np.argmax(y, axis=-1).astype(np.uint8), k - 1
    raise InferenceError(f"`output.type` desconocido en config.yaml: `{kind}`.")


class UNetService(InferenceService):
    """Ejecuta un modelo Keras entrenado según lo declarado en `config.yaml`.

    Puntos de INTEGRACIÓN pendientes de confirmar con el equipo de U-Net:
      * 2D vs 3D (`model.mode`; solo 2D corte a corte está implementado).
      * Orientación de los cortes (`data.slice_axis`, `model.swap_hw`).
      * Normalización (`model.normalization`).
      * Recorte de la región ósea (`model.crop`): NO implementado; hace falta el código
        exacto de entrenamiento para poder revertirlo sin desalinear la máscara.
      * Salida: sigmoid/softmax y significado de cada etiqueta (`model.output`, `model.classes`).
    """

    def __init__(self, cfg: dict, root: Path, model: Any = None):
        self.cfg = cfg["model"]
        self.slice_axis = int(cfg["data"]["slice_axis"])
        self.model_path = resolve_model_path(cfg, root)
        self._model = model  # inyectable para pruebas
        self.root = Path(root)
        self._framework = str(self.cfg.get("framework", "torch")).lower()
        self._device: str | None = None
        stamp = ""
        if self.model_path.exists():
            st = self.model_path.stat()
            stamp = f"{st.st_size}-{st.st_mtime_ns}"
        digest = hashlib.sha1(
            (json.dumps(cfg, sort_keys=True, default=str) + stamp).encode()).hexdigest()[:12]
        self.info = ModelInfo(
            name=str(self.cfg.get("name", "U-Net")), kind="unet", framework="keras",
            description=f"Modelo {self._framework} `{self.model_path.name}` · modo {self.cfg.get('mode', '2d')}",
            config_confirmed=bool(self.cfg.get("confirmed", False)), fingerprint=f"unet-{digest}",
        )

    def _ensure_model(self):
        if self._model is not None:
            return self._model
        if not self.model_path.exists():
            raise InferenceError(f"No se encontró el modelo en `{self.model_path}`.")
        if self._framework == "torch":
            return self._load_torch()
        try:
            try:
                from tensorflow import keras  # type: ignore
            except ImportError:
                import keras  # type: ignore
            self._model = keras.models.load_model(self.model_path, compile=False)
        except Exception as exc:  # noqa: BLE001
            raise InferenceError(
                f"No se pudo cargar el modelo ({type(exc).__name__}: {exc}). Si usa capas o pérdidas "
                "personalizadas, hay que registrarlas en UNetService._ensure_model()."
            ) from exc
        return self._model

    def _load_torch(self):
        """Carga un modelo PyTorch.

        * Con `model.arch` (file + class): checkpoint de pesos (state_dict) sobre la clase de la red del equipo.
        * Sin `model.arch`: se intenta como TorchScript (trae la arquitectura incluida).
        """
        try:
            import torch  # type: ignore

            dev = str(self.cfg.get("device", "auto")).lower()
            self._device = ("cuda" if torch.cuda.is_available() else "cpu") if dev == "auto" else dev
            arch = self.cfg.get("arch") or {}
            if not (arch.get("file") and arch.get("class")):
                self._model = torch.jit.load(str(self.model_path), map_location=self._device).eval()
                return self._model

            import importlib.util as ilu
            import sys

            arch_file = Path(arch["file"])
            arch_file = arch_file if arch_file.is_absolute() else self.root / arch_file
            if not arch_file.exists():
                raise InferenceError(f"No se encontró el archivo de la red en `{arch_file}` (`model.arch.file`).")
            if str(arch_file.parent) not in sys.path:      # por si la red importa módulos vecinos
                sys.path.insert(0, str(arch_file.parent))
            spec = ilu.spec_from_file_location("si_assist_user_arch", arch_file)
            module = ilu.module_from_spec(spec)
            spec.loader.exec_module(module)
            if not hasattr(module, arch["class"]):
                raise InferenceError(f"`{arch_file.name}` no define la clase `{arch['class']}`.")
            net = getattr(module, arch["class"])(**(arch.get("kwargs") or {}))

            try:
                ckpt = torch.load(self.model_path, map_location=self._device,
                                  weights_only=not self.cfg.get("trust_checkpoint", False))
            except Exception as exc:  # noqa: BLE001
                raise InferenceError(
                    f"No se pudo leer el checkpoint ({type(exc).__name__}: {exc}). Si además de tensores "
                    "contiene otros objetos de Python y confías en el archivo, pon `model.trust_checkpoint: true`."
                ) from exc
            state = ckpt
            if isinstance(ckpt, dict):
                key = self.cfg.get("checkpoint_key")
                if key:
                    if key not in ckpt:
                        raise InferenceError(f"El checkpoint no tiene la clave `{key}`; tiene: {list(ckpt)[:10]}.")
                    state = ckpt[key]
                else:
                    for k in ("model_state_dict", "state_dict", "model", "net"):
                        if isinstance(ckpt.get(k), dict):
                            state = ckpt[k]
                            break
            if not isinstance(state, dict):
                raise InferenceError("El checkpoint no contiene un state_dict; define `model.checkpoint_key`.")
            state = {(k[7:] if k.startswith("module.") else k): v for k, v in state.items()}  # DataParallel
            try:
                net.load_state_dict(state)                 # estricto: falla si no coincide con la red
            except Exception as exc:  # noqa: BLE001
                top = list(ckpt)[:8] if isinstance(ckpt, dict) else "(no es un dict)"
                raise InferenceError(
                    f"Los pesos no coinciden con la red ({type(exc).__name__}: {str(exc)[:300]}). Claves del "
                    f"checkpoint: {top}. Si los pesos están bajo otra clave, define `model.checkpoint_key`; "
                    "si no, revisa `model.arch.kwargs` (deben ser los del entrenamiento)."
                ) from exc
            self._model = net.to(self._device).eval()
        except InferenceError:
            raise
        except Exception as exc:  # noqa: BLE001
            raise InferenceError(
                f"No se pudo cargar el modelo PyTorch ({type(exc).__name__}: {exc}). Revisa `model.arch` "
                "(clase y argumentos del constructor, iguales a los del entrenamiento) en config.yaml."
            ) from exc
        return self._model

    def _forward(self, model, x: np.ndarray) -> np.ndarray:
        """x: (lote, alto, ancho, 1) -> salida (lote, alto, ancho, canales), sea Keras o PyTorch."""
        if self._framework != "torch":
            return np.asarray(model.predict(x, verbose=0))
        import torch  # type: ignore

        t = torch.from_numpy(np.ascontiguousarray(x.transpose(0, 3, 1, 2))).to(self._device)  # NCHW
        with torch.no_grad():
            y = model(t)
        if isinstance(y, (tuple, list)):
            y = y[0]
        y = y.detach().float().cpu().numpy()
        if y.ndim == 3:                     # (lote, alto, ancho) -> un canal
            y = y[:, None]
        return y.transpose(0, 2, 3, 1)

    def _predict_mask(self, volume, progress):
        cfg = self.cfg
        if str(cfg.get("mode", "2d")).lower() != "2d":
            raise InferenceError("Solo está implementado el modo 2D corte a corte. El modo 3D requiere el código del equipo.")
        crop = cfg.get("crop") or {}
        roi = None
        if crop.get("enabled"):
            roi = tuple(int(v) for v in (crop.get("roi") or ()))
            if len(roi) != 4 or roi[0] >= roi[1] or roi[2] >= roi[3]:
                raise InferenceError("`model.crop.roi` debe ser [fila_ini, fila_fin, col_ini, col_fin].")
        roi_size = [roi[1] - roi[0], roi[3] - roi[2]] if roi else None

        progress(0.0, "Cargando modelo…")
        model = self._ensure_model()
        if self._framework == "torch":
            size_cfg = cfg.get("input_size") or roi_size
            if not size_cfg:
                raise InferenceError("Con PyTorch define `model.input_size: [alto, ancho]` en config.yaml.")
            shape = (None, *size_cfg, int(cfg.get("channels", 1)))
        else:
            shape = getattr(model, "input_shape", None)
        if not shape or isinstance(shape, list) or len(shape) != 4:
            raise InferenceError(f"Se espera una U-Net 2D con entrada (lote, alto, ancho, canales); detectado: {shape}.")
        size = cfg.get("input_size") or roi_size or [shape[1], shape[2]]
        if None in size:
            raise InferenceError("El modelo no fija el tamaño de entrada; defínelo en `model.input_size` de config.yaml.")
        size = (int(size[0]), int(size[1]))
        channels = int(shape[3] or cfg.get("channels", 1))
        if channels != 1:
            raise InferenceError(f"El front prepara un solo canal y el modelo espera {channels}.")

        progress(0.02, "Preprocesando volumen (puede tardar)…")
        vol = normalize(volume.data, cfg.get("normalization", {}))
        axis, swap = self.slice_axis, bool(cfg.get("swap_hw", False))
        n = vol.shape[axis]
        batch = max(1, int(cfg.get("batch_size", 8)))
        in_plane = tuple(vol.shape[a] for a in range(3) if a != axis)
        full_hw = in_plane[::-1] if swap else in_plane          # forma del corte tal como lo ve la red
        if roi and (roi[1] > full_hw[0] or roi[3] > full_hw[1]):
            raise InferenceError(f"El ROI {list(roi)} no cabe en un corte de {full_hw[0]}x{full_hw[1]} "
                                 "(revisa `model.crop.roi` y `model.swap_hw`).")
        mask = np.zeros(volume.shape, dtype=np.uint8)
        mask_view = np.moveaxis(mask, axis, 0)   # vista: escribir aquí escribe en `mask`
        n_fg = 1

        for start in range(0, n, batch):
            ids = range(start, min(n, start + batch))
            slices = []
            for i in ids:
                sl = np.take(vol, i, axis=axis)
                sl = sl.T if swap else sl
                if roi:
                    sl = sl[roi[0]:roi[1], roi[2]:roi[3]]
                slices.append(_resize(sl, size, order=1))
            x = np.stack(slices)[..., None].astype(np.float32)
            y = self._forward(model, x)
            labels, n_fg = _to_labels(y, cfg.get("output", {}))
            for j, i in enumerate(ids):
                if roi:                                          # la máscara vuelve al corte completo
                    lab = np.zeros(full_hw, dtype=np.uint8)
                    lab[roi[0]:roi[1], roi[2]:roi[3]] = _resize(labels[j], tuple(roi_size), order=0)
                else:
                    lab = _resize(labels[j], full_hw, order=0)
                mask_view[i] = lab.T if swap else lab
            progress(min(1.0, (ids[-1] + 1) / n), f"Procesando corte {ids[-1] + 1} de {n}")

        names = cfg.get("classes", {})
        classes = {k: str(names.get(k, names.get(str(k), f"Clase {k}"))) for k in range(1, n_fg + 1)}
        return mask, classes

def build_service(engine: str, cfg: dict, root: Path) -> InferenceService:
    if engine == "demo":
        return DemoService(slice_axis=int(cfg["data"]["slice_axis"]))
    if engine == "unet":
        return UNetService(cfg, root)
    raise ValueError(f"Motor desconocido: {engine}")
