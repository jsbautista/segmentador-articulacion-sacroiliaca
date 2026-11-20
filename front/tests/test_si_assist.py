"""Pruebas de SI-Assist.

Ejecutar:  python -m pytest tests    o bien    python tests/test_si_assist.py

Importante: la U-Net real no está disponible. Las pruebas de `UNetService` usan un
modelo FALSO inyectado (un umbral sobre la entrada) que solo sirve para comprobar la
tubería del front: normalización, remuestreo, orientación y reconstrucción de la
máscara en el espacio original. NO validan ninguna U-Net real.
Las pruebas que necesitan NiBabel se omiten si no está instalado.
"""
from __future__ import annotations

import gzip
import io
import json
import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from si_assist import data as D          # noqa: E402
from si_assist import inference as I     # noqa: E402
from si_assist import viewer as V        # noqa: E402


# --------------------------------------------------------------------------- #
# Ayudas
# --------------------------------------------------------------------------- #

def npy_bytes(arr: np.ndarray) -> bytes:
    buf = io.BytesIO()
    np.save(buf, arr)
    return buf.getvalue()


def make_volume(arr: np.ndarray, name: str = "caso.npy") -> D.Volume:
    return D.load_volume(npy_bytes(arr.astype(np.float32)), name)


def block_volume(shape=(100, 100, 12), block=(slice(30, 60), slice(20, 50), slice(None)), value=1000.0):
    rng = np.random.default_rng(0)
    arr = rng.normal(0, 5, size=shape).astype(np.float32)
    arr[block] = value
    truth = np.zeros(shape, dtype=bool)
    truth[block] = True
    return arr, truth


def iou(a, b):
    a, b = np.asarray(a, bool), np.asarray(b, bool)
    return float(np.logical_and(a, b).sum() / max(np.logical_or(a, b).sum(), 1))


class FakeModel:
    """Modelo FALSO: clasifica por umbral la intensidad normalizada de entrada."""

    def __init__(self, size=(64, 64), softmax=False, expect_transposed=False):
        self.input_shape = (None, size[0], size[1], 1)
        self.softmax, self.calls = softmax, 0

    def predict(self, x, verbose=0):
        self.calls += 1
        v = x[..., 0]
        if not self.softmax:
            return (v > 0.5)[..., None].astype(np.float32)
        lab = np.where(v > 0.75, 2, np.where(v > 0.25, 1, 0))
        return np.eye(3, dtype=np.float32)[lab]


def make_cfg(**model_overrides):
    cfg = D.load_config(Path("no_existe.yaml"))
    cfg["model"]["normalization"] = {"type": "minmax"}
    cfg["model"].update(model_overrides)
    return cfg


def needs_nibabel():
    try:
        import nibabel  # noqa: F401
    except ImportError:
        raise unittest.SkipTest("nibabel no está instalado")


# --------------------------------------------------------------------------- #
# Configuración
# --------------------------------------------------------------------------- #

def test_config_defaults_and_override():
    cfg = D.load_config(Path("no_existe.yaml"))
    assert cfg["data"]["slice_axis"] == 2 and cfg["model"]["confirmed"] is False
    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp) / "c.yaml"
        p.write_text("data: {slice_axis: 0}\nmodel: {classes: {1: Sacro, 2: Ilíaco}}\n", encoding="utf-8")
        cfg = D.load_config(p)
        assert cfg["data"]["slice_axis"] == 0
        assert cfg["model"]["classes"] == {1: "Sacro", 2: "Ilíaco"}       # reemplaza, no mezcla
        assert cfg["model"]["batch_size"] == 8                              # conserva el resto
        p.write_text("data: {slice_axis: 5}\n", encoding="utf-8")
        try:
            D.load_config(p)
            raise AssertionError("debía fallar")
        except D.ConfigError:
            pass


def test_project_config_yaml_is_valid():
    cfg = D.load_config(ROOT / "config.yaml")
    assert cfg["model"]["confirmed"] is False


# --------------------------------------------------------------------------- #
# Carga y validación
# --------------------------------------------------------------------------- #

def test_load_npy_preserves_shape_and_metadata():
    arr = np.random.default_rng(1).uniform(0, 100, (40, 50, 9)).astype(np.float32)
    vol = D.load_volume(npy_bytes(arr), "estudio.npy")
    assert vol.shape == (40, 50, 9) and vol.data.dtype == np.float32
    assert np.array_equal(vol.data, arr)
    assert len(vol.sha256) == 64 and vol.spacing_known is False
    assert vol.display_range[0] < vol.display_range[1]
    assert D.study_stem("a.b.nii.gz") == "a.b" and D.study_stem("x.npy") == "x"


def test_load_rejects_bad_inputs():
    cases = {
        "2D": (np.zeros((20, 20)) + np.arange(20), "caso.npy", "tridimensional"),
        "4D": (np.random.rand(10, 10, 10, 3), "caso.npy", "tridimensional"),
        "diminuto": (np.random.rand(4, 20, 20), "caso.npy", "pequeño"),
        "constante": (np.ones((16, 16, 16)), "caso.npy", "constante"),
    }
    for label, (arr, name, fragment) in cases.items():
        try:
            D.load_volume(npy_bytes(arr), name)
            raise AssertionError(f"{label}: debía fallar")
        except D.StudyError as exc:
            assert fragment in str(exc), (label, str(exc))
    for raw, name in [(b"basura", "caso.npy"), (b"x", "caso.jpg")]:
        try:
            D.load_volume(raw, name)
            raise AssertionError("debía fallar")
        except D.StudyError:
            pass
    try:
        D.load_volume(npy_bytes(np.random.rand(16, 16, 16)), "caso.npy", max_voxels=1000)
        raise AssertionError("debía fallar")
    except D.StudyError as exc:
        assert "máximo" in str(exc)


def test_squeezes_singleton_fourth_axis_and_handles_nan():
    arr = np.random.rand(16, 16, 16, 1).astype(np.float32)
    arr[0, 0, 0, 0] = np.nan
    vol = D.load_volume(npy_bytes(arr), "caso.npy")
    assert vol.shape == (16, 16, 16) and vol.n_nonfinite == 1 and vol.data[0, 0, 0] == 0.0


def test_reference_validation():
    vol = make_volume(np.random.rand(16, 16, 16))
    ref = (np.random.rand(16, 16, 16) > 0.5).astype(np.uint8)
    out, warns = D.load_reference(npy_bytes(ref), "ref.npy", vol)
    assert out.dtype == np.uint8 and out.shape == vol.shape and warns == []
    for bad, frag in [(np.zeros((16, 16, 15)), "coincidir"), (np.random.rand(16, 16, 16), "etiquetas")]:
        try:
            D.load_reference(npy_bytes(bad), "ref.npy", vol)
            raise AssertionError("debía fallar")
        except D.StudyError as exc:
            assert frag in str(exc)


# --------------------------------------------------------------------------- #
# Visor: orientación y correspondencia espacial
# --------------------------------------------------------------------------- #

def test_orient_moves_a_voxel_to_the_expected_pixel():
    a = np.zeros((30, 40), dtype=np.uint8)
    a[10, 25] = 1                                   # x=10, y=25
    out = V.orient(a, {"transpose": True, "flip_vertical": True})
    assert out.shape == (40, 30) and out[40 - 1 - 25, 10] == 1 and out.sum() == 1
    out2 = V.orient(a, {"transpose": False, "flip_vertical": False})
    assert out2[10, 25] == 1


def test_mask_and_image_stay_aligned_after_orientation():
    arr, truth = block_volume(shape=(60, 80, 10), block=(slice(10, 25), slice(40, 70), slice(None)))
    vol = make_volume(arr)
    result = I.DemoService().predict(vol)
    for axis, idx in {0: 17, 1: 55, 2: 5}.items():      # un corte que atraviesa el bloque en cada eje
        cfg = {"transpose": True, "flip_vertical": True}
        img = V.orient(V.take_slice(vol.data, idx, axis), cfg)
        msk = V.orient(V.take_slice(result.mask, idx, axis), cfg)
        t = V.orient(V.take_slice(truth, idx, axis), cfg)
        assert img.shape == msk.shape == t.shape
        assert t.sum() > 0 and iou(msk > 0, t) > 0.99, axis   # la máscara cae sobre el bloque brillante


def test_class_colors_are_stable_across_slices():
    p = V.ViewParams(display_range=(0, 1), opacity=1.0, visible_labels=(1, 2))
    img = np.full((8, 8), 0.5, dtype=np.float32)
    only2 = np.zeros((8, 8), np.uint8); only2[2:4, 2:4] = 2
    both = np.zeros((8, 8), np.uint8); both[2:4, 2:4] = 2; both[5:7, 5:7] = 1
    a = V.render(img, only2, p)[2, 2]
    b = V.render(img, both, p)[2, 2]
    assert tuple(a) == tuple(b) == V.class_color(2)


def test_render_modes_and_visibility():
    img = np.full((8, 8), 0.5, dtype=np.float32)
    m = np.zeros((8, 8), np.uint8); m[3:5, 3:5] = 1
    base = dict(display_range=(0, 1), visible_labels=(1,), opacity=0.5)
    off = V.render(img, m, V.ViewParams(show_mask=False, **base))
    on = V.render(img, m, V.ViewParams(show_mask=True, **base))
    only = V.render(img, m, V.ViewParams(show_mask=True, mask_only=True, **base))
    hidden = V.render(img, m, V.ViewParams(show_mask=True, **{**base, "visible_labels": ()}))
    assert np.array_equal(off, hidden) and not np.array_equal(off, on)
    assert only[0, 0].sum() == 0 and tuple(only[3, 3]) != (0, 0, 0)          # sin imagen de fondo
    outline = V.render(img, m, V.ViewParams(show_mask=True, outline=True, **base))
    assert outline.shape == (8, 8, 3)


def test_window_brightness_contrast_and_aspect():
    sl = np.linspace(0, 100, 11, dtype=np.float32).reshape(1, 11)
    base = V.window_to_uint8(sl, (0, 100))
    assert base[0, 0] == 0 and base[0, -1] == 255 and np.all(np.diff(base[0].astype(int)) >= 0)
    assert V.window_to_uint8(sl, (0, 100), brightness=0.3).mean() > base.mean()
    assert V.window_to_uint8(sl, (0, 100), contrast=2.0)[0, 7] > base[0, 7]
    assert V.pixel_aspect((1.0, 2.0, 3.0), 2, {"transpose": True}) == 2.0   # filas = eje 1
    rgb = V.render(np.zeros((10, 10), np.float32), None, V.ViewParams(display_range=(0, 1)), aspect=2.0)
    assert rgb.shape == (20, 10, 3)
    assert V.to_png(rgb)[:8] == b"\x89PNG\r\n\x1a\n"


# --------------------------------------------------------------------------- #
# Inferencia: contrato, demo y tubería de la U-Net (con modelo FALSO)
# --------------------------------------------------------------------------- #

def test_demo_service_contract_and_stats():
    arr, _ = block_volume()
    vol = make_volume(arr)
    res = I.DemoService().predict(vol)
    assert res.model.is_demo and res.mask.dtype == np.uint8 and res.mask.shape == vol.shape
    assert set(np.unique(res.mask)) <= {0, 1, 2}
    assert res.stats["total_voxels_segmented"] == int((res.mask > 0).sum())
    assert sum(c["voxels"] for c in res.stats["per_class"]) == res.stats["total_voxels_segmented"]
    assert all(len(v) == res.stats["n_slices"] for v in res.stats["per_slice"].values())
    assert res.volume_sha256 == vol.sha256


def test_contract_violations_are_caught():
    class Bad(I.InferenceService):
        def __init__(self, mask, classes):
            self._m, self._c = mask, classes
            self.info = I.DemoService().info

        def _predict_mask(self, volume, progress):
            return self._m(volume), self._c

    vol = make_volume(np.random.rand(16, 16, 16))
    cases = [
        (lambda v: np.zeros((16, 16, 15), np.uint8), {1: "a"}),
        (lambda v: np.full(v.shape, 0.5), {1: "a"}),
        (lambda v: np.full(v.shape, 3, np.uint8), {1: "a"}),
    ]
    for make, classes in cases:
        try:
            Bad(make, classes).predict(vol)
            raise AssertionError("debía fallar")
        except I.InferenceError:
            pass


def _run_fake_unet(axis, swap=False, softmax=False, shape=(100, 100, 12), block=None, size=(64, 64)):
    block = block or (slice(30, 60), slice(20, 50), slice(None))
    arr, truth = block_volume(shape=shape, block=block)
    if axis != 2:
        arr, truth = np.moveaxis(arr, 2, axis), np.moveaxis(truth, 2, axis)
    cfg = make_cfg(swap_hw=swap, batch_size=5)
    cfg["data"]["slice_axis"] = axis
    model = FakeModel(size=size, softmax=softmax)
    svc = I.UNetService(cfg, ROOT, model=model)
    seen = []
    res = svc.predict(make_volume(arr), lambda f, m: seen.append(f))
    return res, truth, model, seen


def test_unet_pipeline_keeps_mask_in_original_space():
    for axis in (0, 1, 2):
        for swap in (False, True):
            res, truth, model, seen = _run_fake_unet(axis, swap)
            assert res.mask.shape == truth.shape, (axis, swap)
            assert iou(res.mask > 0, truth) > 0.9, (axis, swap, iou(res.mask > 0, truth))
            assert res.classes == {1: "Región segmentada"} and not res.model.is_demo
            assert res.model.config_confirmed is False
    assert model.calls == 3                       # 12 cortes en lotes de 5
    assert seen[-1] == 1.0 and seen == sorted(seen)


def test_unet_identity_size_does_not_resample():
    res, truth, *_ = _run_fake_unet(2, size=(100, 100))
    assert iou(res.mask > 0, truth) > 0.99


def test_unet_softmax_multiclass_labels():
    shape = (80, 80, 8)
    arr = np.zeros(shape, np.float32)
    arr[10:30, 10:30, :] = 500.0
    arr[50:70, 50:70, :] = 1000.0
    cfg = make_cfg(classes={1: "Hueso A"})
    svc = I.UNetService(cfg, ROOT, model=FakeModel(size=(80, 80), softmax=True))
    res = svc.predict(make_volume(arr))
    assert res.classes == {1: "Hueso A", 2: "Clase 2"}
    assert (res.mask[20, 20] == 1).all() and (res.mask[60, 60] == 2).all() and (res.mask[40, 40] == 0).all()


def test_unet_declares_unimplemented_integration_points():
    vol = make_volume(np.random.rand(16, 16, 16))
    for override in ({"mode": "3d"}, {"crop": {"enabled": True}}):
        try:
            I.UNetService(make_cfg(**override), ROOT, model=FakeModel()).predict(vol)
            raise AssertionError("debía fallar")
        except I.InferenceError:
            pass
    bad = FakeModel(); bad.input_shape = (None, 64, 64, 3)
    try:
        I.UNetService(make_cfg(), ROOT, model=bad).predict(vol)
        raise AssertionError("debía fallar")
    except I.InferenceError as exc:
        assert "canal" in str(exc)


def test_fingerprint_changes_with_configuration():
    a = I.UNetService(make_cfg(), ROOT).info.fingerprint
    b = I.UNetService(make_cfg(batch_size=3), ROOT).info.fingerprint
    assert a != b != I.DemoService().info.fingerprint


def test_unet_availability_messages():
    with tempfile.TemporaryDirectory() as tmp:
        ok, msg = I.unet_availability(D.load_config(Path("x.yaml")), Path(tmp))
        assert ok is False and "No se encontró el modelo" in msg


# --------------------------------------------------------------------------- #
# Métricas y exportación
# --------------------------------------------------------------------------- #

def test_dice():
    a = np.zeros((10, 10), bool); a[:5] = True
    assert D.dice_binary(a, a) == 1.0
    assert D.dice_binary(a, ~a) == 0.0
    assert D.dice_binary(np.zeros(5), np.zeros(5)) is None      # indefinido, no se infla a 1.0
    pred = np.zeros((4, 4, 4), np.uint8); pred[:2] = 1
    ref = np.zeros((4, 4, 4), np.uint8); ref[:2] = 1; ref[3] = 2
    rep = D.dice_report(pred, ref, {1: "a", 2: "b"})
    assert rep["per_class"][1] == 1.0 and rep["per_class"][2] == 0.0
    assert 0 < rep["global_binary"] < 1


def test_stats_csv_and_json_are_consistent():
    mask = np.zeros((10, 12, 4), np.uint8)
    mask[2:4, 3:6, 1] = 1                         # 6 px en el corte 2
    mask[0, 0, 3] = 2                             # 1 px en el corte 4
    classes = {1: "A", 2: "B"}
    s = D.compute_stats(mask, classes, (0.5, 2.0, 3.0), slice_axis=2, spacing_known=True)
    assert s["pixel_area_mm2"] == 1.0 and s["voxel_volume_mm3"] == 3.0
    assert s["per_slice"][1] == [0, 6, 0, 0] and s["per_slice"][2] == [0, 0, 0, 1]
    assert s["slices_with_segmentation"] == 2 and s["total_voxels_segmented"] == 7
    pc = {c["label"]: c for c in s["per_class"]}
    assert pc[1]["volume_mm3"] == 18.0 and abs(pc[2]["percent_of_segmented"] - 100 / 7) < 1e-9
    rows = D.per_slice_csv(s, classes).decode("utf-8-sig").splitlines()
    assert rows[0] == "corte,pixeles_A,pixeles_B,area_mm2_A,area_mm2_B" and rows[2].startswith("2,6,0,6.0")
    vol = make_volume(np.random.rand(10, 12, 8))
    payload = json.loads(D.summary_json({"modelo": {"es_demostracion": True}}, s, vol))
    assert payload["segmentacion"]["modelo"]["es_demostracion"] is True
    assert "per_slice" not in payload["resultados"] and payload["estudio"]["dimensiones"] == [10, 12, 8]
    s2 = D.compute_stats(mask, classes, (1, 1, 1), 2, spacing_known=False)
    assert s2["pixel_area_mm2"] is None and "area_mm2" not in D.per_slice_csv(s2, classes).decode()


def test_nifti_roundtrip_and_export():
    needs_nibabel()
    import nibabel as nib

    rng = np.random.default_rng(3)
    data = rng.integers(0, 500, (24, 20, 10)).astype(np.int16)
    affine = np.array([[0.8, 0, 0, -10], [0, 0.8, 0, 5], [0, 0, 3.0, 2], [0, 0, 0, 1]], dtype=float)
    img = nib.Nifti1Image(data, affine)
    img.header.set_zooms((0.8, 0.8, 3.0))
    raw = img.to_bytes()                                   # sin comprimir
    for name, payload in [("a.nii", raw), ("a.nii.gz", gzip.compress(raw)), ("mal_nombre.nii", gzip.compress(raw))]:
        vol = D.load_volume(payload, name)
        assert vol.shape == data.shape and np.array_equal(vol.data, data.astype(np.float32)), name
        assert np.allclose(vol.affine, affine) and vol.spacing == (0.8, 0.8, 3.0), name
        assert vol.axcodes is not None and vol.spacing_known
    mask = (data > 250).astype(np.uint8)
    blob = D.mask_to_nifti_gz(mask, vol, "SI-Assist prueba")
    assert blob[:2] == b"\x1f\x8b"                         # realmente comprimido
    back = nib.Nifti1Image.from_bytes(gzip.decompress(blob))
    assert np.array_equal(np.asarray(back.dataobj), mask) and back.get_data_dtype() == np.uint8
    assert np.allclose(back.affine, affine) and tuple(back.header.get_zooms()) == (0.8, 0.8, 3.0)
    try:
        D.load_volume(b"no es nifti", "x.nii")
        raise AssertionError("debía fallar")
    except D.StudyError as exc:
        assert "NIfTI" in str(exc)


# --------------------------------------------------------------------------- #
# Ejecución directa
# --------------------------------------------------------------------------- #

def _main() -> int:
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_") and callable(f)]
    failed = skipped = 0
    for name, fn in tests:
        try:
            fn()
            print(f"  ok       {name}")
        except unittest.SkipTest as exc:
            skipped += 1
            print(f"  omitida  {name}  ({exc})")
        except Exception as exc:  # noqa: BLE001
            failed += 1
            print(f"  FALLA    {name}: {type(exc).__name__}: {exc}")
    print(f"\n{len(tests) - failed - skipped} ok, {failed} fallan, {skipped} omitidas")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(_main())
