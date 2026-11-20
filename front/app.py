"""Sacroiliaca-Assist — interfaz Streamlit.

Esta capa solo orquesta: la lectura de datos vive en `sacroiliaca_assist.data`, el modelo en
`sacroiliaca_assist.inference` y el dibujado de cortes en `sacroiliaca_assist.viewer`.
"""
from __future__ import annotations

import html
import json
import traceback
from pathlib import Path

import pandas as pd
import streamlit as st

from sacroiliaca_assist import data as D
from sacroiliaca_assist import inference as I
from sacroiliaca_assist import viewer as V

ROOT = Path(__file__).resolve().parent
DISCLAIMER = ("Herramienta académica. No es un dispositivo médico, no realiza diagnósticos "
              "y no está validada clínicamente.")
STUDY_KEYS = ("volume", "volume_key", "load_error", "result", "exports", "run_error", "run_requested",
              "ref", "ref_key", "ref_error", "ref_warnings", "metrics", "metrics_sig", "slice_no", "raw_params")

st.set_page_config(page_title="Sacroilíaca-Assist", page_icon="🩻", layout="wide", initial_sidebar_state="expanded")


# --------------------------------------------------------------------------- #
# Utilidades
# --------------------------------------------------------------------------- #

def _wide(fn, *args, **kwargs):
    """Ancho completo compatible con versiones nuevas y antiguas de Streamlit."""
    try:
        return fn(*args, width="stretch", **kwargs)
    except Exception:  # noqa: BLE001 - versiones anteriores usan use_container_width
        return fn(*args, use_container_width=True, **kwargs)


def _file_key(up) -> str:
    return f"{up.name}|{up.size}|{getattr(up, 'file_id', '')}"


@st.cache_data(show_spinner=False)
def _load_config(stamp: float) -> dict:
    return D.load_config(ROOT / "config.yaml")


def get_config() -> dict:
    path = ROOT / "config.yaml"
    return _load_config(path.stat().st_mtime if path.exists() else 0.0)


@st.cache_resource(show_spinner=False)
def _cached_unet(cfg_json: str, model_stamp: float) -> I.InferenceService:
    # Se cachea el servicio (y con él el modelo cargado) para no recargarlo en cada ejecución.
    return I.build_service("unet", json.loads(cfg_json), ROOT)


def get_service(engine: str, cfg: dict) -> I.InferenceService:
    if engine == "demo":
        return I.build_service("demo", cfg, ROOT)
    path = I.resolve_model_path(cfg, ROOT)
    stamp = path.stat().st_mtime if path.exists() else 0.0
    return _cached_unet(json.dumps(cfg, sort_keys=True, default=str), stamp)


def reset_study() -> None:
    for key in STUDY_KEYS:
        st.session_state.pop(key, None)


def close_study() -> None:
    reset_study()
    st.session_state["nonce"] = st.session_state.get("nonce", 0) + 1  # vacía el cargador de archivos


def request_run() -> None:
    st.session_state["run_requested"] = True


# --------------------------------------------------------------------------- #
# Carga de archivos
# --------------------------------------------------------------------------- #

def raw_params_form(up, cfg: dict) -> dict:
    """Parámetros de un .raw (no tiene cabecera)."""
    rc = cfg["raw"]

    st.caption(
        "Los archivos .raw no traen cabecera: indica cómo se guardaron. "
        "Si los valores no son correctos, el estudio se verá mal o no cargará."
    )

    c1, c2 = st.columns(2)

    width = c1.number_input(
        "Ancho (px)", 1, 8192,
        int(rc["width"]), key="raw_w"
    )

    height = c2.number_input(
        "Alto (px)", 1, 8192,
        int(rc["height"]), key="raw_h"
    )

    dtype = st.selectbox(
        "Tipo de dato",
        D.RAW_DTYPES,
        index=D.RAW_DTYPES.index(rc["dtype"]),
        key="raw_dtype"
    )

    order = st.radio(
        "Orden de bytes",
        ["little", "big"],
        horizontal=True,
        key="raw_order",
        index=0 if rc["byte_order"] == "little" else 1
    )

    spacing = None

    if st.checkbox("Conozco el espaciado (mm)", key="raw_known"):
        s1, s2, s3 = st.columns(3)

        spacing = (
            s1.number_input("X", 0.01, 100.0, 1.0, 0.01, key="raw_sx"),
            s2.number_input("Y", 0.01, 100.0, 1.0, 0.01, key="raw_sy"),
            s3.number_input("Z", 0.01, 100.0, 1.0, 0.01, key="raw_sz")
        )

    params = {
        "width": int(width),
        "height": int(height),
        "dtype": dtype,
        "byte_order": order,
        "spacing": spacing
    }

    n, extra = D.raw_slice_count(up.size, params)

    st.caption(
        f"Con estos valores: {n} cortes de {int(width)}×{int(height)}."
        if not extra and n
        else "Con estos valores el tamaño del archivo no cuadra."
    )

    return params

def handle_upload(up, cfg: dict, raw_params: dict | None = None) -> None:
    ss = st.session_state
    if up is None:
        if "volume_key" in ss:
            reset_study()
        return
    key = _file_key(up) + (
        f"|{json.dumps(raw_params, sort_keys=True)}"
        if raw_params else ""
    )
    if ss.get("volume_key") == key:
        return  # ya cargado: no se vuelve a leer
    reset_study()
    ss["volume_key"] = key
    try:
        with st.spinner("Leyendo estudio…"):
            volume = D.load_volume(up.getvalue(), up.name, cfg["data"]["max_voxels"], raw_params)
    except D.StudyError as exc:
        ss["load_error"] = str(exc)
        return
    except Exception as exc:  # noqa: BLE001
        ss["load_error"] = f"Error inesperado al leer el archivo ({type(exc).__name__}: {exc})."
        return
    ss["volume"] = volume
    ss["raw_params"] = raw_params
    ss["slice_no"] = volume.shape[cfg["data"]["slice_axis"]] // 2 + 1


def handle_reference(up, volume: D.Volume, cfg: dict, raw_params: dict | None = None) -> None:
    ss = st.session_state
    ref_keys = ("ref", "ref_key", "ref_error", "ref_warnings", "metrics", "metrics_sig")
    if up is None:
        for k in ref_keys:
            ss.pop(k, None)
        return
    key = _file_key(up) + (
        f"|{json.dumps(raw_params, sort_keys=True)}"
        if raw_params else ""
    )
    if ss.get("ref_key") == key:
        return
    for k in ref_keys:
        ss.pop(k, None)
    ss["ref_key"] = key
    try:
        ss["ref"], ss["ref_warnings"] = D.load_reference(up.getvalue(), up.name, volume, cfg["data"]["max_voxels"], raw_params)
    except D.StudyError as exc:
        ss["ref_error"] = str(exc)
    except Exception as exc:  # noqa: BLE001
        ss["ref_error"] = f"Error inesperado al leer la referencia ({type(exc).__name__}: {exc})."


# --------------------------------------------------------------------------- #
# Procesamiento
# --------------------------------------------------------------------------- #

def build_exports(volume: D.Volume, result: I.InferenceResult) -> dict:
    stem = D.study_stem(volume.name)
    prefix = "DEMO_" if result.model.is_demo else ""
    desc = "Sacroiliaca-Assist DEMO (no es U-Net)" if result.model.is_demo else "Sacroiliaca-Assist U-Net"
    jobs = {
        "mask": (f"{prefix}{stem}_segmentacion.nii.gz", "application/gzip",
                 lambda: D.mask_to_nifti_gz(result.mask, volume, desc)),
        "csv": (f"{prefix}{stem}_cortes.csv", "text/csv",
                lambda: D.per_slice_csv(result.stats, result.classes)),
        "json": (f"{prefix}{stem}_resumen.json", "application/json",
                 lambda: D.summary_json(result.to_metadata(), result.stats, volume)),
    }
    if volume.source == "raw":
        jobs["mask_raw"] = (
            f"{prefix}{stem}_segmentacion.raw",
            "application/octet-stream",
            lambda: D.mask_to_raw(result.mask)
        )
    out = {}
    for key, (name, mime, make) in jobs.items():
        try:
            out[key] = {"name": name, "mime": mime, "data": make(), "error": None}
        except Exception as exc:  # noqa: BLE001
            out[key] = {"name": name, "mime": mime, "data": None, "error": f"{type(exc).__name__}: {exc}"}
    return out


def run_inference(volume: D.Volume, service: I.InferenceService) -> None:
    ss = st.session_state
    slot = st.empty()
    bar = slot.progress(0.0, text="Preparando volumen…")

    def progress(fraction: float, message: str) -> None:
        bar.progress(float(min(max(fraction, 0.0), 1.0)), text=message)

    try:
        result = service.predict(volume, progress)
    except I.InferenceError as exc:
        ss["run_error"] = str(exc)
        slot.empty()
        return
    except Exception as exc:  # noqa: BLE001
        ss["run_error"] = (f"Error inesperado durante la inferencia ({type(exc).__name__}: {exc}).\n\n"
                           + traceback.format_exc())
        slot.empty()
        return
    slot.empty()
    ss.pop("run_error", None)
    ss["result"] = result
    ss["exports"] = build_exports(volume, result)
    st.rerun()  # refresca la barra lateral con el estado nuevo


# --------------------------------------------------------------------------- #
# Barra lateral
# --------------------------------------------------------------------------- #

def sidebar(cfg: dict) -> I.InferenceService | None:
    ss = st.session_state
    with st.sidebar:
        st.markdown('<div class="brand">Sacroiliaca-Assist</div>'
                    '<div class="brand-sub">Visor de resonancias de articulaciones sacroilíacas '
                    'con segmentación U-Net</div>', unsafe_allow_html=True)

        st.markdown("#### Estudio")
        up = st.file_uploader("Volumen MRI", type=["nii", "nii.gz", "npy", "raw"],
                              key=f"uploader_{ss.get('nonce', 0)}",
                              help="NIfTI (.nii, .nii.gz) o NumPy (.npy), tridimensional. "
                                   "Al cargar otro archivo se reemplaza el estudio actual.")
        raw_params = None

        if up is not None and up.name.lower().endswith(".raw"):
            with st.expander("Parámetros del archivo .raw", expanded=True):
                raw_params = raw_params_form(up, cfg)
        handle_upload(up, cfg, raw_params)
        volume = ss.get("volume")
        if volume is not None:
            st.button("Cerrar estudio", on_click=close_study, key="close_study")

        st.markdown("#### Segmentación")
        ok, reason = I.unet_availability(cfg, ROOT)
        labels = {"unet": "U-Net (modelo entrenado)", "demo": "Demostración (sin modelo)"}
        engine = st.radio("Motor", list(labels), format_func=labels.get, index=0 if ok else 1,
                          key="engine", label_visibility="collapsed")
        service = None
        if engine == "unet" and not ok:
            st.warning(reason)
        else:
            service = get_service(engine, cfg)
        if engine == "demo":
            st.caption("Genera una máscara sintética por umbral de intensidad para probar la interfaz. "
                       "No es una segmentación real.")

        result = ss.get("result")
        if volume is None:
            st.caption("Carga un volumen para habilitar la segmentación.")
        else:
            current = bool(result and service and result.volume_sha256 == volume.sha256
                           and result.model.fingerprint == service.info.fingerprint)
            if current:
                st.success("Segmentación al día.")
                st.button("Volver a ejecutar", on_click=request_run, key="rerun")
            else:
                if result is not None:
                    st.caption("El resultado mostrado proviene de otro motor o configuración.")
                st.button("Ejecutar segmentación", type="primary", on_click=request_run,
                          disabled=service is None, key="run")
        st.markdown(f'<div class="disclaimer">{DISCLAIMER}</div>', unsafe_allow_html=True)
    return service


# --------------------------------------------------------------------------- #
# Panel principal
# --------------------------------------------------------------------------- #

def empty_state() -> None:
    st.markdown(f"""
<div class="empty">
  <h2>Carga un estudio para empezar</h2>
  <ol>
    <li>Carga un volumen MRI tridimensional (NIfTI) desde el panel izquierdo.</li>
    <li>Ejecuta la segmentación con la U-Net.</li>
    <li>Recorre los cortes, ajusta la superposición y revisa los resultados.</li>
  </ol>
  <div class="note">{DISCLAIMER}</div>
</div>""", unsafe_allow_html=True)


def study_bar(volume: D.Volume, result: I.InferenceResult | None, n_slices: int) -> None:
    sx, sy, sz = volume.spacing
    spacing = f"{sx:.2f} × {sy:.2f} × {sz:.2f} mm" if volume.spacing_known else "no disponible"
    orient = "".join(volume.axcodes) if volume.axcodes else "no disponible"
    if result is None:
        status, css = "Sin segmentar", "status-wait"
    elif result.model.is_demo:
        status, css = "Demostración", "status-demo"
    else:
        status, css = "Segmentado · U-Net", "status-ok"
    cells = [("Estudio", volume.name, ""), ("Dimensiones", " × ".join(map(str, volume.shape)), ""),
             ("Cortes", f"{n_slices:,}", ""), ("Espaciado", spacing, ""),
             ("Orientación", orient, ""), ("Estado", status, css)]
    body = "".join(f'<div><div class="k">{k}</div><div class="v {c}" title="{html.escape(v)}">'
                   f'{html.escape(v)}</div></div>' for k, v, c in cells)
    st.markdown(f'<div class="study-bar">{body}</div>', unsafe_allow_html=True)

def display_cfg(volume: D.Volume, cfg: dict) -> dict:
    """Configura la orientación de visualización de archivos RAW."""
    d = dict(cfg["display"])

    if volume.source == "raw":
        d["flip_vertical"] = False

    return d

def _step(delta: int, n: int) -> None:
    ss = st.session_state
    ss["slice_no"] = int(min(max(ss["slice_no"] + delta, 1), n))


def _reset_window() -> None:
    st.session_state["brightness"] = 0.0
    st.session_state["contrast"] = 1.0


@st.fragment
def viewer_panel(volume: D.Volume, result: I.InferenceResult | None, cfg: dict) -> None:
    """Visor. Es un fragmento: mover el control de cortes no reejecuta el resto de la app."""
    ss = st.session_state
    axis, dcfg = cfg["data"]["slice_axis"], display_cfg(volume, cfg)
    n = volume.shape[axis]
    ss["slice_no"] = int(min(max(ss.get("slice_no", n // 2 + 1), 1), n))

    left, right = st.columns([3.3, 1.3], gap="large")
    show, mode, style, opacity, visible = False, "Sobre la imagen", "Relleno", 0.45, ()
    labels = sorted(result.classes) if result else []

    with right:
        st.markdown("**Navegación**")
        a, b = st.columns(2)
        a.button("◀ Anterior", on_click=_step, args=(-1, n), key="prev_slice")
        b.button("Siguiente ▶", on_click=_step, args=(1, n), key="next_slice")
        st.slider("Corte", 1, n, key="slice_no")
        with st.expander("Brillo y contraste"):
            st.slider("Brillo", -0.5, 0.5, 0.0, 0.05, key="brightness")
            st.slider("Contraste", 0.5, 3.0, 1.0, 0.1, key="contrast")
            st.button("Restablecer", on_click=_reset_window, key="reset_window")
        if result is None:
            st.caption("Ejecuta la segmentación desde el panel izquierdo para superponer las regiones.")
        else:
            st.markdown("**Segmentación**")
            show = st.toggle("Mostrar segmentación", value=True, key="show_mask")
            mode = st.radio("Vista", ["Sobre la imagen", "Solo máscara"], horizontal=True,
                            key="mask_mode", disabled=not show)
            style = st.radio("Estilo", ["Relleno", "Contorno"], horizontal=True,
                             key="mask_style", disabled=not show)
            opacity = st.slider("Opacidad", 0.05, 1.0, 0.45, 0.05, key="opacity", disabled=not show)
            visible = st.multiselect("Regiones visibles", labels, default=labels,
                                     format_func=lambda l: result.classes[l],
                                     key=f"visible_{result.created_utc}", disabled=not show)

    idx = ss["slice_no"] - 1
    image2d = V.orient(V.take_slice(volume.data, idx, axis), dcfg)
    mask2d = V.orient(V.take_slice(result.mask, idx, axis), dcfg) if result else None
    brightness, contrast = ss.get("brightness", 0.0), ss.get("contrast", 1.0)
    params = V.ViewParams(display_range=volume.display_range, brightness=brightness, contrast=contrast,
                          show_mask=show, mask_only=(mode == "Solo máscara"), opacity=opacity,
                          outline=(style == "Contorno"), visible_labels=tuple(visible))
    aspect = V.pixel_aspect(volume.spacing, axis, dcfg) if volume.spacing_known else 1.0
    rgb = V.render(image2d, mask2d, params, aspect)

    with left:
        st.markdown(f'<div class="viewer-head"><span>Corte <b>{idx + 1}</b> de {n}</span>'
                    f'<span>Brillo {brightness:+.2f} · Contraste ×{contrast:.1f}</span></div>',
                    unsafe_allow_html=True)
        _wide(st.image, rgb, clamp=True)
        if result is not None:
            pix = result.stats["pixel_area_mm2"]
            rows = []
            for lab in labels:
                cnt = int((mask2d == lab).sum())
                area = f" · {cnt * pix:,.1f} mm²" if pix else ""
                rows.append(f'<div><span class="chip" style="background:{V.hex_color(lab)}"></span>'
                            f'{html.escape(result.classes[lab])} <span class="muted">· {cnt:,} px{area}</span></div>')
            st.markdown('<div class="legend">' + "".join(rows) + "</div>", unsafe_allow_html=True)
            ref = ss.get("ref")
            if ref is not None and not result.model.is_demo:
                d = D.dice_binary(V.take_slice(ref, idx, axis) > 0, V.take_slice(result.mask, idx, axis) > 0)
                st.caption("Dice del corte (binario): " + ("—" if d is None else f"{d:.4f}"))


def results_tab(volume: D.Volume, result: I.InferenceResult | None, cfg: dict) -> None:
    ss = st.session_state
    if result is None:
        st.info("Aún no hay resultados. Ejecuta la segmentación desde el panel izquierdo.")
        return
    s = result.stats
    present = [c for c in s["per_class"] if c["voxels"] > 0]
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Cortes procesados", f"{s['n_slices']:,}")
    m2.metric("Cortes con segmentación", f"{s['slices_with_segmentation']:,}")
    m3.metric("Vóxeles segmentados", f"{s['total_voxels_segmented']:,}")
    m4.metric("Clases identificadas", f"{len(present)} de {len(s['per_class'])}")
    st.caption(f"{result.model.name} · {result.model.description} · {result.elapsed_s:.1f} s · {result.created_utc} UTC")

    df = pd.DataFrame([{
        "Clase": c["name"], "Vóxeles": c["voxels"],
        "% de lo segmentado": round(c["percent_of_segmented"], 2),
        "Volumen (mm³)": None if c["volume_mm3"] is None else round(c["volume_mm3"], 1),
        "Cortes con la clase": c["slices_with_class"],
    } for c in s["per_class"]])
    _wide(st.dataframe, df, hide_index=True)

    labels = sorted(result.classes)
    if present:
        st.markdown("**Distribución por corte** (píxeles segmentados)")
        chart = pd.DataFrame({result.classes[l]: s["per_slice"][l] for l in labels},
                             index=pd.RangeIndex(1, s["n_slices"] + 1, name="Corte"))
        st.bar_chart(chart, color=[V.hex_color(l) for l in labels], x_label="Corte",
                     y_label="Píxeles segmentados")
    else:
        st.info("El modelo no segmentó ningún vóxel en este estudio.")

    with st.expander("Evaluación con máscara de referencia"):
        if result.model.is_demo:
            st.info("No se calculan métricas sobre máscaras de demostración.")
            return
        up = st.file_uploader("Máscara de referencia (.nii, .nii.gz, .npy)", type=["nii", "nii.gz", "npy", "raw"],
                              key=f"ref_uploader_{ss.get('nonce', 0)}",
                              help="Mismo tamaño que el volumen. Solo se usa para calcular Dice.")
        ref_raw = None

        if up is not None and up.name.lower().endswith(".raw"):
            r1, r2 = st.columns(2)

            ref_raw = {
                "width": volume.shape[0],
                "height": volume.shape[1],
                "dtype": r1.selectbox(
                    "Tipo de dato de la referencia",
                    D.RAW_DTYPES,
                    index=0,
                    key="ref_dtype"
                ),
                "byte_order": r2.radio(
                    "Orden de bytes",
                    ["little", "big"],
                    key="ref_order"
                )
            }

            st.caption(
                f"Se asume {volume.shape[0]}×{volume.shape[1]} px, "
                "igual que el volumen."
            )

        handle_reference(up, volume, cfg, ref_raw)
        if ss.get("ref_error"):
            st.error(ss["ref_error"])
        for w in ss.get("ref_warnings", []):
            st.warning(w)
        if ss.get("ref") is not None:
            sig = (ss["ref_key"], result.created_utc)
            if ss.get("metrics_sig") != sig:
                ss["metrics"] = D.dice_report(result.mask, ss["ref"], result.classes)
                ss["metrics_sig"] = sig
            rep = ss["metrics"]
            fmt = lambda v: "—" if v is None else f"{v:.4f}"
            st.metric("Dice global (binario)", fmt(rep["global_binary"]))
            rows = [{"Clase": result.classes.get(l, f"Clase {l}"), "Dice": fmt(v)}
                    for l, v in rep["per_class"].items()]
            _wide(st.dataframe, pd.DataFrame(rows), hide_index=True)
            st.caption("Calculado contra la referencia cargada. “—” significa que ambas máscaras están vacías.")


def export_tab(result: I.InferenceResult | None) -> None:
    if result is None:
        st.info("Ejecuta la segmentación para poder exportar resultados.")
        return
    if result.model.is_demo:
        st.warning("Estos archivos provienen del modo demostración. Los nombres llevan el prefijo DEMO_ "
                   "y el contenido lo identifica como no real.")
    exports = st.session_state.get("exports") or {}
    items = [
        ("mask_raw", "Máscara segmentada (.raw)", "uint8, una etiqueta por píxel, con la misma disposición "
            "que el .raw original."),
        ("mask", "Máscara segmentada", "NIfTI comprimido con las mismas dimensiones, matriz espacial y "
                                         "espaciado del volumen original. Etiquetas enteras por clase."),
        ("csv", "Resultados por corte", "Píxeles (y mm² si hay espaciado) por clase y por corte, desde el corte 1."),
        ("json", "Resumen del estudio", "Estudio, modelo usado, clases y totales en JSON."),
    ]
    for key, title, desc in items:
        item = exports.get(key)
        if key == "mask_raw" and item is None:
            continue
        c1, c2 = st.columns([3, 1.4], gap="large")
        c1.markdown(f"**{title}**  \n<span style='color:var(--muted)'>{desc}</span>", unsafe_allow_html=True)
        if item and item["data"] is not None:
            c2.download_button(f"Descargar {item['name']}", data=item["data"], file_name=item["name"],
                               mime=item["mime"], key=f"dl_{key}")
        else:
            c2.error(f"No se pudo generar: {item['error'] if item else 'sin datos'}")


# --------------------------------------------------------------------------- #
# Programa principal
# --------------------------------------------------------------------------- #

def main() -> None:
    try:
        cfg = get_config()
    except D.ConfigError as exc:
        st.error(f"Error en config.yaml: {exc}")
        st.stop()
    css = ROOT / "assets" / "style.css"
    if css.exists():
        st.markdown(f"<style>{css.read_text(encoding='utf-8')}</style>", unsafe_allow_html=True)

    service = sidebar(cfg)
    ss = st.session_state
    if ss.get("load_error"):
        st.error(ss["load_error"])
    volume = ss.get("volume")
    if volume is None:
        empty_state()
        return

    if ss.pop("run_requested", False) and service is not None:
        run_inference(volume, service)
    result = ss.get("result")

    study_bar(volume, result, volume.shape[cfg["data"]["slice_axis"]])
    if ss.get("run_error"):
        st.error("No se pudo completar la segmentación.")
        with st.expander("Detalle", expanded=True):
            st.code(ss["run_error"])
    if volume.n_nonfinite:
        st.caption(f"{volume.n_nonfinite:,} vóxeles con valores no numéricos (NaN/inf) se mostraron como 0.")
    if result is not None and result.model.is_demo:
        st.markdown('<div class="banner demo"><b>Modo demostración.</b> La máscara es sintética (umbral de '
                    'intensidad) y no proviene de una U-Net ni representa estructuras anatómicas.</div>',
                    unsafe_allow_html=True)
    elif result is not None and not result.model.config_confirmed:
        st.warning("La configuración del modelo (orientación, normalización, recorte, clases) aún no está "
                   "confirmada por el equipo de la U-Net. Las máscaras podrían no ser correctas.")

    tab_view, tab_results, tab_export = st.tabs(["Visor", "Resultados", "Exportar"])
    with tab_view:
        viewer_panel(volume, result, cfg)
    with tab_results:
        results_tab(volume, result, cfg)
    with tab_export:
        export_tab(result)


main()
