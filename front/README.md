# SI-Assist — frontend (Streamlit)

Visor de resonancias magnéticas de las articulaciones sacroilíacas con superposición de la segmentación de una U-Net.
Herramienta académica: no es un dispositivo médico, no diagnostica y no está validada clínicamente.

## Instalación y ejecución

Recomendado: Python 3.10 o 3.11.

```bash
python -m venv .venv
# Windows (PowerShell):  .venv\Scripts\Activate.ps1
# Linux/macOS:           source .venv/bin/activate

pip install -r requirements.txt
streamlit run app.py
```

TensorFlow solo se necesita para la U-Net real. Sin el modelo, la app ofrece el **modo demostración**.

## Estructura

```
app.py                  Interfaz (orquesta; no contiene lógica de datos ni de modelo)
config.yaml             Datos, visualización y parámetros del modelo
si_assist/
  data.py               Carga y validación, métricas, exportación
  inference.py          Contrato InferenceService, DemoService, UNetService
  viewer.py             Ventana de intensidad, superposición, colores por clase
models/                 Aquí va el modelo entrenado (.keras / .h5)
assets/style.css        Tema del visor
tests/test_si_assist.py Pruebas
.streamlit/config.toml  Tema oscuro y límite de subida (1 GB)
```

## Contrato entre la interfaz y el modelo

```python
InferenceService.predict(volume: Volume, progress) -> InferenceResult
```

`InferenceResult` contiene: `mask` (uint8, **misma forma que el volumen, en su espacio original**), `classes`
(`{etiqueta: nombre}`), `shape`, `affine`, `spacing`, `model` (nombre, tipo, si es demostración, si su configuración
está confirmada, huella) y `stats`. `predict()` verifica el contrato (forma, tipo, etiquetas declaradas) antes de
entregar el resultado, de modo que un servicio mal implementado falla con un mensaje claro y no con una máscara desalineada.

Para conectar la U-Net: ajustar `config.yaml` y, si el pipeline del equipo lo exige, completar `UNetService`
(puntos marcados `INTEGRACIÓN`). La interfaz no cambia.

## Qué se necesita del equipo de la U-Net

1. El archivo del modelo (`.keras`, `.h5`) en `models/` (o la ruta en `model.path` / variable `SI_ASSIST_MODEL`).
2. ¿Es 2D o 3D? Hoy solo está implementado 2D corte a corte.
3. Orientación de los cortes: eje de corte (`data.slice_axis`) y si la entrada es `[y, x]` (`model.swap_hw`).
4. Normalización exacta (`none`, `minmax`, `percentile`, `zscore`; por volumen).
5. **Recorte de la región ósea**: el código exacto del recorte y cómo revertirlo. No está implementado a propósito:
   reconstruirlo redimensionando desalinearía la máscara.
6. Salida: sigmoid o softmax, número de canales, si el canal 0 es fondo y qué significa cada etiqueta (`model.classes`).
7. Tamaño de entrada y número de canales.
8. Un volumen de ejemplo con su máscara de referencia, para verificar la alineación espacial.

Cuando todo esté confirmado, poner `model.confirmed: true` en `config.yaml`. Mientras sea `false`, la interfaz avisa
en los resultados reales de que la configuración no está confirmada.

## Modo demostración

Genera una máscara sintética por umbral de intensidad. **No es una U-Net ni segmenta anatomía.** Se identifica con un
aviso permanente, el estado "Demostración", el prefijo `DEMO_` en los archivos exportados y un campo
`es_demostracion` en el JSON. No se calculan métricas (Dice) sobre máscaras de demostración.

## Exportación

- Máscara `.nii.gz` (uint8) con la matriz espacial (affine) y la cabecera del volumen original.
- CSV por corte (píxeles y mm² por clase; cortes numerados desde 1).
- JSON con el estudio, el modelo usado, las clases y los totales.

## Decisiones de diseño

- **Pocos archivos**: 3 módulos + `app.py` + 1 archivo de pruebas, en lugar de un archivo por responsabilidad.
- **La orientación de pantalla (`display`) solo afecta lo que se ve.** Imagen y máscara pasan por las mismas funciones,
  y la máscara exportada no se reorienta.
- **Sin recarga innecesaria**: el volumen no se vuelve a leer si el archivo no cambia; el resultado se reutiliza si el
  volumen y la configuración del modelo son los mismos; el modelo queda cargado entre ejecuciones; el visor es un
  fragmento de Streamlit, así que mover el control de cortes no reejecuta el resto de la app.
- **El modelo ya no se sube por la interfaz**: se coloca en `models/`. Un profesional de la salud no debería tener
  que manejar el archivo del modelo, y evita subidas de cientos de MB en cada sesión.
- Los cortes se numeran desde 1 en toda la interfaz y en el CSV.

## Pruebas

```bash
python -m pytest tests        # o: python tests/test_si_assist.py
```

Las pruebas de `UNetService` usan un modelo **falso** inyectado (un umbral): comprueban la tubería del front
(normalización, remuestreo, orientación, reconstrucción en el espacio original), no ninguna U-Net real.
La prueba de lectura/escritura NIfTI se omite si NiBabel no está instalado.
