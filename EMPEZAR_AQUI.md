# EMPEZAR AQUÍ 👋 (para Hector / cualquier compañero)

Todo desde **un solo sitio**, paso a paso. Sin abrir 500 pestañas.

## 1) Una sola vez — instalar
Necesitas **Python 3.10+** y **Git** instalados. Luego, en PowerShell, dentro de la carpeta del proyecto:

```powershell
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

> Si lo bajas de cero: `git clone https://github.com/amartinezguimon/AI-AD.git` → `cd AI-AD` → y luego lo de arriba. (Ya está todo en `main`, no hay que cambiar de rama.)

## 2) Cada vez que quieras usarlo
**Doble clic en `DEMO.bat`** (o en PowerShell: `python run.py`).

Sale un menú. Escribe el número y Enter:

Sale un menú. Escribe el número y Enter:

| Opción | Qué hace | Qué me envías |
|---|---|---|
| **1) DEMO guiada** | Te lleva de la mano: **calibrar escaparate → dibujar la zona → probar en vivo**. Cada paso puedes saltarlo escribiendo `n`. La prueba en vivo se abre sola en el navegador; pulsa **"Detener sesión"** ahí cuando termines. | El archivo `results\demo_….json` que te indica al final |
| **2) Grabar datos** | Te pide tu nombre y graba ejemplos (**L**=mira, **A**=no, **T**=grabar seguido, **G**=gafas, **H**=gorra, **Q**=guardar). | El archivo `data\raw_sessions\….csv` que te indica al final |
| **3) Solo dibujar la zona** | Marcas con clics la acera que cuenta (**S**=guardar). | (nada; queda guardado) |
| **4) Solo calibrar** | Calibra el escaparate (capturas con **1-5**, **S**=guardar). | (nada; queda guardado) |
| **5) Elegir cámara** | Se abre el navegador con una foto de cada cámara conectada (incluido el móvil si tienes Camo/Continuity Camera activo). Clic en la que sea tu móvil y queda guardada — no hay que escribir ningún número ni adivinar cuál es cuál. | — |
| **6) Importar y etiquetar un vídeo** | Le das la ruta de un vídeo ya grabado (calle, tienda...) y analiza a la gente igual que en vivo. Se abre solo el navegador para que marques quién mira y quién no. | El `…_gaze.csv` que descargas desde el navegador |
| **7) Etiquetar tu última grabación** | Como la 6, pero sin pegar ninguna ruta: coge sola la grabación más reciente (de cualquier prueba en vivo) y abre el navegador. | El `…_gaze.csv` que descargas desde el navegador |
| **0) Salir** | | |

> Lo normal es la **opción 1** (lo hace todo en orden). Las 3 y 4 son por si quieres rehacer solo un paso.

## 3) Enviarme los archivos
Cuando termina, te escribe en pantalla **la ruta exacta** del archivo.
Mándamelo por **WhatsApp o email**. **No hace falta tocar GitHub.**

## Notas
- Al **grabar** (opción 2): recuerda que **“mirar” = mirar a la cámara** (la cámara hace de escaparate). Graba variedad: con/sin gafas y gorra, de cerca y de lejos.
- En la **demo guiada**: la calibración y la zona se guardan juntas, así que la prueba en vivo ya las usa. Si cuenta de más por gente lejana, repite el paso de la zona (opción 3).
- La **opción 6** (importar vídeo) tarda porque analiza el vídeo entero fotograma a fotograma antes de abrir el navegador — no hace falta esperar activo, solo no cerrar la ventana negra hasta que el navegador se abra solo.
- La **prueba en vivo** (dentro de la opción 1) ya no abre una ventana de cámara aparte: se abre sola en el **navegador**, con la cámara y los números (gente, cuántos miran, etc.) al lado. No cierres la ventana negra de PowerShell mientras tanto — solo pulsa "Detener sesión" en la página cuando acabes.
- Cada prueba en vivo **se graba automáticamente** (se ve un ● REC en la esquina mientras corre). Al pulsar "Detener sesión" te dice en la ventana negra dónde quedó guardado el vídeo (`.mp4`, junto al `.json` del reporte). Si lanzaste la prueba desde la **opción 1 (DEMO guiada)** o el menú, te pregunta ahí mismo si quieres etiquetarla ya — di que sí y no hace falta tocar nada más. Si la lanzaste con el acceso directo "VisionMetrics Live" (sin pasar por el menú), usa la **opción 7 (Etiquetar tu última grabación)** o el acceso directo "VisionMetrics Etiquetar Ultima" del Escritorio — encuentra sola el vídeo más reciente.
- Si algo falla, hazme una **captura de la pantalla** con el error y me la mandas.

---

# 🤝 Trabajar los tres con el MISMO dataset (Álvaro, Hector, Cristian)

La idea: cada uno graba y etiqueta en su propio portátil, y **todo acaba en la misma carpeta de entrenamiento** para que el modelo aprenda de los tres.

Cada sesión etiquetada se guarda como un CSV con **fecha y hora en el nombre**
(`data/raw_sessions/live_AAAAMMDD-HHMMSS.csv`), así que **los archivos de dos
personas nunca chocan**: se pueden juntar todos en una carpeta sin pisarse. Además,
en la pantalla de etiquetado ahora eliges tu nombre en el desplegable
(**Álvaro / Hector / Cristian**), y ese nombre queda guardado en cada fila — así se
sabe automáticamente quién etiquetó qué. Lo puedes comprobar en la última pantalla
(**"Model check" → "Training data"**): hay un botón **"⬇ Download training CSV"** y
una **gráfica de barras** con cuántas filas ha aportado cada uno.

## Camino A — con Git (recomendado)
Una vez cada uno tiene el proyecto clonado del repo del equipo
(`git clone https://github.com/amartinezguimon/AI-AD.git`):

1. **Antes de empezar**, trae lo último de los demás:
   ```powershell
   git pull
   ```
2. Elige tu cámara la primera vez: menú **opción 5 (Elegir cámara)** — el `0` que
   viene por defecto es la del móvil de Hector, en tu portátil casi seguro es otra.
   (No subas tu `configs/camera_pref.txt`.)
3. Graba y etiqueta como siempre (opción 1). Tu nombre ya está en el desplegable.
4. Cuando acabes, **sube tus sesiones**:
   ```powershell
   git add data/raw_sessions/
   git commit -m "Cristian: sesiones 26-jul"
   git push
   ```
   No hay conflictos porque tus archivos tienen tu fecha/hora.
5. Para **reentrenar con lo de los tres**: `git pull` y luego el botón
   **"Retrain & compare"** en la pantalla de análisis (o
   `python -m visionmetrics.training.build_dataset` + `python -m visionmetrics.training.train`).

## Camino B — sin Git (más fácil, como hasta ahora)
Si prefieres no tocar Git: al terminar de etiquetar, pulsa
**"⬇ Download training CSV"** (o coge tu `data\raw_sessions\live_….csv`) y
**mándamelo por WhatsApp/email**. Yo (o Álvaro) los metemos todos en
`data/raw_sessions/` y reentrenamos. Más simple, pero hay que juntarlo a mano.

## Ojo con esto
- **No** mandes tu carpeta `venv/` (pesa ~1 GB y es de tu máquina): cada uno crea la suya con `pip install -r requirements.txt`.
- La **primera vez** hace falta internet: se descargan solos los modelos (YOLO + MediaPipe, unos MB).
- En **Mac**: abre con doble clic en `VisionMetrics.command` (lanzarlo desde el Finder/Terminal es lo que da permiso de cámara).
- Solo hacen falta los **CSV** para entrenar; las carpetas de recortes (`_images/`, `_detimages/`) ya no se suben al repo (las ignora `.gitignore`) para mantenerlo ligero.
