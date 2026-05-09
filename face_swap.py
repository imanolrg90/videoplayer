#!/usr/bin/env python3
"""
face_swap.py - Intercambia la cara de una foto en una imagen o video destino.

Uso:
    python face_swap.py

Requisitos:
    pip install insightface onnxruntime opencv-python numpy

Modelo necesario (descargar manualmente, ~500 MB):
    https://huggingface.co/deepinsight/inswapper/resolve/main/inswapper_128.onnx
    Colocarlo en la misma carpeta que este script, o indicar la ruta con --modelo.
"""

import cv2
import numpy as np
import os
import sys
import glob
import shutil
import ssl
import urllib.request
import zipfile

MODEL_URL = "https://huggingface.co/deepinsight/inswapper/resolve/main/inswapper_128.onnx"
BUFFALO_L_URL = "https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip"

# Valores por defecto para pruebas locales (puedes cambiarlos a tus rutas)
DEFAULT_CARA_PATH = r"F:\ORACLE\NETWORK\tnsnames\abby\fotos\cara.jpeg"
DEFAULT_CUERPO_PATH = r"F:\ORACLE\NETWORK\tnsnames\leto\cuerpo.jpeg"
DEFAULT_OUTPUT_DIR = r"F:\down"
VIDEO_EXTS = {".mp4", ".mov", ".mkv", ".avi", ".webm", ".m4v"}
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def contar_imagenes_en_carpeta(path: str) -> int:
    """Cuenta imágenes compatibles dentro de una carpeta (sin recursión)."""
    if not os.path.isdir(path):
        return 0
    total = 0
    for nombre in os.listdir(path):
        _, ext = os.path.splitext(nombre)
        if ext.lower() in IMAGE_EXTS:
            total += 1
    return total


def resolver_max_sources(source_path: str, requested_max_sources):
    """
    Resuelve cuántas fotos usar desde carpeta origen.

    Reglas auto (requested_max_sources=None):
      - > 30 fotos: usar 12
      - <= 30 fotos: usar todas (0)
    """
    if requested_max_sources is not None:
        return requested_max_sources

    if not os.path.isdir(source_path):
        return 0

    total = contar_imagenes_en_carpeta(source_path)
    if total > 30:
        print("Modo auto de fuentes: carpeta grande detectada (>30). Usando top_k=12.")
        return 12

    print("Modo auto de fuentes: carpeta pequena (<=30). Usando todas las fotos.")
    return 0


def asegurar_pack_buffalo_l():
    """Asegura que exista el pack buffalo_l en ~/.insightface/models/buffalo_l."""
    root = os.path.join(os.path.expanduser("~"), ".insightface", "models")
    pack_dir = os.path.join(root, "buffalo_l")
    onnx_files = glob.glob(os.path.join(pack_dir, "*.onnx"))
    if onnx_files:
        # InsightFace 0.2.1 solo enruta bien modelos SCRFD (detección)
        # y ArcFace 112x112 (reconocimiento). Se filtra el resto.
        _filtrar_modelos_compatibles(pack_dir)
        return

    print("No se encontró buffalo_l local. Descargando pack de modelos...")
    os.makedirs(root, exist_ok=True)
    os.makedirs(pack_dir, exist_ok=True)
    zip_path = os.path.join(root, "buffalo_l.zip")

    try:
        # Intento normal con validación TLS
        urllib.request.urlretrieve(BUFFALO_L_URL, zip_path)
    except Exception:
        # Fallback útil en algunos entornos Windows con verificación de revocación bloqueada
        ctx = ssl._create_unverified_context()
        with urllib.request.urlopen(BUFFALO_L_URL, context=ctx) as resp, open(zip_path, "wb") as out:
            shutil.copyfileobj(resp, out)

    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(pack_dir)
    os.remove(zip_path)

    _filtrar_modelos_compatibles(pack_dir)

    onnx_files = glob.glob(os.path.join(pack_dir, "*.onnx"))
    if not onnx_files:
        print("ERROR: Se descargó buffalo_l, pero no se encontraron archivos .onnx.")
        sys.exit(1)


def _filtrar_modelos_compatibles(pack_dir: str):
    """Deja solo los ONNX compatibles con insightface 0.2.1."""
    permitidos = {"det_10g.onnx", "w600k_r50.onnx"}
    for onnx_path in glob.glob(os.path.join(pack_dir, "*.onnx")):
        nombre = os.path.basename(onnx_path)
        if nombre not in permitidos:
            try:
                os.remove(onnx_path)
            except OSError:
                pass


def resolver_ruta_salida(salida_usuario: str, default_output_file: str) -> str:
    """Si el usuario pasa una carpeta, crea nombre de archivo dentro de ella."""
    salida = os.path.abspath(salida_usuario)
    if os.path.isdir(salida):
        return os.path.join(salida, os.path.basename(default_output_file))
    return salida


def es_video(path: str) -> bool:
    """Devuelve True si la ruta parece un archivo de video por extensión."""
    _, ext = os.path.splitext(path)
    return ext.lower() in VIDEO_EXTS


def asegurar_salida_video(path: str) -> str:
    """Asegura que la salida para video tenga extensión de video válida."""
    base, ext = os.path.splitext(path)
    if ext.lower() in VIDEO_EXTS:
        return path
    return f"{base}.mp4"


def crear_writer_video(output_path: str, fps: float, width: int, height: int):
    """Crea VideoWriter con códec según extensión y con fallback."""
    _, ext = os.path.splitext(output_path)
    ext = ext.lower()

    codec_por_ext = {
        ".avi": "XVID",
        ".mp4": "mp4v",
        ".m4v": "mp4v",
        ".mov": "mp4v",
        ".mkv": "XVID",
        ".webm": "VP80",
    }

    codigos = []
    preferido = codec_por_ext.get(ext, "mp4v")
    codigos.append(preferido)
    for alt in ["mp4v", "XVID", "MJPG"]:
        if alt not in codigos:
            codigos.append(alt)

    for code in codigos:
        fourcc = cv2.VideoWriter_fourcc(*code)
        writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        if writer.isOpened():
            return writer
        writer.release()

    return None


def cargar_modelos(model_path: str):
    """Inicializa el analizador de caras y el modelo swapper."""
    try:
        import insightface
        from insightface.app import FaceAnalysis
    except ImportError:
        print("ERROR: Instala las dependencias con:")
        print("  pip install insightface onnxruntime opencv-python numpy")
        sys.exit(1)

    if not os.path.exists(model_path):
        print(f"ERROR: Modelo no encontrado: {model_path}")
        print(f"Descárgalo desde:\n  {MODEL_URL}")
        print("Y colócalo en la misma carpeta que este script o usa --modelo <ruta>.")
        sys.exit(1)

    asegurar_pack_buffalo_l()

    print("Cargando analizador de caras (buffalo_l)...")
    app = FaceAnalysis(name="buffalo_l")
    app.prepare(ctx_id=-1, det_size=(640, 640))

    print("Cargando modelo swapper...")
    swapper = insightface.model_zoo.get_model(
        model_path, download=False, download_zip=False
    )

    return app, swapper


def detectar_cara(app, imagen: np.ndarray, etiqueta: str):
    """Detecta caras en una imagen y devuelve la de mayor confianza."""
    caras = app.get(imagen)
    if not caras:
        print(f"ERROR: No se detectó ninguna cara en la imagen {etiqueta}.")
        sys.exit(1)
    # Ordenar por puntuación de detección y tomar la mejor
    caras = sorted(caras, key=lambda f: f.det_score, reverse=True)
    print(f"  [{etiqueta}] {len(caras)} cara(s) detectada(s). "
          f"Usando la mejor (confianza: {caras[0].det_score:.2f})")
    return caras[0]


def _normalizar_embedding(face):
    """Devuelve embedding normalizado o None si no está disponible."""
    emb = getattr(face, "embedding", None)
    if emb is None:
        return None
    norm = np.linalg.norm(emb)
    if norm == 0:
        return None
    return emb / norm


def _preparar_fuentes_origen(app, source_path: str, max_sources: int = 0):
    """Carga una o varias caras origen desde archivo o carpeta."""
    if os.path.isdir(source_path):
        archivos = []
        for nombre in sorted(os.listdir(source_path)):
            _, ext = os.path.splitext(nombre)
            if ext.lower() in IMAGE_EXTS:
                archivos.append(os.path.join(source_path, nombre))

        if not archivos:
            print(f"ERROR: No se encontraron imágenes en la carpeta origen: {source_path}")
            sys.exit(1)

        fuentes = []
        for ruta in archivos:
            img = cv2.imread(ruta)
            if img is None:
                continue
            caras = app.get(img)
            if not caras:
                continue
            caras = sorted(caras, key=lambda f: f.det_score, reverse=True)
            cara = caras[0]
            fuentes.append({
                "path": ruta,
                "face": cara,
                "embedding": _normalizar_embedding(cara),
            })

        if not fuentes:
            print("ERROR: No se pudo detectar ninguna cara útil en la carpeta origen.")
            sys.exit(1)

        # Ordena por confianza para priorizar mejores fuentes y permite limitar top_k
        fuentes = sorted(
            fuentes,
            key=lambda x: getattr(x["face"], "det_score", 0.0),
            reverse=True,
        )
        if max_sources and max_sources > 0:
            fuentes = fuentes[:max_sources]

        print(f"Origen carpeta: {source_path}")
        print(f"  Imágenes válidas con cara detectada: {len(fuentes)}/{len(archivos)}")
        return fuentes

    img_cara = cv2.imread(source_path)
    if img_cara is None:
        print(f"ERROR: No se pudo abrir la imagen de cara: {source_path}")
        sys.exit(1)

    print(f"Imagen cara:   {source_path}  ({img_cara.shape[1]}x{img_cara.shape[0]})")
    cara = detectar_cara(app, img_cara, "origen")
    return [{
        "path": source_path,
        "face": cara,
        "embedding": _normalizar_embedding(cara),
    }]


def _elegir_fuente_origen(fuentes_origen, cara_destino):
    """Elige la mejor cara origen usando similitud de embedding cuando sea posible."""
    if len(fuentes_origen) == 1:
        return fuentes_origen[0]["face"]

    emb_dest = _normalizar_embedding(cara_destino)
    if emb_dest is None:
        return fuentes_origen[0]["face"]

    mejor = None
    mejor_score = -1.0
    for fuente in fuentes_origen:
        emb_src = fuente["embedding"]
        if emb_src is None:
            continue
        score = float(np.dot(emb_dest, emb_src))
        if score > mejor_score:
            mejor_score = score
            mejor = fuente

    if mejor is None:
        return fuentes_origen[0]["face"]
    return mejor["face"]


def face_swap_imagen(
    source_path: str,
    target_path: str,
    output_path: str,
    model_path: str,
    max_sources: int = 0,
):
    """
    Toma la cara de source_path y la pone en target_path.

    Args:
        source_path:  Imagen con la cara origen.
        target_path:  Imagen del cuerpo/destino.
        output_path:  Ruta de la imagen resultado.
        model_path:   Ruta al archivo inswapper_128.onnx.
    """
    # --- Cargar imágenes ---
    img_cuerpo = cv2.imread(target_path)

    if img_cuerpo is None:
        print(f"ERROR: No se pudo abrir la imagen de cuerpo: {target_path}")
        sys.exit(1)

    print(f"Imagen cuerpo: {target_path}  ({img_cuerpo.shape[1]}x{img_cuerpo.shape[0]})")

    # --- Modelos ---
    app, swapper = cargar_modelos(model_path)
    fuentes_origen = _preparar_fuentes_origen(app, source_path, max_sources=max_sources)

    # --- Detección ---
    print("\nDetectando caras...")
    cara_destino = detectar_cara(app, img_cuerpo, "destino")
    cara_origen = _elegir_fuente_origen(fuentes_origen, cara_destino)

    # --- Swap ---
    print("\nRealizando intercambio de cara...")
    resultado = img_cuerpo.copy()
    resultado = swapper.get(resultado, cara_destino, cara_origen, paste_back=True)

    # --- Guardar ---
    out_dir = os.path.dirname(output_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    ok = cv2.imwrite(output_path, resultado)
    if not ok:
        print(f"ERROR: No se pudo guardar la imagen en: {output_path}")
        sys.exit(1)

    print(f"\nResultado guardado en: {output_path}")
    return output_path


def face_swap_video(
    source_path: str,
    target_path: str,
    output_path: str,
    model_path: str,
    max_sources: int = 0,
):
    """
    Toma la cara de source_path y la pone en cada frame del video target_path.

    Args:
        source_path:  Imagen con la cara origen.
        target_path:  Video destino.
        output_path:  Ruta del video resultado.
        model_path:   Ruta al archivo inswapper_128.onnx.
    """
    cap = cv2.VideoCapture(target_path)
    if not cap.isOpened():
        print(f"ERROR: No se pudo abrir el video destino: {target_path}")
        sys.exit(1)

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if fps <= 0:
        fps = 25.0

    print(f"Video destino: {target_path}  ({width}x{height} @ {fps:.2f} fps)")

    output_path = asegurar_salida_video(output_path)

    out_dir = os.path.dirname(output_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    writer = crear_writer_video(output_path, fps, width, height)
    if writer is None:
        cap.release()
        print(f"ERROR: No se pudo crear el video de salida: {output_path}")
        sys.exit(1)

    app, swapper = cargar_modelos(model_path)
    fuentes_origen = _preparar_fuentes_origen(app, source_path, max_sources=max_sources)

    print("\nProcesando video...")
    procesados = 0
    con_swap = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        caras = app.get(frame)
        if caras:
            caras = sorted(caras, key=lambda f: f.det_score, reverse=True)
            cara_destino = caras[0]
            cara_origen = _elegir_fuente_origen(fuentes_origen, cara_destino)
            resultado = swapper.get(frame.copy(), cara_destino, cara_origen, paste_back=True)
            con_swap += 1
        else:
            resultado = frame

        writer.write(resultado)
        procesados += 1
        if procesados % 30 == 0:
            if total_frames > 0:
                print(f"  Frames procesados: {procesados}/{total_frames}")
            else:
                print(f"  Frames procesados: {procesados}")

    cap.release()
    writer.release()

    print(f"\nVideo guardado en: {output_path}")
    print(f"Frames totales: {procesados} | Frames con swap: {con_swap}")
    return output_path


def face_swap(
    source_path: str,
    target_path: str,
    output_path: str,
    model_path: str,
    max_sources: int = 0,
):
    """Despacha el proceso a imagen o video según el archivo destino."""
    if es_video(target_path):
        return face_swap_video(
            source_path,
            target_path,
            output_path,
            model_path,
            max_sources=max_sources,
        )
    return face_swap_imagen(
        source_path,
        target_path,
        output_path,
        model_path,
        max_sources=max_sources,
    )


def pedir_ruta(
    mensaje: str,
    debe_existir: bool = True,
    valor_por_defecto: str = "",
    permitir_directorio: bool = False,
) -> str:
    """Pide una ruta al usuario por consola, validando que el archivo exista si se requiere."""
    while True:
        ruta = input(mensaje).strip().strip('"').strip("'")
        if not ruta and valor_por_defecto:
            ruta = valor_por_defecto
        if not ruta:
            print("  La ruta no puede estar vacía. Inténtalo de nuevo.")
            continue
        existe = os.path.isfile(ruta) or (permitir_directorio and os.path.isdir(ruta))
        if debe_existir and not existe:
            tipo = "archivo o carpeta" if permitir_directorio else "archivo"
            print(f"  {tipo.capitalize()} no encontrado: {ruta}. Inténtalo de nuevo.")
            continue
        return ruta


def main():
    default_modelo = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                  "models", "inswapper_128.onnx")

    print("=" * 50)
    print("       Face Swap - Intercambio de Cara")
    print("=" * 50)

    cara = pedir_ruta(
        f"Ruta de la CARA origen (foto o carpeta) [{DEFAULT_CARA_PATH}]: ",
        valor_por_defecto=DEFAULT_CARA_PATH,
        permitir_directorio=True,
    )
    cuerpo = pedir_ruta(
        f"Ruta del archivo destino (imagen o video) [{DEFAULT_CUERPO_PATH}]: ",
        valor_por_defecto=DEFAULT_CUERPO_PATH,
    )

    base, ext = os.path.splitext(cuerpo)
    if es_video(cuerpo):
        default_ext = ext if ext else ".mp4"
    else:
        default_ext = ext if ext else ".jpg"
    default_output = f"{base}_faceswap{default_ext}"
    if DEFAULT_OUTPUT_DIR:
        default_output = os.path.join(DEFAULT_OUTPUT_DIR, os.path.basename(default_output))

    output_input = input(f"Ruta de salida [{default_output}]: ").strip().strip('"').strip("'")
    output = resolver_ruta_salida(output_input if output_input else default_output, default_output)

    modelo_input = input(f"Ruta del modelo inswapper_128.onnx [{default_modelo}]: ").strip().strip('"').strip("'")
    modelo = modelo_input if modelo_input else default_modelo

    max_sources_input = input("Máximo de fotos origen a usar (0=todas, Enter=auto): ").strip()
    if not max_sources_input:
        max_sources = None
    else:
        try:
            max_sources = int(max_sources_input)
        except ValueError:
            print("  Valor inválido para máximo de fotos. Se usará modo auto.")
            max_sources = None

    if max_sources is not None and max_sources < 0:
        max_sources = 0

    max_sources = resolver_max_sources(cara, max_sources)

    print()
    face_swap(
        source_path=cara,
        target_path=cuerpo,
        output_path=output,
        model_path=modelo,
        max_sources=max_sources,
    )


if __name__ == "__main__":
    main()

#F:\ORACLE\NETWORK\tnsnames\abby\fotos\cara.jpeg
#f:\ORACLE\NETWORK\tnsnames\leto\cuerpo.jpeg
#F:\down