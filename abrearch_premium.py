"""
Video Manager - Interfaz Explorer + todas las funcionalidades
"""

import sys
import os
import json
import logging
import subprocess
import time
import random
import threading
import re
import shutil
import traceback
import hashlib
from datetime import datetime
from pathlib import Path
from collections import defaultdict, deque

# Silence noisy FFmpeg/OpenCV stderr decoder messages (e.g. Invalid NAL unit).
# Must be set before importing cv2 anywhere in the process.
os.environ.setdefault("OPENCV_FFMPEG_LOGLEVEL", "-8")
os.environ.setdefault("OPENCV_LOG_LEVEL", "SILENT")
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QFileDialog, QMessageBox, QInputDialog, QFrame,
    QTreeWidget, QTreeWidgetItem, QTableWidget, QTableWidgetItem,
    QLabel, QSplitter, QProgressBar, QHeaderView, QAbstractItemView,
    QMenu, QLineEdit, QDialog, QDialogButtonBox, QScrollArea,
    QComboBox, QSlider, QRubberBand, QSizePolicy, QStyle,
    QListWidget, QListWidgetItem, QSpinBox, QStackedWidget, QPlainTextEdit,
    QGraphicsBlurEffect,
    QListView, QTreeView, QStyleOptionSlider
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, pyqtSlot, QSize, QByteArray
from PyQt6.QtCore import QTimer, QMetaObject, Q_ARG, QRect, QPoint, QEvent
from PyQt6.QtGui import QPixmap, QIcon, QShortcut, QKeySequence, QPainter, QPen, QPainterPath
from PyQt6.QtGui import QColor, QDrag, QBrush, QCursor
from PyQt6.QtCore import QMimeData
from PyQt6.QtCore import QUrl
from PyQt6.QtMultimedia import QMediaPlayer, QAudioOutput, QMediaDevices
from PyQt6.QtMultimediaWidgets import QVideoWidget
from database import VideoDatabase
from video_metadata import VideoMetadata

FFPROBE_PATH = r"D:\projects\FINANZAS\tools\ffprobe.exe"
EXTENSIONES_VIDEO = {'.mp4', '.avi', '.mkv', '.mov', '.wmv', '.flv', '.webm', '.m4v'}
EXTENSIONES_IMAGEN = {'.jpg', '.jpeg', '.png', '.webp', '.bmp'}
HASHES_DB = "video_hashes.json"
LOG_DIR_PATH = Path(__file__).with_name("LOG")
APP_LOG_PATH = LOG_DIR_PATH / "abrearch_premium.log"
FOLDER_VIEWS_LOG_BASENAME = "folder_views"
FOLDER_TREE_ICON_SIZE = 150
PRIVACY_UNLOCK_PASSWORD = "MegaGengar"
MAX_SUGGESTED_THUMBS_PER_FOLDER = 10


def _random_frame_indices(total_frames, count, rng=None):
    """Return unique frame indices in fully random order within [0, total_frames)."""
    total_frames = max(0, int(total_frames))
    if total_frames <= 0:
        return []
    sample_size = min(max(0, int(count)), total_frames)
    if sample_size <= 0:
        return []
    rng = rng or random
    return rng.sample(range(total_frames), sample_size)


def _reorder_random_frames_for_hdd(sample_frames, rng=None, block_frames=240):
    """Keep random sample selection but reorder reads to reduce HDD seeks.

    Strategy: randomize block traversal, then read frames in ascending order inside
    each block so head movement is much lower on mechanical disks.
    """
    if not sample_frames:
        return []
    rng = rng or random
    block_frames = max(1, int(block_frames))
    blocks = defaultdict(list)
    for fn in sample_frames:
        blocks[int(fn) // block_frames].append(int(fn))
    block_ids = list(blocks.keys())
    rng.shuffle(block_ids)
    ordered = []
    for bid in block_ids:
        ordered.extend(sorted(blocks[bid]))
    return ordered


def _setup_logging():
    logger = logging.getLogger("abrearch_premium")
    if logger.handlers:
        return logger
    logger.setLevel(logging.INFO)
    LOG_DIR_PATH.mkdir(parents=True, exist_ok=True)
    handler = logging.FileHandler(APP_LOG_PATH, encoding="utf-8")
    handler.setFormatter(logging.Formatter(
        "%(asctime)s [%(levelname)s] %(threadName)s %(message)s"
    ))
    logger.addHandler(handler)
    logger.propagate = False
    return logger


LOGGER = _setup_logging()


def _handle_unhandled_exception(exc_type, exc_value, exc_traceback):
    if issubclass(exc_type, KeyboardInterrupt):
        sys.__excepthook__(exc_type, exc_value, exc_traceback)
        return
    LOGGER.exception("Unhandled exception", exc_info=(exc_type, exc_value, exc_traceback))
    sys.__excepthook__(exc_type, exc_value, exc_traceback)


sys.excepthook = _handle_unhandled_exception


def _handle_thread_exception(args):
    LOGGER.exception(
        "Unhandled thread exception",
        exc_info=(args.exc_type, args.exc_value, args.exc_traceback),
    )


if hasattr(threading, "excepthook"):
    threading.excepthook = _handle_thread_exception


def _silence_opencv_runtime_logs():
    """Best-effort runtime log suppression for OpenCV across versions."""
    try:
        import cv2
    except Exception:
        return

    # Newer OpenCV builds: cv2.setLogLevel(cv2.LOG_LEVEL_*)
    try:
        if hasattr(cv2, "LOG_LEVEL_SILENT") and hasattr(cv2, "setLogLevel"):
            cv2.setLogLevel(cv2.LOG_LEVEL_SILENT)
            return
    except Exception:
        pass

    # Older OpenCV builds: cv2.utils.logging.setLogLevel(...)
    try:
        if hasattr(cv2, "utils") and hasattr(cv2.utils, "logging"):
            lvl = getattr(cv2.utils.logging, "LOG_LEVEL_SILENT", None)
            if lvl is not None:
                cv2.utils.logging.setLogLevel(lvl)
    except Exception:
        pass


_silence_opencv_runtime_logs()

# ---------------------------------------------------------------------------
# Gender classification (Levi & Hassner Caffe model via OpenCV DNN)
# ---------------------------------------------------------------------------
_GENDER_DIR = Path(__file__).parent / "models"
_GENDER_PROTO = _GENDER_DIR / "gender_deploy.prototxt"
_GENDER_MODEL = _GENDER_DIR / "gender_net.caffemodel"
_GENDER_PROTO_URL = (
    "https://raw.githubusercontent.com/smahesh29/Gender-and-Age-Detection"
    "/master/gender_deploy.prototxt"
)
_GENDER_MODEL_URL = (
    "https://raw.githubusercontent.com/smahesh29/Gender-and-Age-Detection"
    "/master/gender_net.caffemodel"
)
_GENDER_MEAN = (78.4263377603, 87.7689143744, 114.895847746)
_GENDER_LABELS = ["Male", "Female"]
_GENDER_PROTOTXT_CONTENT = """\
name: "CaffeNet"
input: "data"
input_dim: 1
input_dim: 3
input_dim: 227
input_dim: 227
layer { name:"conv1" type:"Convolution" bottom:"data" top:"conv1"
  convolution_param { num_output:96 kernel_size:7 stride:4 } }
layer { name:"relu1" type:"ReLU" bottom:"conv1" top:"conv1" }
layer { name:"pool1" type:"Pooling" bottom:"conv1" top:"pool1"
  pooling_param { pool:MAX kernel_size:3 stride:2 } }
layer { name:"norm1" type:"LRN" bottom:"pool1" top:"norm1"
  lrn_param { local_size:5 alpha:0.0001 beta:0.75 } }
layer { name:"conv2" type:"Convolution" bottom:"norm1" top:"conv2"
  convolution_param { num_output:256 pad:2 kernel_size:5 group:2 } }
layer { name:"relu2" type:"ReLU" bottom:"conv2" top:"conv2" }
layer { name:"pool2" type:"Pooling" bottom:"conv2" top:"pool2"
  pooling_param { pool:MAX kernel_size:3 stride:2 } }
layer { name:"conv3" type:"Convolution" bottom:"pool2" top:"conv3"
  convolution_param { num_output:384 pad:1 kernel_size:3 } }
layer { name:"relu3" type:"ReLU" bottom:"conv3" top:"conv3" }
layer { name:"conv4" type:"Convolution" bottom:"conv3" top:"conv4"
  convolution_param { num_output:384 pad:1 kernel_size:3 group:2 } }
layer { name:"relu4" type:"ReLU" bottom:"conv4" top:"conv4" }
layer { name:"conv5" type:"Convolution" bottom:"conv4" top:"conv5"
  convolution_param { num_output:256 pad:1 kernel_size:3 group:2 } }
layer { name:"relu5" type:"ReLU" bottom:"conv5" top:"conv5" }
layer { name:"pool5" type:"Pooling" bottom:"conv5" top:"pool5"
  pooling_param { pool:MAX kernel_size:3 stride:2 } }
layer { name:"fc6" type:"InnerProduct" bottom:"pool5" top:"fc6"
  inner_product_param { num_output:4096 } }
layer { name:"relu6" type:"ReLU" bottom:"fc6" top:"fc6" }
layer { name:"drop6" type:"Dropout" bottom:"fc6" top:"fc6"
  dropout_param { dropout_ratio:0.5 } }
layer { name:"fc7" type:"InnerProduct" bottom:"fc6" top:"fc7"
  inner_product_param { num_output:4096 } }
layer { name:"relu7" type:"ReLU" bottom:"fc7" top:"fc7" }
layer { name:"drop7" type:"Dropout" bottom:"fc7" top:"fc7"
  dropout_param { dropout_ratio:0.5 } }
layer { name:"fc8" type:"InnerProduct" bottom:"fc7" top:"fc8"
  inner_product_param { num_output:2 } }
layer { name:"prob" type:"Softmax" bottom:"fc8" top:"prob" }
"""
_gender_net_cache = [None]   # mutable singleton so it survives module reload


def _ensure_gender_prototxt():
    import urllib.request
    if _GENDER_PROTO.exists() and _GENDER_PROTO.stat().st_size > 0:
        return
    try:
        urllib.request.urlretrieve(_GENDER_PROTO_URL, _GENDER_PROTO)
    except Exception:
        # Fallback embedded prototxt if remote is unavailable.
        _GENDER_PROTO.write_text(_GENDER_PROTOTXT_CONTENT)


def _refresh_gender_prototxt():
    import urllib.request
    try:
        tmp = _GENDER_PROTO.with_suffix(".tmp")
        urllib.request.urlretrieve(_GENDER_PROTO_URL, tmp)
        tmp.replace(_GENDER_PROTO)
        return True
    except Exception:
        return False


def _get_gender_net():
    """Load (and cache) the gender DNN.  Returns None if caffemodel missing."""
    if _gender_net_cache[0] is not None:
        return _gender_net_cache[0]
    _ensure_gender_prototxt()
    if not _GENDER_MODEL.exists():
        return None
    try:
        import cv2
        import numpy as _np
        net = cv2.dnn.readNetFromCaffe(str(_GENDER_PROTO), str(_GENDER_MODEL))
        # Warm-up forward once to validate proto/model compatibility.
        dummy = cv2.dnn.blobFromImage(
            _np.zeros((227, 227, 3), dtype=_np.uint8),
            1.0, (227, 227), _GENDER_MEAN, swapRB=False)
        net.setInput(dummy)
        _ = net.forward()
        _gender_net_cache[0] = net
        return net
    except Exception:
        # Try one refresh of prototxt in case local file is stale/corrupt.
        try:
            import cv2
            import numpy as _np
            if _refresh_gender_prototxt():
                net = cv2.dnn.readNetFromCaffe(str(_GENDER_PROTO), str(_GENDER_MODEL))
                dummy = cv2.dnn.blobFromImage(
                    _np.zeros((227, 227, 3), dtype=_np.uint8),
                    1.0, (227, 227), _GENDER_MEAN, swapRB=False)
                net.setInput(dummy)
                _ = net.forward()
                _gender_net_cache[0] = net
                return net
        except Exception:
            pass
        return None


def _classify_gender(face_bgr):
    """Return 'Female', 'Male', or None if model unavailable / error."""
    import cv2
    net = _get_gender_net()
    if net is None or face_bgr is None or face_bgr.size == 0:
        return None
    try:
        blob = cv2.dnn.blobFromImage(
            face_bgr, 1.0, (227, 227), _GENDER_MEAN, swapRB=False)
        net.setInput(blob)
        preds = net.forward()[0]          # shape (2,)
        return _GENDER_LABELS[int(preds.argmax())]
    except Exception:
        return None


def _download_gender_model_blocking(progress_cb=None):
    """Download gender_net.caffemodel. progress_cb(bytes_done, total) optional."""
    import urllib.request
    _ensure_gender_prototxt()
    tmp = _GENDER_MODEL.with_suffix(".tmp")
    try:
        _refresh_gender_prototxt()
        def _reporthook(count, block, total):
            if progress_cb and total > 0:
                progress_cb(min(count * block, total), total)
        urllib.request.urlretrieve(_GENDER_MODEL_URL, tmp, _reporthook)
        tmp.rename(_GENDER_MODEL)
        _gender_net_cache[0] = None   # force reload
        return True
    except Exception as e:
        if tmp.exists():
            tmp.unlink()
        LOGGER.warning("Error descargando modelo de género: %s", e)
        return False


class _EmbeddedPlaybackState:
    """Estado ligero para mantener compatibilidad con el flujo actual."""

    def __init__(self, ruta_video):
        self.ruta_video = ruta_video
        self._inicio = time.time()
        self._running = True

    def isRunning(self):
        return self._running

    def detener_reproductor(self):
        self._running = False

    def wait(self):
        return


class ThumbnailThread(QThread):
    """Genera miniaturas con ffmpeg y las guarda en BD."""
    thumbnail_ready = pyqtSignal(str, bytes)  # ruta, jpeg_bytes

    def __init__(self, rutas, ffprobe_path):
        super().__init__()
        self.rutas = rutas
        self._stop_flag = False
        ffmpeg_candidate = ""
        if ffprobe_path:
            ffmpeg_candidate = ffprobe_path.replace('ffprobe', 'ffmpeg')
        if ffmpeg_candidate and Path(ffmpeg_candidate).exists():
            self.ffmpeg_path = ffmpeg_candidate
        else:
            # Fallback to ffmpeg from PATH when paired binary is unavailable.
            self.ffmpeg_path = 'ffmpeg'

    def run(self):
        from concurrent.futures import ThreadPoolExecutor, as_completed

        def _extract_one(ruta):
            if self._stop_flag:
                return None, None
            try:
                cmd = [
                    self.ffmpeg_path, '-y',
                    '-ss', '5',            # seek rápido ANTES de -i
                    '-i', str(ruta),
                    '-frames:v', '1',
                    '-vf', 'scale=640:-1',
                    '-q:v', '2',
                    '-f', 'image2pipe',
                    '-vcodec', 'mjpeg',
                    'pipe:1'
                ]
                result = subprocess.run(cmd, capture_output=True, timeout=12)
                if result.returncode == 0 and result.stdout:
                    return str(ruta), result.stdout
            except Exception:
                pass
            return None, None

        with ThreadPoolExecutor(max_workers=3) as pool:
            futures = {pool.submit(_extract_one, r): r for r in self.rutas}
            for fut in as_completed(futures):
                if self._stop_flag:
                    break
                ruta_str, data = fut.result()
                if ruta_str and data:
                    self.thumbnail_ready.emit(ruta_str, data)


class FolderSuggestionThread(QThread):
    """Busca una miniatura sugerida (cara) por carpeta en segundo plano."""

    progress = pyqtSignal(int, int, str)            # done, total, folder_name
    folder_suggested = pyqtSignal(str, int)         # carpeta_str, suggestions_added
    status = pyqtSignal(str)

    def __init__(self, folder_video_map: dict, max_frames_per_video=120, max_suggested_per_folder=10, parent=None):
        super().__init__(parent)
        self.folder_video_map = {
            str(k): list(v) for k, v in (folder_video_map or {}).items()
        }
        self.max_frames_per_video = max(30, int(max_frames_per_video))
        self.max_suggested_per_folder = max(1, int(max_suggested_per_folder))
        self._stop_flag = False

    def stop(self):
        self._stop_flag = True

    def _find_face_thumb(self, video_path: Path, max_frames: int):
        import cv2
        import random as _rnd

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            cap.release()
            return None
        try:
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            if total <= 1:
                return None
            sample = _random_frame_indices(total, max_frames, rng=_rnd)
            sample = _reorder_random_frames_for_hdd(sample, rng=_rnd)
            cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_alt2.xml"
            cascade = cv2.CascadeClassifier(cascade_path)
            if cascade.empty():
                return None
            for fn in sample:
                if self._stop_flag:
                    return None
                cap.set(cv2.CAP_PROP_POS_FRAMES, int(fn))
                ok, frame = cap.read()
                if not ok or frame is None:
                    continue
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                gray = clahe.apply(gray)
                faces = cascade.detectMultiScale(
                    gray, scaleFactor=1.05, minNeighbors=18, minSize=(100, 100)
                )
                if len(faces) == 0:
                    continue
                x, y, w, h = max(faces, key=lambda r: r[2] * r[3])
                pad = int(max(w, h) * 0.35)
                img_h, img_w = frame.shape[:2]
                x1 = max(0, x - pad)
                y1 = max(0, y - pad)
                x2 = min(img_w, x + w + pad)
                y2 = min(img_h, y + h + pad)
                crop = frame[y1:y2, x1:x2]
                if crop is None or crop.size == 0:
                    continue
                ok_jpg, buf = cv2.imencode('.jpg', crop, [cv2.IMWRITE_JPEG_QUALITY, 88])
                if not ok_jpg:
                    continue
                score = float((x2 - x1) * (y2 - y1))
                return int(fn), bytes(buf.tobytes()), score
        finally:
            cap.release()
        return None

    def run(self):
        db = VideoDatabase()
        import random as _rnd
        items = list(self.folder_video_map.items())
        total = len(items)
        done = 0
        self.status.emit(f"Buscando miniaturas sugeridas en {total} carpetas...")

        for carpeta_str, videos in items:
            if self._stop_flag:
                break
            done += 1
            carpeta_name = Path(carpeta_str).name or carpeta_str
            self.progress.emit(done, total, carpeta_name)
            if not videos:
                continue

            added = 0
            # Barajar todos los videos sin preferencia de nombre.
            all_videos = [Path(v) for v in videos]
            _rnd.shuffle(all_videos)
            for vp in all_videos:
                if self._stop_flag:
                    break
                if not vp.exists() or vp.suffix.lower() not in EXTENSIONES_VIDEO:
                    continue
                found = self._find_face_thumb(vp, self.max_frames_per_video)
                if not found:
                    continue
                frame_no, thumb_bytes, score = found
                try:
                    new_id = db.guardar_miniatura_sugerida_carpeta(
                        carpeta_str,
                        str(vp),
                        int(frame_no),
                        thumb_bytes,
                        score,
                    )
                    if new_id:
                        added += 1
                        try:
                            db.recortar_miniaturas_sugeridas_carpeta(
                                carpeta_str,
                                self.max_suggested_per_folder,
                            )
                        except Exception:
                            pass
                        if added >= self.max_suggested_per_folder:
                            break
                except Exception:
                    continue

            if added > 0:
                self.folder_suggested.emit(carpeta_str, added)

        if self._stop_flag:
            self.status.emit("Búsqueda de miniaturas sugeridas pausada")
        else:
            self.status.emit("Búsqueda de miniaturas sugeridas finalizada")


def _calcular_hash_visual_imagen(ruta_str):
    """Calcula hash perceptual de una imagen y lo guarda en BD. Pensado para background.
    Crea su propia conexión a la BD porque SQLite no permite compartir conexiones entre hilos."""
    try:
        db = VideoDatabase()
        if db.tiene_hash(ruta_str):
            return
        import imagehash
        from PIL import Image
        try:
            size_bytes = os.path.getsize(ruta_str)
        except OSError:
            return
        print(f"Hasheando imagen: {os.path.basename(ruta_str)}")
        try:
            img = Image.open(ruta_str).convert('RGB')
            h = imagehash.phash(img)
            lista = [h.hash.flatten().tolist()]  # una sola "frame" de 64 bits
        except Exception:
            return
        db.guardar_hash_visual(ruta_str, size_bytes, lista)
        LOGGER.info("Hash imagen calculado: %s", os.path.basename(ruta_str))
    except Exception as e:
        LOGGER.warning("Error hasheando imagen %s: %s", ruta_str, e)


def _calcular_hash_archivo(ruta_str):
    """Dispatcher: calcula hash visual para video o imagen según extensión."""
    ext = os.path.splitext(ruta_str)[1].lower()
    if ext in EXTENSIONES_IMAGEN:
        _calcular_hash_visual_imagen(ruta_str)
    else:
        _calcular_hash_visual_background(ruta_str)


def _calcular_hash_visual_background(ruta_str):
    """Calcula hash visual de un video y lo guarda en BD. Pensado para background.
    Crea su propia conexión a la BD porque SQLite no permite compartir conexiones entre hilos."""
    try:
        db = VideoDatabase()
        if db.tiene_hash(ruta_str):
            return
        import cv2
        import numpy as np
        import imagehash
        from PIL import Image
        try:
            size_bytes = os.path.getsize(ruta_str)
        except OSError:
            return
        print(f"Hasheando video: {os.path.basename(ruta_str)}")
        lista = []
        # Suppress FFmpeg/OpenCV NAL-unit noise on stderr while capturing frames.
        _devnull_fd = None
        _old_stderr_fd = None
        try:
            _devnull_fd = os.open(os.devnull, os.O_WRONLY)
            _old_stderr_fd = os.dup(2)
            os.dup2(_devnull_fd, 2)
        except OSError:
            _devnull_fd = _old_stderr_fd = None
        try:
            cap = cv2.VideoCapture(ruta_str, cv2.CAP_ANY)
            tf = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if tf > 15:
                for p in np.linspace(tf * 0.05, tf * 0.95, 15).astype(int):
                    cap.set(cv2.CAP_PROP_POS_FRAMES, int(p))
                    ret, frame = cap.read()
                    if ret and frame is not None:
                        try:
                            frm = cv2.resize(frame, (64, 64))
                            img = Image.fromarray(cv2.cvtColor(frm, cv2.COLOR_BGR2RGB))
                            lista.append(imagehash.phash(img).hash.flatten().tolist())
                        except Exception:
                            pass
            cap.release()
        except Exception:
            pass
        finally:
            # Restore stderr regardless of what happened above.
            if _old_stderr_fd is not None:
                try:
                    os.dup2(_old_stderr_fd, 2)
                    os.close(_old_stderr_fd)
                except OSError:
                    pass
            if _devnull_fd is not None:
                try:
                    os.close(_devnull_fd)
                except OSError:
                    pass
        if len(lista) >= 5:
            db.guardar_hash_visual(ruta_str, size_bytes, lista)
            LOGGER.info("Hash calculado: %s", os.path.basename(ruta_str))
    except Exception as e:
        LOGGER.warning("Error hasheando %s: %s", ruta_str, e)


# ---------------------------------------------------------------------------
# Frame-picker: pick a video frame (with optional crop) and save as cover image
# ---------------------------------------------------------------------------

class _CropLabel(QLabel):
    """QLabel subclass that lets the user draw a crop rectangle."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._origin = QPoint()
        self._rubber = QRubberBand(QRubberBand.Shape.Rectangle, self)
        self._rect = QRect()
        self.setMouseTracking(True)
        self.setCursor(Qt.CursorShape.CrossCursor)

    def crop_rect(self):
        """Return the crop rect in label coordinates, or empty rect if none."""
        return self._rect if self._rect.isValid() else QRect()

    def clear_crop(self):
        self._rect = QRect()
        self._rubber.hide()

    def mousePressEvent(self, ev):
        if ev.button() == Qt.MouseButton.LeftButton:
            self._origin = ev.pos()
            self._rubber.setGeometry(QRect(self._origin, QSize()))
            self._rubber.show()

    def mouseMoveEvent(self, ev):
        if not self._origin.isNull():
            self._rubber.setGeometry(QRect(self._origin, ev.pos()).normalized())

    def mouseReleaseEvent(self, ev):
        if ev.button() == Qt.MouseButton.LeftButton:
            self._rect = QRect(self._origin, ev.pos()).normalized()
            self._rubber.setGeometry(self._rect)
            self._origin = QPoint()


class _SeekSlider(QSlider):
    """Slider that jumps directly to the clicked position on the groove."""

    def mousePressEvent(self, ev):
        if ev.button() == Qt.MouseButton.LeftButton:
            # Use groove geometry (not full widget width) so end clicks can
            # reach the real max value, including the last seconds.
            opt = QStyleOptionSlider()
            self.initStyleOption(opt)
            groove = self.style().subControlRect(
                QStyle.ComplexControl.CC_Slider,
                opt,
                QStyle.SubControl.SC_SliderGroove,
                self,
            )
            px = int(ev.position().x())
            pos = max(0, min(px - groove.x(), max(1, groove.width())))
            value = QStyle.sliderValueFromPosition(
                self.minimum(),
                self.maximum(),
                pos,
                max(1, groove.width()),
                self.invertedAppearance(),
            )
            self.setValue(value)
            self.sliderMoved.emit(value)
            ev.accept()
        super().mousePressEvent(ev)


class _ClickableProgressBar(QProgressBar):
    """Progress bar that emits the clicked horizontal ratio (0..1)."""

    seek_requested = pyqtSignal(float)

    def mousePressEvent(self, ev):
        if ev.button() == Qt.MouseButton.LeftButton:
            w = max(1, self.width())
            frac = min(1.0, max(0.0, float(ev.position().x()) / float(w)))
            self.seek_requested.emit(frac)
            ev.accept()
        super().mousePressEvent(ev)


class _VideoOnlyFullscreenWindow(QWidget):
    """Frameless fullscreen host for video-only mode."""

    closed = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent, Qt.WindowType.Window | Qt.WindowType.FramelessWindowHint)
        self.setStyleSheet("background:#000;")
        self.setMouseTracking(True)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(0)
        self.video_widget = QVideoWidget(self)
        self.video_widget.setStyleSheet("background:#000;")
        self.video_widget.setMouseTracking(True)
        lay.addWidget(self.video_widget)

        # Keep controls docked in the fullscreen layout for reliable visibility on Windows.
        self.controls = QFrame(self)
        self.controls.setStyleSheet(
            "QFrame#fsControls {"
            "background: rgba(15, 15, 15, 200);"
            "border: 1px solid rgba(255,255,255,45);"
            "border-radius: 12px;"
            "}"
            "QPushButton {"
            "background: rgba(255,255,255,26);"
            "color: #fff;"
            "border: 1px solid rgba(255,255,255,55);"
            "border-radius: 8px;"
            "padding: 6px 12px;"
            "font-weight: 600;"
            "}"
            "QPushButton:hover { background: rgba(255,255,255,40); }"
            "QSlider::groove:horizontal { height: 8px; background: rgba(255,255,255,38); border-radius: 4px; }"
            "QSlider::sub-page:horizontal { background: rgba(255,255,255,160); border-radius: 4px; }"
            "QSlider::handle:horizontal { width: 16px; margin: -4px 0; border-radius: 8px; background: #ffffff; }"
            "QLabel { color: #fff; font-weight: 600; }"
        )
        self.controls.setObjectName("fsControls")
        self.controls.setMouseTracking(True)
        self.controls.setMinimumHeight(112)
        self.controls.setMaximumHeight(140)

        ctrl_lay = QVBoxLayout(self.controls)
        ctrl_lay.setContentsMargins(14, 8, 14, 10)
        ctrl_lay.setSpacing(6)

        # Title row
        self.lbl_title = QLabel("")
        self.lbl_title.setStyleSheet(
            "color: #fff; font-size: 13px; font-weight: 700;"
            "background: transparent; padding: 0;"
        )
        self.lbl_title.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        ctrl_lay.addWidget(self.lbl_title)

        # Seek slider row
        seek_row = QHBoxLayout()
        seek_row.setContentsMargins(0, 0, 0, 0)
        seek_row.setSpacing(8)
        self.seek_slider = _SeekSlider(Qt.Orientation.Horizontal)
        self.seek_slider.setRange(0, 0)
        self.seek_slider.setMouseTracking(True)
        seek_row.addWidget(self.seek_slider, 1)
        self.lbl_time = QLabel("00:00 / 00:00")
        self.lbl_time.setFixedWidth(130)
        self.lbl_time.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        seek_row.addWidget(self.lbl_time)
        ctrl_lay.addLayout(seek_row)

        # Buttons row
        btn_row = QHBoxLayout()
        btn_row.setContentsMargins(0, 0, 0, 0)
        btn_row.setSpacing(8)

        self.btn_prev = QPushButton("⏮ Anterior")
        self.btn_prev.setCursor(Qt.CursorShape.PointingHandCursor)
        btn_row.addWidget(self.btn_prev)

        self.btn_play_pause = QPushButton("⏸ Pausa")
        self.btn_play_pause.setCursor(Qt.CursorShape.PointingHandCursor)
        btn_row.addWidget(self.btn_play_pause)

        self.btn_next = QPushButton("⏭ Siguiente")
        self.btn_next.setCursor(Qt.CursorShape.PointingHandCursor)
        btn_row.addWidget(self.btn_next)

        btn_row.addStretch(1)

        self.btn_fav = QPushButton("★ Favorito")
        self.btn_fav.setCursor(Qt.CursorShape.PointingHandCursor)
        btn_row.addWidget(self.btn_fav)

        self.btn_delete = QPushButton("🗑 Eliminar")
        self.btn_delete.setCursor(Qt.CursorShape.PointingHandCursor)
        btn_row.addWidget(self.btn_delete)

        self.lbl_vol = QLabel("🔊")
        btn_row.addWidget(self.lbl_vol)

        self.volume_slider = QSlider(Qt.Orientation.Horizontal)
        self.volume_slider.setRange(0, 100)
        self.volume_slider.setValue(100)
        self.volume_slider.setFixedWidth(130)
        self.volume_slider.setMouseTracking(True)
        btn_row.addWidget(self.volume_slider)

        ctrl_lay.addLayout(btn_row)
        lay.addWidget(self.controls)

        # Auto-hide timer
        self._hide_timer = QTimer(self)
        self._hide_timer.setSingleShot(True)
        self._hide_timer.setInterval(2500)
        self._hide_timer.timeout.connect(self._hide_controls_and_cursor)

        # Mouse-position polling timer — reliable on Windows where QVideoWidget's
        # native surface eats mouse events before they reach Qt's event system.
        self._last_cursor_pos = QCursor.pos()
        self._mouse_poll_timer = QTimer(self)
        self._mouse_poll_timer.setInterval(150)
        self._mouse_poll_timer.timeout.connect(self._poll_mouse_pos)
        self._mouse_poll_timer.start()

        for w in (
            self.video_widget,
            self.controls,
            self.seek_slider,
            self.volume_slider,
            self.btn_fav,
            self.btn_delete,
            self.btn_prev,
            self.btn_play_pause,
            self.btn_next,
        ):
            w.installEventFilter(self)

        self.controls.show()
        self.controls.raise_()
        self._restart_controls_timer()

        QShortcut(QKeySequence("Escape"), self, activated=self._emit_closed)
        QShortcut(QKeySequence("F11"), self, activated=self._emit_closed)

    def _emit_closed(self):
        self.closed.emit()

    def _hide_controls_and_cursor(self):
        self.controls.hide()
        self.setCursor(Qt.CursorShape.BlankCursor)

    def _restart_controls_timer(self):
        self.controls.show()
        self.setCursor(Qt.CursorShape.ArrowCursor)
        self._hide_timer.start()

    def _poll_mouse_pos(self):
        pos = QCursor.pos()
        if pos != self._last_cursor_pos:
            self._last_cursor_pos = pos
            if self.geometry().contains(pos):
                self._restart_controls_timer()

    def _layout_controls(self):
        # Controls are docked in the layout; no manual geometry needed.
        return

    def eventFilter(self, obj, event):
        if event.type() in (QEvent.Type.MouseMove, QEvent.Type.MouseButtonPress, QEvent.Type.Wheel):
            self._restart_controls_timer()
        return super().eventFilter(obj, event)

    def mouseMoveEvent(self, event):
        self._restart_controls_timer()
        super().mouseMoveEvent(event)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._layout_controls()

    def mouseDoubleClickEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.closed.emit()
            event.accept()
            return
        super().mouseDoubleClickEvent(event)

    def closeEvent(self, event):
        self._mouse_poll_timer.stop()
        self.setCursor(Qt.CursorShape.ArrowCursor)
        self.closed.emit()
        super().closeEvent(event)


class FacePickerGrid(QDialog):
    """Show a grid of face-crop thumbnails and let the user select one or many."""

    skip_worker_requested = pyqtSignal(int)

    THUMB = 160  # px per cell

    def __init__(self, results, target_faces=None, target_per_video=None, total_videos=None, parent=None):
        """
        results: list of (frame_bgr, video_path, frame_no, (x1,y1,x2,y2))
        """
        super().__init__(parent)
        self.results = list(results)
        self.chosen = None
        self.chosen_many = []
        self._selected_indices = set()
        self._buttons = []
        self._worker_rows = {}
        self._searching = False
        self._stop_callback = None
        self._target_faces = int(target_faces) if target_faces else 0
        self._target_per_video = int(target_per_video) if target_per_video else 0
        self._total_videos = int(total_videos) if total_videos else 0
        self._videos_done = 0
        self._status_detail = ""
        self.setWindowTitle(f"Elegir cara ({len(self.results)} encontradas)")
        self.setModal(True)
        self._build_ui()

    def _build_ui(self):
        from PyQt6.QtGui import QImage
        lay = QVBoxLayout(self)
        lay.setContentsMargins(16, 16, 16, 16)
        lay.setSpacing(12)
        lay.addWidget(QLabel("Haz clic para marcar una o varias caras y pulsa 'Usar seleccionadas':"))
        self.lbl_status = QLabel("")
        self.lbl_status.setWordWrap(True)
        lay.addWidget(self.lbl_status)

        self._workers_box = QFrame()
        self._workers_box.setStyleSheet("QFrame { background: rgba(255,255,255,0.65); border:1px solid #d6ddea; border-radius:8px; }")
        self._workers_lay = QVBoxLayout(self._workers_box)
        self._workers_lay.setContentsMargins(10, 8, 10, 8)
        self._workers_lay.setSpacing(6)
        lay.addWidget(self._workers_box)

        self.grid_widget = QWidget()
        from PyQt6.QtWidgets import QGridLayout
        self.grid = QGridLayout(self.grid_widget)
        self.grid.setContentsMargins(8, 8, 8, 8)
        self.grid.setHorizontalSpacing(14)
        self.grid.setVerticalSpacing(14)

        scroll = QScrollArea()
        scroll.setWidget(self.grid_widget)
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        cols = 5
        cell = self.THUMB + 14 + self.grid.horizontalSpacing()
        fixed_w = cols * cell + self.grid.contentsMargins().left() + self.grid.contentsMargins().right() + 20
        scroll.setFixedWidth(fixed_w)
        scroll.setMinimumHeight(3 * (self.THUMB + 28) + 24)
        lay.addWidget(scroll)

        self.btn_stop = QPushButton("Parar búsqueda")
        self.btn_stop.clicked.connect(self._stop_search)
        self.btn_use_selected = QPushButton("Usar seleccionadas (0)")
        self.btn_use_selected.setEnabled(False)
        self.btn_use_selected.clicked.connect(self._accept_selection)
        self.cancel = QPushButton("Cancelar")
        self.cancel.clicked.connect(self.reject)
        btn_row = QHBoxLayout()
        btn_row.addWidget(self.btn_stop)
        btn_row.addWidget(self.btn_use_selected)
        btn_row.addStretch()
        btn_row.addWidget(self.cancel)
        lay.addLayout(btn_row)
        self._refresh_status()
        for result in self.results:
            self._append_button(result)

    def set_worker_count(self, count):
        # Reset rows when a new search starts.
        while self._workers_lay.count():
            it = self._workers_lay.takeAt(0)
            w = it.widget()
            if w:
                w.deleteLater()
        self._worker_rows = {}

        for slot in range(1, max(1, int(count)) + 1):
            row = QWidget()
            row_lay = QVBoxLayout(row)
            row_lay.setContentsMargins(0, 0, 0, 0)
            row_lay.setSpacing(2)
            lbl = QLabel(f"Hilo {slot}: en espera")
            lbl.setStyleSheet("font-size:11px; color:#444;")
            bar = QProgressBar()
            bar.setRange(0, 100)
            bar.setValue(0)
            info = QLabel("0/0 frames · 0 caras")
            info.setStyleSheet("font-size:10px; color:#666;")
            btn_skip = QPushButton("Saltar vídeo")
            btn_skip.setToolTip("Hace que este hilo abandone el vídeo actual y siga con el siguiente")
            btn_skip.setFixedHeight(22)
            btn_skip.clicked.connect(lambda _checked=False, s=slot: self.skip_worker_requested.emit(int(s)))
            row_lay.addWidget(lbl)
            row_lay.addWidget(bar)
            row_lay.addWidget(info)
            row_lay.addWidget(btn_skip)
            self._workers_lay.addWidget(row)
            self._worker_rows[slot] = {"label": lbl, "bar": bar, "info": info, "skip_btn": btn_skip}

    @pyqtSlot(int, str, int, int, int, int, int)
    def update_worker_progress(self, slot, video_name, current, total, found, videos_done, videos_started):
        row = self._worker_rows.get(int(slot))
        if not row:
            return
        total = max(1, int(total))
        current = max(0, min(int(current), total))
        row["label"].setText(f"Hilo {slot}: {video_name}")
        row["bar"].setRange(0, total)
        row["bar"].setValue(current)
        if self._target_per_video > 0:
            faces_text = f"{int(found)}/{self._target_per_video} caras"
        else:
            faces_text = f"{int(found)} caras"
        row["info"].setText(
            f"{current}/{total} frames · {faces_text} · "
            f"{int(videos_started)} iniciados · {int(videos_done)} completados"
        )

    @pyqtSlot(int, int, int, int)
    def update_totals(self, found_total, target_faces, videos_done, total_videos):
        self._videos_done = max(0, int(videos_done))
        self._total_videos = max(0, int(total_videos))
        self._target_faces = max(0, int(target_faces))
        if int(found_total) > len(self.results):
            # Defensive fallback: source-of-truth for painted items stays self.results.
            found_total = len(self.results)
        self._refresh_status()

    def _append_button(self, result):
        from PyQt6.QtGui import QImage
        idx = len(self.results) - 1
        frame, vid_path, frame_no, (x1, y1, x2, y2) = result
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            return
        rgb = crop[:, :, ::-1].copy()
        h, w = rgb.shape[:2]
        qi = QImage(rgb.data, w, h, w * 3, QImage.Format.Format_RGB888)
        pm = QPixmap.fromImage(qi).scaled(
            self.THUMB, self.THUMB,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation
        )
        btn = QPushButton()
        btn.setCheckable(True)
        btn.setIcon(QIcon(pm))
        btn.setIconSize(QSize(self.THUMB, self.THUMB))
        btn.setFixedSize(self.THUMB + 14, self.THUMB + 14)
        btn.setToolTip(f"{vid_path.name}  frame {frame_no}")
        btn.setStyleSheet(
            "QPushButton { border:2px solid transparent; border-radius:6px; }"
            "QPushButton:checked { border:2px solid #0a84ff; background:rgba(10,132,255,0.14); }"
        )
        btn.clicked.connect(lambda checked, i=idx: self._toggle_select(i, checked))
        self._buttons.append(btn)
        self.grid.addWidget(btn, idx // 5, idx % 5)

    def _refresh_status(self, status_text=None):
        total = len(self.results)
        self.setWindowTitle(f"Elegir cara ({total} encontradas)")
        if status_text is not None:
            self._status_detail = str(status_text)

        if self._total_videos > 0:
            videos_done = min(self._videos_done, self._total_videos)
            videos_left = max(0, self._total_videos - videos_done)
            videos_text = f"Vídeos: {videos_done}/{self._total_videos} completados · faltan {videos_left}"
        else:
            videos_text = "Vídeos: preparando conteo..."

        if self._target_faces > 0:
            faces_text = f"Caras totales: {total}/{self._target_faces}"
        else:
            faces_text = f"Caras totales: {total}"

        if self._searching:
            default_detail = "Buscando caras..."
        else:
            default_detail = f"Búsqueda terminada. {total} caras disponibles."
        detail = self._status_detail or default_detail
        self.lbl_status.setText(f"{videos_text} · {faces_text}\n{detail}")

    def add_result(self, result, status_text=None):
        self.results.append(result)
        self._append_button(result)
        self._refresh_status(status_text)

    def set_searching(self, searching, status_text=None):
        self._searching = searching
        self.btn_stop.setVisible(searching)
        self.btn_stop.setEnabled(searching)
        for row in self._worker_rows.values():
            btn_skip = row.get("skip_btn")
            if btn_skip is not None:
                btn_skip.setEnabled(bool(searching))
        self._refresh_status(status_text)

    def set_stop_callback(self, callback):
        self._stop_callback = callback

    def _stop_search(self):
        if self._stop_callback:
            self._stop_callback()
        self.set_searching(False, f"Búsqueda detenida. {len(self.results)} caras disponibles.")

    def _toggle_select(self, idx, checked):
        if checked:
            self._selected_indices.add(idx)
        else:
            self._selected_indices.discard(idx)
        self.btn_use_selected.setText(f"Usar seleccionadas ({len(self._selected_indices)})")
        self.btn_use_selected.setEnabled(len(self._selected_indices) > 0)

    def _accept_selection(self):
        if not self._selected_indices:
            return
        ordered = sorted(self._selected_indices)
        self.chosen_many = [self.results[i] for i in ordered if i < len(self.results)]
        self.chosen = self.chosen_many[0] if self.chosen_many else None
        self.accept()


class FaceSearchThread(QThread):
    face_found = pyqtSignal(object)
    status_changed = pyqtSignal(str)
    worker_progress = pyqtSignal(int, str, int, int, int, int, int)
    progress_totals = pyqtSignal(int, int, int, int)

    def __init__(self, videos, max_target, max_per_video, probe_frames, only_female,
                 videos_at_once=10, optimize_for_hdd=True, parent=None):
        super().__init__(parent)
        self.videos = list(videos)
        self.max_target = max_target
        self.max_per_video = max_per_video
        self.probe_frames = probe_frames
        self.only_female = only_female
        self.videos_at_once = max(1, int(videos_at_once))
        self.optimize_for_hdd = bool(optimize_for_hdd)
        self._stop_flag = False
        self.results = []
        self._skip_lock = threading.Lock()
        self._skip_slots = set()

    def stop(self):
        self._stop_flag = True

    @pyqtSlot(int)
    def request_skip_slot(self, slot):
        try:
            slot_num = int(slot)
        except Exception:
            return
        if slot_num <= 0:
            return
        with self._skip_lock:
            self._skip_slots.add(slot_num)

    def _consume_skip_slot(self, slot):
        slot_num = int(slot)
        with self._skip_lock:
            if slot_num in self._skip_slots:
                self._skip_slots.discard(slot_num)
                return True
        return False

    def run(self):
        import cv2
        import random as _rnd
        from concurrent.futures import ThreadPoolExecutor, as_completed
        from threading import Lock, get_ident

        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_alt2.xml"
        test_cascade = cv2.CascadeClassifier(cascade_path)
        if test_cascade.empty():
            self.status_changed.emit("No se encontró el clasificador de caras.")
            return

        videos_pool = list(self.videos)

        total_videos = len(videos_pool)
        workers = max(1, min(self.videos_at_once, total_videos if total_videos > 0 else 1))
        slot_lock = Lock()
        results_lock = Lock()
        totals_lock = Lock()
        slot_map = {}
        videos_done_by_slot = {}
        videos_started_by_slot = {}
        free_slots = list(range(1, workers + 1))
        completed_counter = {"value": 0}

        def _process_video(vid_idx, video_path):
            local_results = []
            face_cascade = cv2.CascadeClassifier(cascade_path)
            if face_cascade.empty() or self._stop_flag:
                return vid_idx, video_path, local_results
            tid = get_ident()
            with slot_lock:
                if tid not in slot_map:
                    slot_map[tid] = free_slots.pop(0) if free_slots else ((len(slot_map) % workers) + 1)
                slot = slot_map[tid]
                videos_done_by_slot.setdefault(slot, 0)
                videos_started_by_slot[slot] = videos_started_by_slot.get(slot, 0) + 1
                started_count = videos_started_by_slot[slot]
            try:
                cap = cv2.VideoCapture(str(video_path))
                total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                if total < 2:
                    cap.release()
                    return vid_idx, video_path, local_results

                # Keep random order; sorting biases detections toward early timestamps
                # when max_per_video is reached quickly.
                sample = _random_frame_indices(total, self.probe_frames, rng=_rnd)
                if self.optimize_for_hdd:
                    sample = _reorder_random_frames_for_hdd(sample, rng=_rnd)
                sample_size = len(sample)
                found_this_video = 0
                scanned = 0
                self.worker_progress.emit(
                    slot, video_path.name, 0, sample_size, 0,
                    videos_done_by_slot.get(slot, 0), started_count
                )

                for fn in sample:
                    if self._stop_flag:
                        break
                    if self._consume_skip_slot(slot):
                        break
                    if found_this_video >= self.max_per_video:
                        break

                    cap.set(cv2.CAP_PROP_POS_FRAMES, fn)
                    ok, frame = cap.read()
                    scanned += 1
                    if not ok:
                        if scanned % 10 == 0 or scanned == sample_size:
                            self.worker_progress.emit(
                                slot, video_path.name, scanned, sample_size,
                                found_this_video, videos_done_by_slot.get(slot, 0), started_count
                            )
                        continue
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                    gray = clahe.apply(gray)
                    faces = face_cascade.detectMultiScale(
                        gray, scaleFactor=1.05, minNeighbors=18, minSize=(100, 100))
                    if len(faces) == 0:
                        continue

                    x, y, w, h = max(faces, key=lambda r: r[2] * r[3])
                    if self.only_female:
                        img_h, img_w = frame.shape[:2]
                        fx1 = max(0, x)
                        fx2 = min(img_w, x + w)
                        fy1 = max(0, y)
                        fy2 = min(img_h, y + h)
                        if _classify_gender(frame[fy1:fy2, fx1:fx2]) != "Female":
                            continue

                    pad = int(max(w, h) * 0.35)
                    img_h, img_w = frame.shape[:2]
                    x1 = max(0, x - pad)
                    y1 = max(0, y - pad)
                    x2 = min(img_w, x + w + pad)
                    y2 = min(img_h, y + h + pad)
                    result = (frame.copy(), video_path, fn, (x1, y1, x2, y2))
                    local_results.append(result)
                    found_this_video += 1
                    should_emit_face = False
                    with results_lock:
                        if not self._stop_flag and len(self.results) < self.max_target:
                            self.results.append(result)
                            should_emit_face = True
                            found_total = len(self.results)
                            if len(self.results) >= self.max_target:
                                self._stop_flag = True
                    if should_emit_face:
                        self.face_found.emit(result)
                        with totals_lock:
                            done_snapshot = completed_counter["value"]
                        self.progress_totals.emit(found_total, self.max_target, done_snapshot, total_videos)
                    self.worker_progress.emit(
                        slot, video_path.name, scanned, sample_size,
                        found_this_video, videos_done_by_slot.get(slot, 0), started_count
                    )
                    if self._stop_flag:
                        break

                cap.release()
                with slot_lock:
                    videos_done_by_slot[slot] = videos_done_by_slot.get(slot, 0) + 1
                    done_count = videos_done_by_slot[slot]
                self.worker_progress.emit(
                    slot, video_path.name, scanned, sample_size, found_this_video,
                    done_count, started_count
                )
            except Exception:
                return vid_idx, video_path, local_results

            return vid_idx, video_path, local_results

        self.status_changed.emit(
            f"Analizando {total_videos} vídeos en paralelo ({workers} a la vez)..."
        )
        self.progress_totals.emit(0, self.max_target, 0, total_videos)

        completed = 0
        executor = ThreadPoolExecutor(max_workers=workers)
        futures = [
            executor.submit(_process_video, vid_idx, video_path)
            for vid_idx, video_path in enumerate(videos_pool, 1)
        ]
        try:
            for fut in as_completed(futures):
                if self._stop_flag:
                    break
                try:
                    vid_idx, video_path, local_results = fut.result()
                except Exception:
                    completed += 1
                    with totals_lock:
                        completed_counter["value"] = completed
                    with results_lock:
                        found_total = len(self.results)
                    self.progress_totals.emit(found_total, self.max_target, completed, total_videos)
                    continue

                completed += 1
                with totals_lock:
                    completed_counter["value"] = completed
                self.status_changed.emit(
                    f"Analizando {video_path.name} ({completed}/{total_videos})... {len(self.results)} caras encontradas"
                )
                with results_lock:
                    found_total = len(self.results)
                self.progress_totals.emit(found_total, self.max_target, completed, total_videos)
                if len(self.results) >= self.max_target:
                    self._stop_flag = True
                    break
        finally:
            if self._stop_flag:
                executor.shutdown(wait=False, cancel_futures=True)
            else:
                executor.shutdown(wait=True)

        if self._stop_flag:
            self.status_changed.emit(f"Búsqueda cancelada. {len(self.results)} caras encontradas.")
        else:
            self.status_changed.emit(f"Búsqueda terminada. {len(self.results)} caras encontradas.")


class FramePickerDialog(QDialog):
    """
    Opens a modal dialog to pick a frame from a video and save it as the
    folder cover image (cover.jpg inside the folder).
    """

    PREVIEW_W = 720
    PREVIEW_H = 405

    def __init__(self, carpeta: Path, videos: list, db=None, parent=None):
        super().__init__(parent)
        self.carpeta = carpeta
        self.videos = videos          # list of Path objects
        self._db = db                 # VideoDatabase o None
        self._cap = None              # cv2.VideoCapture
        self._total_frames = 0
        self._current_frame = None    # numpy array (BGR)
        self._pixmap_orig = None      # QPixmap at native resolution

        self.setWindowTitle("Elegir miniatura de carpeta")
        self.setMinimumSize(780, 600)
        self._build_ui()
        if videos:
            self._load_video(self.videos[0])

    # ------------------------------------------------------------------
    def _build_ui(self):
        lay = QVBoxLayout(self)
        lay.setSpacing(8)

        # --- video selector ---
        top = QHBoxLayout()
        top.addWidget(QLabel("Video:"))
        self.combo_video = QComboBox()
        for v in self.videos:
            try:
                label = str(v.relative_to(self.carpeta)).replace("\\", "/")
            except Exception:
                label = v.name
            self.combo_video.addItem(label, userData=v)
        self.combo_video.currentIndexChanged.connect(self._on_video_changed)
        top.addWidget(self.combo_video, 1)
        lay.addLayout(top)

        # --- preview label (custom for crop) ---
        self.lbl_preview = _CropLabel()
        self.lbl_preview.setFixedSize(self.PREVIEW_W, self.PREVIEW_H)
        self.lbl_preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lbl_preview.setStyleSheet("background:#111;border:1px solid #333;")
        self.lbl_preview.setText("Cargando…")

        scroll = QScrollArea()
        scroll.setWidget(self.lbl_preview)
        scroll.setWidgetResizable(False)
        scroll.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lay.addWidget(scroll, 1)

        # --- frame slider ---
        slider_row = QHBoxLayout()
        self.lbl_frame = QLabel("Frame: 0 / 0")
        self.lbl_frame.setMinimumWidth(110)
        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setMinimum(0)
        self.slider.setMaximum(0)
        self.slider.valueChanged.connect(self._on_slider)
        slider_row.addWidget(self.lbl_frame)
        slider_row.addWidget(self.slider, 1)
        lay.addLayout(slider_row)

        # --- crop hint ---
        hint = QLabel("Arrastra sobre la imagen para seleccionar un área (opcional). "
                      "Si no hay selección se guarda el frame completo.")
        hint.setWordWrap(True)
        hint.setStyleSheet("color:#888;font-size:11px;")
        lay.addWidget(hint)

        # --- buttons ---
        btn_clear = QPushButton("Borrar selección")
        btn_clear.clicked.connect(self._clear_crop)
        self.btn_face = QPushButton("🎲 Cara aleatoria")
        self.btn_face.setToolTip("Buscar un frame con cara detectada y usarlo como miniatura")
        self.btn_face.clicked.connect(self._pick_random_face)
        self.btn_detect = QPushButton("🔍 Buscar cara")
        self.btn_detect.setToolTip("Detectar cara en el frame actual y recortarla")
        self.btn_detect.clicked.connect(self._detect_face_in_current_frame)
        self.btn_all = QPushButton("🎨 Buscar en todos los vídeos")
        self.btn_all.setToolTip("Escanear todos los vídeos y proponer caras")
        self.btn_all.clicked.connect(self._browse_all_faces)

        from PyQt6.QtWidgets import QSpinBox, QLabel as _QLabel
        self.spn_faces = QSpinBox()
        self.spn_faces.setRange(1, 2147483647)
        self.spn_faces.setValue(1000)
        self.spn_faces.setSuffix(" caras")
        self.spn_faces.setToolTip("Número de caras a buscar en total")
        self.spn_per_video = QSpinBox()
        self.spn_per_video.setRange(1, 2147483647)
        self.spn_per_video.setValue(50)
        self.spn_per_video.setSuffix("/video")
        self.spn_per_video.setToolTip("Máximo de caras a tomar de cada video")

        self.spn_probe_frames = QSpinBox()
        self.spn_probe_frames.setRange(30, 5000)
        self.spn_probe_frames.setValue(3000)
        self.spn_probe_frames.setSuffix(" frames")
        self.spn_probe_frames.setToolTip("Cuántos frames se prueban por video (menos = más rápido)")

        self.spn_parallel_videos = QSpinBox()
        self.spn_parallel_videos.setRange(1, 2147483647)
        self.spn_parallel_videos.setValue(2)
        self.spn_parallel_videos.setSuffix(" vídeos")
        self.spn_parallel_videos.setToolTip("Cuántos vídeos se analizan en paralelo")

        self._lbl_faces = _QLabel("Buscar:")
        self._lbl_per_video = _QLabel("Max/video:")
        self._lbl_probe = _QLabel("Muestras:")
        self._lbl_parallel = _QLabel("Videos a la vez:")
        btn_save = QPushButton("Guardar como miniatura")
        btn_save.setDefault(True)
        btn_save.clicked.connect(self._save)
        btn_cancel = QPushButton("Cancelar")
        btn_cancel.clicked.connect(self.reject)

        from PyQt6.QtWidgets import QCheckBox
        self.chk_female = QCheckBox("Solo mujeres")
        self.chk_female.setChecked(True)
        self.chk_female.toggled.connect(self._sync_female_label)
        self.chk_female.setToolTip(
            "Filtrar caras por género (requiere gender_net.caffemodel ~28 MB,\n"
            "se descarga automáticamente la primera vez)")
        self._sync_female_label(self.chk_female.isChecked())

        self.chk_hdd = QCheckBox("⚙ Modo HDD rápido")
        self.chk_hdd.setChecked(True)
        self.chk_hdd.setToolTip(
            "Optimiza lectura para discos mecánicos: menos saltos de cabezal,\n"
            "manteniendo selección aleatoria de frames."
        )

        from PyQt6.QtWidgets import QGridLayout

        controls_card = QFrame()
        controls_card.setStyleSheet(
            "QFrame { background:rgba(255, 255, 255, 0.92); border:1px solid #d6ddea; border-radius:10px; }"
        )
        controls_grid = QGridLayout(controls_card)
        controls_grid.setContentsMargins(14, 12, 14, 12)
        controls_grid.setHorizontalSpacing(10)
        controls_grid.setVerticalSpacing(10)
        controls_grid.setColumnStretch(8, 1)

        controls_grid.addWidget(btn_clear, 0, 0)
        controls_grid.addWidget(self.btn_face, 0, 1)
        controls_grid.addWidget(self.btn_detect, 0, 2)
        controls_grid.addWidget(self.btn_all, 0, 3, 1, 2)
        controls_grid.addWidget(self.chk_hdd, 0, 5, 1, 2)

        controls_grid.addWidget(self._lbl_faces, 1, 0)
        controls_grid.addWidget(self.spn_faces, 1, 1)
        controls_grid.addWidget(self._lbl_per_video, 1, 2)
        controls_grid.addWidget(self.spn_per_video, 1, 3)
        controls_grid.addWidget(self._lbl_probe, 1, 4)
        controls_grid.addWidget(self.spn_probe_frames, 1, 5)
        controls_grid.addWidget(self._lbl_parallel, 1, 6)
        controls_grid.addWidget(self.spn_parallel_videos, 1, 7)
        controls_grid.addWidget(self.chk_female, 1, 8)

        actions_row = QHBoxLayout()
        actions_row.setSpacing(8)
        actions_row.addStretch()
        actions_row.addWidget(btn_cancel)
        actions_row.addWidget(btn_save)
        controls_grid.addLayout(actions_row, 2, 0, 1, 9)

        lay.addWidget(controls_card)

    # ------------------------------------------------------------------
    def _sync_female_label(self, checked):
        if checked:
            self.chk_female.setText("✓ Solo mujeres")
        else:
            self.chk_female.setText("Solo mujeres")

    # ------------------------------------------------------------------
    def _load_video(self, path: Path):
        import cv2
        if self._cap:
            self._cap.release()
        self._cap = cv2.VideoCapture(str(path))
        self._total_frames = max(0, int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT)) - 1)
        self.slider.blockSignals(True)
        self.slider.setMaximum(self._total_frames)
        self.slider.setValue(0)
        self.slider.blockSignals(False)
        self.lbl_preview.clear_crop()
        self._seek(0)

    def _on_video_changed(self, idx):
        path = self.combo_video.itemData(idx)
        if path:
            self._load_video(path)

    def _on_slider(self, value):
        self._seek(value)

    def _seek(self, frame_no):
        import cv2
        if not self._cap or not self._cap.isOpened():
            return
        self._cap.set(cv2.CAP_PROP_POS_FRAMES, frame_no)
        ok, frame = self._cap.read()
        if not ok:
            return
        self._current_frame = frame
        self.lbl_frame.setText(f"Frame: {frame_no} / {self._total_frames}")
        # Convert BGR → RGB then to QPixmap
        rgb = frame[:, :, ::-1].copy()
        h, w, _ = rgb.shape
        from PyQt6.QtGui import QImage
        qi = QImage(rgb.data, w, h, w * 3, QImage.Format.Format_RGB888)
        self._pixmap_orig = QPixmap.fromImage(qi)
        scaled = self._pixmap_orig.scaled(
            self.PREVIEW_W, self.PREVIEW_H,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation
        )
        self.lbl_preview.clear_crop()
        self.lbl_preview.setPixmap(scaled)
        # Store the scale factors for crop mapping
        self._scale_x = w / scaled.width()
        self._scale_y = h / scaled.height()
        # Offset of the pixmap inside the label (centered)
        self._px_offset_x = (self.PREVIEW_W - scaled.width()) // 2
        self._px_offset_y = (self.PREVIEW_H - scaled.height()) // 2

    def _clear_crop(self):
        self.lbl_preview.clear_crop()

    def _maybe_download_gender_model(self):
        """If gender filter is on and model missing, offer to download it. Returns True if ready."""
        if not self.chk_female.isChecked():
            return True   # filter off, no model needed
        if _GENDER_MODEL.exists():
            return True
        reply = QMessageBox.question(
            self, "Modelo de género",
            "Para filtrar por mujeres se necesita descargar el modelo de clasificación "
            "de género (~28 MB).\n¿Descargar ahora?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )
        if reply != QMessageBox.StandardButton.Yes:
            return False
        # Download in foreground with status updates
        btn = self.sender() or self.btn_all
        orig = btn.text() if btn else ""
        if btn:
            btn.setEnabled(False)
            btn.setText("Descargando modelo…")
            QApplication.processEvents()
        ok = _download_gender_model_blocking(
            lambda done, total: (
                btn.setText(f"Descargando… {done//1024//1024}/{total//1024//1024} MB"),
                QApplication.processEvents()
            ) if btn else None
        )
        if btn:
            btn.setEnabled(True)
            btn.setText(orig)
        if not ok:
            QMessageBox.warning(self, "Error",
                                "No se pudo descargar el modelo.\n"
                                "Descárgalo manualmente desde:\n" + _GENDER_MODEL_URL +
                                f"\ny ponlo en:\n{_GENDER_MODEL}")
        return ok

    def _pick_random_face(self):
        """Scan frames randomly to find one with a detected face and jump to it."""
        import cv2
        if not self._cap or not self._cap.isOpened() or self._total_frames < 2:
            QMessageBox.information(self, "Sin vídeo", "Carga un vídeo primero.")
            return
        if not self._maybe_download_gender_model():
            return

        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_alt2.xml"
        face_cascade = cv2.CascadeClassifier(cascade_path)
        if face_cascade.empty():
            QMessageBox.warning(self, "Error", "No se encontró el clasificador de caras.")
            return
        only_female = self.chk_female.isChecked() and _GENDER_MODEL.exists()

        self.btn_face.setEnabled(False)
        self.btn_face.setText("Buscando…")
        QApplication.processEvents()

        import random as _random
        candidates = _random_frame_indices(self._total_frames, 120, rng=_random)
        found_frame = None

        for fn in candidates:
            self._cap.set(cv2.CAP_PROP_POS_FRAMES, fn)
            ok, frame = self._cap.read()
            if not ok:
                continue
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            gray = clahe.apply(gray)
            faces = face_cascade.detectMultiScale(
                gray, scaleFactor=1.05, minNeighbors=18, minSize=(100, 100)
            )
            if len(faces) == 0:
                continue
            x, y, w, h = max(faces, key=lambda r: r[2] * r[3])
            if only_female:
                img_h, img_w = frame.shape[:2]
                fx1, fx2 = max(0, x), min(img_w, x + w)
                fy1, fy2 = max(0, y), min(img_h, y + h)
                gender = _classify_gender(frame[fy1:fy2, fx1:fx2])
                if gender != "Female":
                    continue
            found_frame = fn
            pad = int(max(w, h) * 0.30)
            img_h, img_w = frame.shape[:2]
            x1 = max(0, x - pad);  y1 = max(0, y - pad)
            x2 = min(img_w, x + w + pad);  y2 = min(img_h, y + h + pad)
            self.slider.blockSignals(True)
            self.slider.setValue(fn)
            self.slider.blockSignals(False)
            self._seek(fn)
            sx = 1.0 / self._scale_x;  sy = 1.0 / self._scale_y
            lx1 = int(x1 * sx) + self._px_offset_x
            ly1 = int(y1 * sy) + self._px_offset_y
            lx2 = int(x2 * sx) + self._px_offset_x
            ly2 = int(y2 * sy) + self._px_offset_y
            self.lbl_preview._rect = QRect(QPoint(lx1, ly1), QPoint(lx2, ly2))
            self.lbl_preview._rubber.setGeometry(self.lbl_preview._rect)
            self.lbl_preview._rubber.show()
            break

        self.btn_face.setEnabled(True)
        self.btn_face.setText("🎲 Cara aleatoria")
        if found_frame is None:
            QMessageBox.information(self, "Sin cara",
                                    "No se encontró ninguna cara en el muestreo.\n"
                                    "Prueba a elegir el frame manualmente.")

    def _detect_face_in_current_frame(self):
        """Detect a face in the currently displayed frame and set the crop rect around it."""
        import cv2
        if self._current_frame is None:
            QMessageBox.information(self, "Sin frame", "No hay ningún frame cargado.")
            return
        if not self._maybe_download_gender_model():
            return
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_alt2.xml"
        face_cascade = cv2.CascadeClassifier(cascade_path)
        if face_cascade.empty():
            QMessageBox.warning(self, "Error", "No se encontró el clasificador de caras.")
            return
        only_female = self.chk_female.isChecked() and _GENDER_MODEL.exists()
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray = clahe.apply(cv2.cvtColor(self._current_frame, cv2.COLOR_BGR2GRAY))
        faces = face_cascade.detectMultiScale(
            gray, scaleFactor=1.05, minNeighbors=18, minSize=(100, 100)
        )
        if len(faces) == 0:
            QMessageBox.information(self, "Sin cara",
                                    "No se detectó ninguna cara en este frame.\n"
                                    "Prueba con otro frame o usa 'Cara aleatoria'.")
            return
        img_h, img_w = self._current_frame.shape[:2]
        picked = None
        for fx, fy, fw, fh in sorted(faces, key=lambda r: -(r[2]*r[3])):
            if only_female:
                crop = self._current_frame[max(0,fy):min(img_h,fy+fh),
                                           max(0,fx):min(img_w,fx+fw)]
                if _classify_gender(crop) != "Female":
                    continue
            picked = (fx, fy, fw, fh)
            break
        if picked is None:
            QMessageBox.information(self, "Sin cara de mujer",
                                    "No se detectó ninguna cara femenina en este frame.")
            return
        x, y, w, h = picked
        pad = int(max(w, h) * 0.30)
        x1 = max(0, x - pad);  y1 = max(0, y - pad)
        x2 = min(img_w, x + w + pad);  y2 = min(img_h, y + h + pad)
        sx = 1.0 / self._scale_x;  sy = 1.0 / self._scale_y
        lx1 = int(x1 * sx) + self._px_offset_x
        ly1 = int(y1 * sy) + self._px_offset_y
        lx2 = int(x2 * sx) + self._px_offset_x
        ly2 = int(y2 * sy) + self._px_offset_y
        self.lbl_preview._rect = QRect(QPoint(lx1, ly1), QPoint(lx2, ly2))
        self.lbl_preview._rubber.setGeometry(self.lbl_preview._rect)
        self.lbl_preview._rubber.show()

    def _browse_all_faces(self):
        """Scan all videos in the folder with configurable caps and sampling."""
        import cv2
        if not self._maybe_download_gender_model():
            return
        only_female = self.chk_female.isChecked() and _GENDER_MODEL.exists()

        self.btn_all.setEnabled(False)
        self.btn_all.setText("Buscando…")
        QApplication.processEvents()
        MAX_TARGET = self.spn_faces.value()
        MAX_PER_VIDEO = self.spn_per_video.value()
        PROBE_FRAMES = self.spn_probe_frames.value()
        VIDEOS_AT_ONCE = self.spn_parallel_videos.value()
        OPTIMIZE_FOR_HDD = self.chk_hdd.isChecked()
        videos_for_search = list(self.videos)
        if self._db is not None and videos_for_search:
            # Batch fetch evita N consultas SQL al ordenar (crítico con muchos videos).
            stats_map = self._db.obtener_stats_batch([str(v) for v in videos_for_search])
            def _video_priority(v_path):
                key = str(v_path).replace('\\', '/')
                s = stats_map.get(key)
                if s:
                    return (
                        -int(s.get('reproducciones', 0)),
                        -int(s.get('tiempo_visto_seg', 0)),
                        str(v_path).lower(),
                    )
                return (0, 0, str(v_path).lower())
            videos_for_search.sort(key=_video_priority)

        dlg = FacePickerGrid(
            [],
            target_faces=MAX_TARGET,
            target_per_video=MAX_PER_VIDEO,
            total_videos=len(videos_for_search),
            parent=self,
        )
        dlg.set_worker_count(VIDEOS_AT_ONCE)
        dlg.set_searching(True, "Preparando búsqueda de caras...")
        search_thread = FaceSearchThread(
            videos_for_search, MAX_TARGET, MAX_PER_VIDEO, PROBE_FRAMES, only_female,
            videos_at_once=VIDEOS_AT_ONCE,
            optimize_for_hdd=OPTIMIZE_FOR_HDD,
            parent=self
        )
        dlg.set_stop_callback(search_thread.stop)

        def _update_search_button():
            self.btn_all.setText(f"Buscando… ({len(dlg.results)}/{MAX_TARGET} caras)")

        search_thread.face_found.connect(
            lambda result: (
                dlg.add_result(result),
                _update_search_button()
            )
        )
        search_thread.status_changed.connect(dlg._refresh_status)
        search_thread.worker_progress.connect(dlg.update_worker_progress)
        search_thread.progress_totals.connect(dlg.update_totals)
        dlg.skip_worker_requested.connect(search_thread.request_skip_slot)
        search_thread.status_changed.connect(lambda _text: _update_search_button())
        search_thread.finished.connect(
            lambda: (
                dlg.set_searching(False, f"Búsqueda terminada. {len(dlg.results)} caras disponibles."),
                dlg.reject() if not dlg.results else None
            )
        )
        search_thread.start()

        try:
            accepted = dlg.exec() == QDialog.DialogCode.Accepted and bool(dlg.chosen_many)
        finally:
            if search_thread.isRunning():
                search_thread.stop()
                search_thread.wait()
            self.btn_all.setEnabled(True)
            self.btn_all.setText("🎨 Buscar en todos los vídeos")

        results = dlg.results
        if not results:
            QMessageBox.information(self, "Sin caras",
                                    "No se encontró ninguna cara en los vídeos.")
            return

        if accepted:
            selected_faces = dlg.chosen_many
            frame, vid_path, frame_no, (x1, y1, x2, y2) = selected_faces[0]
            # Switch combo to the chosen video
            for i in range(self.combo_video.count()):
                if self.combo_video.itemData(i) == vid_path:
                    self.combo_video.blockSignals(True)
                    self.combo_video.setCurrentIndex(i)
                    self.combo_video.blockSignals(False)
                    break
            # Load the video and seek to the exact frame
            self._load_video(vid_path)
            self.slider.blockSignals(True)
            self.slider.setValue(frame_no)
            self.slider.blockSignals(False)
            self._seek(frame_no)
            # Set crop rect
            sx = 1.0 / self._scale_x;  sy = 1.0 / self._scale_y
            lx1 = int(x1 * sx) + self._px_offset_x
            ly1 = int(y1 * sy) + self._px_offset_y
            lx2 = int(x2 * sx) + self._px_offset_x
            ly2 = int(y2 * sy) + self._px_offset_y
            self.lbl_preview._rect = QRect(QPoint(lx1, ly1), QPoint(lx2, ly2))
            self.lbl_preview._rubber.setGeometry(self.lbl_preview._rect)
            self.lbl_preview._rubber.show()

            # Save all selected faces directly as folder thumbnails (multi-select workflow)
            saved_count = 0
            if self._db is not None:
                for frame_bgr, v_path, fn, (rx1, ry1, rx2, ry2) in selected_faces:
                    try:
                        crop = frame_bgr[ry1:ry2, rx1:rx2]
                        if crop is None or crop.size == 0:
                            continue
                        ok, buf = cv2.imencode('.jpg', crop, [cv2.IMWRITE_JPEG_QUALITY, 88])
                        if not ok:
                            continue
                        self._db.guardar_miniatura_carpeta(
                            str(self.carpeta),
                            str(v_path),
                            int(fn),
                            buf.tobytes(),
                        )
                        saved_count += 1
                    except Exception:
                        continue
            if saved_count > 0:
                QMessageBox.information(
                    self,
                    "Caras seleccionadas",
                    f"Se guardaron {saved_count} miniaturas de carpeta."
                )

    def _save(self):
        import cv2
        if self._current_frame is None:
            QMessageBox.warning(self, "Sin frame", "No hay ningún frame cargado.")
            return

        frame = self._current_frame  # BGR numpy
        crop = self.lbl_preview.crop_rect()

        if crop.isValid() and not crop.isEmpty():
            # Map label coords → original image coords
            x1 = int((crop.left()   - self._px_offset_x) * self._scale_x)
            y1 = int((crop.top()    - self._px_offset_y) * self._scale_y)
            x2 = int((crop.right()  - self._px_offset_x) * self._scale_x)
            y2 = int((crop.bottom() - self._px_offset_y) * self._scale_y)
            h, w = frame.shape[:2]
            x1, x2 = max(0, x1), min(w, x2)
            y1, y2 = max(0, y1), min(h, y2)
            if x2 > x1 and y2 > y1:
                frame = frame[y1:y2, x1:x2]

        dest = self.carpeta / "cover.jpg"
        try:
            cv2.imwrite(str(dest), frame, [cv2.IMWRITE_JPEG_QUALITY, 92])
        except Exception as e:
            QMessageBox.critical(self, "Error", f"No se pudo guardar:\n{e}")
            return

        # Guardar también en la galería de miniaturas de carpeta
        if self._db is not None:
            try:
                ok, buf = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 88])
                if ok:
                    video_path = self.combo_video.currentData()
                    frame_no = self.slider.value()
                    self._db.guardar_miniatura_carpeta(
                        str(self.carpeta),
                        str(video_path) if video_path else "",
                        frame_no,
                        buf.tobytes()
                    )
            except Exception:
                pass

        self.accept()

    def closeEvent(self, ev):
        if self._cap:
            self._cap.release()
        super().closeEvent(ev)


# ---------------------------------------------------------------------------
# Folder thumbnails gallery
# ---------------------------------------------------------------------------

class _ThumbCard(QFrame):
    """Tarjeta clickable que muestra una miniatura de carpeta."""
    double_clicked = pyqtSignal(dict)
    delete_requested = pyqtSignal(dict)

    _W = 240
    _H = 135

    def __init__(self, thumb_data: dict, parent=None):
        super().__init__(parent)
        self._data = thumb_data
        self.setFixedWidth(self._W + 20)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setStyleSheet(
            "QFrame { background:#f5f5f7; border:1px solid #d0d0d8; border-radius:8px; }"
            "QFrame:hover { border-color:#0a84ff; background:#eef3ff; }"
        )
        vl = QVBoxLayout(self)
        vl.setContentsMargins(8, 8, 8, 6)
        vl.setSpacing(4)

        lbl_img = QLabel()
        lbl_img.setFixedSize(self._W, self._H)
        lbl_img.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lbl_img.setStyleSheet("background:#111; border-radius:4px; border:none;")
        pm = QPixmap()
        pm.loadFromData(QByteArray(thumb_data['thumbnail_blob']))
        if not pm.isNull():
            lbl_img.setPixmap(pm.scaled(
                self._W, self._H,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation
            ))
        vl.addWidget(lbl_img)

        bot = QHBoxLayout()
        bot.setContentsMargins(0, 0, 0, 0)
        name_lbl = QLabel(Path(thumb_data['video_ruta']).name)
        name_lbl.setStyleSheet("font-size:10px; color:#444; border:none;")
        name_lbl.setFixedWidth(self._W - 28)
        bot.addWidget(name_lbl, 1)
        btn_del = QPushButton("✕")
        btn_del.setFixedSize(22, 22)
        btn_del.setStyleSheet(
            "QPushButton { background:transparent; color:#888; border:none; font-size:12px; padding:0; }"
            "QPushButton:hover { color:#d11a2a; }"
        )
        btn_del.clicked.connect(lambda: self.delete_requested.emit(self._data))
        bot.addWidget(btn_del)
        vl.addLayout(bot)

    def mouseDoubleClickEvent(self, ev):
        self.double_clicked.emit(self._data)
        super().mouseDoubleClickEvent(ev)


class FolderThumbnailsDialog(QDialog):
    """Galería de todas las miniaturas asociadas a una carpeta."""

    def __init__(self, carpeta: Path, db, videos_en_carpeta: list, parent=None):
        super().__init__(parent)
        self.carpeta = carpeta
        self.db = db
        self.videos = videos_en_carpeta  # list[Path]
        self.setWindowTitle(f"Miniaturas — {carpeta.name}")
        self.setMinimumSize(760, 480)
        self._build_ui()
        self._load()

    def _build_ui(self):
        from PyQt6.QtWidgets import QGridLayout
        lay = QVBoxLayout(self)
        lay.setSpacing(8)

        self.lbl_info = QLabel()
        self.lbl_info.setStyleSheet("font-weight:bold; font-size:13px;")
        lay.addWidget(self.lbl_info)

        hint = QLabel("Doble clic → abrir en editor de miniatura  ·  ✕ → eliminar de la galería")
        hint.setStyleSheet("color:#888; font-size:11px;")
        lay.addWidget(hint)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        self._grid_widget = QWidget()
        self._grid_layout = QGridLayout(self._grid_widget)
        self._grid_layout.setSpacing(10)
        self._grid_layout.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft)
        scroll.setWidget(self._grid_widget)
        lay.addWidget(scroll, 1)

        foot = QHBoxLayout()
        foot.addStretch()
        btn_close = QPushButton("Cerrar")
        btn_close.clicked.connect(self.accept)
        foot.addWidget(btn_close)
        lay.addLayout(foot)

    def _load(self):
        while self._grid_layout.count():
            item = self._grid_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        thumbs = self.db.obtener_miniaturas_carpeta(str(self.carpeta))
        n = len(thumbs)
        self.lbl_info.setText(
            f"{n} miniatura{'s' if n != 1 else ''} guardada{'s' if n != 1 else ''} "
            f"para \"{self.carpeta.name}\""
        )
        if not thumbs:
            empty = QLabel("No hay miniaturas guardadas aún.\nUsa «Elegir miniatura de carpeta» para añadir.")
            empty.setAlignment(Qt.AlignmentFlag.AlignCenter)
            empty.setStyleSheet("color:#999; font-size:12px;")
            self._grid_layout.addWidget(empty, 0, 0)
            return

        cols = max(1, (self.width() - 40) // (_ThumbCard._W + 30))
        for idx, t in enumerate(thumbs):
            card = _ThumbCard(t)
            card.double_clicked.connect(self._on_dblclick)
            card.delete_requested.connect(self._on_delete)
            self._grid_layout.addWidget(card, idx // cols, idx % cols)

    def _on_dblclick(self, thumb_data: dict):
        video_path = Path(thumb_data['video_ruta'])
        frame_no = int(thumb_data['frame_no'])
        if not video_path.exists():
            QMessageBox.warning(self, "Video no encontrado",
                                f"No se encontró el archivo:\n{video_path}")
            return
        videos = list(self.videos)
        if video_path not in videos:
            videos.insert(0, video_path)
        dlg = FramePickerDialog(self.carpeta, videos, db=self.db, parent=self)
        # Seleccionar el video correcto en el combo
        for i in range(dlg.combo_video.count()):
            if dlg.combo_video.itemData(i) == video_path:
                dlg.combo_video.blockSignals(True)
                dlg.combo_video.setCurrentIndex(i)
                dlg.combo_video.blockSignals(False)
                dlg._load_video(video_path)
                break
        # Ir al frame guardado
        dlg.slider.setValue(frame_no)
        dlg.exec()
        self._load()  # refrescar por si se añadió otra miniatura

    def _on_delete(self, thumb_data: dict):
        reply = QMessageBox.question(
            self, "Eliminar miniatura",
            "¿Eliminar esta miniatura de la galería?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )
        if reply == QMessageBox.StandardButton.Yes:
            self.db.eliminar_miniatura_carpeta(thumb_data['id'])
            self._load()


class SuggestedFolderThumbnailsDialog(QDialog):
    """Lista de miniaturas sugeridas para elegir una como miniatura de carpeta."""

    def __init__(self, carpeta: Path, suggestions: list, db=None, parent=None):
        super().__init__(parent)
        self.carpeta = carpeta
        self.suggestions = list(suggestions or [])
        self.db = db
        self.selected_thumb = None
        self.selected_thumbs = []
        self.want_manual_picker = False
        self.setWindowTitle(f"Miniaturas sugeridas — {carpeta.name}")
        self.setMinimumSize(900, 540)
        self._build_ui()
        self._populate()

    def _build_ui(self):
        lay = QVBoxLayout(self)
        lay.setSpacing(8)

        self.lbl_info = QLabel()
        self.lbl_info.setStyleSheet("font-weight:bold; font-size:13px;")
        lay.addWidget(self.lbl_info)

        hint = QLabel("Selecciona una o varias miniaturas. «Usar seleccionadas» las asigna a la galería (la primera se aplica como portada). «Borrar seleccionadas» las elimina de las sugerencias. Doble clic aplica una.")
        hint.setStyleSheet("color:#888; font-size:11px;")
        hint.setWordWrap(True)
        lay.addWidget(hint)

        self.listw = QListWidget()
        self.listw.setViewMode(QListView.ViewMode.IconMode)
        self.listw.setResizeMode(QListView.ResizeMode.Adjust)
        self.listw.setMovement(QListView.Movement.Static)
        self.listw.setWrapping(True)
        self.listw.setSpacing(10)
        self.listw.setIconSize(QSize(240, 135))
        self.listw.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self.listw.itemDoubleClicked.connect(self._accept_selected)
        lay.addWidget(self.listw, 1)

        foot = QHBoxLayout()
        btn_manual = QPushButton("Editor manual…")
        btn_manual.clicked.connect(self._open_manual)
        foot.addWidget(btn_manual)
        btn_delete = QPushButton("Borrar seleccionadas")
        btn_delete.clicked.connect(self._delete_selected)
        foot.addWidget(btn_delete)
        btn_delete_all = QPushButton("Borrar todas")
        btn_delete_all.clicked.connect(self._delete_all)
        foot.addWidget(btn_delete_all)
        foot.addStretch()
        btn_cancel = QPushButton("Cancelar")
        btn_cancel.clicked.connect(self.reject)
        btn_use = QPushButton("Usar seleccionadas")
        btn_use.setDefault(True)
        btn_use.clicked.connect(self._accept_selected)
        foot.addWidget(btn_cancel)
        foot.addWidget(btn_use)
        lay.addLayout(foot)

    def _populate(self):
        self.listw.clear()
        n = len(self.suggestions)
        self.lbl_info.setText(
            f"{n} miniatura{'s' if n != 1 else ''} sugerida{'s' if n != 1 else ''} para \"{self.carpeta.name}\""
        )
        for s in self.suggestions:
            pm = QPixmap()
            pm.loadFromData(QByteArray(s.get('thumbnail_blob', b'')))
            icon = QIcon(pm.scaled(
                240, 135,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )) if not pm.isNull() else QIcon()
            label = f"{Path(s.get('video_ruta', '')).name}\nframe {int(s.get('frame_no', 0))}"
            it = QListWidgetItem(icon, label)
            it.setData(Qt.ItemDataRole.UserRole, s)
            self.listw.addItem(it)

    def _accept_selected(self):
        selected_items = self.listw.selectedItems()
        if not selected_items:
            QMessageBox.information(self, "Selecciona una miniatura", "Elige al menos una miniatura sugerida primero.")
            return
        self.selected_thumbs = [it.data(Qt.ItemDataRole.UserRole) for it in selected_items]
        self.selected_thumb = self.selected_thumbs[0]  # backward compat
        self.accept()

    def _open_manual(self):
        self.want_manual_picker = True
        self.reject()

    def _delete_selected(self):
        selected_items = self.listw.selectedItems()
        if not selected_items:
            QMessageBox.information(self, "Selecciona una miniatura", "Elige al menos una miniatura sugerida para borrar.")
            return
        if self.db is None:
            QMessageBox.warning(self, "No disponible", "No se pudo borrar la selección.")
            return
        n = len(selected_items)
        reply = QMessageBox.question(
            self,
            "Borrar miniaturas sugeridas",
            f"¿Eliminar {n} miniatura{'s' if n != 1 else ''} sugerida{'s' if n != 1 else ''}?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        if reply != QMessageBox.StandardButton.Yes:
            return
        ids_removed = []
        errors = []
        for it in selected_items:
            s = it.data(Qt.ItemDataRole.UserRole) or {}
            sid = s.get('id')
            if not sid:
                continue
            try:
                self.db.eliminar_miniatura_sugerida_carpeta(int(sid))
                ids_removed.append(int(sid))
            except Exception as e:
                errors.append(str(e))
        self.suggestions = [x for x in self.suggestions if int(x.get('id', -1)) not in ids_removed]
        self._populate()
        if errors:
            QMessageBox.warning(self, "Error", f"No se pudo borrar alguna sugerencia:\n{errors[0]}")

    def _delete_all(self):
        if self.db is None:
            QMessageBox.warning(self, "No disponible", "No se pudo borrar las sugerencias.")
            return
        n = len(self.suggestions)
        if n == 0:
            return
        reply = QMessageBox.question(
            self,
            "Borrar todas las sugeridas",
            f"¿Eliminar las {n} miniaturas sugeridas de esta carpeta?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        if reply != QMessageBox.StandardButton.Yes:
            return
        try:
            self.db.eliminar_todas_sugeridas_carpeta(str(self.carpeta))
            self.suggestions = []
            self._populate()
        except Exception as e:
            QMessageBox.warning(self, "Error", f"No se pudo borrar las sugerencias:\n{e}")


# ---------------------------------------------------------------------------
# Loading spinner
# ---------------------------------------------------------------------------

class _LoadingSpinner(QWidget):
    """Spinner animado dibujado con QPainter. No requiere archivos externos."""

    def __init__(self, size: int = 56, parent=None):
        super().__init__(parent)
        self.setFixedSize(size, size)
        self._angle = 0
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._tick)

    def _tick(self):
        self._angle = (self._angle + 10) % 360
        self.update()

    def showEvent(self, ev):
        super().showEvent(ev)
        self._timer.start(25)

    def hideEvent(self, ev):
        super().hideEvent(ev)
        self._timer.stop()

    def paintEvent(self, ev):
        from PyQt6.QtCore import QRectF
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        margin = 6
        rect = QRectF(margin, margin,
                      self.width() - 2 * margin, self.height() - 2 * margin)
        # Arco de fondo
        pen_bg = QPen(QColor(255, 255, 255, 45))
        pen_bg.setWidth(5)
        pen_bg.setCapStyle(Qt.PenCapStyle.RoundCap)
        p.setPen(pen_bg)
        p.drawEllipse(rect)
        # Arco animado
        pen_fg = QPen(QColor(100, 180, 255))
        pen_fg.setWidth(5)
        pen_fg.setCapStyle(Qt.PenCapStyle.RoundCap)
        p.setPen(pen_fg)
        p.drawArc(rect, int((-self._angle + 90) * 16), int(-270 * 16))
        p.end()


# ---------------------------------------------------------------------------
# Photo Slideshow
# ---------------------------------------------------------------------------

PANEL_MIME = "application/x-photo-panel-id"


class _DragHandle(QPushButton):
    """Botón que inicia un QDrag con el id del panel para reordenar splits."""

    def __init__(self, panel, text="⋮⋮", parent=None):
        super().__init__(text, parent)
        self._panel = panel
        self._press_pos = None
        self.setCursor(Qt.CursorShape.OpenHandCursor)

    def mousePressEvent(self, ev):
        if ev.button() == Qt.MouseButton.LeftButton:
            self._press_pos = ev.position().toPoint()
            self.setCursor(Qt.CursorShape.ClosedHandCursor)
        super().mousePressEvent(ev)

    def mouseMoveEvent(self, ev):
        if self._press_pos is None:
            return super().mouseMoveEvent(ev)
        if (ev.position().toPoint() - self._press_pos).manhattanLength() < QApplication.startDragDistance():
            return super().mouseMoveEvent(ev)
        drag = QDrag(self)
        mime = QMimeData()
        mime.setData(PANEL_MIME, str(id(self._panel)).encode("utf-8"))
        drag.setMimeData(mime)
        try:
            pm = self._panel.grab().scaledToWidth(220, Qt.TransformationMode.SmoothTransformation)
            drag.setPixmap(pm)
            drag.setHotSpot(QPoint(pm.width() // 2, pm.height() // 2))
        except Exception:
            pass
        self._press_pos = None
        self.setCursor(Qt.CursorShape.OpenHandCursor)
        drag.exec(Qt.DropAction.MoveAction)

    def mouseReleaseEvent(self, ev):
        self._press_pos = None
        self.setCursor(Qt.CursorShape.OpenHandCursor)
        super().mouseReleaseEvent(ev)


class _PhotoPanel(QFrame):
    """Panel que muestra fotos y videos de una carpeta con controles propios."""

    # Registro id->panel para resolver drops entre instancias.
    _panel_registry: dict = {}

    _BTN_CSS = (
        "QPushButton { background:#1e1e1e; color:#ccc; border:1px solid #333; "
        "border-radius:4px; padding:1px 8px; font-size:11px; min-height:22px; }"
        "QPushButton:hover { background:#2e2e2e; border-color:#666; }"
    )

    def __init__(self, carpeta: Path, db=None, photo_seconds=5, log_callback=None, parent=None):
        super().__init__(parent)
        self._carpeta = carpeta
        self._db = db
        _PhotoPanel._panel_registry[id(self)] = self
        self.setAcceptDrops(True)
        self._swap_callback = None  # set by PhotoSlideshowWindow
        self._log_callback = log_callback
        self._photo_seconds = max(1, int(photo_seconds))
        self._items: list = []
        self._idx = 0
        self._current_pm = None
        self._current_ruta: Path = None
        self._panel_paused = False
        self._solo_favoritos = False
        self._view_started_at = None
        self._view_elapsed = 0.0
        self._ignore_end_of_media_until = 0.0
        self.setStyleSheet("background:#111;")
        _lay = QVBoxLayout(self)
        _lay.setContentsMargins(0, 0, 0, 0)
        _lay.setSpacing(0)

        # ── content stack ──
        self._stack = QStackedWidget()

        self._img_lbl = QLabel()
        self._img_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._img_lbl.setStyleSheet("background:#111; color:#666;")
        self._img_lbl.setCursor(Qt.CursorShape.PointingHandCursor)
        self._img_lbl.installEventFilter(self)
        self._stack.addWidget(self._img_lbl)           # index 0

        self._video_widget = QVideoWidget()
        self._video_widget.setStyleSheet("background:#111;")
        self._video_widget.setCursor(Qt.CursorShape.PointingHandCursor)
        self._video_widget.installEventFilter(self)
        self._stack.addWidget(self._video_widget)       # index 1

        self._audio = QAudioOutput()
        try:
            self._audio.setDevice(QMediaDevices.defaultAudioOutput())
        except Exception:
            pass
        self._audio.setMuted(True)
        self._player = QMediaPlayer()
        self._player.setAudioOutput(self._audio)
        self._player.setVideoOutput(self._video_widget)
        self._player.mediaStatusChanged.connect(self._on_media_status_changed)

        _lay.addWidget(self._stack, 1)

        # ── caption ──
        self._caption = QLabel(carpeta.name)
        self._caption.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._caption.setStyleSheet(
            "background:rgba(0,0,0,200); color:#bbb; font-size:9px; padding:2px 8px;"
        )
        self._caption.setFixedHeight(18)
        _lay.addWidget(self._caption)

        time_row = QHBoxLayout()
        time_row.setContentsMargins(6, 2, 6, 2)
        time_row.setSpacing(6)
        self._lbl_remaining = QLabel("00:00")
        self._lbl_remaining.setStyleSheet("color:#9aa3ad; font-size:10px;")
        self._lbl_remaining.setFixedWidth(58)
        time_row.addWidget(self._lbl_remaining)
        self._progress = _ClickableProgressBar()
        self._progress.setRange(0, 1000)
        self._progress.setValue(0)
        self._progress.setTextVisible(False)
        self._progress.setFixedHeight(10)
        self._progress.setCursor(Qt.CursorShape.PointingHandCursor)
        self._progress.setStyleSheet(
            "QProgressBar { background:#242424; border:none; border-radius:3px; }"
            "QProgressBar::chunk { background:#5a8fd4; border-radius:3px; }"
            "QProgressBar:hover { background:#2e2e2e; }"
            "QProgressBar:hover::chunk { background:#7aaff0; }"
        )
        self._progress.setToolTip("Haz clic para ir a ese punto del vídeo o foto")
        self._progress.seek_requested.connect(self._on_progress_seek_requested)
        time_row.addWidget(self._progress, 1)
        _lay.addLayout(time_row)

        # ── per-panel controls (overlay on top-left of content area) ──
        OBTN = (
            "QPushButton { background:rgba(15,15,15,190); color:#ccc; border:none; "
            "border-radius:4px; font-size:14px; padding:2px; }"
            "QPushButton:hover { background:rgba(50,80,120,230); }"
            "QPushButton:checked { background:rgba(80,130,200,200); color:#fff; }"
        )
        self._ctrl_overlay = QWidget(self, Qt.WindowType.Window | Qt.WindowType.FramelessWindowHint | Qt.WindowType.Tool | Qt.WindowType.WindowStaysOnTopHint | Qt.WindowType.WindowDoesNotAcceptFocus)
        self._ctrl_overlay.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
        self._ctrl_overlay.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating, True)
        ov_lay = QVBoxLayout(self._ctrl_overlay)
        ov_lay.setContentsMargins(3, 3, 3, 3)
        ov_lay.setSpacing(2)

        self._btn_drag = _DragHandle(self, "⋮⋮")
        self._btn_drag.setToolTip("Arrastra para reordenar este split")
        self._btn_drag.setFixedSize(28, 28)
        self._btn_drag.setStyleSheet(OBTN)
        ov_lay.addWidget(self._btn_drag)

        self._btn_pp = QPushButton("⏸")
        self._btn_pp.setToolTip("Pausar/reanudar este panel")
        self._btn_pp.setFixedSize(28, 28)
        self._btn_pp.setStyleSheet(OBTN)
        self._btn_pp.clicked.connect(self._toggle_panel_pause)
        ov_lay.addWidget(self._btn_pp)

        self._btn_audio = QPushButton("🔇")
        self._btn_audio.setToolTip("Activar/desactivar audio de este split")
        self._btn_audio.setFixedSize(28, 28)
        self._btn_audio.setStyleSheet(OBTN)
        self._btn_audio.clicked.connect(self._toggle_panel_audio)
        ov_lay.addWidget(self._btn_audio)

        self._btn_fav = QPushButton("⭐")
        self._btn_fav.setToolTip("Marcar/quitar favorito (añade «top » al nombre)")
        self._btn_fav.setFixedSize(28, 28)
        self._btn_fav.setStyleSheet(OBTN)
        self._btn_fav.clicked.connect(self._toggle_favorito)
        ov_lay.addWidget(self._btn_fav)

        self._btn_del = QPushButton("🗑")
        self._btn_del.setToolTip("Borrar este archivo del disco")
        self._btn_del.setFixedSize(28, 28)
        self._btn_del.setStyleSheet(OBTN.replace("color:#ccc", "color:#ff6060"))
        self._btn_del.clicked.connect(self._borrar_actual)
        ov_lay.addWidget(self._btn_del)

        self._btn_next = QPushButton("⏭")
        self._btn_next.setToolTip("Siguiente ítem en este split")
        self._btn_next.setFixedSize(28, 28)
        self._btn_next.setStyleSheet(OBTN)
        self._btn_next.clicked.connect(self._next_item_manual)
        ov_lay.addWidget(self._btn_next)

        self._btn_only_fav = QPushButton("★")
        self._btn_only_fav.setCheckable(True)
        self._btn_only_fav.setToolTip("Mostrar solo archivos favoritos (top)")
        self._btn_only_fav.setFixedSize(28, 28)
        self._btn_only_fav.setStyleSheet(OBTN)
        self._btn_only_fav.clicked.connect(self._toggle_solo_favoritos)
        ov_lay.addWidget(self._btn_only_fav)

        self._btn_change_folder = QPushButton("📁")
        self._btn_change_folder.setToolTip("Cambiar carpeta de este split")
        self._btn_change_folder.setFixedSize(28, 28)
        self._btn_change_folder.setStyleSheet(OBTN)
        self._btn_change_folder.clicked.connect(self._on_change_folder)
        ov_lay.addWidget(self._btn_change_folder)

        self._btn_remove_split = QPushButton("✖")
        self._btn_remove_split.setToolTip("Quitar este split")
        self._btn_remove_split.setFixedSize(28, 28)
        self._btn_remove_split.setStyleSheet(OBTN.replace("color:#ccc", "color:#ff9090"))
        self._btn_remove_split.clicked.connect(self._on_remove_split)
        ov_lay.addWidget(self._btn_remove_split)

        self._ctrl_overlay.adjustSize()
        self._ctrl_overlay.raise_()

        # callbacks set externally by PhotoSlideshowWindow
        self._change_folder_callback = None
        self._remove_split_callback = None

        self._sync_audio_button_text()
        self._load_items()

    # ── helpers ──

    def _load_items(self):
        exts = EXTENSIONES_IMAGEN | EXTENSIONES_VIDEO
        try:
            self._items = sorted(
                [f for f in self._carpeta.rglob("*")
                 if f.is_file() and f.suffix.lower() in exts],
                key=lambda p: p.name.lower(),
            )
        except (PermissionError, OSError):
            self._items = []
        random.shuffle(self._items)
        self._idx = 0

    def _is_image(self, ruta: Path):
        return ruta.suffix.lower() in EXTENSIONES_IMAGEN

    def _is_video(self, ruta: Path):
        return ruta.suffix.lower() in EXTENSIONES_VIDEO

    def _is_top(self, ruta: Path):
        return ruta.name.lower().startswith("top ")

    def _items_visibles(self):
        if not self._solo_favoritos:
            return self._items
        return [it for it in self._items if self._is_top(it)]

    def _is_viewed(self, ruta: Path):
        if self._is_image(ruta):
            return ruta.name.lower().startswith("rwd ") or ruta.name.lower().startswith("top rwd ")
        return ruta.stem.lower().endswith("_rwd")

    def _photo_parse_prefixes(self, ruta: Path):
        """Return (core_stem, is_top, is_rwd) for photo names with front prefixes."""
        stem = ruta.stem
        low = stem.lower()
        is_top = False
        is_rwd = False
        changed = True
        while changed:
            changed = False
            if low.startswith("top "):
                is_top = True
                stem = stem[4:]
                low = stem.lower()
                changed = True
            if low.startswith("rwd "):
                is_rwd = True
                stem = stem[4:]
                low = stem.lower()
                changed = True
        return stem, is_top, is_rwd

    def _photo_build_name(self, core_stem: str, suffix: str, is_top: bool, is_rwd: bool):
        parts = []
        if is_top:
            parts.append("top")
        if is_rwd:
            parts.append("rwd")
        parts.append(core_stem)
        return " ".join(parts) + suffix

    def _pause_view_clock(self):
        if self._current_ruta and self._view_started_at is not None:
            self._view_elapsed += max(0.0, time.time() - self._view_started_at)
            self._view_started_at = None

    def _resume_view_clock(self):
        if self._current_ruta and self._view_started_at is None and not self._panel_paused:
            self._view_started_at = time.time()

    def _rename_path_and_sync_db(self, old_path: Path, new_path: Path):
        if new_path == old_path:
            return old_path
        if new_path.exists():
            return old_path
        last_err = None
        for _ in range(12):
            try:
                old_path.rename(new_path)
                last_err = None
                break
            except OSError as e:
                last_err = e
                QApplication.processEvents()
                time.sleep(0.08)
        if last_err is not None:
            raise last_err
        for i, it in enumerate(self._items):
            if it == old_path:
                self._items[i] = new_path
                break
        if self._db:
            self._db.renombrar_ruta(old_path, new_path)
        if self._current_ruta == old_path:
            self._current_ruta = new_path
        return new_path

    def _mark_as_viewed(self, ruta: Path):
        if not ruta or not ruta.exists() or self._is_viewed(ruta):
            return ruta
        try:
            if self._is_image(ruta):
                core, is_top, _ = self._photo_parse_prefixes(ruta)
                nueva = ruta.with_name(self._photo_build_name(core, ruta.suffix, is_top, True))
                return self._rename_path_and_sync_db(ruta, nueva)
            if self._is_video(ruta):
                base = ruta.stem
                c = 0
                nueva = ruta.with_name(f"{base}_rwd{ruta.suffix}")
                while nueva.exists():
                    c += 1
                    nueva = ruta.with_name(f"{base}_rwd_{c}{ruta.suffix}")
                return self._rename_path_and_sync_db(ruta, nueva)
        except OSError:
            return ruta
        return ruta

    def _finalize_current_view(self):
        if not self._current_ruta:
            return
        self._pause_view_clock()
        ruta = self._current_ruta
        elapsed = max(1, int(round(self._view_elapsed)))
        if self._db:
            try:
                self._db.registrar_visualizacion(str(ruta), elapsed)
            except Exception:
                pass
        if self._log_callback:
            try:
                self._log_callback(str(ruta.parent), elapsed)
            except Exception:
                pass
        nueva = self._mark_as_viewed(ruta)
        self._current_ruta = nueva
        self._view_elapsed = 0.0
        self._view_started_at = None

    def eventFilter(self, obj, event):
        if obj in (self._img_lbl, self._video_widget):
            if event.type() == QEvent.Type.MouseButtonPress and event.button() == Qt.MouseButton.LeftButton:
                self._toggle_panel_pause()
                event.accept()
                return True
        return super().eventFilter(obj, event)

    def _on_media_status_changed(self, status):
        # Videos do not obey the photo interval; advance only when they end.
        if time.time() < self._ignore_end_of_media_until:
            return
        if status == QMediaPlayer.MediaStatus.EndOfMedia and not self._panel_paused:
            self.avanzar()

    def _stop_video_without_auto_advance(self):
        """Stop current video without triggering EndOfMedia-driven next."""
        self._ignore_end_of_media_until = time.time() + 0.8
        self._player.stop()

    def _release_video_file_handle(self):
        """Ensure file handle is released before rename on Windows."""
        self._stop_video_without_auto_advance()
        self._player.setSource(QUrl())
        QApplication.processEvents()
        time.sleep(0.05)
        QApplication.processEvents()

    def _fmt_sec(self, sec):
        sec = max(0, int(sec))
        m, s = divmod(sec, 60)
        return f"{m:02d}:{s:02d}"

    def _current_elapsed_seconds(self):
        if not self._current_ruta:
            return 0.0
        elapsed = self._view_elapsed
        if self._view_started_at is not None:
            elapsed += max(0.0, time.time() - self._view_started_at)
        return elapsed

    def _update_timer_ui(self):
        if not self._current_ruta:
            self._lbl_remaining.setText("00:00")
            self._progress.setValue(0)
            return

        if self._is_video(self._current_ruta) and self._stack.currentIndex() == 1:
            dur_ms = max(0, int(self._player.duration()))
            pos_ms = max(0, int(self._player.position()))
            if dur_ms > 0:
                rem = max(0, (dur_ms - pos_ms) / 1000.0)
                frac = min(1.0, max(0.0, pos_ms / dur_ms))
                self._lbl_remaining.setText(self._fmt_sec(rem))
                self._progress.setValue(int(frac * 1000))
                return

        total = float(self._photo_seconds)
        elapsed = self._current_elapsed_seconds()
        rem = max(0.0, total - elapsed)
        frac = min(1.0, max(0.0, elapsed / total))
        self._lbl_remaining.setText(self._fmt_sec(rem))
        self._progress.setValue(int(frac * 1000))

    def _on_progress_seek_requested(self, frac):
        """Seek within current video or photo countdown when user clicks timeline bar."""
        frac = min(1.0, max(0.0, frac))
        if not self._current_ruta:
            return
        if self._is_video(self._current_ruta):
            dur = max(0, int(self._player.duration()))
            if dur <= 0:
                return
            self._player.setPosition(int(dur * frac))
        else:
            # For photos: jump to that fraction of the photo display time.
            total = float(self._photo_seconds)
            self._view_elapsed = total * frac
            self._view_started_at = time.time() if not self._panel_paused else None
        self._update_timer_ui()

    def _update_fav_button(self):
        is_fav = bool(
            self._current_ruta
            and self._is_top(self._current_ruta)
        )
        self._btn_fav.setText("💔" if is_fav else "⭐")
        self._btn_fav.setToolTip(
            "Quitar de favoritos (elimina «top »)"
            if is_fav else
            "Marcar como favorito (añade «top »)"
        )

    def _set_caption(self, ruta: Path):
        prefix = "🎬 " if ruta.suffix.lower() in EXTENSIONES_VIDEO else ""
        top_mark = "⭐ " if self._is_top(ruta) else ""
        self._caption.setText(
            f"  {top_mark}{self._carpeta.name}  ·  {prefix}{ruta.name}  "
        )

    # ── public ──

    def avanzar(self):
        """Muestra el siguiente ítem. No hace nada si el panel está pausado."""
        if self._panel_paused:
            return
        self._finalize_current_view()
        self._stop_video_without_auto_advance()
        visibles = self._items_visibles()
        if not visibles:
            self._stack.setCurrentIndex(0)
            if self._solo_favoritos:
                self._img_lbl.setText(f"⭐  Sin favoritos\n{self._carpeta.name}")
            else:
                self._img_lbl.setText(f"📂  Sin contenido\n{self._carpeta.name}")
            self._current_ruta = None
            self._view_elapsed = 0.0
            self._view_started_at = None
            self._update_timer_ui()
            self._update_fav_button()
            return
        for _ in range(len(self._items)):
            ruta = self._items[self._idx % len(self._items)]
            self._idx += 1
            if self._solo_favoritos and not self._is_top(ruta):
                continue
            if ruta.suffix.lower() in EXTENSIONES_IMAGEN:
                pm = QPixmap(str(ruta))
                if not pm.isNull():
                    self._current_pm = pm
                    self._current_ruta = ruta
                    self._set_caption(ruta)
                    self._stack.setCurrentIndex(0)
                    self._render()
                    self._view_elapsed = 0.0
                    self._resume_view_clock()
                    self._update_timer_ui()
                    self._update_fav_button()
                    QTimer.singleShot(0, self._ctrl_overlay.raise_)
                    return
            elif ruta.suffix.lower() in EXTENSIONES_VIDEO:
                self._current_ruta = ruta
                self._set_caption(ruta)
                self._stack.setCurrentIndex(1)
                self._player.setSource(QUrl.fromLocalFile(str(ruta)))
                self._player.play()
                self._view_elapsed = 0.0
                self._resume_view_clock()
                self._update_timer_ui()
                self._update_fav_button()
                QTimer.singleShot(0, self._ctrl_overlay.raise_)
                return
        self._img_lbl.setText("⚠  Sin contenido válido")
        self._current_ruta = None
        self._view_elapsed = 0.0
        self._view_started_at = None
        self._update_timer_ui()

    def _toggle_solo_favoritos(self):
        self._solo_favoritos = self._btn_only_fav.isChecked()
        self._btn_only_fav.setText("solo favoritos ✓" if self._solo_favoritos else "solo favoritos")
        self.avanzar()

    def _on_change_folder(self):
        if callable(self._change_folder_callback):
            self._change_folder_callback(self)

    def _on_remove_split(self):
        if callable(self._remove_split_callback):
            self._remove_split_callback(self)

    def _next_item_manual(self):
        """Manual next for this panel only."""
        paused_backup = self._panel_paused
        self._panel_paused = False
        self.avanzar()
        self._panel_paused = paused_backup

    def avanzar_si_toca(self, segundos: int):
        """Advance only photos by timer; videos advance on EndOfMedia."""
        self._update_timer_ui()
        if self._panel_paused or not self._current_ruta:
            return
        if self._is_video(self._current_ruta):
            return
        elapsed = self._current_elapsed_seconds()
        if elapsed >= max(1, int(segundos)):
            self.avanzar()

    def stop_player(self):
        self._finalize_current_view()
        self._stop_video_without_auto_advance()

    # ── per-panel actions ──

    def _toggle_panel_pause(self):
        self._panel_paused = not self._panel_paused
        if self._panel_paused:
            self._btn_pp.setText("▶")
            self._pause_view_clock()
            self._player.pause()
        else:
            self._btn_pp.setText("⏸")
            self._resume_view_clock()
            if self._stack.currentIndex() == 1:
                self._player.play()

    def _sync_audio_button_text(self):
        if self._audio.isMuted():
            self._btn_audio.setText("🔇")
            self._btn_audio.setToolTip("Activar audio de este split")
        else:
            self._btn_audio.setText("🔊")
            self._btn_audio.setToolTip("Silenciar audio de este split")

    def _toggle_panel_audio(self):
        self._audio.setMuted(not self._audio.isMuted())
        self._sync_audio_button_text()

    def _toggle_favorito(self):
        if not self._current_ruta or not self._current_ruta.exists():
            return
        ruta = self._current_ruta
        if self._is_image(ruta):
            core, is_top, is_rwd = self._photo_parse_prefixes(ruta)
            nueva = ruta.with_name(self._photo_build_name(core, ruta.suffix, not is_top, is_rwd))
            is_fav = is_top
        else:
            is_fav = self._is_top(ruta)
            new_name = ruta.name[4:] if is_fav else f"top {ruta.name}"
            nueva = ruta.with_name(new_name)
        if nueva.exists():
            QMessageBox.warning(self, "Favorito", f"Ya existe:\n{nueva.name}")
            return
        try:
            # If this is the video currently playing, stop it first
            if self._stack.currentIndex() == 1:
                self._release_video_file_handle()
            self._current_ruta = self._rename_path_and_sync_db(ruta, nueva)
            if self._db:
                self._db.marcar_favorito(str(self._current_ruta), not is_fav)
            self._set_caption(self._current_ruta)
            self._update_fav_button()
            # Resume video from new path if it was playing
            if self._stack.currentIndex() == 1 and not self._panel_paused:
                self._player.setSource(QUrl.fromLocalFile(str(self._current_ruta)))
                self._player.play()
        except OSError as e:
            QMessageBox.warning(self, "Error al renombrar", str(e))

    def _borrar_actual(self):
        if not self._current_ruta or not self._current_ruta.exists():
            return
        ruta = self._current_ruta
        _MBOX_CSS = (
            "QMessageBox { background:#1b1f2a; }"
            "QMessageBox QLabel { color:#e8eeff; font-size:11pt; background:transparent; }"
            "QMessageBox QPushButton { background:#2a3c6e; color:#fff; border:1px solid #3a4f86; "
            "border-radius:6px; padding:6px 16px; min-width:70px; font-weight:600; }"
            "QMessageBox QPushButton:hover { background:#3a5298; }"
        )
        box = QMessageBox(self)
        box.setStyleSheet(_MBOX_CSS)
        box.setWindowTitle("Borrar archivo")
        box.setIcon(QMessageBox.Icon.Question)
        box.setText(f"¿Eliminar permanentemente?\n\n{ruta.name}")
        box.setStandardButtons(QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
        box.setDefaultButton(QMessageBox.StandardButton.No)
        if box.exec() != QMessageBox.StandardButton.Yes:
            return
        try:
            self._finalize_current_view()
            ruta = self._current_ruta or ruta
            self._release_video_file_handle()
            ruta.unlink()
            if ruta in self._items:
                self._items.remove(ruta)
            self._current_ruta = None
            # Advance to next item immediately (ignores pause so user sees something)
            paused_backup = self._panel_paused
            self._panel_paused = False
            self.avanzar()
            self._panel_paused = paused_backup
        except OSError as e:
            err = QMessageBox(self)
            err.setStyleSheet(_MBOX_CSS)
            err.setIcon(QMessageBox.Icon.Warning)
            err.setWindowTitle("Error al borrar")
            err.setText(str(e))
            err.exec()

    # ── render ──

    def _render(self):
        if (self._current_pm and not self._current_pm.isNull()
                and self._img_lbl.width() > 0 and self._img_lbl.height() > 0):
            scaled = self._current_pm.scaled(
                self._img_lbl.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
            self._img_lbl.setPixmap(scaled)

    def dragEnterEvent(self, ev):
        if ev.mimeData().hasFormat(PANEL_MIME):
            ev.acceptProposedAction()
            self.setStyleSheet("background:#111; border:3px solid #4a8af0;")
        else:
            ev.ignore()

    def dragLeaveEvent(self, ev):
        self.setStyleSheet("background:#111;")
        super().dragLeaveEvent(ev)

    def dropEvent(self, ev):
        self.setStyleSheet("background:#111;")
        if not ev.mimeData().hasFormat(PANEL_MIME):
            ev.ignore()
            return
        try:
            src_id = int(bytes(ev.mimeData().data(PANEL_MIME)).decode("utf-8"))
        except Exception:
            ev.ignore()
            return
        src = _PhotoPanel._panel_registry.get(src_id)
        if src is None or src is self:
            ev.ignore()
            return
        if callable(self._swap_callback):
            self._swap_callback(src, self)
        ev.acceptProposedAction()

    def resizeEvent(self, ev):
        super().resizeEvent(ev)
        self._render()
        self._reposition_overlay()

    def moveEvent(self, ev):
        super().moveEvent(ev)
        self._reposition_overlay()

    def showEvent(self, ev):
        super().showEvent(ev)
        if hasattr(self, "_ctrl_overlay"):
            self._ctrl_overlay.show()
        self._reposition_overlay()

    def hideEvent(self, ev):
        super().hideEvent(ev)
        if hasattr(self, "_ctrl_overlay"):
            self._ctrl_overlay.hide()

    def _reposition_overlay(self):
        if not hasattr(self, "_ctrl_overlay"):
            return
        try:
            top_left = self.mapToGlobal(QPoint(0, 0))
            self._ctrl_overlay.move(top_left)
            self._ctrl_overlay.adjustSize()
            self._ctrl_overlay.raise_()
        except Exception:
            pass


class PhotoSlideshowWindow(QWidget):
    """Ventana con N paneles en grid que muestran fotos/vídeos rotando cada N segundos.

    Layout rules:
      • 1-2 carpetas → 1 fila, N columnas.
      • 3-4 carpetas → 2 filas, ceil(N/2) columnas.
      • 5+ carpetas  → rows = ceil(sqrt(N)), cols = ceil(N/rows).
    Cada celda tiene un tamaño fijo e igual: los paneles no se redimensionan
    al cambiar de foto ni de vídeo.
    """

    def __init__(self, carpetas: list, segundos: int, db=None, log_callback=None, parent=None):
        super().__init__(parent)
        self._carpetas = [Path(c) for c in carpetas]
        self._segundos = max(1, int(segundos))
        self._db = db
        self._log_callback = log_callback
        self._panels: list = []
        self._paused = False
        self.setWindowTitle("Presentación de Fotos")
        self.setStyleSheet("background:#000;")
        self.resize(1400, 840)
        self._build_ui()
        self._timer = QTimer(self)
        self._timer.setInterval(200)
        self._timer.timeout.connect(self._tick)
        self._avanzar_todos()   # primera imagen inmediata
        self._timer.start()

    @staticmethod
    def _grid_dims(n: int):
        """Return (rows, cols) for n panels. Maximum 2 rows."""
        import math
        if n <= 1:
            return 1, 1
        rows = min(2, n)
        cols = math.ceil(n / rows)
        return rows, cols

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        # ── splitters: outer vertical (rows), each row horizontal (cols) ──
        splitter_css = (
            "QSplitter::handle { background:#222; }"
            "QSplitter::handle:horizontal { width:4px; }"
            "QSplitter::handle:vertical   { height:4px; }"
            "QSplitter::handle:hover { background:#3a6ec0; }"
        )
        rows, cols = self._grid_dims(len(self._carpetas))
        outer = QSplitter(Qt.Orientation.Vertical)
        outer.setStyleSheet(splitter_css)
        outer.setChildrenCollapsible(False)
        outer.setHandleWidth(4)
        self._outer_splitter = outer
        self._row_splitters = []
        for r in range(rows):
            row_split = QSplitter(Qt.Orientation.Horizontal)
            row_split.setStyleSheet(splitter_css)
            row_split.setChildrenCollapsible(False)
            row_split.setHandleWidth(4)
            self._row_splitters.append(row_split)
            outer.addWidget(row_split)
        for i, c in enumerate(self._carpetas):
            panel = _PhotoPanel(
                c,
                db=self._db,
                photo_seconds=self._segundos,
                log_callback=self._log_callback,
            )
            panel.setSizePolicy(
                QSizePolicy.Policy.Ignored,
                QSizePolicy.Policy.Ignored,
            )
            panel._swap_callback = self._swap_panels
            panel._change_folder_callback = self._change_panel_folder
            panel._remove_split_callback = self._remove_panel
            self._panels.append(panel)
            self._row_splitters[i // cols].addWidget(panel)

        # Equal initial sizes
        for rs in self._row_splitters:
            rs.setSizes([1] * rs.count())
        outer.setSizes([1] * outer.count())

        root.addWidget(outer, 1)

        # ── bottom bar ──
        _btn_css = (
            "QPushButton { background:#2a2a2a; color:#ddd; border:1px solid #444; "
            "border-radius:6px; padding:3px 14px; font-size:10px; }"
            "QPushButton:hover { background:#3a3a3a; border-color:#888; }"
        )
        bar = QWidget()
        bar.setFixedHeight(40)
        bar.setStyleSheet("background:rgba(12,12,12,230);")
        bar_lay = QHBoxLayout(bar)
        bar_lay.setContentsMargins(10, 4, 10, 4)
        bar_lay.setSpacing(8)

        self._btn_pause = QPushButton("⏸ Pausa todo")
        self._btn_pause.setStyleSheet(_btn_css)
        self._btn_pause.clicked.connect(self._toggle_pause)
        bar_lay.addWidget(self._btn_pause)

        btn_fs = QPushButton("⛶ Pantalla completa")
        btn_fs.setStyleSheet(_btn_css)
        btn_fs.clicked.connect(self._toggle_fullscreen)
        bar_lay.addWidget(btn_fs)

        btn_add = QPushButton("＋ Añadir split")
        btn_add.setStyleSheet(_btn_css)
        btn_add.clicked.connect(self._add_panel)
        bar_lay.addWidget(btn_add)

        self._lbl_info = QLabel(
            f"{len(self._carpetas)} carpeta{'s' if len(self._carpetas) != 1 else ''}  ·  "
            f"Cambio cada {self._segundos}s  ·  Espacio: pausa  ·  F11: pantalla completa  ·  "
            f"Arrastra ⋮⋮ para reordenar  ·  ESC: cerrar"
        )
        self._lbl_info.setStyleSheet("color:#555; font-size:9px;")
        bar_lay.addWidget(self._lbl_info, 1)

        btn_close = QPushButton("✕ Cerrar")
        btn_close.setStyleSheet(_btn_css.replace("color:#ddd", "color:#ff7070"))
        btn_close.clicked.connect(self.close)
        bar_lay.addWidget(btn_close)
        root.addWidget(bar)

    def _swap_panels(self, src, dst):
        """Intercambia las posiciones de dos _PhotoPanel en los splitters."""
        if src is dst:
            return
        sa = src.parentWidget()
        sb = dst.parentWidget()
        if sa is None or sb is None:
            return
        if sa is sb:
            sizes = sa.sizes()
            widgets = [sa.widget(i) for i in range(sa.count())]
            ia = widgets.index(src)
            ib = widgets.index(dst)
            widgets[ia], widgets[ib] = widgets[ib], widgets[ia]
            for w in widgets:
                w.setParent(None)
            for w in widgets:
                sa.addWidget(w)
            sa.setSizes(sizes)
        else:
            sa_widgets = [sa.widget(i) for i in range(sa.count())]
            sb_widgets = [sb.widget(i) for i in range(sb.count())]
            ia = sa_widgets.index(src)
            ib = sb_widgets.index(dst)
            sa_sizes = sa.sizes()
            sb_sizes = sb.sizes()
            sa_widgets[ia] = dst
            sb_widgets[ib] = src
            for w in list(sa_widgets) + list(sb_widgets):
                w.setParent(None)
            for w in sa_widgets:
                sa.addWidget(w)
            for w in sb_widgets:
                sb.addWidget(w)
            if len(sa_sizes) == sa.count():
                sa.setSizes(sa_sizes)
            if len(sb_sizes) == sb.count():
                sb.setSizes(sb_sizes)
        # Reposicionar los overlays tras el reparenting.
        for p in (src, dst):
            if hasattr(p, "_reposition_overlay"):
                QTimer.singleShot(0, p._reposition_overlay)

    # ── add / remove / change-folder ──

    def _make_panel(self, carpeta: Path) -> "_PhotoPanel":
        p = _PhotoPanel(
            carpeta,
            db=self._db,
            photo_seconds=self._segundos,
            log_callback=self._log_callback,
        )
        p.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Ignored)
        p._swap_callback = self._swap_panels
        p._change_folder_callback = self._change_panel_folder
        p._remove_split_callback = self._remove_panel
        return p

    def _rebuild_grid(self):
        """Destroy all splitters and recreate the grid from self._panels."""
        # Hide all overlay windows first
        for p in self._panels:
            if hasattr(p, "_ctrl_overlay"):
                p._ctrl_overlay.hide()

        # Remove outer splitter from layout
        layout = self.layout()
        if self._outer_splitter is not None:
            layout.removeWidget(self._outer_splitter)
            self._outer_splitter.hide()
            self._outer_splitter.deleteLater()
            self._outer_splitter = None

        self._row_splitters = []
        splitter_css = (
            "QSplitter::handle { background:#222; }"
            "QSplitter::handle:horizontal { width:4px; }"
            "QSplitter::handle:vertical   { height:4px; }"
            "QSplitter::handle:hover { background:#3a6ec0; }"
        )
        rows, cols = self._grid_dims(len(self._panels))
        outer = QSplitter(Qt.Orientation.Vertical)
        outer.setStyleSheet(splitter_css)
        outer.setChildrenCollapsible(False)
        outer.setHandleWidth(4)
        self._outer_splitter = outer
        for _ in range(rows):
            rs = QSplitter(Qt.Orientation.Horizontal)
            rs.setStyleSheet(splitter_css)
            rs.setChildrenCollapsible(False)
            rs.setHandleWidth(4)
            self._row_splitters.append(rs)
            outer.addWidget(rs)
        for i, panel in enumerate(self._panels):
            panel.setParent(None)
            self._row_splitters[i // cols].addWidget(panel)
        for rs in self._row_splitters:
            rs.setSizes([1] * rs.count())
        outer.setSizes([1] * outer.count())
        # Insert before last widget (the bottom bar)
        layout.insertWidget(0, outer, 1)
        outer.show()
        for p in self._panels:
            p.show()
            QTimer.singleShot(0, p._reposition_overlay)

    def _add_panel(self):
        folder = QFileDialog.getExistingDirectory(self, "Seleccionar carpeta para nuevo split")
        if not folder:
            return
        panel = self._make_panel(Path(folder))
        self._panels.append(panel)
        self._rebuild_grid()
        panel.avanzar()

    def _remove_panel(self, panel):
        if len(self._panels) <= 1:
            return  # no quitar el último
        if hasattr(panel, "_ctrl_overlay"):
            panel._ctrl_overlay.hide()
            panel._ctrl_overlay.deleteLater()
        _PhotoPanel._panel_registry.pop(id(panel), None)
        self._panels.remove(panel)
        panel.setParent(None)
        panel.deleteLater()
        self._rebuild_grid()

    def _change_panel_folder(self, panel):
        folder = QFileDialog.getExistingDirectory(self, "Seleccionar nueva carpeta")
        if not folder:
            return
        panel._carpeta = Path(folder)
        panel._caption.setText(panel._carpeta.name)
        panel._items = []
        panel._idx = 0
        panel._current_ruta = None
        panel._current_pm = None
        panel._load_items()
        panel.avanzar()

    def _avanzar_todos(self):
        for p in self._panels:
            p.avanzar()

    def _tick(self):
        for p in self._panels:
            p.avanzar_si_toca(self._segundos)

    def _toggle_pause(self):
        if self._paused:
            self._timer.start()
            self._btn_pause.setText("⏸ Pausa")
        else:
            self._timer.stop()
            self._btn_pause.setText("▶ Reanudar")
        self._paused = not self._paused

    def _toggle_fullscreen(self):
        if self.isFullScreen():
            self.showNormal()
        else:
            self.showFullScreen()

    def keyPressEvent(self, ev):
        k = ev.key()
        if k == Qt.Key.Key_Escape:
            if self.isFullScreen():
                self.showNormal()
            else:
                self.close()
        elif k == Qt.Key.Key_Space:
            self._toggle_pause()
        elif k == Qt.Key.Key_F11:
            self._toggle_fullscreen()
        super().keyPressEvent(ev)

    def closeEvent(self, ev):
        self._timer.stop()
        for p in self._panels:
            p.stop_player()
        super().closeEvent(ev)


class PhotoSlideshowConfigDialog(QDialog):
    """Diálogo para configurar N carpetas y N segundos para la presentación de fotos."""

    _SETTINGS_FOLDERS_KEY = "last_photo_slideshow_folders"

    def __init__(self, db=None, parent=None):
        super().__init__(parent)
        self._db = db
        self.setWindowTitle("Presentación de Fotos")
        self.setMinimumWidth(540)
        self._build_ui()
        self._restore_saved_folders()

    def _build_ui(self):
        lay = QVBoxLayout(self)
        lay.setSpacing(12)
        lay.setContentsMargins(16, 16, 16, 16)

        title = QLabel("📷  Presentación de Fotos en Pantalla Dividida")
        title.setStyleSheet("font-weight:700; font-size:13px;")
        lay.addWidget(title)

        hint = QLabel(
            "Añade N carpetas. Al reproducir, la pantalla se divide en N paneles "
            "y las fotos rotan automáticamente cada N segundos. Los vídeos se reproducen completos."
        )
        hint.setWordWrap(True)
        hint.setStyleSheet("color:#666; font-size:10px;")
        lay.addWidget(hint)

        self._list = QListWidget()
        self._list.setMinimumHeight(200)
        lay.addWidget(self._list)

        fbtn_row = QHBoxLayout()
        btn_add = QPushButton("+ Añadir carpeta")
        btn_add.setObjectName("accent")
        btn_add.clicked.connect(self._add_carpeta)
        fbtn_row.addWidget(btn_add)
        btn_remove = QPushButton("− Quitar seleccionada")
        btn_remove.clicked.connect(self._remove_carpeta)
        fbtn_row.addWidget(btn_remove)
        fbtn_row.addStretch()
        lay.addLayout(fbtn_row)

        sep = QFrame()
        sep.setFrameShape(QFrame.Shape.HLine)
        sep.setStyleSheet("color:#ddd;")
        lay.addWidget(sep)

        sec_row = QHBoxLayout()
        sec_row.addWidget(QLabel("Segundos entre fotos:"))
        self._spn = QSpinBox()
        self._spn.setRange(1, 300)
        self._spn.setValue(5)
        self._spn.setSuffix(" s")
        self._spn.setFixedWidth(90)
        sec_row.addWidget(self._spn)
        sec_row.addStretch()
        lay.addLayout(sec_row)

        btns = QHBoxLayout()
        btns.addStretch()
        btn_cancel = QPushButton("Cancelar")
        btn_cancel.clicked.connect(self.reject)
        btns.addWidget(btn_cancel)
        btn_play = QPushButton("▶  Iniciar presentación")
        btn_play.setObjectName("accent")
        btn_play.clicked.connect(self._on_play)
        btns.addWidget(btn_play)
        lay.addLayout(btns)

    def _add_carpeta(self):
        dlg = QFileDialog(self, "Seleccionar carpetas")
        dlg.setFileMode(QFileDialog.FileMode.Directory)
        dlg.setOption(QFileDialog.Option.ShowDirsOnly, True)
        # Required on Windows to allow selecting multiple directories.
        dlg.setOption(QFileDialog.Option.DontUseNativeDialog, True)
        start_dir = str(Path.home())
        if self._db:
            saved = self._db.obtener_setting("last_photo_folder", start_dir)
            if saved and Path(saved).exists():
                start_dir = saved
        dlg.setDirectory(start_dir)
        # In the non-native dialog, enforce multi-select on both views.
        for view in dlg.findChildren(QListView):
            view.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        for view in dlg.findChildren(QTreeView):
            view.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        if not dlg.exec():
            return
        selected = dlg.selectedFiles()
        for d in selected:
            self._add_carpeta_item(d)
        if self._db and selected:
            # Persist last used folder across app restarts.
            self._db.guardar_setting("last_photo_folder", selected[-1])

    def _add_carpeta_item(self, d):
        for i in range(self._list.count()):
            if self._list.item(i).data(Qt.ItemDataRole.UserRole) == d:
                return
        item = QListWidgetItem(f"📁  {Path(d).name}    {d}")
        item.setData(Qt.ItemDataRole.UserRole, d)
        self._list.addItem(item)

    def _remove_carpeta(self):
        row = self._list.currentRow()
        if row >= 0:
            self._list.takeItem(row)

    def _restore_saved_folders(self):
        if not self._db:
            return
        raw = self._db.obtener_setting(self._SETTINGS_FOLDERS_KEY, "")
        if not raw:
            return
        try:
            saved_folders = json.loads(raw)
        except Exception:
            return
        if not isinstance(saved_folders, list):
            return
        for folder in saved_folders:
            try:
                folder_str = str(folder)
            except Exception:
                continue
            if folder_str and Path(folder_str).exists():
                self._add_carpeta_item(folder_str)

    def _save_selected_folders(self):
        if not self._db:
            return
        self._db.guardar_setting(
            self._SETTINGS_FOLDERS_KEY,
            json.dumps(self.carpetas(), ensure_ascii=False),
        )

    def carpetas(self) -> list:
        return [
            self._list.item(i).data(Qt.ItemDataRole.UserRole)
            for i in range(self._list.count())
        ]

    def segundos(self) -> int:
        return self._spn.value()

    def _on_play(self):
        if not self.carpetas():
            QMessageBox.warning(self, "Sin carpetas",
                                "Añade al menos una carpeta antes de iniciar.")
            return
        self._save_selected_folders()
        self.accept()


# ---------------------------------------------------------------------------

class VideoBrowserApp(QMainWindow):

    _hover_frame_signal = pyqtSignal(int, bytes)  # row, raw RGB bytes (w,h packed in first 8 bytes)

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Video Manager")
        self.resize(1200, 750)
        # Maximize on first show (deferred so the window handle exists).
        QTimer.singleShot(0, self.showMaximized)
        self._privacy_locked = False
        self._privacy_blur_effect = None
        self._privacy_overlay = None
        self._privacy_pass_input = None
        self._privacy_info_label = None
        self._loading_overlay = None

        self.db = VideoDatabase()
        self.ruta_raiz = None
        self.carpeta_actual = None
        self.video_elegido = None
        self.player_thread = None
        self.videos_base = []
        self.lista_actual = []
        self.modo_actual = "1"
        self.carpeta_fijada = None
        self.historial_raices = []
        self.carpetas_vetadas = set(self.db.obtener_carpetas_vetadas())
        self.ffprobe_path = (
            FFPROBE_PATH if os.path.exists(FFPROBE_PATH)
            else (shutil.which("ffprobe") or shutil.which("ffprobe.exe"))
        )
        self.thumb_thread = None
        self.mode_buttons = {}
        self.filtro_texto = ""
        self.incluir_fotos_gestion = False
        self._photo_autonext_token = 0
        self._reco_top_pool = []
        self._reco_top_cursor = 0
        self._reco_top_chunk_size = 18
        self._reco_extra_pool = []
        self._reco_extra_cursor = 0
        self._reco_extra_chunk_size = 10
        self._reco_thumb_cache = {}
        self._reco_stats_map = {}
        self._reco_loading_chunk = False
        self._dashboard_home_active = True
        self._dashboard_thumb_cache = {}
        self._dashboard_wheel_targets = set()
        self._dashboard_block_queue = []
        self._dashboard_block_queue_name = ""
        self._dashboard_prewarm_target = ""
        self._dashboard_prewarm_thread = None
        self._preview_10s_queue: list = []
        self._preview_10s_active: bool = False
        self._preview_10s_timer: "QTimer | None" = None
        self._preview_10s_seek_ms: int = 0
        self._preview_10s_block_title: str | None = None
        self._player_autonext_active: bool = False
        self._player_autonext_timer: "QTimer | None" = None
        self.duration_cache = {}
        self.idle_hash_queue = []
        self.idle_hash_in_progress = False
        self.idle_hash_count = 0
        self.duplicate_paths = set()
        self.duplicate_map = {}
        self.duplicate_folder_prefixes = set()
        self.folder_agg_cache = {}
        self.folder_icon_cache = {}
        self._folder_icon_minute = int(time.time() // 60)
        self._folder_thumb_ids_cache = None
        self._folder_suggest_count_cache = {}
        # Caches para acelerar el escaneo en HDD: evitan miles de stat()/rglob() repetidos
        self._video_size_cache = {}        # {ruta_str: int(bytes)}
        self._all_files_cache = []         # lista cruda de Path por _fast_walk
        self._all_files_cache_root = None
        self._all_files_cache_ts = 0.0
        self._all_files_cache_ttl = 30.0   # segundos antes de re-escanear
        self._folder_image_cache = {}      # {carpeta_str: Path|None}
        self._tree_build_timer = None
        self._tree_build_queue = deque()
        self._tree_build_nodes = 0
        self._tree_chunk_items = 240
        self._tree_chunk_min_items = 80
        self._tree_chunk_max_items = 700
        self._tree_chunk_target_ms = 24.0
        self.thumb_total = 0
        self.thumb_done = 0
        # Regular table thumbnails should be generated unless explicitly stopped.
        self._thumbs_stop = False
        self._suggestions_stop = True
        self._folder_suggest_thread = None
        self._folder_suggestion_queue = {}
        self._suggest_cycle = 0
        self._suggest_done = 0
        self._suggest_total = 0
        self._hash_stop = True
        self.audio_output = QAudioOutput(self)
        self._media_devices = QMediaDevices(self)
        self.media_player = QMediaPlayer(self)
        self.media_player.setAudioOutput(self.audio_output)
        self._apply_system_audio_output_device()
        try:
            self._media_devices.audioOutputsChanged.connect(self._on_audio_outputs_changed)
        except Exception:
            pass
        self.audio_output.setVolume(1.0)
        self.audio_output.setMuted(True)
        self._last_nonzero_volume = 1.0
        self.playback_slider_dragging = False
        self._active_ruta_reproduccion = None
        self._foto_window = None
        self._ignore_next_stop_event = False
        self._folder_view_path = None
        self._folder_view_started_at = None
        self._daily_folder_log_date = datetime.now().strftime("%Y-%m-%d")
        self._daily_folder_stats = defaultdict(lambda: {"vistas": 0, "segundos": 0})
        self._load_daily_folder_log()
        self._folder_log_dirty = False
        self._folder_log_write_timer = QTimer(self)
        self._folder_log_write_timer.setSingleShot(True)
        self._folder_log_write_timer.setInterval(900)
        self._folder_log_write_timer.timeout.connect(self._flush_pending_daily_folder_log)
        self._meta_check_thread = None
        self._meta_check_stop = False
        self._meta_check_total = 0
        self._meta_check_done = 0
        self._meta_check_aligned = 0
        self._restore_last_error = ""
        self._pending_random_seek_ratio = None
        self._pending_random_pause = False
        self._pending_random_seek_tries = 0
        self._pending_random_seek_scheduled = False
        self._window_was_maximized_before_fullscreen = False
        self._video_only_fs_window = None
        self._fs_seek_dragging = False
        self._fs_play_pause_conn = None
        # Seek absoluto pendiente (ms) para reproducir un frame concreto al cargar el video
        self._pending_seek_ms = None
        self._pending_seek_ms_tries = 0
        self._pending_seek_ms_scheduled = False
        # Cache del thumb_id elegido por carpeta (para que el doble-click reproduzca
        # el video y frame de la miniatura visible).
        self._folder_chosen_thumb_cache = {}
        self._pending_rwd_renames = set()
        self._pending_rwd_attempts = {}  # ruta_str → nº reintentos en esta sesión
        self._pending_rwd_timer = QTimer(self)
        self._pending_rwd_timer.setInterval(5000)
        self._pending_rwd_timer.timeout.connect(self._retry_pending_rwd_renames)
        self._pending_rwd_max_session_attempts = 3
        self._pending_delete_paths = set()
        self._pending_fav_renames = {}
        self._pending_file_ops_timer = QTimer(self)
        self._pending_file_ops_timer.setInterval(5000)
        self._pending_file_ops_timer.timeout.connect(self._retry_pending_file_ops)

        # Hover preview state
        self._hover_row = -1
        self._hover_video = None
        self._hover_orig_icon = None
        self._hover_timer = None
        self._photo_preview_pm = None
        self._play_history = []
        self._play_history_idx = -1

        self._build_ui()
        self._setup_app_icon()
        self._setup_privacy_overlay()
        self._setup_loading_overlay()
        QTimer.singleShot(0, self._lock_privacy)
        self._sync_mute_button_text()
        self._apply_style()
        self.media_player.mediaStatusChanged.connect(self._on_media_status_changed)
        self.media_player.positionChanged.connect(self._on_player_position_changed)
        self.media_player.durationChanged.connect(self._on_player_duration_changed)
        self.video_widget.fullScreenChanged.connect(self._on_video_fullscreen_changed)
        self._setup_idle_hashing()
        self._setup_tree_building()
        self._setup_folder_thumbnail_rotation()
        self._setup_hover_preview()
        self._load_pending_file_ops()
        self._refresh_pending_ops_badge()
        # Mostrar dashboard de inicio nada más arrancar.
        QTimer.singleShot(0, self._go_to_dashboard)

    def _apply_style(self):
        self.setStyleSheet("""
            * {
                font-family: "Segoe UI", "Segoe UI Variable", "Inter", sans-serif;
                font-size: 10pt;
            }
            QMainWindow, QWidget {
                background: #f8f8f8;
                color: #0f0f0f;
            }

            QLabel#ytLogo, QPushButton#ytLogo {
                background: #ff0000;
                color: #ffffff;
                border: none;
                border-radius: 11px;
                padding: 5px 10px;
                font-size: 10pt;
                font-weight: 800;
                letter-spacing: 0.2px;
            }
            QPushButton#ytLogo:hover { background: #e60000; }
            QPushButton#ytLogo:pressed { background: #cc0000; }

            QPushButton#hamburger {
                background: transparent;
                border: none;
                border-radius: 18px;
                font-size: 16pt;
                color: #0f0f0f;
                padding: 0px;
            }
            QPushButton#hamburger:hover  { background: #ececec; }
            QPushButton#hamburger:pressed,
            QPushButton#hamburger:checked { background: #e0e0e0; }

            /* ── Árbol ── */
            QTreeWidget {
                background: #ffffff;
                border: 1px solid #e5e5e5;
                border-radius: 12px;
                color: #0f0f0f;
                outline: none;
                padding: 6px;
            }
            QTreeWidget#sidebarTree {
                background: #ffffff;
                border: 1px solid #d0d0d0;
                border-radius: 14px;
                font-size: 12pt;
            }
            QTreeWidget#sidebarTree::item {
                padding: 4px 4px;
                margin: 0px 1px;
            }
            QTreeWidget#sidebarTree QHeaderView::section {
                padding: 3px 6px;
                font-size: 10pt;
            }
            QTreeWidget::item {
                padding: 10px 6px;
                border-radius: 8px;
                margin: 1px 2px;
            }
            QTreeWidget::item:hover {
                background: #f2f2f2;
            }
            QTreeWidget::item:selected {
                background: #ffeceb;
                color: #8b1010;
            }
            QTreeWidget::branch:has-children:!has-siblings:closed,
            QTreeWidget::branch:closed:has-children:has-siblings {
                border-image: none;
            }

            /* ── Tabla ── */
            QTableWidget {
                background: #ffffff;
                border: 1px solid #e5e5e5;
                border-radius: 12px;
                color: #0f0f0f;
                outline: none;
                gridline-color: #f1f1f1;
                alternate-background-color: #fcfcfc;
                selection-background-color: #ffeceb;
                selection-color: #8b1010;
            }
            QTableWidget::item {
                padding: 6px 8px;
                border: none;
            }
            QTableWidget::item:hover { background: #f7f7f7; }
            QTableWidget::item:selected {
                background: #ffeceb;
                color: #8b1010;
            }
            QHeaderView::section {
                background: #fafafa;
                color: #606060;
                border: none;
                border-bottom: 1px solid #ececec;
                padding: 8px 10px;
                font-weight: 600;
                font-size: 9pt;
                letter-spacing: 0.3px;
            }
            QHeaderView::section:first { border-top-left-radius: 12px; }
            QHeaderView::section:last  { border-top-right-radius: 12px; }

            QFrame#channelBar {
                background: #ffffff;
                border: 1px solid #e8e8e8;
                border-radius: 14px;
            }
            QFrame#dashboardBlock {
                background: #ffffff;
                border: 1px solid #e6e6e6;
                border-radius: 12px;
            }
            QLabel#dashboardTitle {
                font-size: 11pt;
                font-weight: 700;
                color: #151515;
            }
            QPushButton#dashboardNavButton {
                background: #ffffff;
                border: 1px solid #dddddd;
                border-radius: 18px;
                color: #151515;
                font-size: 14pt;
                font-weight: 800;
                padding: 0px;
                min-width: 36px;
                max-width: 36px;
                min-height: 36px;
                max-height: 36px;
            }
            QPushButton#dashboardNavButton:hover {
                background: #f4f4f4;
                border-color: #cfcfcf;
            }
            QPushButton#dashboardNavButton:pressed { background: #e8e8e8; }
            QPushButton#dashboardNavButton:disabled {
                background: #f7f7f7;
                color: #b7b7b7;
                border-color: #ebebeb;
            }
            QLabel#channelAvatar {
                background: transparent;
                border: none;
            }
            QLabel#channelName {
                font-size: 12pt;
                font-weight: 700;
                color: #0f0f0f;
            }
            QLabel#channelMeta {
                font-size: 9pt;
                color: #6b6b6b;
            }
            QPushButton#channelAction {
                background: #efefef;
                border: 1px solid #dfdfdf;
                border-radius: 14px;
                padding: 5px 12px;
                color: #222222;
                font-weight: 600;
            }
            QPushButton#channelAction:hover { background: #e4e4e4; }
            QPushButton#channelAction:pressed { background: #d8d8d8; }

            /* ── Botones ── */
            QPushButton {
                background: #f2f2f2;
                border: 1px solid #d8d8d8;
                border-radius: 18px;
                padding: 6px 14px;
                color: #0f0f0f;
                font-weight: 600;
            }
            QPushButton:hover {
                background: #e9e9e9;
                border-color: #cfcfcf;
            }
            QPushButton:pressed { background: #dfdfdf; }
            QPushButton:disabled { color: #a8a8a8; background: #f5f5f5; }

            QPushButton#accent {
                background: #ff0000;
                color: #ffffff;
                border: 1px solid #e20000;
                border-radius: 18px;
                font-weight: 600;
            }
            QPushButton#accent:hover {
                background: #e60000;
            }
            QPushButton#accent:pressed {
                background: #cc0000;
            }

            QPushButton#danger {
                background: #fff5f5;
                color: #d30000;
                border: 1px solid #ffc1c1;
            }
            QPushButton#danger:hover {
                background: #ffe9e9;
                border-color: #ff9f9f;
            }

            QPushButton#mode {
                background: #f2f2f2;
                border: 1px solid #dfdfdf;
                padding: 5px 14px;
                font-size: 9pt;
                border-radius: 14px;
                color: #444;
            }
            QPushButton#mode:hover {
                background: #ebebeb;
                border-color: #cfcfcf;
                color: #202020;
            }
            QPushButton#mode:checked {
                background: #0f0f0f;
                border-color: #0f0f0f;
                color: #ffffff;
                font-weight: 600;
            }

            QListWidget#recoList {
                background: #ffffff;
                border: 1px solid #e8e8e8;
                border-radius: 10px;
                padding: 5px;
            }
            QListWidget#recoList::item {
                border: 1px solid #efefef;
                border-radius: 10px;
                padding: 4px;
                margin: 2px;
                background: #fcfcfc;
            }
            QListWidget#recoList::item:selected {
                border-color: #ff6b6b;
                background: #fff1f1;
                color: #0f0f0f;
            }
            QListWidget#recoList::item:hover {
                border-color: #ffd0d0;
                background: #fff8f8;
            }

            /* ── Checkboxes ── */
            QCheckBox {
                color: #0f0f0f;
                spacing: 8px;
                font-weight: 600;
            }
            QCheckBox::indicator {
                width: 16px;
                height: 16px;
                border: 1px solid #bfbfbf;
                border-radius: 4px;
                background: #ffffff;
            }
            QCheckBox::indicator:checked {
                background: #ff0000;
                border-color: #ff0000;
            }
            QCheckBox::indicator:hover {
                border-color: #ff4a4a;
            }

            /* ── Splitter ── */
            QSplitter::handle {
                background: transparent;
                width: 6px;
            }
            QSplitter::handle:hover {
                background: #e3e3e3;
            }

            /* ── Labels ── */
            QLabel#path {
                color: #5f5f5f;
                font-size: 9pt;
                padding: 3px 9px;
                background: #ffffff;
                border: 1px solid #e1e1e1;
                border-radius: 8px;
            }
            QLabel#count {
                color: #606060;
                font-size: 9pt;
                font-weight: 500;
            }
            QLabel#detail {
                color: #0f0f0f;
                font-size: 10pt;
                font-weight: 500;
                padding: 2px 0;
            }

            /* ── ProgressBar ── */
            QProgressBar {
                background: #efefef;
                border: none;
                border-radius: 5px;
                text-align: center;
                color: #202020;
                max-height: 10px;
                font-size: 8pt;
            }
            QProgressBar::chunk {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #ff3b30, stop:1 #ff0000);
                border-radius: 5px;
            }

            /* ── Detail frame ── */
            QFrame#detailFrame {
                background: #ffffff;
                border: 1px solid #e5e5e5;
                border-radius: 12px;
                padding: 6px;
            }

            /* ── Scrollbars ── */
            QScrollBar:vertical {
                background: transparent;
                width: 10px;
                margin: 4px 2px;
            }
            QScrollBar::handle:vertical {
                background: #c8c8c8;
                border-radius: 4px;
                min-height: 30px;
            }
            QScrollBar::handle:vertical:hover { background: #ababab; }
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0; }
            QScrollBar:horizontal {
                background: transparent;
                height: 10px;
                margin: 2px 4px;
            }
            QScrollBar::handle:horizontal {
                background: #c8c8c8;
                border-radius: 4px;
                min-width: 30px;
            }
            QScrollBar::handle:horizontal:hover { background: #ababab; }
            QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal { width: 0; }

            /* ── Menus ── */
            QMenu {
                background: #ffffff;
                border: 1px solid #e3e3e3;
                border-radius: 8px;
                padding: 4px;
            }
            QMenu::item {
                padding: 6px 18px;
                border-radius: 6px;
            }
            QMenu::item:selected {
                background: #ffeceb;
                color: #8b1010;
            }

            QToolTip {
                background: #ffffff;
                color: #202020;
                border: 1px solid #d8d8d8;
                padding: 6px 8px;
                border-radius: 6px;
            }

            QStatusBar {
                background: #ffffff;
                color: #555;
                border-top: 1px solid #e8e8e8;
            }

            QTabWidget::pane {
                border: 1px solid #e5e5e5;
                background: #ffffff;
                border-radius: 10px;
            }

            QTabBar::tab {
                background: #f2f2f2;
                border: 1px solid #dddddd;
                border-bottom: none;
                border-top-left-radius: 8px;
                border-top-right-radius: 8px;
                padding: 6px 12px;
                margin-right: 4px;
                color: #444;
            }
            QTabBar::tab:selected {
                background: #ffffff;
                color: #111;
                border-color: #e1e1e1;
            }

            QSlider::groove:horizontal {
                height: 6px;
                background: #ececec;
                border-radius: 3px;
            }
            QSlider::handle:horizontal {
                width: 14px;
                margin: -4px 0;
                border-radius: 7px;
                background: #ff0000;
                border: 1px solid #d40000;
            }

            QComboBox {
                background: #ffffff;
                border: 1px solid #d8d8d8;
                border-radius: 8px;
                padding: 6px 10px;
                color: #1a1d23;
                selection-background-color: #ffeceb;
            }
            QComboBox:focus { border-color: #ff6b6b; }
            QComboBox::drop-down {
                border: none;
                width: 24px;
            }
            QComboBox QAbstractItemView {
                background: #ffffff;
                border: 1px solid #d8d8d8;
                border-radius: 6px;
                color: #1a1d23;
                selection-background-color: #ffeceb;
                selection-color: #8b1010;
                outline: none;
            }

            QLineEdit, QInputDialog QLineEdit {
                background: #ffffff;
                border: 1px solid #d8d8d8;
                border-radius: 10px;
                padding: 8px 10px;
                selection-background-color: #ffeceb;
            }
            QLineEdit:focus { border-color: #ff6b6b; }

            /* keep readability on custom black preview widgets */
            QVideoWidget {
                background: #000000;
            }

            QPlainTextEdit, QTextEdit {
                background: #ffffff;
                border: 1px solid #e0e0e0;
                border-radius: 8px;
                color: #1a1a1a;
            }

            QGroupBox {
                border: 1px solid #e3e3e3;
                border-radius: 10px;
                margin-top: 8px;
                padding: 8px;
                background: #ffffff;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 8px;
                padding: 0 4px;
                color: #666;
            }

            QListWidget {
                background: #ffffff;
                border: 1px solid #e5e5e5;
                border-radius: 10px;
            }
            QListWidget::item {
                padding: 6px;
                border-radius: 6px;
            }
            QListWidget::item:selected {
                background: #ffeceb;
                color: #8b1010;
            }

            QMenu::separator {
                height: 1px;
                background: #ededed;
                margin: 4px 8px;
            }

            QSpinBox, QDoubleSpinBox {
                background: #ffffff;
                border: 1px solid #d8d8d8;
                border-radius: 8px;
                padding: 4px 8px;
            }
            QSpinBox:focus, QDoubleSpinBox:focus {
                border-color: #ff6b6b;
            }

            QCalendarWidget QWidget {
                background: #ffffff;
                color: #1a1a1a;
            }
            QCalendarWidget QToolButton {
                background: #f2f2f2;
                border: 1px solid #dddddd;
                border-radius: 6px;
                color: #ffffff;
            }
        """)

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(12, 12, 12, 12)
        root.setSpacing(8)

        # -- toolbar --
        bar = QHBoxLayout()
        bar.setSpacing(8)
        self.btn_hamburger = QPushButton("☰")
        self.btn_hamburger.setObjectName("hamburger")
        self.btn_hamburger.setToolTip("Mostrar / ocultar carpetas")
        self.btn_hamburger.setFixedSize(40, 36)
        self.btn_hamburger.setCheckable(True)
        self.btn_hamburger.pressed.connect(self._toggle_sidebar)
        bar.addWidget(self.btn_hamburger)
        self.lbl_yt_logo = QPushButton("▶ YouTube")
        self.lbl_yt_logo.setObjectName("ytLogo")
        self.lbl_yt_logo.setCursor(Qt.CursorShape.PointingHandCursor)
        self.lbl_yt_logo.setToolTip("Ir al inicio (dashboard)")
        self.lbl_yt_logo.clicked.connect(self._go_to_dashboard)
        bar.addWidget(self.lbl_yt_logo)
        btn_root = QPushButton("Abrir carpeta")
        btn_root.setObjectName("accent")
        btn_root.clicked.connect(self.seleccionar_raiz)
        bar.addWidget(btn_root)
        self.btn_volver = QPushButton("← Volver")
        self.btn_volver.clicked.connect(self.volver_carpeta_anterior)
        self.btn_volver.setEnabled(False)
        bar.addWidget(self.btn_volver)
        btn_refrescar = QPushButton("↻ Refrescar")
        btn_refrescar.clicked.connect(self.refrescar)
        bar.addWidget(btn_refrescar)
        btn_refrescar_carpeta = QPushButton("↺ Refrescar carpeta")
        btn_refrescar_carpeta.setToolTip("Refresca solo los datos de la carpeta actual (sin reconstruir toda la estructura)")
        btn_refrescar_carpeta.clicked.connect(self.refrescar_carpeta_actual)
        bar.addWidget(btn_refrescar_carpeta)
        btn_log = QPushButton("Ver Log")
        btn_log.clicked.connect(self.abrir_carpeta_log)
        bar.addWidget(btn_log)
        btn_restaurar = QPushButton("🔄 Restaurar BD")
        btn_restaurar.setToolTip("Rellenar la BD con los datos guardados en los metadatos de los videos")
        btn_restaurar.clicked.connect(self._restaurar_bd_desde_metadatos)
        bar.addWidget(btn_restaurar)
        self.btn_stop_hash = QPushButton("▶ Hash")
        self.btn_stop_hash.setCheckable(True)
        self.btn_stop_hash.setChecked(True)
        self.btn_stop_hash.setToolTip("Pausar / reanudar el hashing en segundo plano")
        self.btn_stop_hash.clicked.connect(self._toggle_hash_stop)
        bar.addWidget(self.btn_stop_hash)
        self.btn_stop_thumbs = QPushButton("▶ Miniaturas")
        self.btn_stop_thumbs.setCheckable(True)
        self.btn_stop_thumbs.setChecked(True)
        self.btn_stop_thumbs.setToolTip("Pausar / reanudar búsqueda de miniaturas sugeridas por caras en segundo plano")
        self.btn_stop_thumbs.clicked.connect(self._toggle_thumb_stop)
        bar.addWidget(self.btn_stop_thumbs)
        self.btn_privacy = QPushButton("🔒 Privacidad")
        self.btn_privacy.setToolTip("Oculta totalmente la app y pide contraseña para volver")
        self.btn_privacy.clicked.connect(self._toggle_privacy_lock)
        bar.addWidget(self.btn_privacy)
        self.lbl_ruta = QLabel("")
        self.lbl_ruta.setObjectName("path")
        bar.addWidget(self.lbl_ruta, 1)
        self.lbl_count = QLabel("")
        self.lbl_count.setObjectName("count")
        bar.addWidget(self.lbl_count)
        self.lbl_pending_ops = QLabel("")
        self.lbl_pending_ops.setObjectName("count")
        self.lbl_pending_ops.setToolTip("Operaciones diferidas para aplicar al iniciar la app")
        bar.addWidget(self.lbl_pending_ops)
        self.btn_run_pending_ops = QPushButton("▶ Ejecutar pendientes")
        self.btn_run_pending_ops.setToolTip("Ejecuta ahora las operaciones diferidas de inicio")
        self.btn_run_pending_ops.clicked.connect(self._run_pending_ops_now)
        bar.addWidget(self.btn_run_pending_ops)
        root.addLayout(bar)

        # -- buscador --
        search_row = QHBoxLayout()
        search_row.setSpacing(8)
        self.search_box = QLineEdit()
        self.search_box.setPlaceholderText("Buscar por nombre en la carpeta actual...")
        self.search_box.textChanged.connect(self._on_search_changed)
        search_row.addWidget(self.search_box, 1)
        from PyQt6.QtWidgets import QCheckBox
        self.chk_incluir_fotos = QCheckBox("Solo fotos")
        self.chk_incluir_fotos.setToolTip("Mostrar solo fotos en la tabla (sin vídeos). Gestión completa: favoritos, revisadas, duplicados, etc.")
        self.chk_incluir_fotos.toggled.connect(self._on_toggle_incluir_fotos)
        search_row.addWidget(self.chk_incluir_fotos)
        self.lbl_hint = QLabel("Enter: reproducir  |  F11: pantalla completa  |  Arrastra divisor para agrandar video")
        self.lbl_hint.setObjectName("count")
        search_row.addWidget(self.lbl_hint)
        root.addLayout(search_row)

        # -- filtros --
        filtros_row = QHBoxLayout()
        filtros_row.setSpacing(4)
        for texto, modo in [("Todos", "1"), ("Favoritos", "2"), ("Sin Revisar", "8"), ("- Vistos", "3"),
                    ("+ Vistos", "4"), ("+ Tiempo", "7"), ("Pesados", "6")]:
            b = QPushButton(texto)
            b.setObjectName("mode")
            b.setCheckable(True)
            b.clicked.connect(lambda _, m=modo: self.iniciar_modo(m))
            self.mode_buttons[modo] = b
            filtros_row.addWidget(b)
        sep_fotos = QFrame()
        sep_fotos.setFrameShape(QFrame.Shape.VLine)
        sep_fotos.setStyleSheet("color:#ccc;")
        filtros_row.addWidget(sep_fotos)
        btn_fotos = QPushButton("\U0001f4f7 Fotos")
        btn_fotos.setObjectName("mode")
        btn_fotos.setToolTip("Presentaci\u00f3n de fotos en pantalla dividida")
        btn_fotos.clicked.connect(self._abrir_modo_fotos)
        filtros_row.addWidget(btn_fotos)
        filtros_row.addStretch()
        self.lbl_hash = QLabel("")
        self.lbl_hash.setObjectName("count")
        filtros_row.addWidget(self.lbl_hash)
        root.addLayout(filtros_row)

        # -- splitter: reproductor (centro) + lista (derecha) --
        # El árbol de carpetas NO va en el splitter: es un overlay flotante
        # que se muestra encima del contenido al pulsar la hamburguesa.
        splitter = QSplitter(Qt.Orientation.Horizontal)
        self._main_splitter = splitter

        self.tree = QTreeWidget(None)
        self.tree.setObjectName("sidebarTree")
        self.tree.setWindowFlag(Qt.WindowType.Popup, False)
        self.tree.setWindowFlag(Qt.WindowType.Tool, True)
        self.tree.setWindowFlag(Qt.WindowType.WindowStaysOnTopHint, True)
        self.tree.setWindowFlag(Qt.WindowType.FramelessWindowHint, True)
        self.tree.setHeaderLabels(["", "Carpeta", "Vistas", "Tiempo", "Peso", "% Rev", "% Hash", "Mini"])
        self.tree.setIndentation(16)
        self.tree.setIconSize(QSize(192, 192))
        self.tree.header().setStretchLastSection(False)
        self.tree.header().setSectionResizeMode(0, QHeaderView.ResizeMode.Fixed)
        self.tree.header().setSectionResizeMode(1, QHeaderView.ResizeMode.Interactive)
        self.tree.setColumnWidth(1, 280)
        self.tree.setColumnWidth(0, 236)
        for c in (2, 3, 4, 5, 6, 7):
            self.tree.header().setSectionResizeMode(c, QHeaderView.ResizeMode.ResizeToContents)
            self.tree.setColumnHidden(c, False)
        self.tree.setHeaderHidden(False)
        self.tree.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.tree.setHorizontalScrollMode(QAbstractItemView.ScrollMode.ScrollPerPixel)
        self.tree.currentItemChanged.connect(self._on_tree_select)
        self.tree.itemDoubleClicked.connect(self._on_tree_double_click)
        self.tree.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.tree.customContextMenuRequested.connect(self._tree_context_menu)
        self.tree.viewport().setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.tree.viewport().customContextMenuRequested.connect(self._tree_context_menu)
        # Overlay flotante: popup de nivel superior, oculto por defecto
        self.tree.setVisible(False)
        self.tree.raise_()
        try:
            from PyQt6.QtWidgets import QGraphicsDropShadowEffect
            _shadow = QGraphicsDropShadowEffect(self)
            _shadow.setBlurRadius(28)
            _shadow.setOffset(0, 4)
            _shadow.setColor(QColor(0, 0, 0, 90))
            self.tree.setGraphicsEffect(_shadow)
        except Exception:
            pass

        # ===== Centro: reproductor grande =====
        center = QWidget()
        center.setObjectName("centerCol")
        ccol = QVBoxLayout(center)
        ccol.setContentsMargins(0, 0, 0, 0)
        ccol.setSpacing(6)

        self.lbl_folder = QLabel("")
        self.lbl_folder.setObjectName("count")
        ccol.addWidget(self.lbl_folder)

        # ===== Columna derecha: recomendados + lista de vídeos =====
        right = QWidget()
        right.setObjectName("rightCol")
        rcol = QVBoxLayout(right)
        rcol.setContentsMargins(0, 0, 0, 0)
        rcol.setSpacing(6)

        top_panel = right  # alias para mantener nombre legacy en el resto del método
        top_layout = rcol

        # -- panel visual de recomendados --
        self.reco_panel = QFrame()
        self.reco_panel.setObjectName("detailFrame")
        reco_layout = QVBoxLayout(self.reco_panel)
        reco_layout.setContentsMargins(8, 6, 8, 6)
        reco_layout.setSpacing(6)

        self.lbl_reco_title = QLabel("Recomendados Top")
        self.lbl_reco_title.setObjectName("detail")
        reco_layout.addWidget(self.lbl_reco_title)

        self.reco_list = QListWidget()
        self.reco_list.setObjectName("recoList")
        self.reco_list.setViewMode(QListView.ViewMode.IconMode)
        self.reco_list.setFlow(QListView.Flow.LeftToRight)
        self.reco_list.setResizeMode(QListView.ResizeMode.Adjust)
        self.reco_list.setMovement(QListView.Movement.Static)
        self.reco_list.setWrapping(False)
        self.reco_list.setSpacing(8)
        self.reco_list.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.reco_list.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.reco_list.setIconSize(QSize(172, 96))
        self.reco_list.setGridSize(QSize(204, 156))
        self.reco_list.setWordWrap(True)
        self.reco_list.setTextElideMode(Qt.TextElideMode.ElideRight)
        self.reco_list.setMaximumHeight(182)
        self.reco_list.itemClicked.connect(self._on_reco_item_clicked)
        self.reco_list.horizontalScrollBar().valueChanged.connect(self._on_reco_scroll_changed)
        reco_layout.addWidget(self.reco_list)

        top_layout.addWidget(self.reco_panel)

        self.dashboard_scroll = QScrollArea()
        self.dashboard_scroll.setWidgetResizable(True)
        self.dashboard_scroll.setFrameShape(QFrame.Shape.NoFrame)
        self.dashboard_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.dashboard_scroll.setObjectName("homeDashboard")

        dash_host = QWidget()
        dash_col = QVBoxLayout(dash_host)
        dash_col.setContentsMargins(0, 0, 0, 0)
        dash_col.setSpacing(8)

        def _mk_dash_block(title: str):
            frame = QFrame()
            frame.setObjectName("dashboardBlock")
            lay = QVBoxLayout(frame)
            lay.setContentsMargins(8, 6, 8, 6)
            lay.setSpacing(6)

            top_row = QHBoxLayout()
            top_row.setContentsMargins(0, 0, 0, 0)
            top_row.setSpacing(8)
            lbl = QLabel(title)
            lbl.setObjectName("dashboardTitle")
            top_row.addWidget(lbl, 1)

            btn_play_block = QPushButton("▶ Play")
            btn_play_block.setCursor(Qt.CursorShape.PointingHandCursor)
            btn_play_block.setToolTip("Reproducir videos de este bloque")
            top_row.addWidget(btn_play_block, 0, Qt.AlignmentFlag.AlignRight)

            btn_10s = QPushButton("▶ 10s")
            btn_10s.setCursor(Qt.CursorShape.PointingHandCursor)
            btn_10s.setToolTip("Reproducir 10 segundos de cada vídeo del bloque en posición aleatoria")
            top_row.addWidget(btn_10s, 0, Qt.AlignmentFlag.AlignRight)

            lay.addLayout(top_row)

            nav_row = QHBoxLayout()
            nav_row.setContentsMargins(0, 0, 0, 0)
            nav_row.setSpacing(8)

            btn_left = QPushButton("‹")
            btn_left.setObjectName("dashboardNavButton")
            btn_left.setCursor(Qt.CursorShape.PointingHandCursor)
            btn_left.setAutoRepeat(True)

            lst = QListWidget()
            lst.setViewMode(QListView.ViewMode.IconMode)
            lst.setFlow(QListView.Flow.LeftToRight)
            lst.setObjectName(title)
            lst.setResizeMode(QListView.ResizeMode.Adjust)
            lst.setMovement(QListView.Movement.Static)
            lst.setWrapping(False)
            lst.setSpacing(12)
            lst.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
            lst.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
            lst.setIconSize(QSize(344, 192))
            lst.setGridSize(QSize(392, 316))
            lst.setWordWrap(True)
            lst.setTextElideMode(Qt.TextElideMode.ElideRight)
            lst.setMinimumHeight(356)
            lst.setMaximumHeight(356)
            lst.itemClicked.connect(self._on_dashboard_item_clicked)
            self._dashboard_wheel_targets.add(lst)
            lst.installEventFilter(self)
            lst.viewport().installEventFilter(self)

            btn_right = QPushButton("›")
            btn_right.setObjectName("dashboardNavButton")
            btn_right.setCursor(Qt.CursorShape.PointingHandCursor)
            btn_right.setAutoRepeat(True)

            nav_row.addWidget(btn_left, 0, Qt.AlignmentFlag.AlignVCenter)
            nav_row.addWidget(lst, 1)
            nav_row.addWidget(btn_right, 0, Qt.AlignmentFlag.AlignVCenter)
            lay.addLayout(nav_row)

            btn_left.clicked.connect(lambda _=False, w=lst: self._scroll_dashboard_carousel(w, -1))
            btn_right.clicked.connect(lambda _=False, w=lst: self._scroll_dashboard_carousel(w, +1))
            lst.horizontalScrollBar().valueChanged.connect(
                lambda _v, w=lst, bl=btn_left, br=btn_right: self._sync_dashboard_nav_buttons(w, bl, br)
            )
            lst.horizontalScrollBar().rangeChanged.connect(
                lambda _a, _b, w=lst, bl=btn_left, br=btn_right: self._sync_dashboard_nav_buttons(w, bl, br)
            )
            QTimer.singleShot(0, lambda w=lst, bl=btn_left, br=btn_right: self._sync_dashboard_nav_buttons(w, bl, br))
            btn_play_block.clicked.connect(lambda _=False, w=lst, t=title: self._play_dashboard_block(w, t))
            btn_10s.clicked.connect(lambda _=False, w=lst, t=title: self._play_block_10s_preview(w, t))
            return frame, lst

        block1, self.dash_recommended_list = _mk_dash_block("Bloque 1 · Videos recomendados")
        block2, self.dash_last_seen_list = _mk_dash_block("Bloque 2 · Últimos videos vistos")
        block3, self.dash_channels_mix_list = _mk_dash_block("Bloque 3 · Canales más vistos (mix)")
        block4, self.dash_unreviewed_list = _mk_dash_block("Bloque 4 · Videos sin revisar")
        block5, self.dash_channels_least_list = _mk_dash_block("Bloque 5 · Canales menos vistos (mix)")
        block6, self.dash_discovery_list = _mk_dash_block("Bloque 6 · Videos por descubrir")
        block7, self.dash_short_list = _mk_dash_block("Bloque 7 · Videos cortos  (<1.5 min)")
        dash_col.addWidget(block1)
        dash_col.addWidget(block2)
        dash_col.addWidget(block3)
        dash_col.addWidget(block4)
        dash_col.addWidget(block5)
        dash_col.addWidget(block6)
        dash_col.addWidget(block7)
        dash_col.addStretch()
        self.dashboard_scroll.setWidget(dash_host)
        top_layout.addWidget(self.dashboard_scroll, 1)
        self.dashboard_scroll.setVisible(False)

        self.tabla = QTableWidget(0, 7)
        self.tabla.setHorizontalHeaderLabels(["", "Nombre", "Vistas", "Tiempo", "Peso", "✓", "#"])
        self.tabla.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.tabla.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.tabla.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.tabla.setSortingEnabled(True)
        self.tabla.setAlternatingRowColors(True)
        self.tabla.setShowGrid(False)
        self.tabla.setIconSize(QSize(220, 124))
        self.tabla.verticalHeader().setVisible(False)
        self.tabla.verticalHeader().setDefaultSectionSize(132)
        self.tabla.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Fixed)
        self.tabla.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        self.tabla.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
        self.tabla.horizontalHeader().setSectionResizeMode(3, QHeaderView.ResizeMode.ResizeToContents)
        self.tabla.horizontalHeader().setSectionResizeMode(4, QHeaderView.ResizeMode.ResizeToContents)
        self.tabla.horizontalHeader().setSectionResizeMode(5, QHeaderView.ResizeMode.ResizeToContents)
        self.tabla.horizontalHeader().setSectionResizeMode(6, QHeaderView.ResizeMode.ResizeToContents)
        self.tabla.setColumnWidth(0, 232)
        self.tabla.currentCellChanged.connect(self._on_row_changed)
        self.tabla.cellClicked.connect(self._on_table_cell_clicked)
        self.tabla.cellDoubleClicked.connect(lambda r, c: self.reproducir_video_actual())
        self.tabla.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.tabla.customContextMenuRequested.connect(self._tabla_context_menu)
        top_layout.addWidget(self.tabla, 1)

        # progress (al pie de la columna derecha)
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        top_layout.addWidget(self.progress_bar)
        self.lbl_progreso = QLabel("")
        self.lbl_progreso.setObjectName("path")
        self.lbl_progreso.setVisible(False)
        top_layout.addWidget(self.lbl_progreso)
        self.thumb_progress_bar = QProgressBar()
        self.thumb_progress_bar.setVisible(False)
        top_layout.addWidget(self.thumb_progress_bar)
        self.lbl_thumb_progreso = QLabel("")
        self.lbl_thumb_progreso.setObjectName("count")
        self.lbl_thumb_progreso.setVisible(False)
        top_layout.addWidget(self.lbl_thumb_progreso)

        # detalle + botones (en la columna central, debajo del reproductor)
        detail = QFrame()
        detail.setObjectName("detailFrame")
        dl = QVBoxLayout(detail)
        dl.setContentsMargins(8, 4, 8, 4)
        dl.setSpacing(4)

        self.preview_stack = QStackedWidget()
        self.preview_stack.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

        self.video_widget = QVideoWidget()
        self.video_widget.setMinimumHeight(220)
        self.video_widget.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.video_widget.installEventFilter(self)
        self.preview_stack.addWidget(self.video_widget)

        self.photo_preview_label = QLabel()
        self.photo_preview_label.setMinimumHeight(220)
        self.photo_preview_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.photo_preview_label.setStyleSheet("background:#101010; color:#666; border:1px solid #2a2a2a;")
        self.photo_preview_label.setText("Haz clic en una foto para previsualizarla")
        self.photo_preview_label.installEventFilter(self)
        self.preview_stack.addWidget(self.photo_preview_label)

        dl.addWidget(self.preview_stack)
        self.media_player.setVideoOutput(self.video_widget)

        # Barra "canal" estilo YouTube justo bajo el video.
        self.channel_bar = QFrame()
        self.channel_bar.setObjectName("channelBar")
        channel_row = QHBoxLayout(self.channel_bar)
        channel_row.setContentsMargins(10, 8, 10, 8)
        channel_row.setSpacing(10)

        self.lbl_channel_avatar = QLabel()
        self.lbl_channel_avatar.setObjectName("channelAvatar")
        self.lbl_channel_avatar.setFixedSize(54, 54)
        self.lbl_channel_avatar.setScaledContents(False)
        channel_row.addWidget(self.lbl_channel_avatar)

        channel_text_col = QVBoxLayout()
        channel_text_col.setSpacing(1)
        self.lbl_channel_name = QLabel("Canal")
        self.lbl_channel_name.setObjectName("channelName")
        self.lbl_channel_meta = QLabel("0 videos")
        self.lbl_channel_meta.setObjectName("channelMeta")
        channel_text_col.addWidget(self.lbl_channel_name)
        channel_text_col.addWidget(self.lbl_channel_meta)
        channel_row.addLayout(channel_text_col, 1)

        self.btn_channel_action = QPushButton("Opciones del canal")
        self.btn_channel_action.setObjectName("channelAction")
        self.btn_channel_action.setToolTip("Abrir opciones de la carpeta actual")
        self.btn_channel_action.clicked.connect(self._show_selected_folder_options)
        channel_row.addWidget(self.btn_channel_action)

        dl.addWidget(self.channel_bar)
        self._refresh_channel_bar(None, 0)

        player_row = QHBoxLayout()
        player_row.setSpacing(6)
        self.btn_pause_resume = QPushButton("⏯ Pausa/Reanudar")
        self.btn_pause_resume.clicked.connect(self._toggle_pause_resume)
        player_row.addWidget(self.btn_pause_resume)
        self.btn_stop_player = QPushButton("⏹ Detener")
        self.btn_stop_player.clicked.connect(self._on_stop_player_clicked)
        player_row.addWidget(self.btn_stop_player)
        self.btn_mute = QPushButton("🔇 Mute")
        self.btn_mute.clicked.connect(self._toggle_mute)
        player_row.addWidget(self.btn_mute)
        self.btn_fullscreen = QPushButton("⛶ Pantalla completa")
        self.btn_fullscreen.clicked.connect(self._toggle_fullscreen)
        player_row.addWidget(self.btn_fullscreen)
        self.lbl_seek_step = QLabel("Paso:")
        self.lbl_seek_step.setObjectName("count")
        player_row.addWidget(self.lbl_seek_step)
        self.cmb_seek_step = QComboBox()
        self.cmb_seek_step.setEditable(True)
        self.cmb_seek_step.setInsertPolicy(QComboBox.InsertPolicy.NoInsert)
        self.cmb_seek_step.setToolTip("Segundos que avanzan/retroceden las flechas ← →. Puedes elegir 10, 20 o escribir X.")
        for sec in (2, 5, 10, 20, 30):
            self.cmb_seek_step.addItem(f"{sec}s", sec)
        if self.cmb_seek_step.lineEdit() is not None:
            self.cmb_seek_step.lineEdit().setPlaceholderText("Xs")
            self.cmb_seek_step.lineEdit().editingFinished.connect(self._normalize_seek_step_combo)
        idx_5 = self.cmb_seek_step.findData(5)
        self.cmb_seek_step.setCurrentIndex(idx_5 if idx_5 >= 0 else 0)
        self.cmb_seek_step.setMinimumWidth(74)
        player_row.addWidget(self.cmb_seek_step)
        self.lbl_player_autonext = QLabel("Auto sig.:")
        self.lbl_player_autonext.setObjectName("count")
        player_row.addWidget(self.lbl_player_autonext)
        self.cmb_player_autonext = QComboBox()
        self.cmb_player_autonext.setEditable(True)
        self.cmb_player_autonext.setInsertPolicy(QComboBox.InsertPolicy.NoInsert)
        self.cmb_player_autonext.setToolTip("Segundos para saltar automáticamente al siguiente vídeo. Puedes elegir un valor o escribir cualquier número.")
        for sec in (3, 10, 20, 30, 60):
            self.cmb_player_autonext.addItem(f"{sec}s", sec)
        if self.cmb_player_autonext.lineEdit() is not None:
            self.cmb_player_autonext.lineEdit().setPlaceholderText("Xs")
            self.cmb_player_autonext.lineEdit().editingFinished.connect(self._on_player_autonext_step_changed)
        idx_3 = self.cmb_player_autonext.findData(3)
        self.cmb_player_autonext.setCurrentIndex(idx_3 if idx_3 >= 0 else 0)
        self.cmb_player_autonext.setMinimumWidth(78)
        self.cmb_player_autonext.currentIndexChanged.connect(lambda _idx: self._on_player_autonext_step_changed())
        player_row.addWidget(self.cmb_player_autonext)
        self.btn_player_autonext = QPushButton("▶ Auto 3s")
        self.btn_player_autonext.setCheckable(True)
        self.btn_player_autonext.setToolTip("Activar o desactivar el salto automático al siguiente vídeo usando los segundos elegidos")
        self.btn_player_autonext.clicked.connect(self._toggle_player_autonext)
        player_row.addWidget(self.btn_player_autonext)
        lbl_vol = QLabel("🔊")
        lbl_vol.setObjectName("count")
        player_row.addWidget(lbl_vol)
        self.sld_volume = QSlider(Qt.Orientation.Horizontal)
        self.sld_volume.setRange(0, 100)
        self.sld_volume.setValue(100)
        self.sld_volume.setFixedWidth(90)
        self.sld_volume.setToolTip("Volumen (0-100). También con Ctrl+↑/↓")
        self.sld_volume.valueChanged.connect(self._on_volume_slider_changed)
        player_row.addWidget(self.sld_volume)
        self.player_pos = QLabel("00:00 / 00:00")
        self.player_pos.setObjectName("count")
        player_row.addWidget(self.player_pos)
        dl.addLayout(player_row)

        self.player_slider = _SeekSlider(Qt.Orientation.Horizontal)
        self.player_slider.setRange(0, 0)
        self.player_slider.sliderPressed.connect(self._on_player_slider_pressed)
        self.player_slider.sliderReleased.connect(self._on_player_slider_released)
        self.player_slider.sliderMoved.connect(self._on_player_slider_moved)
        dl.addWidget(self.player_slider)

        self.lbl_detail = QLabel("Selecciona un video")
        self.lbl_detail.setObjectName("detail")
        self.lbl_detail.setWordWrap(True)
        dl.addWidget(self.lbl_detail)

        btns = QHBoxLayout()
        btns.setSpacing(6)

        btn_play = QPushButton("▶ Reproducir")
        btn_play.setObjectName("accent")
        btn_play.clicked.connect(self.reproducir_video_actual)
        btns.addWidget(btn_play)

        self.btn_prev = QPushButton("Anterior")
        self.btn_prev.clicked.connect(self.video_anterior)
        btns.addWidget(self.btn_prev)

        self.btn_next = QPushButton("Siguiente")
        self.btn_next.clicked.connect(self.proximo_video)
        btns.addWidget(self.btn_next)

        self.btn_repeat = QPushButton("🔁 Repetir actual")
        self.btn_repeat.setCheckable(True)
        self.btn_repeat.setChecked(False)
        self.btn_repeat.setToolTip("Si está activo, al terminar el vídeo vuelve a reproducir el mismo")
        btns.addWidget(self.btn_repeat)

        self.chk_autoplay_next = QCheckBox("Autoplay siguiente")
        self.chk_autoplay_next.setChecked(True)
        self.chk_autoplay_next.setToolTip("Si está activo, al terminar el vídeo reproduce automáticamente el siguiente")
        btns.addWidget(self.chk_autoplay_next)

        self.btn_random_frame = QPushButton("🎲 Frame aleatorio")
        self.btn_random_frame.setToolTip(
            "Ir a un video aleatorio de la biblioteca y colocarlo en un frame aleatorio"
        )
        self.btn_random_frame.clicked.connect(self.ir_a_frame_aleatorio_biblioteca)
        btns.addWidget(self.btn_random_frame)

        self.chk_random_start_play = QCheckBox("Play")
        self.chk_random_start_play.setToolTip(
            "Si está activado, el salto aleatorio empieza reproduciendo; si no, queda en pausa"
        )
        self.chk_random_start_play.setChecked(False)
        btns.addWidget(self.chk_random_start_play)

        self.btn_fav = QPushButton("⭐ Favorito")
        self.btn_fav.clicked.connect(self.toggle_favorito_y_renombrar)
        btns.addWidget(self.btn_fav)

        btn_fix = QPushButton("Fijar")
        btn_fix.clicked.connect(self.fijar_carpeta)
        btns.addWidget(btn_fix)

        btn_veto = QPushButton("Vetar")
        btn_veto.clicked.connect(self.vetar_carpeta)
        btns.addWidget(btn_veto)

        btn_del = QPushButton("Borrar")
        btn_del.setObjectName("danger")
        btn_del.clicked.connect(self.borrar_video)
        btns.addWidget(btn_del)

        btn_rename = QPushButton("Renombrar")
        btn_rename.clicked.connect(self.renombrar_video)
        btns.addWidget(btn_rename)

        btn_duplis = QPushButton("Duplicados")
        btn_duplis.clicked.connect(self.borrar_duplicados_carpeta)
        btns.addWidget(btn_duplis)

        dl.addLayout(btns)
        ccol.addWidget(detail, 1)

        splitter.addWidget(center)
        splitter.addWidget(right)
        self._center_widget = center
        self._right_widget = right
        splitter.setStretchFactor(0, 6)
        splitter.setStretchFactor(1, 4)
        splitter.setSizes([820, 480])
        splitter.setCollapsible(0, False)
        splitter.setCollapsible(1, False)
        root.addWidget(splitter, 1)

        # filtro global de eventos para auto-cerrar el sidebar al pulsar fuera
        try:
            QApplication.instance().installEventFilter(self)
        except Exception:
            pass

        self._set_active_mode_button(self.modo_actual)
        self._setup_shortcuts()

    def _setup_shortcuts(self):
        QShortcut(QKeySequence("Ctrl+O"), self, activated=self.seleccionar_raiz)
        QShortcut(QKeySequence("F5"), self, activated=self.refrescar)
        QShortcut(QKeySequence("Shift+F5"), self, activated=self.refrescar_carpeta_actual)
        QShortcut(QKeySequence("Ctrl+F"), self, activated=self.search_box.setFocus)
        QShortcut(QKeySequence("Ctrl+J"), self, activated=self.ir_a_frame_aleatorio_biblioteca)
        QShortcut(QKeySequence("Ctrl+Shift+X"), self, activated=self._cancel_10s_preview_shortcut)
        QShortcut(QKeySequence("Ctrl+Alt+L"), self, activated=self._toggle_privacy_lock)
        QShortcut(QKeySequence("Enter"), self, activated=self.reproducir_video_actual)
        QShortcut(QKeySequence("Return"), self, activated=self.reproducir_video_actual)
        QShortcut(QKeySequence("Backspace"), self, activated=self.video_anterior)
        QShortcut(QKeySequence("R"), self, activated=lambda: self.btn_repeat.toggle() if hasattr(self, "btn_repeat") else None)
        QShortcut(QKeySequence(Qt.Key.Key_Left), self, activated=lambda: self._seek_relative_seconds(-self._get_seek_step_seconds()))
        QShortcut(QKeySequence(Qt.Key.Key_Right), self, activated=lambda: self._seek_relative_seconds(self._get_seek_step_seconds()))
        QShortcut(QKeySequence("Space"), self, activated=self.proximo_video)
        QShortcut(QKeySequence("Delete"), self, activated=self.borrar_video)
        QShortcut(QKeySequence("F11"), self, activated=self._toggle_fullscreen)
        QShortcut(QKeySequence("Escape"), self, activated=self._exit_fullscreen)
        QShortcut(QKeySequence("M"), self, activated=self._toggle_mute)
        QShortcut(QKeySequence("Ctrl+Up"), self, activated=lambda: self._change_volume(10))
        QShortcut(QKeySequence("Ctrl+Down"), self, activated=lambda: self._change_volume(-10))

    def _setup_app_icon(self):
        """Load app icon for title bar and taskbar, with fallback."""
        icon = QIcon()
        candidates = [
            Path(__file__).with_name("app_icon.ico"),
            Path(__file__).with_name("app_icon.png"),
            Path(__file__).with_name("app_icon.svg"),
        ]
        for p in candidates:
            try:
                if p.exists():
                    icon = QIcon(str(p))
                    if not icon.isNull():
                        break
            except Exception:
                continue
        if icon.isNull():
            icon = self.style().standardIcon(QStyle.StandardPixmap.SP_ComputerIcon)
        self.setWindowIcon(icon)
        try:
            QApplication.instance().setWindowIcon(icon)
        except Exception:
            pass

    def _setup_privacy_overlay(self):
        self._privacy_overlay = QFrame(self)
        self._privacy_overlay.setVisible(False)
        self._privacy_overlay.setStyleSheet(
            "QFrame#privacyOverlay { background: #f9f9f9; }"
            "QLabel { color: #0f0f0f; background: transparent; border: none; }"
            "QLineEdit { background:#ffffff; color:#0f0f0f; "
            "  border:1px solid #c6c6c6; border-radius:4px; "
            "  padding:14px 14px; font-size:11pt; min-width:300px; }"
            "QLineEdit:focus { border:1.5px solid #1a73e8; }"
            "QPushButton#btnUnlock { background:#ff0000; color:#ffffff; "
            "  border:none; border-radius:18px; padding:10px 28px; "
            "  font-size:10pt; font-weight:700; }"
            "QPushButton#btnUnlock:hover  { background:#e60000; }"
            "QPushButton#btnUnlock:pressed { background:#cc0000; }"
            "QLabel#ytLogoBig { background:#ff0000; color:#ffffff; "
            "  border-radius:14px; padding:8px 16px; "
            "  font-size:18pt; font-weight:800; letter-spacing:0.5px; }"
        )
        self._privacy_overlay.setObjectName("privacyOverlay")

        outer = QHBoxLayout(self._privacy_overlay)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addStretch(2)

        # ── card estilo Google/YouTube ──
        card = QFrame()
        card.setObjectName("privacyCard")
        card.setFixedWidth(440)
        card.setStyleSheet(
            "QFrame#privacyCard { background:#ffffff; "
            "border:1px solid #dadce0; border-radius:12px; }"
        )
        card_lay = QVBoxLayout(card)
        card_lay.setContentsMargins(48, 44, 48, 36)
        card_lay.setSpacing(0)

        # logo ▶ YouTube centrado
        logo_row = QHBoxLayout()
        logo_row.addStretch()
        lbl_icon = QLabel("▶ YouTube")
        lbl_icon.setObjectName("ytLogoBig")
        lbl_icon.setAlignment(Qt.AlignmentFlag.AlignCenter)
        logo_row.addWidget(lbl_icon)
        logo_row.addStretch()
        card_lay.addLayout(logo_row)
        card_lay.addSpacing(18)

        lbl_app = QLabel("Iniciar sesión")
        lbl_app.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lbl_app.setStyleSheet(
            "font-size:20pt; font-weight:500; color:#202124; "
            "background:transparent; border:none;"
        )
        card_lay.addWidget(lbl_app)
        card_lay.addSpacing(6)

        lbl_sub = QLabel("Continuar a Video Manager")
        lbl_sub.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lbl_sub.setStyleSheet(
            "font-size:10pt; color:#5f6368; "
            "background:transparent; border:none;"
        )
        card_lay.addWidget(lbl_sub)
        card_lay.addSpacing(28)

        self._privacy_user_input = QLineEdit()
        self._privacy_user_input.setPlaceholderText("Correo electrónico o usuario")
        self._privacy_user_input.returnPressed.connect(self._attempt_unlock_privacy)
        card_lay.addWidget(self._privacy_user_input)
        card_lay.addSpacing(14)

        self._privacy_pass_input = QLineEdit()
        self._privacy_pass_input.setEchoMode(QLineEdit.EchoMode.Password)
        self._privacy_pass_input.setPlaceholderText("Contraseña")
        self._privacy_pass_input.returnPressed.connect(self._attempt_unlock_privacy)
        card_lay.addWidget(self._privacy_pass_input)
        card_lay.addSpacing(28)

        # fila inferior: enlace ayuda + botón rojo a la derecha
        actions = QHBoxLayout()
        actions.setContentsMargins(0, 0, 0, 0)
        lbl_help = QLabel("¿Olvidaste tu contraseña?")
        lbl_help.setStyleSheet(
            "color:#1a73e8; font-size:9pt; font-weight:600; "
            "background:transparent; border:none;"
        )
        actions.addWidget(lbl_help)
        actions.addStretch()
        btn_unlock = QPushButton("Siguiente")
        btn_unlock.setObjectName("btnUnlock")
        btn_unlock.setCursor(Qt.CursorShape.PointingHandCursor)
        btn_unlock.clicked.connect(self._attempt_unlock_privacy)
        actions.addWidget(btn_unlock)
        card_lay.addLayout(actions)
        card_lay.addSpacing(14)

        self._privacy_info_label = QLabel("")
        self._privacy_info_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._privacy_info_label.setStyleSheet(
            "color:#d93025; font-size:9pt; font-weight:600; "
            "background:transparent; border:none;"
        )
        card_lay.addWidget(self._privacy_info_label)

        outer.addWidget(card, alignment=Qt.AlignmentFlag.AlignVCenter)
        outer.addStretch(2)
        self._sync_privacy_overlay_geometry()

    def _sync_privacy_overlay_geometry(self):
        if self._privacy_overlay is not None:
            self._privacy_overlay.setGeometry(self.rect())

    def _setup_loading_overlay(self):
        ov = QFrame(self)
        ov.setObjectName("loadingOverlay")
        ov.setVisible(False)
        ov.setStyleSheet(
            "QFrame#loadingOverlay { background: rgba(10, 12, 22, 210); }"
            "QLabel { color: #c8d8f8; font-size: 13pt; font-weight: 600; }"
        )
        lay = QVBoxLayout(ov)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addStretch()
        spinner = _LoadingSpinner(64, ov)
        lay.addWidget(spinner, alignment=Qt.AlignmentFlag.AlignHCenter)
        lay.addSpacing(18)
        lbl = QLabel("Cargando…")
        lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lay.addWidget(lbl)
        lay.addStretch()
        self._loading_overlay = ov
        self._loading_label = lbl
        ov.setGeometry(self.rect())

    def _show_loading(self, msg: str = "Cargando…"):
        if self._loading_overlay is None:
            return
        self._loading_label.setText(msg)
        self._loading_overlay.setGeometry(self.rect())
        self._loading_overlay.show()
        self._loading_overlay.raise_()
        QApplication.processEvents()

    def _hide_loading(self):
        if self._loading_overlay is not None:
            self._loading_overlay.hide()

    def _toggle_privacy_lock(self):
        if self._privacy_locked:
            if self._privacy_pass_input is not None:
                self._privacy_pass_input.setFocus()
                self._privacy_pass_input.selectAll()
            return
        self._lock_privacy()

    def _lock_privacy(self):
        if self._privacy_locked:
            return
        self._privacy_locked = True
        self._privacy_info_label.setText("")
        self._privacy_pass_input.clear()
        if hasattr(self, "_privacy_user_input") and self._privacy_user_input is not None:
            self._privacy_user_input.clear()
        self._sync_privacy_overlay_geometry()
        self._privacy_overlay.show()
        self._privacy_overlay.raise_()
        if self._privacy_blur_effect is None:
            self._privacy_blur_effect = QGraphicsBlurEffect(self)
            self._privacy_blur_effect.setBlurRadius(28.0)
        cw = self.centralWidget()
        if cw is not None:
            cw.setGraphicsEffect(self._privacy_blur_effect)
        # QVideoWidget es nativo en Windows: pinta por encima del overlay.
        # Paramos el player y ocultamos todo su stack para que no se vea el rectángulo negro.
        if hasattr(self, "media_player") and self.media_player is not None:
            self.media_player.pause()
        if hasattr(self, "preview_stack") and self.preview_stack is not None:
            self.preview_stack.hide()
        if hasattr(self, "video_widget") and self.video_widget is not None:
            self.video_widget.hide()
        self.btn_privacy.setText("🔓 Bloqueada")
        self.btn_privacy.setEnabled(False)
        self._privacy_pass_input.setFocus()
        self.statusBar().showMessage("Aplicación bloqueada en modo privacidad", 2000)

    def _attempt_unlock_privacy(self):
        if not self._privacy_locked:
            return
        entered = self._privacy_pass_input.text() if self._privacy_pass_input else ""
        if entered != PRIVACY_UNLOCK_PASSWORD:
            self._privacy_info_label.setText("Contraseña incorrecta")
            if self._privacy_pass_input is not None:
                self._privacy_pass_input.selectAll()
                self._privacy_pass_input.setFocus()
            return
        self._unlock_privacy()

    def _unlock_privacy(self):
        self._privacy_locked = False
        cw = self.centralWidget()
        if cw is not None:
            cw.setGraphicsEffect(None)
        if hasattr(self, "preview_stack") and self.preview_stack is not None:
            self.preview_stack.show()
        if hasattr(self, "video_widget") and self.video_widget is not None:
            self.video_widget.show()
        if self._privacy_overlay is not None:
            self._privacy_overlay.hide()
        if self._privacy_pass_input is not None:
            self._privacy_pass_input.clear()
        if hasattr(self, "_privacy_user_input") and self._privacy_user_input is not None:
            self._privacy_user_input.clear()
        if self._privacy_info_label is not None:
            self._privacy_info_label.setText("")
        self.btn_privacy.setEnabled(True)
        self.btn_privacy.setText("🔒 Privacidad")
        self.statusBar().showMessage("Privacidad desactivada", 2000)

    def _notify(self, texto, ms=2500):
        self.statusBar().showMessage(texto, ms)

    def resizeEvent(self, ev):
        super().resizeEvent(ev)
        self._sync_privacy_overlay_geometry()
        if self._loading_overlay is not None:
            self._loading_overlay.setGeometry(self.rect())
        self._sync_sidebar_overlay_geometry()

    def _sync_sidebar_overlay_geometry(self):
        tree = getattr(self, "tree", None)
        btn_h = getattr(self, "btn_hamburger", None)
        if tree is None:
            return
        # Recalcular anchos visibles para mostrar todas las columnas del árbol.
        try:
            for col in (0, 1, 2, 3, 4, 5, 6, 7):
                tree.resizeColumnToContents(col)
        except Exception:
            pass
        icon_w = max(220, tree.iconSize().width() + 28)
        tree.setColumnWidth(0, icon_w)
        # Limitar Carpeta a un ancho razonable para evitar espacio vacío exagerado.
        folder_w = min(320, max(180, tree.columnWidth(1) + 8))
        tree.setColumnWidth(1, folder_w)
        for col in (2, 3, 4, 5, 6, 7):
            tree.setColumnWidth(col, max(52, tree.columnWidth(col) + 2))
        # Posicionar bajo el botón hamburguesa
        x = 12
        y = 60
        if btn_h is not None:
            try:
                pt = btn_h.mapToGlobal(btn_h.rect().bottomLeft())
                x = pt.x()
                y = pt.y() + 6
            except Exception:
                pass
        try:
            screen_geo = btn_h.screen().availableGeometry() if btn_h is not None and btn_h.screen() is not None else QApplication.primaryScreen().availableGeometry()
            visible_cols = [c for c in range(tree.columnCount()) if not tree.isColumnHidden(c)]
            content_w = sum(tree.columnWidth(c) for c in visible_cols)
            # Ajuste ceñido al contenido: evita hueco blanco extra al final.
            w = content_w + 18
            w = max(360, min(w, max(360, screen_geo.width() - 24)))
            h = max(200, screen_geo.bottom() - y - 20)
        except Exception:
            visible_cols = [c for c in range(tree.columnCount()) if not tree.isColumnHidden(c)]
            content_w = sum(tree.columnWidth(c) for c in visible_cols)
            w = max(360, min(content_w + 18, 980))
            h = max(200, self.height() - 80)
        tree.setGeometry(x, y, w, h)

    def _update_thumb_progress(self, actual=None):
        if self.thumb_total <= 0:
            self.thumb_progress_bar.setVisible(False)
            self.lbl_thumb_progreso.setVisible(False)
            self.lbl_thumb_progreso.setText("")
            return
        self.thumb_progress_bar.setRange(0, self.thumb_total)
        self.thumb_progress_bar.setValue(min(self.thumb_done, self.thumb_total))
        texto = f"Miniaturas: {self.thumb_done}/{self.thumb_total}"
        if actual:
            texto += f"  -  {actual}"
        self.lbl_thumb_progreso.setText(texto)
        self.thumb_progress_bar.setVisible(True)
        self.lbl_thumb_progreso.setVisible(True)

    def _finish_thumb_progress(self, mensaje=None):
        if mensaje:
            self.statusBar().showMessage(mensaje, 2500)
        self.thumb_total = 0
        self.thumb_done = 0
        self.thumb_progress_bar.setVisible(False)
        self.lbl_thumb_progreso.setVisible(False)
        self.lbl_thumb_progreso.setText("")

    def _fmt_ms(self, ms):
        total = max(0, int(ms // 1000))
        m, s = divmod(total, 60)
        return f"{m:02d}:{s:02d}"

    def _is_playing(self):
        if self.player_thread and self.player_thread.isRunning():
            return True
        return self.media_player.playbackState() != QMediaPlayer.PlaybackState.StoppedState

    def _on_player_position_changed(self, pos):
        if not self.playback_slider_dragging:
            self.player_slider.setValue(pos)
        self.player_pos.setText(f"{self._fmt_ms(pos)} / {self._fmt_ms(self.media_player.duration())}")
        fsw = self._video_only_fs_window
        if fsw is not None:
            if not self._fs_seek_dragging:
                fsw.seek_slider.setValue(pos)
            fsw.lbl_time.setText(f"{self._fmt_ms(pos)} / {self._fmt_ms(self.media_player.duration())}")

    def _on_player_duration_changed(self, dur):
        self.player_slider.setRange(0, max(0, dur))
        self.player_pos.setText(f"{self._fmt_ms(self.media_player.position())} / {self._fmt_ms(dur)}")
        fsw = self._video_only_fs_window
        if fsw is not None:
            fsw.seek_slider.setRange(0, max(0, dur))
            if not self._fs_seek_dragging:
                fsw.seek_slider.setValue(self.media_player.position())
            fsw.lbl_time.setText(f"{self._fmt_ms(self.media_player.position())} / {self._fmt_ms(dur)}")
        if self._pending_random_seek_ratio is not None:
            self._schedule_pending_random_seek(40)
        if self._pending_seek_ms is not None:
            self._schedule_pending_seek_ms(40)

    def _schedule_pending_seek_ms(self, delay_ms=120):
        if self._pending_seek_ms is None:
            return
        if self._pending_seek_ms_scheduled:
            return
        self._pending_seek_ms_scheduled = True
        QTimer.singleShot(max(0, int(delay_ms)), self._run_pending_seek_ms)

    def _run_pending_seek_ms(self):
        self._pending_seek_ms_scheduled = False
        if self._pending_seek_ms is None:
            return
        dur = int(self.media_player.duration())
        if dur <= 0:
            if self._pending_seek_ms_tries > 0:
                self._pending_seek_ms_tries -= 1
                self._schedule_pending_seek_ms(120)
            return
        target = max(0, min(int(self._pending_seek_ms), max(0, dur - 1)))
        self.media_player.setPosition(target)
        self.player_slider.setValue(target)
        current = int(self.media_player.position())
        if abs(current - target) > 1500 and self._pending_seek_ms_tries > 0:
            self._pending_seek_ms_tries -= 1
            self.media_player.play()
            self.media_player.pause()
            self._schedule_pending_seek_ms(120)
            return
        self._pending_seek_ms = None

    def _schedule_pending_random_seek(self, delay_ms=120):
        if self._pending_random_seek_ratio is None:
            return
        if self._pending_random_seek_scheduled:
            return
        self._pending_random_seek_scheduled = True
        QTimer.singleShot(max(0, int(delay_ms)), self._run_pending_random_seek)

    def _run_pending_random_seek(self):
        self._pending_random_seek_scheduled = False
        if self._pending_random_seek_ratio is None:
            return
        dur = int(self.media_player.duration())
        if dur <= 0:
            if self._pending_random_seek_tries > 0:
                self._pending_random_seek_tries -= 1
                self._schedule_pending_random_seek(120)
            return

        ratio = max(0.0, min(float(self._pending_random_seek_ratio), 1.0))
        target = int(max(0, dur - 1) * ratio)
        self.media_player.setPosition(target)
        self.player_slider.setValue(target)

        # Some formats on Qt/WMF ignore first seek; nudge and retry briefly.
        current = int(self.media_player.position())
        if abs(current - target) > 1500 and self._pending_random_seek_tries > 0:
            self._pending_random_seek_tries -= 1
            self.media_player.play()
            self.media_player.pause()
            self._schedule_pending_random_seek(120)
            return

        self._pending_random_seek_ratio = None
        if self._pending_random_pause:
            self.media_player.pause()
        self._pending_random_pause = False

    def ir_a_frame_aleatorio_biblioteca(self):
        """Jump to a random video in library and seek to a random playback point."""
        if not self.ruta_raiz:
            QMessageBox.information(self, "Sin biblioteca", "Primero abre una carpeta raíz.")
            return

        if not self.videos_base:
            self._scan()
        candidates = [
            v for v in self.videos_base
            if v.exists() and v.is_file() and v.suffix.lower() in EXTENSIONES_VIDEO
        ]
        if not candidates:
            QMessageBox.information(self, "Sin vídeos", "No hay vídeos en la biblioteca actual.")
            return

        elegido = random.choice(candidates)
        # Keep away from very beginning/end and land paused on the random frame.
        self._pending_random_seek_ratio = random.uniform(0.03, 0.97)
        start_in_play = bool(getattr(self, "chk_random_start_play", None) and self.chk_random_start_play.isChecked())
        self._pending_random_pause = not start_in_play
        self._pending_random_seek_tries = 12
        self._pending_random_seek_scheduled = False
        self.forzar_guardado_tiempo_actual()
        self.video_elegido = elegido
        self._reproducir_elegido()
        if self._pending_random_pause:
            # Pause immediately (usually still at t=0) and keep paused after deferred seek.
            self.media_player.pause()
        self._schedule_pending_random_seek(80)
        self._notify(f"Salto aleatorio: {elegido.name}")

    def _on_player_slider_pressed(self):
        self.playback_slider_dragging = True

    def _on_player_slider_moved(self, value):
        self.player_pos.setText(f"{self._fmt_ms(value)} / {self._fmt_ms(self.media_player.duration())}")

    def _clamp_manual_seek_target(self, target_ms):
        """Clamp manual seek away from exact EOF to keep last seconds seekable."""
        dur = int(self.media_player.duration())
        if dur <= 0:
            return max(0, int(target_ms))
        # Avoid exact EOF, which immediately emits EndOfMedia and skips next.
        max_target = max(0, dur - 250)
        return max(0, min(int(target_ms), max_target))

    def _get_seek_step_seconds(self):
        try:
            if hasattr(self, "cmb_seek_step") and self.cmb_seek_step is not None:
                v = self.cmb_seek_step.currentData()
                if v is not None:
                    return max(1, int(v))
                txt = self.cmb_seek_step.currentText()
                custom_v = self._parse_seek_step_value(txt)
                if custom_v is not None:
                    return custom_v
        except Exception:
            pass
        return 5

    def _get_player_autonext_seconds(self):
        try:
            if hasattr(self, "cmb_player_autonext") and self.cmb_player_autonext is not None:
                v = self.cmb_player_autonext.currentData()
                if v is not None:
                    return max(1, int(v))
                txt = self.cmb_player_autonext.currentText()
                custom_v = self._parse_seek_step_value(txt)
                if custom_v is not None:
                    return max(1, custom_v)
        except Exception:
            pass
        return 3

    def _parse_seek_step_value(self, value):
        text = str(value or "").strip().lower()
        if not text:
            return None
        m = re.search(r"\d+", text)
        if not m:
            return None
        return max(1, int(m.group(0)))

    def _normalize_seek_step_combo(self):
        if not hasattr(self, "cmb_seek_step") or self.cmb_seek_step is None:
            return
        step = self._parse_seek_step_value(self.cmb_seek_step.currentData())
        if step is None:
            step = self._parse_seek_step_value(self.cmb_seek_step.currentText())
        if step is None:
            step = 5
        idx = self.cmb_seek_step.findData(step)
        if idx >= 0:
            self.cmb_seek_step.setCurrentIndex(idx)
            return
        self.cmb_seek_step.setEditText(f"{step}s")

    def _on_player_autonext_step_changed(self):
        if not hasattr(self, "cmb_player_autonext") or self.cmb_player_autonext is None:
            return
        step = self._parse_seek_step_value(self.cmb_player_autonext.currentData())
        if step is None:
            step = self._parse_seek_step_value(self.cmb_player_autonext.currentText())
        if step is None:
            step = 3
        step = max(1, step)
        idx = self.cmb_player_autonext.findData(step)
        if idx >= 0:
            self.cmb_player_autonext.setCurrentIndex(idx)
        else:
            self.cmb_player_autonext.setEditText(f"{step}s")
        self._sync_player_autonext_button()
        if self._player_autonext_active:
            self._restart_player_autonext_timer()

    def _sync_player_autonext_button(self):
        if not hasattr(self, "btn_player_autonext") or self.btn_player_autonext is None:
            return
        secs = self._get_player_autonext_seconds()
        self.btn_player_autonext.blockSignals(True)
        self.btn_player_autonext.setChecked(self._player_autonext_active)
        self.btn_player_autonext.setText(f"■ Auto {secs}s" if self._player_autonext_active else f"▶ Auto {secs}s")
        self.btn_player_autonext.blockSignals(False)

    def _ensure_player_autonext_timer(self):
        if self._player_autonext_timer is None:
            self._player_autonext_timer = QTimer(self)
            self._player_autonext_timer.setSingleShot(True)
            self._player_autonext_timer.timeout.connect(self._fire_player_autonext)
        return self._player_autonext_timer

    def _stop_player_autonext(self, notify=False):
        self._player_autonext_active = False
        if self._player_autonext_timer is not None:
            try:
                self._player_autonext_timer.stop()
            except Exception:
                pass
        self._sync_player_autonext_button()
        if notify:
            self._notify("Auto salto desactivado", 1800)

    def _toggle_player_autonext(self, checked=None):
        if checked is None:
            checked = not self._player_autonext_active
        if not checked:
            self._stop_player_autonext(notify=True)
            return
        ruta_actual = Path(self.video_elegido) if self.video_elegido else None
        if not ruta_actual or not ruta_actual.exists() or ruta_actual.suffix.lower() not in EXTENSIONES_VIDEO:
            self._stop_player_autonext(notify=False)
            self._notify("Abre un vídeo para activar el auto salto", 2200)
            return
        self._stop_10s_preview()
        self._player_autonext_active = True
        self._restart_player_autonext_timer()
        secs = self._get_player_autonext_seconds()
        self._notify(f"Auto salto activado: siguiente en {secs}s", 2200)

    def _restart_player_autonext_timer(self):
        if not self._player_autonext_active:
            self._sync_player_autonext_button()
            return
        ruta_actual = Path(self.video_elegido) if self.video_elegido else None
        if not ruta_actual or not ruta_actual.exists() or ruta_actual.suffix.lower() not in EXTENSIONES_VIDEO:
            self._stop_player_autonext(notify=False)
            return
        secs = self._get_player_autonext_seconds()
        timer = self._ensure_player_autonext_timer()
        timer.stop()
        timer.start(max(1, secs) * 1000)
        self._sync_player_autonext_button()

    def _fire_player_autonext(self):
        if not self._player_autonext_active:
            return
        ruta_actual = Path(self.video_elegido) if self.video_elegido else None
        if not ruta_actual or not ruta_actual.exists() or ruta_actual.suffix.lower() not in EXTENSIONES_VIDEO:
            self._stop_player_autonext(notify=False)
            return
        self._pending_random_seek_ratio = random.uniform(0.03, 0.97)
        self._pending_random_pause = False
        self._pending_random_seek_tries = 12
        self._pending_random_seek_scheduled = False
        self.proximo_video()

    def _seek_relative_seconds(self, delta_seconds):
        if self.media_player.source().isEmpty():
            return
        now = int(self.media_player.position())
        delta = int(float(delta_seconds) * 1000.0)
        target = self._clamp_manual_seek_target(now + delta)
        self.media_player.setPosition(target)
        self.player_slider.setValue(target)

    def _on_player_slider_released(self):
        self.playback_slider_dragging = False
        target = self._clamp_manual_seek_target(self.player_slider.value())
        self.player_slider.setValue(target)
        self.media_player.setPosition(target)

    def _on_fs_seek_slider_pressed(self):
        self._fs_seek_dragging = True

    def _on_fs_seek_slider_moved(self, value):
        fsw = self._video_only_fs_window
        if fsw is not None:
            fsw.lbl_time.setText(f"{self._fmt_ms(value)} / {self._fmt_ms(self.media_player.duration())}")

    def _on_fs_seek_slider_released(self):
        fsw = self._video_only_fs_window
        if fsw is None:
            self._fs_seek_dragging = False
            return
        self._fs_seek_dragging = False
        target = self._clamp_manual_seek_target(fsw.seek_slider.value())
        fsw.seek_slider.setValue(target)
        self.player_slider.setValue(target)
        self.media_player.setPosition(target)

    def _toggle_pause_resume(self):
        if self.media_player.playbackState() == QMediaPlayer.PlaybackState.PlayingState:
            self.media_player.pause()
            return
        if self.media_player.source().isEmpty() and self.video_elegido and self.video_elegido.exists():
            self.reproducir_video_actual()
            return
        self.media_player.play()

    def _on_stop_player_clicked(self):
        # Persist playback stats and force an immediate folder-log checkpoint.
        self.forzar_guardado_tiempo_actual()

    def _log_playback_checkpoint(self, base_folder, motivo="playback_checkpoint"):
        """Write folder_views immediately and continue tracking from base_folder."""
        if not base_folder:
            return
        self._start_folder_view_log(base_folder)
        self._flush_folder_view_log(motivo)
        self._start_folder_view_log(base_folder)

    def _add_folder_log_seconds(self, carpeta, segundos, vistas=1):
        if not carpeta:
            return
        segundos = max(0, int(segundos))
        if segundos <= 0:
            return
        self._ensure_daily_log_date()
        bucket = self._folder_log_bucket(carpeta)
        if not bucket:
            return
        self._daily_folder_stats[bucket]["vistas"] += max(0, int(vistas))
        self._daily_folder_stats[bucket]["segundos"] += segundos
        self._folder_log_dirty = True
        self._folder_log_write_timer.start()

    def _add_photo_folder_log_seconds(self, carpeta, segundos, vistas=1):
        """Log accumulator for photo-slideshow mode: stores full folder path."""
        if not carpeta:
            return
        segundos = max(0, int(segundos))
        if segundos <= 0:
            return
        self._ensure_daily_log_date()
        try:
            bucket = str(Path(carpeta).resolve(strict=False)).replace("\\", "/").rstrip("/")
        except Exception:
            bucket = str(carpeta).replace("\\", "/").rstrip("/")
        if not bucket:
            return
        self._daily_folder_stats[bucket]["vistas"] += max(0, int(vistas))
        self._daily_folder_stats[bucket]["segundos"] += segundos
        self._folder_log_dirty = True
        self._folder_log_write_timer.start()

    def _flush_pending_daily_folder_log(self):
        if not getattr(self, "_folder_log_dirty", False):
            return
        self._write_daily_folder_log()
        self._folder_log_dirty = False

    def _on_volume_slider_changed(self, value):
        vol = value / 100.0
        self.audio_output.setVolume(vol)
        if value > 0:
            self._last_nonzero_volume = vol
        if value == 0:
            self.audio_output.setMuted(True)
        else:
            self.audio_output.setMuted(False)
        self._sync_mute_button_text()
        fsw = self._video_only_fs_window
        if fsw is not None and fsw.volume_slider.value() != int(value):
            fsw.volume_slider.blockSignals(True)
            fsw.volume_slider.setValue(int(value))
            fsw.volume_slider.blockSignals(False)

    def _on_fs_volume_slider_changed(self, value):
        if hasattr(self, "sld_volume") and self.sld_volume.value() != int(value):
            self.sld_volume.setValue(int(value))

    def _change_volume(self, delta):
        if hasattr(self, "sld_volume"):
            self.sld_volume.setValue(max(0, min(100, self.sld_volume.value() + delta)))

    def _sync_mute_button_text(self):
        if hasattr(self, "btn_mute"):
            self.btn_mute.setText("🔇 Mute" if self.audio_output.isMuted() else "🔊 Sonido")
        if hasattr(self, "sld_volume"):
            # Keep slider in sync when mute is toggled externally
            if self.audio_output.isMuted():
                self.sld_volume.blockSignals(True)
                self.sld_volume.setValue(0)
                self.sld_volume.blockSignals(False)
            elif self.sld_volume.value() == 0:
                self.sld_volume.blockSignals(True)
                restore_vol = max(1, int(getattr(self, "_last_nonzero_volume", 1.0) * 100))
                self.sld_volume.setValue(restore_vol)
                self.sld_volume.blockSignals(False)
        fsw = self._video_only_fs_window
        if fsw is not None and hasattr(self, "sld_volume"):
            fsw.volume_slider.blockSignals(True)
            fsw.volume_slider.setValue(int(self.sld_volume.value()))
            fsw.volume_slider.blockSignals(False)

    def _apply_system_audio_output_device(self):
        """Route playback to the current default audio output configured in Windows."""
        try:
            device = QMediaDevices.defaultAudioOutput()
            if device and not device.isNull():
                self.audio_output.setDevice(device)
        except Exception as e:
            LOGGER.warning("No se pudo aplicar dispositivo de audio por defecto: %s", e)

    def _on_audio_outputs_changed(self):
        """When Windows output device changes, follow the new default device."""
        self._apply_system_audio_output_device()

    def _toggle_mute(self):
        if self.audio_output.isMuted():
            restore_vol = max(0.01, float(getattr(self, "_last_nonzero_volume", 1.0)))
            self.audio_output.setVolume(restore_vol)
            self.audio_output.setMuted(False)
        else:
            current_vol = float(self.audio_output.volume())
            if current_vol > 0:
                self._last_nonzero_volume = current_vol
            self.audio_output.setMuted(True)
        self._sync_mute_button_text()

    def _toggle_fullscreen(self):
        video_widget = getattr(self, "video_widget", None)
        preview_stack = getattr(self, "preview_stack", None)
        if video_widget is None or preview_stack is None:
            return
        if self._video_only_fs_window is not None:
            self._exit_fullscreen()
            return

        preview_stack.setCurrentWidget(video_widget)
        fs_window = _VideoOnlyFullscreenWindow()
        fs_window.closed.connect(self._exit_fullscreen)
        fs_window.btn_fav.clicked.connect(self.toggle_favorito_y_renombrar)
        fs_window.btn_delete.clicked.connect(self.borrar_video)
        fs_window.btn_next.clicked.connect(self.proximo_video)
        fs_window.btn_prev.clicked.connect(self.video_anterior)
        fs_window.btn_play_pause.clicked.connect(self._toggle_pause_resume)
        # Update play/pause button label when playback state changes
        def _sync_fs_play_pause(state):
            if self._video_only_fs_window is None:
                return
            w = self._video_only_fs_window
            if state == QMediaPlayer.PlaybackState.PlayingState:
                w.btn_play_pause.setText("⏸ Pausa")
            else:
                w.btn_play_pause.setText("▶ Reanudar")
        self._fs_play_pause_conn = self.media_player.playbackStateChanged.connect(_sync_fs_play_pause)
        # Set initial play/pause label
        if self.media_player.playbackState() == QMediaPlayer.PlaybackState.PlayingState:
            fs_window.btn_play_pause.setText("⏸ Pausa")
        else:
            fs_window.btn_play_pause.setText("▶ Reanudar")
        # Title
        if self.video_elegido:
            fs_window.lbl_title.setText(Path(self.video_elegido).name)
        if self.video_elegido and Path(self.video_elegido).name.lower().startswith("top "):
            fs_window.btn_fav.setText("★ Quitar Top")
        else:
            fs_window.btn_fav.setText("★ Favorito")
        fs_window.volume_slider.setValue(self.sld_volume.value() if hasattr(self, "sld_volume") else 100)
        fs_window.volume_slider.valueChanged.connect(self._on_fs_volume_slider_changed)
        fs_window.seek_slider.sliderPressed.connect(self._on_fs_seek_slider_pressed)
        fs_window.seek_slider.sliderMoved.connect(self._on_fs_seek_slider_moved)
        fs_window.seek_slider.sliderReleased.connect(self._on_fs_seek_slider_released)
        fs_window.seek_slider.setRange(0, max(0, int(self.media_player.duration())))
        fs_window.seek_slider.setValue(int(self.media_player.position()))
        fs_window.lbl_time.setText(
            f"{self._fmt_ms(self.media_player.position())} / {self._fmt_ms(self.media_player.duration())}"
        )
        self._video_only_fs_window = fs_window
        try:
            self.media_player.setVideoOutput(fs_window.video_widget)
        except Exception:
            pass
        fs_window.showFullScreen()
        QTimer.singleShot(0, fs_window._restart_controls_timer)
        QTimer.singleShot(0, fs_window._layout_controls)
        try:
            if self.media_player.playbackState() == QMediaPlayer.PlaybackState.PlayingState:
                self.media_player.play()
        except Exception:
            pass
        if hasattr(self, "btn_fullscreen"):
            self.btn_fullscreen.setText("🗗 Salir pantalla completa")

    def _exit_fullscreen(self):
        video_widget = getattr(self, "video_widget", None)
        if video_widget is not None and video_widget.isFullScreen():
            video_widget.setFullScreen(False)

        # Disconnect play/pause state signal if connected
        conn = getattr(self, "_fs_play_pause_conn", None)
        if conn is not None:
            try:
                self.media_player.playbackStateChanged.disconnect(conn)
            except Exception:
                pass
            self._fs_play_pause_conn = None

        fs_window = self._video_only_fs_window
        self._video_only_fs_window = None
        self._fs_seek_dragging = False
        if fs_window is not None:
            try:
                fs_window.closed.disconnect(self._exit_fullscreen)
            except Exception:
                pass
            try:
                fs_window.hide()
                fs_window.close()
            except Exception:
                pass
            fs_window.deleteLater()

        try:
            self.media_player.setVideoOutput(video_widget)
        except Exception:
            pass
        self._restore_video_widget_after_fullscreen()
        try:
            if self.media_player.playbackState() == QMediaPlayer.PlaybackState.PlayingState:
                self.media_player.play()
        except Exception:
            pass

        if self.isFullScreen():
            if self._window_was_maximized_before_fullscreen:
                self.showMaximized()
            else:
                self.showNormal()
        if hasattr(self, "btn_fullscreen"):
            self.btn_fullscreen.setText("⛶ Pantalla completa")
        QTimer.singleShot(0, self._restore_video_widget_after_fullscreen)

    def _restore_video_widget_after_fullscreen(self):
        """Ensure the embedded video widget is visible again after leaving fullscreen."""
        video_widget = getattr(self, "video_widget", None)
        preview_stack = getattr(self, "preview_stack", None)
        if video_widget is None or preview_stack is None:
            return

        # On some Windows/Qt paths, leaving fullscreen may keep the video surface detached.
        if preview_stack.indexOf(video_widget) < 0:
            preview_stack.addWidget(video_widget)
        preview_stack.setCurrentWidget(video_widget)
        video_widget.showNormal()
        video_widget.show()
        video_widget.raise_()
        try:
            self.media_player.setVideoOutput(video_widget)
        except Exception:
            pass

    def _on_video_fullscreen_changed(self, fullscreen):
        if self._video_only_fs_window is not None:
            return
        if hasattr(self, "btn_fullscreen"):
            self.btn_fullscreen.setText("🗗 Salir pantalla completa" if fullscreen else "⛶ Pantalla completa")
        if not fullscreen:
            QTimer.singleShot(0, self._restore_video_widget_after_fullscreen)

    def _on_media_status_changed(self, status):
        if self._pending_random_seek_ratio is not None and status in (
            QMediaPlayer.MediaStatus.LoadedMedia,
            QMediaPlayer.MediaStatus.BufferedMedia,
            QMediaPlayer.MediaStatus.BufferingMedia,
        ):
            self._schedule_pending_random_seek(10)
        if status == QMediaPlayer.MediaStatus.EndOfMedia:
            if self.player_thread and self.player_thread.isRunning():
                self.player_thread.detener_reproductor()
                ruta = self._active_ruta_reproduccion or (str(self.video_elegido) if self.video_elegido else "")
                inicio = getattr(self.player_thread, '_inicio', time.time())
                t_sesion = max(0, int(time.time() - inicio))
                self.player_thread = None
                if ruta:
                    self._on_play_done(ruta, t_sesion)
                repetir_actual = bool(getattr(self, "btn_repeat", None) and self.btn_repeat.isChecked())
                autoplay_next = bool(getattr(self, "chk_autoplay_next", None) and self.chk_autoplay_next.isChecked())
                if repetir_actual and self.video_elegido and Path(self.video_elegido).exists():
                    QTimer.singleShot(0, lambda: self._reproducir_elegido(push_history=False))
                elif autoplay_next:
                    QTimer.singleShot(0, self._autoplay_next_video)
                else:
                    self._notify("Fin del video (autoplay desactivado)", 1800)
            return
        if status == QMediaPlayer.MediaStatus.InvalidMedia:
            self._notify("No se pudo reproducir este video", 2500)

    def _set_active_mode_button(self, modo):
        for m, b in self.mode_buttons.items():
            b.setChecked(m == modo)

    def _on_search_changed(self, texto):
        self.filtro_texto = texto.strip().lower()
        self._apply_table_filter()

    def _on_toggle_incluir_fotos(self, checked):
        self.incluir_fotos_gestion = bool(checked)
        # Recompute folder stats/counts for current mode (videos vs solo fotos).
        self._build_folder_aggregate_cache()
        self._build_tree()
        if self.carpeta_actual:
            self._refresh_list()

    def _apply_table_filter(self):
        if not hasattr(self, "tabla"):
            return
        visibles = 0
        for row in range(self.tabla.rowCount()):
            name_item = self.tabla.item(row, 1)
            nombre = name_item.text().lower() if name_item else ""
            mostrar = (not self.filtro_texto) or (self.filtro_texto in nombre)
            self.tabla.setRowHidden(row, not mostrar)
            if mostrar:
                visibles += 1
        if self.carpeta_actual:
            self.lbl_folder.setText(f"{self.carpeta_actual.name}  —  {visibles} visibles")

    def _round_avatar_pixmap(self, source_pm: QPixmap, size=54):
        if source_pm.isNull():
            return QPixmap()
        side = max(24, int(size))
        dpr = 2.0
        px = int(side * dpr)
        target = QPixmap(px, px)
        target.fill(Qt.GlobalColor.transparent)
        p = QPainter(target)
        p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        p.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, True)
        clip = QPainterPath()
        clip.addEllipse(0.0, 0.0, float(px), float(px))
        p.setClipPath(clip)
        scaled = source_pm.scaled(
            px,
            px,
            Qt.AspectRatioMode.KeepAspectRatioByExpanding,
            Qt.TransformationMode.SmoothTransformation,
        )
        sx = max(0, (scaled.width() - px) // 2)
        sy = max(0, (scaled.height() - px) // 2)
        p.drawPixmap(0, 0, scaled, sx, sy, px, px)
        p.setClipping(False)
        p.setPen(QPen(QColor("#ffffff"), max(1, int(2 * dpr))))
        p.drawEllipse(1, 1, px - 2, px - 2)
        p.end()
        target.setDevicePixelRatio(dpr)
        return target

    def _refresh_channel_bar(self, carpeta, video_count=0):
        if not hasattr(self, "lbl_channel_name"):
            return
        if carpeta is None:
            self.lbl_channel_name.setText("Canal")
            self.lbl_channel_meta.setText("Selecciona una carpeta")
            self.lbl_channel_avatar.clear()
            return
        try:
            carpeta = Path(carpeta)
        except Exception:
            self.lbl_channel_name.setText("Canal")
            self.lbl_channel_meta.setText("Carpeta no válida")
            self.lbl_channel_avatar.clear()
            return

        self.lbl_channel_name.setText(carpeta.name or "Canal")
        n = max(0, int(video_count or 0))
        self.lbl_channel_meta.setText(f"{n} video{'s' if n != 1 else ''}")

        icon = self._folder_icon(carpeta)
        avatar_base = icon.pixmap(96, 96) if icon is not None else QPixmap()
        avatar_pm = self._round_avatar_pixmap(avatar_base, self.lbl_channel_avatar.width())
        if avatar_pm.isNull():
            fallback = QPixmap(self.lbl_channel_avatar.size())
            fallback.fill(QColor("#d9d9d9"))
            avatar_pm = self._round_avatar_pixmap(fallback, self.lbl_channel_avatar.width())
        self.lbl_channel_avatar.setPixmap(avatar_pm)

    def _duration_text(self, ruta):
        ruta_str = str(ruta)
        if ruta_str in self.duration_cache:
            return self.duration_cache[ruta_str]
        dur = "N/A"
        if self.ffprobe_path:
            try:
                cmd = [self.ffprobe_path, '-v', 'error', '-show_entries', 'format=duration',
                       '-of', 'default=noprint_wrappers=1:nokey=1', ruta_str]
                seg = float(subprocess.check_output(cmd, timeout=5).decode().strip())
                dm, ds = divmod(int(seg), 60)
                dur = f"{dm}m {ds}s"
            except Exception:
                pass
        self.duration_cache[ruta_str] = dur
        return dur

    def _format_duration_badge(self, total_seconds: int):
        total_seconds = max(0, int(total_seconds or 0))
        hh, rem = divmod(total_seconds, 3600)
        mm, ss = divmod(rem, 60)
        if hh > 0:
            return f"{hh}:{mm:02d}:{ss:02d}"
        return f"{mm}:{ss:02d}"

    def _duration_badge_text(self, ruta: Path):
        ruta_str = str(ruta)
        if not hasattr(self, "_duration_badge_cache"):
            self._duration_badge_cache = {}
        cached = self._duration_badge_cache.get(ruta_str)
        if cached is not None:
            return cached

        secs = None
        txt = self.duration_cache.get(ruta_str)
        if txt:
            m = re.search(r"^(\d+)m\s+(\d+)s$", str(txt).strip())
            if m:
                secs = int(m.group(1)) * 60 + int(m.group(2))
            else:
                m2 = re.search(r"^(\d+)s$", str(txt).strip())
                if m2:
                    secs = int(m2.group(1))

        if secs is None and self.ffprobe_path:
            try:
                cmd = [
                    self.ffprobe_path,
                    '-v', 'error',
                    '-show_entries', 'format=duration',
                    '-of', 'default=noprint_wrappers=1:nokey=1',
                    ruta_str,
                ]
                seg = float(subprocess.check_output(cmd, timeout=2).decode().strip())
                secs = max(0, int(seg))
                dm, ds = divmod(secs, 60)
                self.duration_cache[ruta_str] = f"{dm}m {ds}s"
            except Exception:
                secs = None

        badge = self._format_duration_badge(secs) if secs is not None else ""
        self._duration_badge_cache[ruta_str] = badge
        return badge

    def _item_visto_marcador(self, ruta: Path):
        if ruta.suffix.lower() in EXTENSIONES_IMAGEN:
            n = ruta.name.lower()
            return "✓" if (n.startswith("rwd ") or n.startswith("top rwd ")) else ""
        return "✓" if ruta.stem.lower().endswith("_rwd") else ""

    def _is_video_revisado(self, ruta: Path):
        if not ruta:
            return False
        ext = ruta.suffix.lower()
        if ext in EXTENSIONES_IMAGEN:
            n = ruta.name.lower()
            return n.startswith("rwd ") or n.startswith("top rwd ")
        if ext not in EXTENSIONES_VIDEO:
            return False
        stem = ruta.stem.lower()
        return stem.endswith("_rwd") or bool(re.search(r"_rwd_\d+$", stem))

    def _setup_idle_hashing(self):
        self.idle_hash_timer = QTimer(self)
        self.idle_hash_timer.setInterval(2500)
        self.idle_hash_timer.timeout.connect(self._tick_idle_hashing)
        self.idle_hash_timer.start()

    def _setup_tree_building(self):
        self._tree_build_timer = QTimer(self)
        self._tree_build_timer.setInterval(0)
        self._tree_build_timer.timeout.connect(self._process_tree_build_chunk)

    def _setup_folder_thumbnail_rotation(self):
        self._folder_thumb_timer = QTimer(self)
        self._folder_thumb_timer.setInterval(60000)
        self._folder_thumb_timer.timeout.connect(self._on_folder_thumbnail_rotation_tick)
        self._folder_thumb_timer.start()

    def _on_folder_thumbnail_rotation_tick(self):
        minute_key = int(time.time() // 60)
        if minute_key == self._folder_icon_minute:
            return
        self._folder_icon_minute = minute_key
        self.folder_icon_cache.clear()
        # También refresca el conteo de miniaturas por carpeta para que el
        # badge refleje altas/bajas hechas desde otras ventanas.
        self._refresh_folder_thumb_ids_cache()
        self._refresh_tree_folder_icons()

    def _refresh_tree_folder_icons(self):
        if not hasattr(self, "tree"):
            return
        # Refresca conteo de miniaturas por carpeta para que el badge inferior
        # derecho refleje cambios hechos desde otras ventanas.
        self._refresh_folder_thumb_ids_cache()
        self.folder_icon_cache.clear()
        for i in range(self.tree.topLevelItemCount()):
            item = self.tree.topLevelItem(i)
            if item:
                self._refresh_tree_folder_icons_from_item(item)

    def _refresh_tree_folder_icons_from_item(self, item):
        ruta = item.data(0, Qt.ItemDataRole.UserRole)
        if ruta:
            icon = self._folder_icon(Path(ruta))
            if icon:
                item.setIcon(0, icon)
            else:
                item.setIcon(0, QIcon())
        for i in range(item.childCount()):
            self._refresh_tree_folder_icons_from_item(item.child(i))

    def _cancel_tree_build(self):
        if self._tree_build_timer and self._tree_build_timer.isActive():
            self._tree_build_timer.stop()
        self._tree_build_queue.clear()
        self._tree_build_nodes = 0

    def _process_tree_build_chunk(self):
        if not self._tree_build_queue:
            if self._tree_build_timer and self._tree_build_timer.isActive():
                self._tree_build_timer.stop()
            self._hide_loading()
            self.statusBar().showMessage("Estructura cargada", 1200)
            return

        vetadas_lower = {v.lower() for v in self.carpetas_vetadas}
        start = time.perf_counter()
        time_budget = max(0.008, self._tree_chunk_target_ms / 1000.0)
        item_limit = max(self._tree_chunk_min_items, min(self._tree_chunk_max_items, int(self._tree_chunk_items)))
        processed = 0
        while self._tree_build_queue and processed < item_limit:
            parent_item, carpeta, dirs, idx = self._tree_build_queue.popleft()
            if dirs is None:
                try:
                    dirs = sorted(
                        [d for d in carpeta.iterdir() if d.is_dir() and not d.name.startswith('.')],
                        key=lambda x: x.name.lower()
                    )
                except (PermissionError, OSError):
                    dirs = []
                idx = 0

            while idx < len(dirs):
                d = dirs[idx]
                if d.name.lower() in vetadas_lower:
                    idx += 1
                    continue
                try:
                    key = str(d).replace('\\', '/').rstrip('/')
                    n = int(self.folder_agg_cache.get(key, {}).get('total_videos', 0))
                    label = f"{d.name}  ({n})"
                    has_dups = self._folder_has_duplicates(d)
                    if has_dups:
                        label += " !"
                    item = QTreeWidgetItem([" ", label, "", "", "", "", "", ""])
                    icon = self._folder_icon(d)
                    if icon:
                        item.setIcon(0, icon)
                    item.setData(0, Qt.ItemDataRole.UserRole, str(d))
                    self._set_folder_stats(item, d)
                    self._paint_folder_duplicate_state(item, has_dups)
                    parent_item.addChild(item)
                    self._tree_build_queue.append((item, d, None, 0))
                    self._tree_build_nodes += 1
                    processed += 1
                    if processed >= item_limit:
                        idx += 1
                        # Requeue this parent to continue with remaining child dirs.
                        self._tree_build_queue.appendleft((parent_item, carpeta, dirs, idx))
                        break
                except Exception as e:
                    LOGGER.warning("Error creando nodo de carpeta %s: %s", d, e)
                idx += 1

            if (time.perf_counter() - start) >= time_budget:
                break

        elapsed_ms = (time.perf_counter() - start) * 1000.0
        if elapsed_ms > (self._tree_chunk_target_ms * 1.35):
            self._tree_chunk_items = max(self._tree_chunk_min_items, int(self._tree_chunk_items * 0.75))
        elif elapsed_ms < (self._tree_chunk_target_ms * 0.65) and processed >= int(item_limit * 0.85):
            self._tree_chunk_items = min(self._tree_chunk_max_items, int(self._tree_chunk_items * 1.25))

        if self._tree_build_queue:
            self.statusBar().showMessage(
                f"Cargando estructura... {self._tree_build_nodes} carpetas (lote {int(self._tree_chunk_items)})"
            )

    # ------------------------------------------------------------------
    # Hover thumbnail cycling
    # ------------------------------------------------------------------
    def _setup_hover_preview(self):
        self._hover_frame_signal.connect(self._on_hover_frame_ready)
        self._hover_timer = QTimer(self)
        self._hover_timer.setInterval(800)
        self._hover_timer.timeout.connect(self._cycle_hover_frame)
        self.tabla.viewport().installEventFilter(self)
        self.tabla.viewport().setMouseTracking(True)

    def _toggle_sidebar(self):
        tree = getattr(self, "tree", None)
        if tree is None:
            return
        new_state = not tree.isVisible()
        if new_state:
            self._sync_sidebar_overlay_geometry()
            tree.show()
            tree.raise_()
            tree.setFocus()
        else:
            tree.hide()
        btn_h = getattr(self, "btn_hamburger", None)
        if btn_h is not None:
            try:
                btn_h.setChecked(new_state)
            except Exception:
                pass

    def eventFilter(self, obj, event):
        from PyQt6.QtCore import QEvent
        if event.type() == QEvent.Type.Wheel:
            dash_targets = getattr(self, "_dashboard_wheel_targets", set())
            dashboard_src = None
            if obj in dash_targets:
                dashboard_src = obj
            else:
                for w in dash_targets:
                    if obj is w.viewport():
                        dashboard_src = w
                        break
            dashboard_scroll = getattr(self, "dashboard_scroll", None)
            if (
                dashboard_src is not None
                and dashboard_scroll is not None
                and self._dashboard_home_active
                and dashboard_scroll.isVisible()
            ):
                vbar = dashboard_scroll.verticalScrollBar()
                delta_y = int(event.angleDelta().y())
                if delta_y == 0:
                    delta_y = int(event.pixelDelta().y())
                if delta_y != 0:
                    step = max(24, int(vbar.singleStep() or 36))
                    ticks = delta_y / 120.0
                    if ticks == 0:
                        ticks = 1 if delta_y > 0 else -1
                    vbar.setValue(vbar.value() - int(round(ticks * step * 3)))
                return True
        if obj is getattr(self, "tree", None) and event.type() == QEvent.Type.Hide:
            btn_h = getattr(self, "btn_hamburger", None)
            if btn_h is not None:
                try:
                    btn_h.setChecked(False)
                except Exception:
                    pass
        # Auto-cerrar el árbol (sidebar) al pulsar fuera de él o del botón hamburguesa
        try:
            if event.type() == QEvent.Type.MouseButtonPress:
                tree = getattr(self, "tree", None)
                btn_h = getattr(self, "btn_hamburger", None)
                if tree is not None and tree.isVisible() and isinstance(obj, QWidget):
                    inside_tree = (obj is tree) or tree.isAncestorOf(obj)
                    inside_btn = btn_h is not None and (obj is btn_h or btn_h.isAncestorOf(obj))
                    if not inside_tree and not inside_btn:
                        tree.hide()
                        if btn_h is not None:
                            try:
                                btn_h.setChecked(False)
                            except Exception:
                                pass
        except Exception:
            pass
        photo_preview_label = getattr(self, "photo_preview_label", None)
        video_widget = getattr(self, "video_widget", None)
        tabla = getattr(self, "tabla", None)
        if tabla is not None and obj is tabla.viewport():
            t = event.type()
            if t == QEvent.Type.MouseMove:
                row = tabla.rowAt(event.pos().y())
                if row >= 0 and row != self._hover_row:
                    self._stop_hover_preview()
                    self._start_hover_preview(row)
                elif row < 0:
                    self._stop_hover_preview()
            elif t == QEvent.Type.Leave:
                self._stop_hover_preview()
        elif photo_preview_label is not None and obj is photo_preview_label and event.type() == QEvent.Type.Resize:
            self._render_photo_preview()
        elif video_widget is not None and obj is video_widget and event.type() == QEvent.Type.MouseButtonDblClick:
            if event.button() == Qt.MouseButton.LeftButton:
                self._toggle_fullscreen()
                return True
        elif video_widget is not None and obj is video_widget and event.type() == QEvent.Type.MouseButtonPress:
            if event.button() == Qt.MouseButton.LeftButton:
                self._toggle_pause_resume()
                return True
        return super().eventFilter(obj, event)

    def _render_photo_preview(self):
        if not hasattr(self, "photo_preview_label"):
            return
        if self._photo_preview_pm is None or self._photo_preview_pm.isNull():
            return
        scaled = self._photo_preview_pm.scaled(
            self.photo_preview_label.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self.photo_preview_label.setPixmap(scaled)

    def _mostrar_foto_en_panel(self, ruta: Path):
        pm = QPixmap(str(ruta))
        if pm.isNull():
            QMessageBox.warning(self, "Foto", "No se pudo cargar la imagen seleccionada.")
            return
        self._photo_preview_pm = pm
        self.photo_preview_label.setToolTip(str(ruta))
        self._render_photo_preview()
        self.preview_stack.setCurrentWidget(self.photo_preview_label)
        self.player_slider.setRange(0, 0)
        self.player_slider.setValue(0)
        self.player_pos.setText("00:00 / 00:00")

    def _start_hover_preview(self, row):
        item = self.tabla.item(row, 0)
        if not item:
            return
        ruta_str = item.data(Qt.ItemDataRole.UserRole)
        if not ruta_str or not Path(ruta_str).is_file():
            return
        self._hover_row = row
        self._hover_video = ruta_str
        self._hover_orig_icon = item.icon()
        self._cycle_hover_frame()   # immediate first frame
        self._hover_timer.start()

    def _stop_hover_preview(self):
        if self._hover_timer and self._hover_timer.isActive():
            self._hover_timer.stop()
        if self._hover_row >= 0 and self._hover_orig_icon is not None:
            item = self.tabla.item(self._hover_row, 0)
            if item:
                item.setIcon(self._hover_orig_icon)
        self._hover_row = -1
        self._hover_video = None
        self._hover_orig_icon = None

    def _cycle_hover_frame(self):
        if self._hover_row < 0 or not self._hover_video:
            return
        ruta = self._hover_video
        row = self._hover_row

        def worker():
            try:
                import cv2
                import random as _rnd
                cap = cv2.VideoCapture(ruta)
                total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                if total < 2:
                    cap.release()
                    return
                cap.set(cv2.CAP_PROP_POS_FRAMES, _rnd.randint(0, total - 1))
                ok, frame = cap.read()
                cap.release()
                if not ok:
                    return
                rgb = frame[:, :, ::-1].copy()
                h, w = rgb.shape[:2]
                # Pack w,h into first 8 bytes so the slot can reconstruct QImage
                import struct
                header = struct.pack('>II', w, h)
                self._hover_frame_signal.emit(row, header + rgb.tobytes())
            except Exception:
                pass

        threading.Thread(target=worker, daemon=True).start()

    @pyqtSlot(int, bytes)
    def _on_hover_frame_ready(self, row, data):
        if row != self._hover_row:
            return  # user already moved away
        import struct
        from PyQt6.QtGui import QImage
        w, h = struct.unpack('>II', data[:8])
        rgb_bytes = data[8:]
        qi = QImage(rgb_bytes, w, h, w * 3, QImage.Format.Format_RGB888)
        pm = QPixmap.fromImage(qi)
        pm_hi = pm.scaled(320, 180, Qt.AspectRatioMode.KeepAspectRatio,
                          Qt.TransformationMode.SmoothTransformation)
        pm_hi.setDevicePixelRatio(2.0)
        item = self.tabla.item(row, 0)
        if item:
            item.setIcon(QIcon(pm_hi))


    def _rebuild_idle_hash_queue(self):
        self.idle_hash_queue = []
        if not self.ruta_raiz:
            return
        # Reutiliza el último escaneo y una sola consulta por lote en vez de
        # un nuevo rglob + tiene_hash() por archivo.
        self._ensure_all_files_cache()
        try:
            hash_set = self.db.obtener_rutas_con_hash()
        except Exception:
            hash_set = set()
        all_cache = getattr(self, '_all_files_cache', None) or []
        exts_hasheable = EXTENSIONES_VIDEO | EXTENSIONES_IMAGEN
        self.idle_hash_queue = [
            str(f) for f in all_cache
            if f.suffix.lower() in exts_hasheable
            and str(f).replace('\\', '/') not in hash_set
        ]

    def _rebuild_duplicate_index(self):
        self.duplicate_paths = set()
        self.duplicate_map = {}
        self.duplicate_folder_prefixes = set()
        self.duplicate_folder_prefixes_video = set()
        self.duplicate_folder_prefixes_image = set()
        # Set de rutas conocidas (calculado en _scan) para evitar Path.exists() por archivo.
        existing = getattr(self, '_existing_paths_norm', None)
        grupos = defaultdict(list)
        for row in self.db.obtener_todos_hashes_visuales():
            ruta = row.get('ruta')
            hash_data = row.get('hash_visual')
            if not ruta or not hash_data:
                continue
            ruta_norm = str(ruta).replace('\\', '/')
            if existing is not None:
                if ruta_norm not in existing:
                    continue
            else:
                if not Path(ruta_norm).exists():
                    continue
            firma = self._hash_signature(hash_data)
            if not firma:
                continue
            grupos[firma].append(ruta_norm)

        for rutas in grupos.values():
            rutas_unicas = list(dict.fromkeys(rutas))
            if len(rutas_unicas) < 2:
                continue
            for ruta in rutas_unicas:
                self.duplicate_paths.add(ruta)
                self.duplicate_map[ruta] = [r for r in rutas_unicas if r != ruta]

        for ruta in self.duplicate_paths:
            try:
                p = Path(ruta)
                ext = p.suffix.lower()
                for anc in p.parents:
                    pref = str(anc).replace('\\', '/').rstrip('/')
                    self.duplicate_folder_prefixes.add(pref)
                    if ext in EXTENSIONES_IMAGEN:
                        self.duplicate_folder_prefixes_image.add(pref)
                    elif ext in EXTENSIONES_VIDEO:
                        self.duplicate_folder_prefixes_video.add(pref)
            except Exception:
                continue

    def _hash_signature(self, hash_data):
        """Normaliza hash visual para evitar falsos duplicados por datos viejos/malformados.
        Acepta tanto hashes de video (≥5 frames) como de imagen (1 frame)."""
        if not isinstance(hash_data, list):
            return None
        n = len(hash_data)
        if n < 1:
            return None
        # Videos requieren al menos 5 frames; imágenes usan exactamente 1 frame
        if n < 5 and n != 1:
            return None
        frames_norm = []
        for frame in hash_data:
            if not isinstance(frame, list) or len(frame) < 16:
                return None
            bits = []
            for v in frame:
                bits.append('1' if v in (1, True, '1', 'true', 'True') else '0')
            frames_norm.append(''.join(bits))
        return '|'.join(frames_norm)

    def _folder_has_duplicates(self, carpeta):
        pref = str(carpeta).replace('\\', '/').rstrip('/')
        if self.incluir_fotos_gestion:
            return pref in self.duplicate_folder_prefixes_image
        return pref in self.duplicate_folder_prefixes_video

    def _paint_folder_duplicate_state(self, item, has_duplicates):
        """Paint folder row red when it contains duplicates."""
        if not has_duplicates:
            return
        rojo = QColor("#d11a2a")
        for c in range(0, 8):
            item.setForeground(c, rojo)

    def _find_folder_image(self, carpeta):
        # Cache: en HDD, listar cada carpeta repetidamente cuando se redibujan
        # iconos del árbol es muy costoso.
        key = str(carpeta)
        if key in self._folder_image_cache:
            return self._folder_image_cache[key]
        preferidas = (
            'folder.jpg', 'folder.jpeg', 'folder.png',
            'cover.jpg', 'cover.jpeg', 'cover.png',
            'poster.jpg', 'poster.jpeg', 'poster.png'
        )
        result = None
        for nombre in preferidas:
            p = carpeta / nombre
            if p.exists() and p.is_file():
                result = p
                break
        if result is None:
            try:
                with os.scandir(carpeta) as it:
                    for entry in it:
                        try:
                            if entry.is_file(follow_symlinks=False):
                                ext = os.path.splitext(entry.name)[1].lower()
                                if ext in EXTENSIONES_IMAGEN:
                                    result = Path(entry.path)
                                    break
                        except OSError:
                            continue
            except (PermissionError, OSError):
                result = None
        self._folder_image_cache[key] = result
        return result

    def _folder_daily_log_path(self, fecha_str=None):
        fecha = fecha_str or self._daily_folder_log_date
        LOG_DIR_PATH.mkdir(parents=True, exist_ok=True)
        return LOG_DIR_PATH / f"{FOLDER_VIEWS_LOG_BASENAME}_{fecha}.log"

    def abrir_carpeta_log(self):
        try:
            self._flush_pending_daily_folder_log()
            LOG_DIR_PATH.mkdir(parents=True, exist_ok=True)
            override_totals_path = LOG_DIR_PATH / f"{FOLDER_VIEWS_LOG_BASENAME}_totales_override.log"
            log_files = sorted(
                [p for p in LOG_DIR_PATH.glob("*.log") if p.is_file() and p != override_totals_path],
                key=lambda p: p.name.lower(),
                reverse=True,
            )
            if not log_files and not override_totals_path.exists():
                QMessageBox.information(self, "LOG", "No hay archivos .log en la carpeta LOG.")
                return

            dlg = QDialog(self)
            dlg.setWindowTitle("Ver / Editar LOG")
            dlg.resize(980, 780)

            lay = QVBoxLayout(dlg)
            lay.setContentsMargins(10, 10, 10, 10)
            lay.setSpacing(8)

            # ── Part 1: Cumulative totals across all log files ──
            lbl_totales = QLabel("Minutos totales por carpeta (todos los días):")
            lbl_totales.setObjectName("count")
            lay.addWidget(lbl_totales)

            txt_totales = QPlainTextEdit()
            txt_totales.setReadOnly(False)
            txt_totales.setLineWrapMode(QPlainTextEdit.LineWrapMode.NoWrap)
            txt_totales.setFixedHeight(180)
            txt_totales.setStyleSheet(
                "QPlainTextEdit { background:#141414; color:#c8d0da; font-family:Consolas,monospace; font-size:12px; }"
            )
            txt_totales.setToolTip("Editable: aquí puedes ajustar manualmente los totales acumulados mostrados.")
            lay.addWidget(txt_totales)

            top_totals_row = QHBoxLayout()
            top_totals_row.setSpacing(6)
            btn_save_totales = QPushButton("Guardar totales")
            btn_save_totales.setToolTip("Guarda esta parte superior como acumulado persistente")
            top_totals_row.addWidget(btn_save_totales)
            top_totals_row.addStretch()
            lay.addLayout(top_totals_row)

            def _secs_to_points(secs):
                # Puntos del log: minutos redondeados hacia arriba si >=30s.
                import math as _math
                secs = max(0, int(secs))
                if secs < 30:
                    return 0
                return int(_math.ceil(secs / 60.0))

            def _parse_log_lines_to_secs(text):
                import re as _re
                totals = {}  # carpeta -> segundos acumulados
                for raw in text.splitlines():
                    line = raw.strip()
                    if not line or " - " not in line:
                        continue
                    carpeta, resumen = line.split(" - ", 1)
                    carpeta = carpeta.strip()
                    resumen = resumen.strip()
                    if not carpeta:
                        continue
                    m_s = _re.search(r"^(\d+)s$", resumen)
                    m_m = _re.search(r"^(\d+)$", resumen)
                    if m_s:
                        secs = int(m_s.group(1))
                    elif m_m:
                        secs = int(m_m.group(1)) * 60
                    else:
                        continue
                    totals[carpeta] = totals.get(carpeta, 0) + max(0, secs)
                return totals

            def _format_secs_sorted_lines(totals, as_seconds=True):
                if not totals:
                    return "(sin datos)" if as_seconds else ""
                ordered = sorted(
                    totals.items(),
                    key=lambda kv: (-_secs_to_points(kv[1]), -int(kv[1]), str(kv[0]).lower())
                )
                if as_seconds:
                    return "\n".join(f"{carpeta} - {int(secs)}s" for carpeta, secs in ordered)
                return "\n".join(f"{carpeta} - {_secs_to_points(secs)}" for carpeta, secs in ordered)

            def _build_totales():
                # Si existe override, ese acumulado manda sobre el cálculo por archivos diarios.
                if override_totals_path.exists():
                    try:
                        override_totals = _parse_log_lines_to_secs(
                            override_totals_path.read_text(encoding="utf-8", errors="replace")
                        )
                        return _format_secs_sorted_lines(override_totals, as_seconds=True)
                    except Exception:
                        pass
                totals = {}  # carpeta -> total_seconds
                for p in log_files:
                    try:
                        file_totals = _parse_log_lines_to_secs(
                            p.read_text(encoding="utf-8", errors="replace")
                        )
                        for carpeta, secs in file_totals.items():
                            totals[carpeta] = totals.get(carpeta, 0) + secs
                    except Exception:
                        pass
                return _format_secs_sorted_lines(totals, as_seconds=True)

            txt_totales.setPlainText(_build_totales())

            # ── Part 2: Current daily log (editable) ──
            lbl_diario = QLabel("Log del día (minutos, >30 s):")
            lbl_diario.setObjectName("count")
            lay.addWidget(lbl_diario)

            top = QHBoxLayout()
            top.setSpacing(6)
            top.addWidget(QLabel("Archivo:"))
            cmb = QComboBox()
            for p in log_files:
                cmb.addItem(p.name, str(p))
            top.addWidget(cmb, 1)
            btn_reload = QPushButton("Recargar")
            btn_save = QPushButton("Guardar")
            btn_del_total = QPushButton("Borrar total carpeta")
            top.addWidget(btn_reload)
            top.addWidget(btn_save)
            top.addWidget(btn_del_total)
            lay.addLayout(top)

            editor = QPlainTextEdit()
            editor.setLineWrapMode(QPlainTextEdit.LineWrapMode.NoWrap)
            lay.addWidget(editor, 1)

            status = QLabel("")
            status.setObjectName("count")
            lay.addWidget(status)

            btns = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
            btns.rejected.connect(dlg.reject)
            lay.addWidget(btns)

            def _current_path():
                data = cmb.currentData()
                return Path(data) if data else None

            def _load_selected():
                p = _current_path()
                if not p or not p.exists():
                    editor.setPlainText("")
                    status.setText("Archivo no encontrado")
                    return
                try:
                    totals = _parse_log_lines_to_secs(p.read_text(encoding="utf-8", errors="replace"))
                    editor.setPlainText(_format_secs_sorted_lines(totals, as_seconds=False))
                    status.setText(f"Cargado: {p.name}")
                except Exception as ex:
                    status.setText(f"Error al leer: {ex}")

            def _save_selected():
                p = _current_path()
                if not p:
                    return
                try:
                    totals = _parse_log_lines_to_secs(editor.toPlainText())
                    # Guarda siempre ordenado por puntos descendentes.
                    p.write_text(_format_secs_sorted_lines(totals, as_seconds=False), encoding="utf-8")
                    editor.setPlainText(_format_secs_sorted_lines(totals, as_seconds=False))
                    status.setText(f"Guardado: {p.name}")
                    txt_totales.setPlainText(_build_totales())
                except Exception as ex:
                    status.setText(f"Error al guardar: {ex}")

            def _delete_total_folder():
                try:
                    totals = _parse_log_lines_to_secs(txt_totales.toPlainText())
                    carpetas = sorted(totals.keys(), key=lambda s: s.lower())
                    if not carpetas:
                        QMessageBox.information(self, "LOG", "No hay carpetas acumuladas para borrar.")
                        return
                    carpeta, ok = QInputDialog.getItem(
                        self,
                        "Borrar total por carpeta",
                        "Carpeta a eliminar del acumulado:",
                        carpetas,
                        0,
                        False,
                    )
                    if not ok or not carpeta:
                        return
                    borradas = 0
                    target_files = [override_totals_path] if override_totals_path.exists() else list(log_files)
                    for p in target_files:
                        try:
                            file_totals = _parse_log_lines_to_secs(
                                p.read_text(encoding="utf-8", errors="replace")
                            )
                            if carpeta not in file_totals:
                                continue
                            file_totals.pop(carpeta, None)
                            p.write_text(
                                _format_secs_sorted_lines(file_totals, as_seconds=(p == override_totals_path)),
                                encoding="utf-8",
                            )
                            borradas += 1
                        except Exception:
                            continue
                    txt_totales.setPlainText(_build_totales())
                    _load_selected()
                    status.setText(f"Carpeta borrada del acumulado: {carpeta} ({borradas} archivo(s) de log)")
                except Exception as ex:
                    status.setText(f"Error borrando acumulado: {ex}")

            def _save_totales_override():
                try:
                    totals = _parse_log_lines_to_secs(txt_totales.toPlainText())
                    override_totals_path.write_text(
                        _format_secs_sorted_lines(totals, as_seconds=True),
                        encoding="utf-8",
                    )
                    txt_totales.setPlainText(_build_totales())
                    status.setText(f"Totales guardados: {override_totals_path.name}")
                except Exception as ex:
                    status.setText(f"Error guardando totales: {ex}")

            cmb.currentIndexChanged.connect(lambda _i: _load_selected())
            btn_reload.clicked.connect(_load_selected)
            btn_save.clicked.connect(_save_selected)
            btn_del_total.clicked.connect(_delete_total_folder)
            btn_save_totales.clicked.connect(_save_totales_override)

            _load_selected()
            dlg.exec()
        except Exception as e:
            QMessageBox.warning(self, "LOG", f"No se pudo abrir el visor de LOG:\n{e}")

    def _restaurar_bd_desde_metadatos(self):
        if not self.ruta_raiz:
            QMessageBox.warning(self, "Restaurar BD", "Primero abre una carpeta raíz.")
            return
        if not self.ffprobe_path:
            QMessageBox.warning(
                self,
                "Restaurar BD",
                "No se encontró ffprobe.\n\n"
                "Configura FFPROBE_PATH o instala ffprobe en el PATH de Windows.",
            )
            return
        reply = QMessageBox.question(
            self, "Restaurar BD desde metadatos",
            f"Se leerá el tag 'comment' de todos los videos bajo:\n{self.ruta_raiz}\n\n"
            "Los datos de reproduciones y tiempo visto de cada video se copiarán a la BD "
            "(sin sobreescribir valores mayores ya existentes).\n\n¿Continuar?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        if reply != QMessageBox.StandardButton.Yes:
            return

        self.progress_bar.setRange(0, 0)
        self.progress_bar.setVisible(True)
        self.lbl_progreso.setText("Leyendo metadatos…")
        self.lbl_progreso.setVisible(True)
        QApplication.processEvents()

        def worker():
            def cb(actual, total, nombre):
                QMetaObject.invokeMethod(
                    self, "_on_restaurar_progress",
                    Qt.ConnectionType.QueuedConnection,
                    Q_ARG(int, actual), Q_ARG(int, total), Q_ARG(str, nombre),
                )
            try:
                self._restore_last_error = ""
                restaurados, sin_datos, errores = self.db.restaurar_desde_metadatos(
                    self.ruta_raiz, progress_callback=cb, ffprobe_path=self.ffprobe_path
                )
                QMetaObject.invokeMethod(
                    self, "_on_restaurar_done",
                    Qt.ConnectionType.QueuedConnection,
                    Q_ARG(int, restaurados), Q_ARG(int, sin_datos), Q_ARG(int, errores),
                )
            except Exception as e:
                self._restore_last_error = f"{type(e).__name__}: {e}\n\n{traceback.format_exc()}"
                QMetaObject.invokeMethod(
                    self, "_on_restaurar_done",
                    Qt.ConnectionType.QueuedConnection,
                    Q_ARG(int, -1), Q_ARG(int, 0), Q_ARG(int, 0),
                )
                LOGGER.exception("Error en restaurar_desde_metadatos: %s", e)

        threading.Thread(target=worker, daemon=True).start()

    @pyqtSlot(int, int, str)
    def _on_restaurar_progress(self, actual, total, nombre):
        self.progress_bar.setRange(0, total)
        self.progress_bar.setValue(actual)
        self.lbl_progreso.setText(f"Leyendo metadatos {actual}/{total}: {nombre}")

    @pyqtSlot(int, int, int)
    def _on_restaurar_done(self, restaurados, sin_datos, errores):
        self.progress_bar.setVisible(False)
        self.lbl_progreso.setVisible(False)
        if restaurados == -1:
            detalle = self._restore_last_error.strip()
            if not detalle:
                detalle = "No se recibieron detalles del error."
            msg = (
                "Error durante la restauración.\n\n"
                "Detalle:\n"
                f"{detalle[:3000]}"
            )
            QMessageBox.critical(self, "Restaurar BD", msg)
            return
        msg_extra = ""
        if restaurados == 0 and errores == 0:
            msg_extra = (
                "\n\nNo se restauró ningún video. "
                "Verifica que esos archivos tengan metadatos en el tag 'comment'."
            )
        QMessageBox.information(
            self, "Restaurar BD",
            f"Restauración completada:\n\n"
            f"  ✅ Restaurados: {restaurados}\n"
            f"  ⚪ Sin datos en metadatos: {sin_datos}\n"
            f"  ⚠️ Errores: {errores}"
            f"{msg_extra}",
        )
        if self.carpeta_actual:
            self._refresh_list()
        self._build_tree()

    def _folder_log_label(self, carpeta_str):
        p = Path(carpeta_str)
        if p.is_absolute():
            try:
                return str(p.resolve(strict=False))
            except Exception:
                return str(p)
        if self.ruta_raiz:
            try:
                return str((self.ruta_raiz / p).resolve(strict=False))
            except Exception:
                pass
        return str(carpeta_str).replace("\\", "/")

    def _folder_log_bucket(self, carpeta_str):
        """Map a visited folder path to the compact log key shown in daily stats.

        Rules:
        - Use immediate folder under current root.
        - Skip root itself (no bucket).
        """
        raw = str(carpeta_str or "").strip()
        if not raw:
            return None
        raw_norm = raw.replace("\\", "/").strip()

        # Already-compact keys from log file, e.g. "joi" or "tnsnames/abby".
        if not re.match(r"^[A-Za-z]:/", raw_norm) and not raw_norm.startswith("/"):
            first = raw_norm.strip("/").split("/", 1)[0].strip()
            return first or None

        # Never emit drive roots as buckets, e.g. "D:/" or "D:".
        if re.match(r"^[A-Za-z]:/?$", raw_norm):
            return None

        if not self.ruta_raiz:
            return None

        try:
            p = Path(raw_norm).resolve(strict=False)
        except Exception:
            p = Path(raw_norm)

        try:
            root = Path(self.ruta_raiz).resolve(strict=False)
        except Exception:
            root = Path(self.ruta_raiz)

        p_norm = str(p).replace("\\", "/").rstrip("/")
        root_norm = str(root).replace("\\", "/").rstrip("/")
        if not root_norm:
            return None
        if p_norm.lower() == root_norm.lower():
            return None
        prefix = root_norm + "/"
        if not p_norm.lower().startswith(prefix.lower()):
            return None
        rel = p_norm[len(prefix):]
        first = rel.split("/", 1)[0].strip()
        if re.match(r"^[A-Za-z]:$", first):
            return None
        return first

    def _load_daily_folder_log(self):
        self._daily_folder_stats = defaultdict(lambda: {"vistas": 0, "segundos": 0})
        path = self._folder_daily_log_path()
        if not path.exists():
            return
        try:
            with open(path, "r", encoding="utf-8") as f:
                for raw in f:
                    line = raw.strip()
                    if not line or line == "Carpetas -" or " - " not in line:
                        continue
                    carpeta, resumen = line.split(" - ", 1)
                    # Format: <carpeta> - Xs (seconds) or legacy <carpeta> - N (minutes)
                    resumen_s = resumen.strip()
                    m_secs = re.search(r"^(\d+)s$", resumen_s)
                    m_mins = re.search(r"^(\d+)$", resumen_s)
                    if m_secs:
                        score_secs = int(m_secs.group(1))
                    elif m_mins:
                        score_secs = int(m_mins.group(1)) * 60
                    else:
                        continue
                    bucket = self._folder_log_bucket(carpeta)
                    if not bucket:
                        continue
                    self._daily_folder_stats[bucket]["segundos"] += score_secs
        except Exception as e:
            LOGGER.warning("No se pudo leer log diario de carpetas: %s", e)

    def _write_daily_folder_log(self):
        path = self._folder_daily_log_path()
        lineas = []
        normalized = defaultdict(lambda: {"vistas": 0, "segundos": 0})
        for carpeta, data in self._daily_folder_stats.items():
            bucket = self._folder_log_bucket(carpeta)
            if not bucket:
                continue
            normalized[bucket]["vistas"] += int(data.get("vistas", 0))
            normalized[bucket]["segundos"] += int(data.get("segundos", 0))
        self._daily_folder_stats = normalized
        items = sorted(
            self._daily_folder_stats.items(),
            key=lambda kv: (-(kv[1]["vistas"] + kv[1]["segundos"] // 60), str(kv[0]).lower())
        )
        for carpeta, data in items:
            secs = int(data.get("segundos", 0))
            if secs < 1:
                continue
            lineas.append(f"{carpeta} - {secs}s")
        try:
            with open(path, "w", encoding="utf-8") as f:
                f.write("\n".join(lineas) + "\n")
        except Exception as e:
            LOGGER.warning("No se pudo escribir log diario de carpetas: %s", e)

    def _ensure_daily_log_date(self):
        hoy = datetime.now().strftime("%Y-%m-%d")
        if hoy == self._daily_folder_log_date:
            return
        self._daily_folder_log_date = hoy
        self._load_daily_folder_log()

    def _flush_folder_view_log(self, motivo="switch"):
        if not self._folder_view_path or not self._folder_view_started_at:
            return
        self._ensure_daily_log_date()
        # Respect manual edits/deletions in the .log file made while app is running.
        # Reload disk state first so stale in-memory counters are not written back.
        self._load_daily_folder_log()
        end_ts = time.time()
        elapsed = max(0, int(end_ts - self._folder_view_started_at))
        bucket = self._folder_log_bucket(self._folder_view_path)
        if bucket:
            self._daily_folder_stats[bucket]["vistas"] += 1
            self._daily_folder_stats[bucket]["segundos"] += elapsed
        self._write_daily_folder_log()
        self._folder_view_path = None
        self._folder_view_started_at = None

    def _start_folder_view_log(self, carpeta):
        self._ensure_daily_log_date()
        carpeta_str = str(Path(carpeta).resolve(strict=False))
        if self._folder_view_path == carpeta_str and self._folder_view_started_at:
            return
        self._flush_folder_view_log("switch")
        self._folder_view_path = carpeta_str
        self._folder_view_started_at = time.time()

    def _refresh_folder_thumb_ids_cache(self):
        """Cachea en una sola query los IDs de miniaturas por carpeta (sin BLOB).

        Esto evita que `_folder_icon` haga un SELECT con BLOBs por cada carpeta
        del árbol; basta una consulta global y luego sólo se carga el blob de la
        miniatura escogida (rotación por minuto).
        """
        try:
            cur = self.db.conn.cursor() if self.db else None
            if cur is None:
                self._folder_thumb_ids_cache = {}
                return
            cur.execute(
                "SELECT carpeta, id FROM folder_thumbnails ORDER BY carpeta, fecha ASC"
            )
            cache = defaultdict(list)
            for row in cur.fetchall():
                cache[row['carpeta']].append(int(row['id']))
            self._folder_thumb_ids_cache = dict(cache)
            # Conteo de sugeridas en bloque para columna Mini (asignadas/sugeridas).
            try:
                self._folder_suggest_count_cache = self.db.obtener_conteos_sugeridas_carpetas()
            except Exception:
                self._folder_suggest_count_cache = {}
        except Exception:
            self._folder_thumb_ids_cache = {}
            self._folder_suggest_count_cache = {}

    def _folder_icon(self, carpeta):
        minute_key = int(time.time() // 60)
        if self._folder_icon_minute != minute_key or self._folder_thumb_ids_cache is None:
            self._folder_icon_minute = minute_key
            self.folder_icon_cache.clear()
            if not hasattr(self, "_folder_chosen_thumb_cache"):
                self._folder_chosen_thumb_cache = {}
            self._folder_chosen_thumb_cache.clear()
            self._refresh_folder_thumb_ids_cache()
        if not hasattr(self, "_folder_chosen_thumb_cache"):
            self._folder_chosen_thumb_cache = {}
        key = str(carpeta)
        if key in self.folder_icon_cache:
            return self.folder_icon_cache[key]

        carpeta_norm = key.replace('\\', '/')
        thumb_ids = (
            self._folder_thumb_ids_cache.get(carpeta_norm, [])
            if self._folder_thumb_ids_cache else []
        )
        thumb_count = len(thumb_ids)

        icon = None
        thumb_data = None
        chosen_id = None
        if thumb_count > 0:
            try:
                seed = f"{key}|{minute_key}".encode("utf-8", errors="ignore")
                idx = int(hashlib.sha1(seed).hexdigest(), 16) % thumb_count
                chosen_id = thumb_ids[idx]
                thumb_data = self.db.obtener_miniatura_por_id(chosen_id)
            except Exception:
                thumb_data = None
                chosen_id = None
        self._folder_chosen_thumb_cache[key] = chosen_id

        if thumb_data:
            pm = QPixmap()
            pm.loadFromData(QByteArray(thumb_data))
        else:
            img = self._find_folder_image(carpeta)
            pm = QPixmap(str(img)) if img else QPixmap()

        if not pm.isNull():
            icon = self._make_premium_folder_channel_icon(pm, self.tree.iconSize())

        self.folder_icon_cache[key] = icon
        return icon

    def _make_premium_folder_channel_icon(self, source_pm: QPixmap, target_size: QSize):
        """Render a folder thumbnail as a premium 'channel card' icon."""
        if source_pm.isNull():
            return QIcon()
        w = max(96, int(target_size.width()))
        h = max(96, int(target_size.height()))
        dpr = 2.0
        ww = int(w * dpr)
        hh = int(h * dpr)

        canvas = QPixmap(ww, hh)
        canvas.fill(Qt.GlobalColor.transparent)

        p = QPainter(canvas)
        p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        p.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, True)

        radius = int(16 * dpr)
        card_path = QPainterPath()
        card_path.addRoundedRect(1.0, 1.0, float(ww - 2), float(hh - 2), float(radius), float(radius))

        # Base card background.
        p.fillPath(card_path, QColor("#141414"))
        p.setClipPath(card_path)

        # Centered crop to avoid portrait images sticking to one side.
        scaled = source_pm.scaled(
            ww,
            hh,
            Qt.AspectRatioMode.KeepAspectRatioByExpanding,
            Qt.TransformationMode.SmoothTransformation,
        )
        sx = max(0, (scaled.width() - ww) // 2)
        sy = max(0, (scaled.height() - hh) // 2)
        p.drawPixmap(0, 0, scaled, sx, sy, ww, hh)

        # Subtle top-to-bottom overlay for premium channel-card look.
        p.fillRect(0, 0, ww, hh, QColor(0, 0, 0, 26))

        p.setClipping(False)
        p.setPen(QPen(QColor("#f1f1f1"), 2))
        p.drawRoundedRect(1, 1, ww - 2, hh - 2, radius, radius)
        p.end()

        canvas.setDevicePixelRatio(dpr)
        return QIcon(canvas)

    def _is_tmpmeta_file(self, path_obj: Path):
        """Return True for temporary ffmpeg metadata files that must be auto-removed."""
        try:
            return path_obj.is_file() and "_tmpmeta" in path_obj.stem.lower()
        except Exception:
            return False

    def _cleanup_tmpmeta_in_iterable(self, files_iterable):
        """Delete tmpmeta files immediately, no user confirmation required."""
        borrados = 0
        for f in files_iterable:
            try:
                if self._is_tmpmeta_file(f):
                    f.unlink()
                    borrados += 1
            except (FileNotFoundError, PermissionError, OSError):
                continue
        if borrados:
            self.statusBar().showMessage(f"Limpieza automática: {borrados} archivo(s) tmpmeta eliminado(s)", 2500)

    def _toggle_hash_stop(self, checked):
        self._hash_stop = checked
        if checked:
            self.btn_stop_hash.setText("▶ Hash")
            self._notify("Hashing en segundo plano pausado", 2500)
        else:
            self.btn_stop_hash.setText("⏸ Hash")
            self._notify("Hashing en segundo plano reanudado", 2500)

    def _toggle_meta_check_stop(self, checked):
        self._meta_check_stop = checked
        if not hasattr(self, "btn_stop_meta"):
            return
        if checked:
            self.btn_stop_meta.setText("▶ Comprobación")
            if self._meta_check_total > 0:
                self.lbl_progreso.setVisible(True)
                self.lbl_progreso.setText(
                    f"Comprobación pausada {self._meta_check_done}/{self._meta_check_total}"
                )
                self.statusBar().showMessage(
                    f"Comprobación pausada ({self._meta_check_done}/{self._meta_check_total})",
                    2500,
                )
            else:
                self._notify("Comprobación de metadatos pausada", 2500)
        else:
            self._update_meta_check_button_text()
            self._notify("Comprobación de metadatos reanudada", 2500)
            # Re-launch for the current folder if there's no active check
            if self.carpeta_actual and (
                self._meta_check_thread is None or not self._meta_check_thread.is_alive()
            ):
                try:
                    videos = [
                        v for v in self.carpeta_actual.rglob("*")
                        if v.is_file() and v.suffix.lower() in EXTENSIONES_VIDEO
                    ]
                    self._launch_meta_check(videos)
                except Exception:
                    pass

    def _toggle_thumb_stop(self, checked):
        self._suggestions_stop = checked
        if checked:
            self.btn_stop_thumbs.setText("▶ Miniaturas")
            self._suggest_cycle = 0
            if self._folder_suggest_thread and self._folder_suggest_thread.isRunning():
                self._folder_suggest_thread.stop()
                self._folder_suggest_thread.wait(2000)
            self._notify("Búsqueda de miniaturas sugeridas pausada", 2500)
            return

        self.btn_stop_thumbs.setText("⏸ Miniaturas")
        self._notify("Búsqueda de miniaturas sugeridas reanudada", 2500)
        self._start_folder_suggestions_background()

    def _build_folder_suggestion_queue(self):
        """Construye cola de carpetas (con vídeos) para un ciclo de sugerencias."""
        if not self.videos_base:
            self._scan()
        folder_map = defaultdict(list)
        for v in self.videos_base:
            if not v.exists() or v.suffix.lower() not in EXTENSIONES_VIDEO:
                continue
            folder_map[str(v.parent)].append(str(v))

        queue = {carpeta_str: videos for carpeta_str, videos in folder_map.items() if videos}
        self._folder_suggestion_queue = queue
        self._suggest_total = len(queue)
        self._suggest_done = 0

    def _start_folder_suggestions_background(self):
        if self._suggestions_stop or not self.ruta_raiz:
            return
        if self._folder_suggest_thread and self._folder_suggest_thread.isRunning():
            return
        self._suggest_cycle += 1
        self._build_folder_suggestion_queue()
        if not self._folder_suggestion_queue:
            self._notify("No hay carpetas con vídeos para sugerir miniaturas", 2500)
            return
        self.thumb_progress_bar.setVisible(True)
        self.lbl_thumb_progreso.setVisible(True)
        self.thumb_progress_bar.setRange(0, self._suggest_total)
        self.thumb_progress_bar.setValue(0)
        self.lbl_thumb_progreso.setText(
            f"Sugeridas ciclo {self._suggest_cycle}: 0/{self._suggest_total}"
        )
        self._folder_suggest_thread = FolderSuggestionThread(
            self._folder_suggestion_queue,
            max_frames_per_video=120,
            max_suggested_per_folder=MAX_SUGGESTED_THUMBS_PER_FOLDER,
            parent=self,
        )
        self._folder_suggest_thread.progress.connect(self._on_folder_suggest_progress)
        self._folder_suggest_thread.folder_suggested.connect(self._on_folder_suggested)
        self._folder_suggest_thread.status.connect(lambda txt: self.statusBar().showMessage(txt, 2500))
        self._folder_suggest_thread.finished.connect(self._on_folder_suggest_finished)
        self._folder_suggest_thread.start()

    @pyqtSlot(int, int, str)
    def _on_folder_suggest_progress(self, done, total, folder_name):
        self._suggest_done = int(done)
        self._suggest_total = int(total)
        self.thumb_progress_bar.setVisible(True)
        self.lbl_thumb_progreso.setVisible(True)
        self.thumb_progress_bar.setRange(0, max(1, self._suggest_total))
        self.thumb_progress_bar.setValue(min(self._suggest_done, self._suggest_total))
        self.lbl_thumb_progreso.setText(
            f"Sugeridas ciclo {self._suggest_cycle}: {self._suggest_done}/{self._suggest_total}  -  {folder_name}"
        )

    @pyqtSlot(str, int)
    def _on_folder_suggested(self, carpeta_str, _added):
        # Refresca iconos/conteos si la carpeta ahora tiene miniaturas sugeridas disponibles.
        try:
            carpeta = Path(carpeta_str)
            self.folder_icon_cache.pop(str(carpeta), None)
            self._refresh_tree_folder_icons()
        except Exception:
            pass

    @pyqtSlot()
    def _on_folder_suggest_finished(self):
        self._refresh_folder_thumb_ids_cache()
        self._refresh_tree_folder_icons()
        if self._suggestions_stop:
            self.thumb_progress_bar.setVisible(False)
            self.lbl_thumb_progreso.setVisible(False)
            self.lbl_thumb_progreso.setText("")
            self._notify("Sugerencias de miniaturas pausadas", 2200)
        else:
            self._notify(
                f"Ciclo {self._suggest_cycle} completado. Iniciando siguiente...",
                1600,
            )
            QTimer.singleShot(250, self._start_folder_suggestions_background)

    def _tick_idle_hashing(self):
        if not self.ruta_raiz:
            return
        if self._hash_stop:
            return
        if self.idle_hash_in_progress:
            return
        if self.progress_bar.isVisible():
            return
        if self._is_playing():
            return
        if not self.idle_hash_queue:
            self._rebuild_idle_hash_queue()
            if not self.idle_hash_queue:
                return

        ruta_str = self.idle_hash_queue.pop(0)
        self.idle_hash_in_progress = True
        LOGGER.info("Idle hash+sync iniciado: %s", ruta_str)
        self.statusBar().showMessage(f"Hasheando en segundo plano: {Path(ruta_str).name}")
        threading.Thread(target=self._idle_hash_worker, args=(ruta_str,), daemon=True).start()

    def _idle_hash_worker(self, ruta_str):
        try:
            self._hash_video_and_postcheck_worker(ruta_str, "hash_idle")
        finally:
            QMetaObject.invokeMethod(
                self, "_on_idle_hash_done",
                Qt.ConnectionType.QueuedConnection,
                Q_ARG(str, ruta_str)
            )

    def _hash_video_and_postcheck_worker(self, ruta_str, source):
        LOGGER.info("Hash worker (%s) iniciado: %s", source, ruta_str)
        try:
            _calcular_hash_archivo(ruta_str)
        finally:
            try:
                QMetaObject.invokeMethod(
                    self, "_on_hash_background_done",
                    Qt.ConnectionType.QueuedConnection,
                    Q_ARG(str, ruta_str),
                    Q_ARG(str, source or "hash_bg"),
                )
            except Exception:
                pass
        LOGGER.info("Hash worker (%s) finalizado: %s", source, ruta_str)

    @pyqtSlot(str, str)
    def _on_hash_background_done(self, ruta_str, source):
        try:
            LOGGER.info("Post-hash sync programada (%s): %s", source, ruta_str)
            self._launch_meta_check_for_video(Path(ruta_str), source=source or "hash_bg")
        except Exception:
            LOGGER.exception("Error en post-hash sync (%s): %s", source, ruta_str)
            pass

    @pyqtSlot(str)
    def _on_idle_hash_done(self, ruta_str):
        self.idle_hash_in_progress = False
        self.idle_hash_count += 1
        LOGGER.info("Idle hash+sync completado (%d): %s", self.idle_hash_count, ruta_str)
        if self.idle_hash_count % 5 == 0:
            self._rebuild_duplicate_index()
            self.mostrar_porcentaje_hashes()
            self.statusBar().showMessage(f"Hashing en segundo plano: {self.idle_hash_count} completados", 1500)

    # ── Raíz y escaneo ──

    def seleccionar_raiz(self):
        d = QFileDialog.getExistingDirectory(self, "Seleccionar Carpeta")
        if d:
            self._flush_folder_view_log("cambio_raiz")
            self.ruta_raiz = Path(d)
            self.carpeta_actual = None
            self.folder_icon_cache.clear()
            self.duration_cache.clear()
            self.idle_hash_queue = []
            self.idle_hash_count = 0
            self.lbl_ruta.setText(str(self.ruta_raiz))
            self._show_loading(f"Cargando {self.ruta_raiz.name}…")
            self._scan()
            self._build_tree()
            self._refresh_list()
            self.mostrar_porcentaje_hashes()
            self._notify(f"Carpeta cargada: {self.ruta_raiz.name}")

    def _scan(self):
        if not self.ruta_raiz:
            return
        vetadas_lower = {v.lower() for v in self.carpetas_vetadas}
        videos = []
        sizes = {}
        all_files = []
        # os.scandir es ~5-10× más rápido que pathlib.rglob en Windows porque
        # devuelve los stats en la misma syscall del directory listing.
        for f, entry in self._fast_walk(self.ruta_raiz, vetadas_lower):
            all_files.append(f)
            if self._is_tmpmeta_file(f):
                try:
                    f.unlink()
                except (FileNotFoundError, PermissionError, OSError):
                    pass
                continue
            if f.suffix.lower() not in EXTENSIONES_VIDEO:
                continue
            try:
                sizes[str(f).replace('\\', '/')] = int(entry.stat().st_size)
            except OSError:
                pass
            videos.append(f)
        self.videos_base = videos
        self._video_size_cache = sizes
        self._all_files_cache = all_files
        self._all_files_cache_root = str(self.ruta_raiz)
        self._all_files_cache_ts = time.time()
        # Existencia conocida (set normalizado) → evita Path.exists() en bucles
        self._existing_paths_norm = {str(f).replace('\\', '/') for f in all_files}
        # Limpia caches dependientes (estructura de carpetas pudo cambiar).
        self._folder_image_cache.clear()
        self._build_folder_aggregate_cache()
        self._rebuild_duplicate_index()
        self.lbl_count.setText(f"{len(self.videos_base)} videos")
        if not self._suggestions_stop:
            QTimer.singleShot(100, self._start_folder_suggestions_background)

    def _fast_walk(self, root, vetadas_lower=None):
        """Generador (Path, DirEntry) usando os.scandir iterativo.

        Mucho más rápido que ``Path.rglob`` en HDD porque cada DirEntry trae
        el tipo y el stat sin syscalls extra. Filtra carpetas vetadas en la
        propia bajada para no entrar en ellas.
        """
        vetadas_lower = vetadas_lower or set()
        stack = [str(root)]
        while stack:
            current = stack.pop()
            try:
                with os.scandir(current) as it:
                    for entry in it:
                        try:
                            name_lower = entry.name.lower()
                            if entry.is_dir(follow_symlinks=False):
                                if name_lower in vetadas_lower:
                                    continue
                                stack.append(entry.path)
                            elif entry.is_file(follow_symlinks=False):
                                yield Path(entry.path), entry
                        except OSError:
                            continue
            except (PermissionError, OSError):
                continue

    def _ensure_all_files_cache(self):
        """Garantiza que self._all_files_cache esté fresco (TTL + ruta correcta)."""
        if not self.ruta_raiz:
            return []
        now = time.time()
        if (
            self._all_files_cache
            and self._all_files_cache_root == str(self.ruta_raiz)
            and (now - self._all_files_cache_ts) < self._all_files_cache_ttl
        ):
            return self._all_files_cache
        # Re-escanea (con sizes incluidos) para refrescar todas las caches.
        self._scan()
        return self._all_files_cache

    def _build_folder_aggregate_cache(self):
        """Precompute per-folder totals for current management mode.

        - Normal mode: only videos
        - Solo fotos mode: only images
        """
        self.folder_agg_cache = {}
        if not self.ruta_raiz:
            return

        root_norm = str(self.ruta_raiz).replace('\\', '/').rstrip('/')
        cache = defaultdict(lambda: {
            'total_videos': 0,
            'peso_total': 0,
            'revisados': 0,
            'hasheados': 0,
            'total_vistas': 0,
            'total_tiempo': 0,
        })

        # Select file universe for current mode.
        if self.incluir_fotos_gestion:
            managed_exts = set(EXTENSIONES_IMAGEN)
        else:
            managed_exts = set(EXTENSIONES_VIDEO)
        files_pool = [f for f in (self._all_files_cache or []) if f.suffix.lower() in managed_exts]

        # Una sola consulta para todos los hashes (evita N round-trips a SQLite).
        try:
            hash_set = self.db.obtener_rutas_con_hash()
        except Exception:
            hash_set = set()
        # Y una sola consulta para todas las stats de los archivos del árbol.
        try:
            stats_map = self.db.obtener_stats_batch([str(f) for f in files_pool])
        except Exception:
            stats_map = {}

        for f in files_pool:
            ruta_norm = str(f).replace('\\', '/')
            size_bytes = self._video_size_cache.get(ruta_norm)
            if size_bytes is None:
                try:
                    size_bytes = int(f.stat().st_size)
                except OSError:
                    size_bytes = 0
            revisado = self._is_video_revisado(f)
            hasheado = ruta_norm in hash_set
            s = stats_map.get(ruta_norm)
            v_repros = int(s['reproducciones']) if s else 0
            v_tiempo = int(s['tiempo_visto_seg']) if s else 0

            parent = f.parent
            while True:
                p_norm = str(parent).replace('\\', '/').rstrip('/')
                if not p_norm.startswith(root_norm):
                    break
                agg = cache[p_norm]
                agg['total_videos'] += 1
                agg['peso_total'] += size_bytes
                if revisado:
                    agg['revisados'] += 1
                if hasheado:
                    agg['hasheados'] += 1
                agg['total_vistas'] += v_repros
                agg['total_tiempo'] += v_tiempo
                if parent == self.ruta_raiz:
                    break
                new_parent = parent.parent
                if new_parent == parent:
                    break
                parent = new_parent

        self.folder_agg_cache = dict(cache)

    # ── Árbol de carpetas ──

    def _build_tree(self):
        self._cancel_tree_build()
        self.tree.clear()
        if not self.ruta_raiz:
            return
        root_has_dups = self._folder_has_duplicates(self.ruta_raiz)
        root_text = self.ruta_raiz.name + (" !" if root_has_dups else "")
        root = QTreeWidgetItem([" ", root_text, "", "", "", "", "", ""])
        root_icon = self._folder_icon(self.ruta_raiz)
        if root_icon:
            root.setIcon(0, root_icon)
        root.setData(0, Qt.ItemDataRole.UserRole, str(self.ruta_raiz))
        self._set_folder_stats(root, self.ruta_raiz)
        self._paint_folder_duplicate_state(root, root_has_dups)
        self.tree.addTopLevelItem(root)
        self.tree.collapseAll()
        root.setExpanded(True)
        # Inicio tipo dashboard: no seleccionar carpeta automáticamente.
        self.tree.setCurrentItem(None)
        self._tree_build_nodes = 1
        self._tree_build_queue.append((root, self.ruta_raiz, None, 0))
        self.statusBar().showMessage("Cargando estructura...", 0)
        self._process_tree_build_chunk()
        if self._tree_build_queue and self._tree_build_timer:
            self._tree_build_timer.start()

    def _set_folder_stats(self, item, carpeta):
        # Usa solo la cache pre-agregada (construida una vez en _scan); evita
        # consultas SQL `LIKE` por cada nodo del árbol, que son full-table-scans.
        if self._folder_thumb_ids_cache is None:
            self._refresh_folder_thumb_ids_cache()
        key = str(carpeta).replace('\\', '/').rstrip('/')
        folder_stats = self.folder_agg_cache.get(key, {})
        peso_total = int(folder_stats.get('peso_total', 0))
        total_videos = int(folder_stats.get('total_videos', 0))
        revisados = int(folder_stats.get('revisados', 0))
        hasheados = int(folder_stats.get('hasheados', 0))
        item.setText(2, str(int(folder_stats.get('total_vistas', 0))))
        tm = int(folder_stats.get('total_tiempo', 0)) // 60
        item.setText(3, f"{tm}m" if tm < 60 else f"{tm // 60}h {tm % 60}m")
        peso_mb = peso_total / (1024 * 1024)
        item.setText(4, f"{peso_mb / 1024:.1f} GB" if peso_mb >= 1024 else f"{peso_mb:.0f} MB")
        if total_videos > 0:
            pct = revisados * 100 // total_videos
            item.setText(5, f"{pct}%")
            pct_h = hasheados * 100 // total_videos
            item.setText(6, f"{pct_h}%")
        else:
            item.setText(5, "-")
            item.setText(6, "-")
        # Col 7: nº de miniaturas asignadas a esta carpeta (cache global por minuto)
        thumb_ids = (
            self._folder_thumb_ids_cache.get(key, [])
            if self._folder_thumb_ids_cache else []
        )
        n_thumbs = len(thumb_ids)
        n_suggest = int(self._folder_suggest_count_cache.get(key, 0)) if self._folder_suggest_count_cache else 0
        item.setText(7, f"{n_thumbs}/{n_suggest}")
        item.setToolTip(7, f"{n_thumbs} asignadas / {n_suggest} sugeridas")

        # Estado visual de completitud: verde cuando todo está revisado y hasheado.
        # El rojo por duplicados se aplica después y tiene prioridad.
        for c in range(0, 8):
            item.setForeground(c, QBrush())
        completo = (total_videos > 0 and revisados >= total_videos and hasheados >= total_videos)
        if completo:
            verde = QColor("#1f9d55")
            for c in range(0, 8):
                item.setForeground(c, verde)

        for c in (2, 3, 4, 5, 6, 7):
            item.setTextAlignment(c, Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)

    def _add_children(self, parent_item, carpeta):
        try:
            dirs = sorted(
                [d for d in carpeta.iterdir() if d.is_dir() and not d.name.startswith('.')],
                key=lambda x: x.name.lower()
            )
        except (PermissionError, OSError):
            return
        vetadas_lower = {v.lower() for v in self.carpetas_vetadas}
        for d in dirs:
            if d.name.lower() in vetadas_lower:
                continue
            key = str(d).replace('\\', '/').rstrip('/')
            n = int(self.folder_agg_cache.get(key, {}).get('total_videos', 0))
            label = f"{d.name}  ({n})"
            has_dups = self._folder_has_duplicates(d)
            if has_dups:
                label += " !"
            item = QTreeWidgetItem([" ", label, "", "", "", "", "", ""])
            icon = self._folder_icon(d)
            if icon:
                item.setIcon(0, icon)
            item.setData(0, Qt.ItemDataRole.UserRole, str(d))
            self._set_folder_stats(item, d)
            self._paint_folder_duplicate_state(item, has_dups)
            parent_item.addChild(item)
            self._add_children(item, d)

    def _on_tree_select(self, current, _prev):
        if not current:
            return
        ruta = current.data(0, Qt.ItemDataRole.UserRole)
        if ruta:
            self.carpeta_actual = Path(ruta)
            self._start_folder_view_log(self.carpeta_actual)
            self._refresh_list()
            tree = getattr(self, "tree", None)
            if tree is not None and tree.isVisible():
                tree.hide()
            btn_h = getattr(self, "btn_hamburger", None)
            if btn_h is not None:
                try:
                    btn_h.setChecked(False)
                except Exception:
                    pass

    def _on_tree_double_click(self, item, _column):
        ruta = item.data(0, Qt.ItemDataRole.UserRole)
        if not ruta:
            return
        carpeta = Path(ruta)
        # Asegura que la cache de iconos esté poblada (y por tanto el thumb elegido).
        try:
            self._folder_icon(carpeta)
        except Exception:
            pass
        chosen_id = None
        if hasattr(self, "_folder_chosen_thumb_cache"):
            chosen_id = self._folder_chosen_thumb_cache.get(str(carpeta))

        elegido = None
        seek_ms = None
        if chosen_id:
            try:
                origen = self.db.obtener_origen_miniatura(chosen_id)
            except Exception:
                origen = None
            if origen:
                video_ruta, frame_no = origen
                p = Path(video_ruta)
                if p.exists() and p.suffix.lower() in EXTENSIONES_VIDEO:
                    elegido = p
                    seek_ms = self._frame_to_ms(p, frame_no)

        if elegido is None:
            try:
                videos = [
                    f for f in carpeta.iterdir()
                    if f.is_file() and f.suffix.lower() in EXTENSIONES_VIDEO
                ]
            except (PermissionError, OSError):
                videos = []
            if not videos:
                QMessageBox.information(self, "Sin videos", f"No hay videos en {carpeta.name}")
                return
            elegido = random.choice(videos)

        self.video_elegido = elegido
        self.forzar_guardado_tiempo_actual()
        # Programar seek absoluto si tenemos frame de la miniatura
        self._pending_random_seek_ratio = None
        if seek_ms is not None and seek_ms > 0:
            self._pending_seek_ms = int(seek_ms)
            self._pending_seek_ms_tries = 6
            self._pending_random_pause = False
        else:
            self._pending_seek_ms = None
        self._reproducir_elegido()

    def _frame_to_ms(self, video_path: Path, frame_no: int):
        """Convierte frame_no → milisegundos usando el fps real del video."""
        if frame_no is None or frame_no <= 0:
            return 0
        fps = 0.0
        try:
            import cv2
            cap = cv2.VideoCapture(str(video_path))
            try:
                fps = float(cap.get(cv2.CAP_PROP_FPS) or 0)
            finally:
                cap.release()
        except Exception:
            fps = 0.0
        if fps <= 0:
            fps = 30.0
        return int(frame_no * 1000.0 / fps)

    def closeEvent(self, ev):
        # Si se cierra con un video en reproducción, primero persistimos esa sesión.
        if self.video_elegido and self.video_elegido.exists() and not self._folder_view_path:
            self._start_folder_view_log(self.video_elegido.parent)
        self.forzar_guardado_tiempo_actual()
        self._flush_pending_daily_folder_log()
        self._flush_folder_view_log("salida_app")
        super().closeEvent(ev)

    def _tree_context_menu(self, pos):
        tree = getattr(self, "tree", None)
        if tree is None:
            return
        # Resolver siempre en coordenadas de viewport para soportar señales
        # emitidas tanto por tree como por tree.viewport().
        global_pos = QCursor.pos()
        view_pos = tree.viewport().mapFromGlobal(global_pos)
        item = tree.itemAt(view_pos)
        if item is None:
            # Fallback por si el backend no actualizó aún el cursor global.
            sender_obj = self.sender()
            try:
                if sender_obj is tree.viewport():
                    item = tree.itemAt(pos)
                    global_pos = tree.viewport().mapToGlobal(pos)
                else:
                    vp_pos = tree.viewport().mapFrom(tree, pos)
                    item = tree.itemAt(vp_pos)
                    global_pos = tree.mapToGlobal(pos)
            except Exception:
                item = None
        if not item:
            return
        ruta = item.data(0, Qt.ItemDataRole.UserRole)
        if not ruta:
            return
        carpeta = Path(ruta)
        self._show_folder_options_menu(carpeta, global_pos, tree)

    def _bump_menu_font(self, menu, delta_points=1):
        """Increase menu text size by a small amount for readability."""
        try:
            f = menu.font()
            if f.pointSize() > 0:
                f.setPointSize(f.pointSize() + int(delta_points))
            elif f.pointSizeF() > 0:
                f.setPointSizeF(f.pointSizeF() + float(delta_points))
            menu.setFont(f)
        except Exception:
            pass

    def _show_folder_options_menu(self, carpeta: Path, global_pos: QPoint, parent_widget=None):
        if not carpeta:
            return
        menu_parent = parent_widget if parent_widget is not None else self
        menu = QMenu(menu_parent)
        self._bump_menu_font(menu, 1)
        act_entrar = menu.addAction("Entrar en esta carpeta")
        act_rename_folder = menu.addAction("Renombrar carpeta…")
        act_hashear = menu.addAction("Hashear carpeta")
        act_check_folder = menu.addAction("Comprobar carpeta")
        act_dup_manage = menu.addAction("Mostrar duplicados (gestionar)")
        act_dup_other_folders = menu.addAction("Buscar duplicados en otras carpetas")
        menu.addSeparator()
        act_miniatura = menu.addAction("Elegir miniatura de carpeta…")
        act_mini_sugeridas = menu.addAction("Miniaturas sugeridas…")
        act_ver_thumbs = menu.addAction("Ver miniaturas asociadas…")
        accion = menu.exec(global_pos)
        if accion == act_entrar:
            self.entrar_en_carpeta(carpeta)
        elif accion == act_rename_folder:
            self._renombrar_carpeta(carpeta)
        elif accion == act_hashear:
            self.hashear_carpeta(carpeta)
        elif accion == act_check_folder:
            self._comprobar_carpeta(carpeta)
        elif accion == act_dup_manage:
            self._gestionar_duplicados_carpeta(carpeta)
        elif accion == act_dup_other_folders:
            self._buscar_duplicados_carpeta_otras_carpetas(carpeta)
        elif accion == act_miniatura:
            self._elegir_miniatura_carpeta(carpeta)
        elif accion == act_mini_sugeridas:
            self._ver_miniaturas_sugeridas_carpeta(carpeta)
        elif accion == act_ver_thumbs:
            self._ver_miniaturas_carpeta(carpeta)

    def _show_selected_folder_options(self):
        tree = getattr(self, "tree", None)
        carpeta = None
        if tree is not None:
            item = tree.currentItem()
            if item is not None:
                ruta = item.data(0, Qt.ItemDataRole.UserRole)
                if ruta:
                    carpeta = Path(ruta)
        if carpeta is None and getattr(self, "ruta_actual", None):
            try:
                carpeta = Path(self.ruta_actual)
            except Exception:
                carpeta = None
        if carpeta is None or not carpeta.exists() or not carpeta.is_dir():
            QMessageBox.information(self, "Opciones de carpeta", "Selecciona una carpeta primero.")
            return
        btn = self.sender() if isinstance(self.sender(), QPushButton) else None
        if btn is None:
            btn = getattr(self, "btn_channel_action", None)
        if btn is not None:
            global_pos = btn.mapToGlobal(btn.rect().bottomLeft())
            global_pos.setY(global_pos.y() + 6)
        else:
            global_pos = QCursor.pos()
        self._show_folder_options_menu(carpeta, global_pos, self)

    def _comprobar_carpeta(self, carpeta: Path):
        try:
            videos = [
                f for f in carpeta.rglob("*")
                if f.is_file() and f.suffix.lower() in EXTENSIONES_VIDEO
            ]
        except (PermissionError, OSError) as e:
            QMessageBox.critical(self, "Comprobar carpeta", f"No se pudo leer la carpeta:\n{e}")
            return

        if not videos:
            QMessageBox.information(self, "Comprobar carpeta", "No hay videos en esta carpeta.")
            return

        revisados = [v for v in videos if self._is_video_revisado(v)]
        if not revisados:
            QMessageBox.information(
                self,
                "Comprobar carpeta",
                "No hay videos revisados para comprobar en esta carpeta.",
            )
            return

        self._notify(f"Comprobando carpeta: {carpeta.name} ({len(revisados)} revisados)", 2200)
        LOGGER.info("Comprobación manual de carpeta: %s (videos_revisados=%d)", carpeta, len(revisados))
        self._launch_meta_check(revisados)

    def _remap_path_after_folder_rename(self, path_value, old_base: Path, new_base: Path):
        if path_value is None:
            return None
        try:
            p = Path(path_value).resolve(strict=False)
            rel = p.relative_to(old_base.resolve(strict=False))
            return (new_base / rel)
        except Exception:
            return path_value

    def _renombrar_carpeta(self, carpeta: Path):
        if not carpeta or not carpeta.exists() or not carpeta.is_dir():
            QMessageBox.warning(self, "Renombrar carpeta", "La carpeta seleccionada no existe.")
            return

        nuevo_nombre, ok = QInputDialog.getText(
            self,
            "Renombrar carpeta",
            "Nuevo nombre de carpeta:",
            text=carpeta.name,
        )
        if not ok:
            return
        nuevo_nombre = (nuevo_nombre or "").strip()
        if not nuevo_nombre or nuevo_nombre == carpeta.name:
            return
        if any(ch in nuevo_nombre for ch in ('\\', '/')):
            QMessageBox.warning(self, "Renombrar carpeta", "El nombre no puede contener barras.")
            return

        destino = carpeta.with_name(nuevo_nombre)
        if destino.exists():
            QMessageBox.warning(self, "Renombrar carpeta", "Ya existe una carpeta con ese nombre.")
            return

        old_base = carpeta.resolve(strict=False)
        new_base = destino.resolve(strict=False)

        # Persist pending playback/log state before changing paths on disk.
        self.forzar_guardado_tiempo_actual()
        self._flush_folder_view_log("rename_folder")

        try:
            carpeta.rename(destino)
            self.db.renombrar_prefijo_ruta(old_base, new_base)
        except Exception as e:
            QMessageBox.critical(self, "Renombrar carpeta", f"No se pudo renombrar:\n{e}")
            return

        # Update in-memory paths so current UI/session state remains coherent.
        if self.ruta_raiz:
            self.ruta_raiz = self._remap_path_after_folder_rename(self.ruta_raiz, old_base, new_base)
            if isinstance(self.ruta_raiz, Path):
                self.lbl_ruta.setText(str(self.ruta_raiz))
        if self.carpeta_actual:
            self.carpeta_actual = self._remap_path_after_folder_rename(self.carpeta_actual, old_base, new_base)
        if self.carpeta_fijada:
            self.carpeta_fijada = self._remap_path_after_folder_rename(self.carpeta_fijada, old_base, new_base)
        if self.video_elegido:
            self.video_elegido = self._remap_path_after_folder_rename(self.video_elegido, old_base, new_base)
        if self.historial_raices:
            self.historial_raices = [
                self._remap_path_after_folder_rename(p, old_base, new_base)
                for p in self.historial_raices
            ]
        if self._active_ruta_reproduccion:
            remapped = self._remap_path_after_folder_rename(Path(self._active_ruta_reproduccion), old_base, new_base)
            if isinstance(remapped, Path):
                self._active_ruta_reproduccion = str(remapped)
        if self._folder_view_path:
            remapped_view = self._remap_path_after_folder_rename(Path(self._folder_view_path), old_base, new_base)
            if isinstance(remapped_view, Path):
                self._folder_view_path = str(remapped_view)

        self.folder_icon_cache.clear()
        self.duration_cache.clear()
        self._scan()
        self._build_tree()
        self.mostrar_porcentaje_hashes()
        if self.carpeta_actual:
            self._start_folder_view_log(self.carpeta_actual)
            self._refresh_list()
        self._notify(f"Carpeta renombrada: {carpeta.name} → {destino.name}")

    def _gestionar_duplicados_carpeta(self, carpeta: Path):
        """Launch existing duplicate-management flow scoped to selected folder."""
        if not self._folder_has_duplicates(carpeta):
            QMessageBox.information(self, "Duplicados", "No se detectan duplicados en esta carpeta.")
            return
        prev_fijada = self.carpeta_fijada
        try:
            self.carpeta_fijada = carpeta
            self.borrar_duplicados_carpeta()
        finally:
            self.carpeta_fijada = prev_fijada

    def _buscar_duplicados_carpeta_otras_carpetas(self, carpeta: Path):
        """Find duplicates in selected folder against other folders (mode-aware)."""
        exts_objetivo = set(EXTENSIONES_IMAGEN) if self.incluir_fotos_gestion else set(EXTENSIONES_VIDEO)
        tipo_txt = "fotos" if self.incluir_fotos_gestion else "videos"
        try:
            videos = [
                f for f in carpeta.rglob("*")
                if f.suffix.lower() in exts_objetivo and f.is_file()
            ]
        except (PermissionError, OSError) as e:
            QMessageBox.critical(self, "Duplicados", f"No se pudo leer la carpeta:\n{e}")
            return

        if not videos:
            QMessageBox.information(self, "Duplicados", f"No hay {tipo_txt} en esta carpeta.")
            return

        rutas_carpeta = [str(v).replace('\\', '/') for v in videos]
        pendientes_hash = [r for r in rutas_carpeta if not self.db.tiene_hash(r)]
        total_pendientes = len(pendientes_hash)
        if total_pendientes:
            for i, ruta in enumerate(pendientes_hash):
                self._set_progress(i, total_pendientes, f"Hasheando: {Path(ruta).name}")
                _calcular_hash_archivo(ruta)
            self._hide_progress()

        self._rebuild_duplicate_index()

        pref = str(carpeta).replace('\\', '/').rstrip('/')
        pref_slash = pref + '/'
        resultados = []
        total_coincidencias = 0
        pares_gestion = []

        for ruta in rutas_carpeta:
            dups = []
            for d in self.duplicate_map.get(ruta, []):
                if d == pref or d.startswith(pref_slash):
                    continue
                dups.append(d)
            dups = list(dict.fromkeys(dups))
            if dups:
                resultados.append((ruta, dups))
                total_coincidencias += len(dups)
                for d in dups:
                    pares_gestion.append((Path(ruta), Path(d)))

        if not resultados:
            QMessageBox.information(
                self,
                "Duplicados",
                "No se encontraron duplicados de esta carpeta en otras carpetas.",
            )
            return

        lineas = []
        max_videos = 12
        max_dups_por_video = 3
        for ruta, dups in resultados[:max_videos]:
            lineas.append(f"• {Path(ruta).name}")
            for d in dups[:max_dups_por_video]:
                lineas.append(f"    - {Path(d).name}  ({d})")
            resto = len(dups) - max_dups_por_video
            if resto > 0:
                lineas.append(f"    - ... y {resto} más")

        resto_videos = len(resultados) - max_videos
        if resto_videos > 0:
            lineas.append(f"... y {resto_videos} {tipo_txt} más")

        mensaje = (
            f"Carpeta: {carpeta.name}\n"
            f"{tipo_txt.capitalize()} con duplicados fuera de esta carpeta: {len(resultados)}\n"
            f"Coincidencias totales en otras carpetas: {total_coincidencias}\n\n"
            + "\n".join(lineas)
        )
        decision = QMessageBox.question(
            self,
            "Duplicados en otras carpetas",
            mensaje + "\n\n¿Quieres gestionarlos ahora?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.Yes,
        )
        if decision != QMessageBox.StandardButton.Yes:
            return

        borrados_n = self._gestionar_pares_duplicados(
            pares_gestion,
            titulo="Duplicado entre carpetas",
        )
        self._scan()
        self._refresh_list()
        QMessageBox.information(self, "Listo", f"Duplicados eliminados: {borrados_n}")

    def hashear_carpeta(self, carpeta):
        try:
            archivos = [
                f for f in carpeta.rglob("*")
                if f.suffix.lower() in (EXTENSIONES_VIDEO | EXTENSIONES_IMAGEN) and f.is_file()
            ]
        except (PermissionError, OSError) as e:
            QMessageBox.critical(self, "Error", f"No se pudo leer la carpeta:\n{e}")
            return
        pendientes = [str(f) for f in archivos if not self.db.tiene_hash(str(f))]
        total = len(pendientes)
        if total == 0:
            QMessageBox.information(self, "Hashear carpeta",
                                    f"Todos los {len(archivos)} archivos ya tienen hash.")
            return
        self.progress_bar.setRange(0, total)
        self.progress_bar.setValue(0)
        self.progress_bar.setVisible(True)
        self.lbl_progreso.setText(f"Hasheando 0/{total} en {carpeta.name}…")
        self.lbl_progreso.setVisible(True)

        def worker():
            for i, ruta_str in enumerate(pendientes, 1):
                if self._hash_stop:
                    break
                try:
                    _calcular_hash_archivo(ruta_str)
                except Exception as e:
                    LOGGER.warning("Error hasheando %s: %s", ruta_str, e)
                # Actualizar UI desde el hilo principal
                QMetaObject.invokeMethod(
                    self, "_on_hash_progress",
                    Qt.ConnectionType.QueuedConnection,
                    Q_ARG(int, i), Q_ARG(int, total), Q_ARG(str, carpeta.name), Q_ARG(str, ruta_str)
                )

        threading.Thread(target=worker, daemon=True).start()

    @pyqtSlot(int, int, str, str)
    def _on_hash_progress(self, i, total, nombre_carpeta, ruta_str):
        self.progress_bar.setValue(i)
        self.lbl_progreso.setText(f"Hasheando {i}/{total} en {nombre_carpeta}…")
        try:
            self._launch_meta_check_for_video(Path(ruta_str), source="hash_manual")
        except Exception:
            pass
        if i >= total:
            self.progress_bar.setVisible(False)
            self.lbl_progreso.setVisible(False)
            self._rebuild_duplicate_index()
            self._build_tree()
            self.mostrar_porcentaje_hashes()
            if self.carpeta_actual:
                self._refresh_list()

    def _ver_miniaturas_carpeta(self, carpeta: Path):
        """Muestra la galería de miniaturas asociadas a la carpeta."""
        try:
            videos = sorted(
                [f for f in carpeta.iterdir()
                 if f.is_file() and f.suffix.lower() in EXTENSIONES_VIDEO],
                key=lambda p: p.name.lower()
            )
        except (PermissionError, OSError):
            videos = []
        dlg = FolderThumbnailsDialog(carpeta, self.db, videos, parent=self)
        dlg.exec()

    def _ver_miniaturas_sugeridas_carpeta(self, carpeta: Path):
        """Muestra sugerencias automáticas y permite aplicar una de ellas."""
        try:
            sugeridas = self.db.obtener_miniaturas_sugeridas_carpeta(str(carpeta), limit=200)
        except Exception:
            sugeridas = []
        if not sugeridas:
            QMessageBox.information(
                self,
                "Miniaturas sugeridas",
                "No hay sugerencias aún para esta carpeta.\n"
                "Activa el botón ▶ Miniaturas para buscarlas en segundo plano.",
            )
            return
        dlg = SuggestedFolderThumbnailsDialog(carpeta, sugeridas, db=self.db, parent=self)
        if dlg.exec() == QDialog.DialogCode.Accepted and dlg.selected_thumbs:
            primary, *rest = dlg.selected_thumbs
            self._aplicar_miniatura_desde_sugerida(carpeta, primary, set_cover=True)
            for sug in rest:
                try:
                    self._aplicar_miniatura_desde_sugerida(carpeta, sug, set_cover=False)
                except Exception:
                    pass
            # Borrar todas las sugerencias restantes (descartadas)
            try:
                self.db.eliminar_todas_sugeridas_carpeta(str(carpeta))
            except Exception:
                pass
        else:
            self._refresh_folder_thumb_ids_cache()
            self._refresh_tree_folder_icons()

    def _elegir_miniatura_carpeta(self, carpeta: Path):
        """Permite elegir miniatura sugerida; si no, abre editor manual."""
        sugeridas = []
        try:
            sugeridas = self.db.obtener_miniaturas_sugeridas_carpeta(str(carpeta), limit=200)
        except Exception:
            sugeridas = []

        if sugeridas:
            dlg_s = SuggestedFolderThumbnailsDialog(carpeta, sugeridas, db=self.db, parent=self)
            if dlg_s.exec() == QDialog.DialogCode.Accepted and dlg_s.selected_thumbs:
                primary, *rest = dlg_s.selected_thumbs
                try:
                    self._aplicar_miniatura_desde_sugerida(carpeta, primary, set_cover=True)
                    for sug in rest:
                        try:
                            self._aplicar_miniatura_desde_sugerida(carpeta, sug, set_cover=False)
                        except Exception:
                            pass
                    # Borrar todas las sugerencias restantes (descartadas)
                    try:
                        self.db.eliminar_todas_sugeridas_carpeta(str(carpeta))
                    except Exception:
                        pass
                    return
                except Exception as e:
                    QMessageBox.warning(
                        self,
                        "Miniatura sugerida",
                        f"No se pudo aplicar la miniatura sugerida:\n{e}",
                    )
            elif not dlg_s.want_manual_picker:
                self._refresh_folder_thumb_ids_cache()
                self._refresh_tree_folder_icons()
                return

        self._abrir_editor_miniatura_manual(carpeta)

    def _abrir_editor_miniatura_manual(self, carpeta: Path):
        """Open the FramePickerDialog for the given folder."""
        try:
            videos = sorted(
                [f for f in carpeta.rglob("*")
                 if f.is_file() and f.suffix.lower() in EXTENSIONES_VIDEO],
                key=lambda p: str(p.relative_to(carpeta)).lower()
            )
        except (PermissionError, OSError) as e:
            QMessageBox.critical(self, "Error", f"No se pudo leer la carpeta:\n{e}")
            return
        if not videos:
            QMessageBox.information(self, "Sin vídeos",
                                    "Esta carpeta y sus subcarpetas no contienen vídeos.")
            return
        dlg = FramePickerDialog(carpeta, videos, db=self.db, parent=self)
        if dlg.exec() == QDialog.DialogCode.Accepted:
            # Invalidate cached icons and refresh them in place; no full tree rebuild needed.
            key = str(carpeta)
            self.folder_icon_cache.pop(key, None)
            self._refresh_tree_folder_icons()
            self.statusBar().showMessage(
                f"Miniatura guardada en {carpeta.name}/cover.jpg", 4000)

    def _aplicar_miniatura_desde_sugerida(self, carpeta: Path, sug: dict, *, set_cover: bool = True):
        """Asigna una miniatura sugerida a la carpeta.

        Si *set_cover* es True (por defecto) escribe cover.jpg y refresca iconos.
        En cualquier caso guarda en la galería y elimina de sugerencias.
        """
        thumb_blob = sug.get('thumbnail_blob') or b''
        if not thumb_blob:
            raise ValueError("La sugerencia no contiene imagen")
        video_ruta = str(sug.get('video_ruta') or "")
        frame_no = int(sug.get('frame_no') or 0)
        if set_cover:
            dest = carpeta / "cover.jpg"
            with open(dest, 'wb') as fh:
                fh.write(thumb_blob)
        self.db.guardar_miniatura_carpeta(str(carpeta), video_ruta, frame_no, thumb_blob)
        # Eliminar de sugerencias para que no vuelva a aparecer
        sid = sug.get('id')
        if sid:
            try:
                self.db.eliminar_miniatura_sugerida_carpeta(int(sid))
            except Exception:
                pass
        if set_cover:
            key = str(carpeta)
            self.folder_icon_cache.pop(key, None)
            self._refresh_tree_folder_icons()
            self.statusBar().showMessage(
                f"Miniatura sugerida aplicada en {carpeta.name}/cover.jpg", 4000
            )

    def _abrir_modo_fotos(self):
        """Abre el configurador de presentaci\u00f3n de fotos y lanza la ventana."""
        dlg = PhotoSlideshowConfigDialog(db=self.db, parent=self)
        if dlg.exec() == QDialog.DialogCode.Accepted:
            if self._foto_window is not None:
                self._foto_window.close()
            self._foto_window = PhotoSlideshowWindow(
                dlg.carpetas(),
                dlg.segundos(),
                db=self.db,
                log_callback=self._add_photo_folder_log_seconds,
                parent=None,
            )
            self._foto_window.show()
            self._foto_window.raise_()

    def entrar_en_carpeta(self, nueva_raiz):
        if self.ruta_raiz:
            self.historial_raices.append(self.ruta_raiz)
            self.btn_volver.setEnabled(True)
        self.ruta_raiz = nueva_raiz
        self.folder_icon_cache.clear()
        self.duration_cache.clear()
        self.idle_hash_queue = []
        self.idle_hash_count = 0
        self.lbl_ruta.setText(str(self.ruta_raiz))
        self.statusBar().showMessage("Cargando estructura...", 0)
        QApplication.processEvents()
        self._scan()
        self._build_tree()
        self.carpeta_actual = None
        self._refresh_list()
        self.mostrar_porcentaje_hashes()
        self._notify(f"Entraste a: {self.ruta_raiz.name}")

    def volver_carpeta_anterior(self):
        if not self.historial_raices:
            return
        self.ruta_raiz = self.historial_raices.pop()
        self.folder_icon_cache.clear()
        self.duration_cache.clear()
        self.idle_hash_queue = []
        self.idle_hash_count = 0
        self.btn_volver.setEnabled(bool(self.historial_raices))
        self.lbl_ruta.setText(str(self.ruta_raiz))
        self.statusBar().showMessage("Cargando estructura...", 0)
        QApplication.processEvents()
        self._scan()
        self._build_tree()
        self.carpeta_actual = None
        self._refresh_list()
        self.mostrar_porcentaje_hashes()
        self._notify(f"Volviste a: {self.ruta_raiz.name}")

    def refrescar(self):
        if not self.ruta_raiz:
            return
        self.folder_icon_cache.clear()
        self.duration_cache.clear()
        self.idle_hash_queue = []
        self.idle_hash_count = 0
        self.statusBar().showMessage("Refrescando estructura...", 0)
        QApplication.processEvents()
        self._scan()
        self._build_tree()
        if not self.carpeta_actual:
            self._refresh_list()
        self.mostrar_porcentaje_hashes()
        self._notify("Vista actualizada")

    def refrescar_carpeta_actual(self):
        """Refresca solo lista/datos de la carpeta actual, sin reescanear toda la raíz."""
        if not self.carpeta_actual:
            self._notify("Selecciona una carpeta primero", 1800)
            return
        carpeta_norm = str(self.carpeta_actual).replace('\\', '/').rstrip('/') + '/'
        # Limpiar cachés puntuales de esa carpeta para forzar recálculo de datos visibles.
        self.duration_cache = {
            k: v for k, v in self.duration_cache.items()
            if not str(k).replace('\\', '/').startswith(carpeta_norm)
        }
        self._video_size_cache = {
            k: v for k, v in self._video_size_cache.items()
            if not str(k).replace('\\', '/').startswith(carpeta_norm)
        }
        self.statusBar().showMessage(f"Refrescando carpeta: {self.carpeta_actual.name}...", 0)
        QApplication.processEvents()
        self._refresh_list()
        self.statusBar().showMessage(f"Carpeta actualizada: {self.carpeta_actual.name}", 1500)
        self._notify("Datos de carpeta actual actualizados", 1500)

    # ── Lista de videos ──

    def _refresh_list(self):
        self._finish_thumb_progress()
        self.tabla.setSortingEnabled(False)
        self.tabla.setRowCount(0)
        self.video_elegido = None
        self.lbl_detail.setText("Selecciona un video para ver detalles y reproducir")
        if not self.carpeta_actual:
            self._show_home_dashboard()
            self.lbl_folder.setText("Selecciona una carpeta en el panel izquierdo")
            self._refresh_channel_bar(None, 0)
            self._refresh_visual_recommendations([], {}, {})
            self._refresh_home_dashboard()
            return
        self._show_folder_lists()
        try:
            exts = set(EXTENSIONES_VIDEO)
            if self.incluir_fotos_gestion:
                exts = set(EXTENSIONES_IMAGEN)
            # Reutiliza el escaneo cacheado (HDD-friendly) cuando la carpeta
            # actual está bajo la raíz; si no, usa scandir rápido.
            self._ensure_all_files_cache()
            carpeta_str = str(self.carpeta_actual)
            carpeta_norm = carpeta_str.replace('\\', '/').rstrip('/') + '/'
            base_root = self._all_files_cache_root
            if (
                self._all_files_cache
                and base_root
                and (carpeta_str == base_root or carpeta_str.startswith(base_root.rstrip('\\/') + os.sep))
            ):
                raw_files = [
                    f for f in self._all_files_cache
                    if str(f).replace('\\', '/').startswith(carpeta_norm) or str(f) == carpeta_str
                ]
            else:
                raw_files = [
                    f for f, _e in self._fast_walk(self.carpeta_actual,
                                                    {v.lower() for v in self.carpetas_vetadas})
                ]
            self._cleanup_tmpmeta_in_iterable(raw_files)
            items = sorted(
                [v for v in raw_files if v.suffix.lower() in exts and not self._is_tmpmeta_file(v)],
                key=lambda x: str(x.relative_to(self.carpeta_actual)).lower()
            )
        except (PermissionError, OSError):
            items = []
        tipo_txt = "fotos" if self.incluir_fotos_gestion else "videos"
        self.lbl_folder.setText(f"{self.carpeta_actual.name}  —  {len(items)} {tipo_txt} (incluye subcarpetas)")
        self._refresh_channel_bar(self.carpeta_actual, len(items))
        if not items:
            self.lbl_detail.setText("No hay elementos en esta carpeta")
            self._notify("Carpeta sin elementos", 1800)
        sin_thumb = []
        # Batch fetches: una sola consulta por tipo en lugar de 4 por video.
        # Esto es crítico en HDD porque evita miles de round-trips a SQLite.
        rutas_str = [str(v) for v in items]
        stats_map = self.db.obtener_stats_batch(rutas_str)
        thumbs_map = self.db.obtener_miniaturas_batch(rutas_str)
        rutas_video = [str(v) for v in items if v.suffix.lower() in EXTENSIONES_VIDEO]
        hash_set = self.db.obtener_rutas_con_hash(rutas_video) if rutas_video else set()
        _stats_default = {
            'reproducciones': 0, 'tiempo_visto_seg': 0,
            'ultima_reproduccion': '', 'favorito': False, 'fue_visto': False,
        }
        for v in items:
            ruta_norm = str(v).replace('\\', '/')
            peso_bytes = self._video_size_cache.get(ruta_norm)
            if peso_bytes is None:
                try:
                    peso_bytes = os.path.getsize(v)
                    self._video_size_cache[ruta_norm] = int(peso_bytes)
                except OSError:
                    continue
            stats = stats_map.get(ruta_norm, _stats_default)
            peso_mb = peso_bytes / (1024 * 1024)
            visto = self._item_visto_marcador(v)
            r = self.tabla.rowCount()
            self.tabla.insertRow(r)

            # Col 0: miniatura
            thumb_item = QTableWidgetItem()
            thumb_item.setData(Qt.ItemDataRole.UserRole, str(v))
            thumb_data = thumbs_map.get(ruta_norm)
            if thumb_data:
                pm = QPixmap()
                pm.loadFromData(QByteArray(thumb_data))
                if not pm.isNull():
                    badge = self._duration_badge_text(v) if v.suffix.lower() in EXTENSIONES_VIDEO else ""
                    progress_ratio = None
                    if v.suffix.lower() in EXTENSIONES_VIDEO:
                        dur_secs = self._duration_seconds_cached(v)
                        if isinstance(dur_secs, int) and dur_secs > 0 and dur_secs < 10**9:
                            progress_ratio = max(
                                0.0,
                                min(1.0, float(int(stats.get('tiempo_visto_seg', 0) or 0)) / float(dur_secs)),
                            )
                    thumb_item.setIcon(
                        self._make_youtube_thumb_icon(
                            pm,
                            self.tabla.iconSize(),
                            badge_text=badge,
                            progress_ratio=progress_ratio,
                        )
                    )
            else:
                if v.suffix.lower() in EXTENSIONES_VIDEO:
                    sin_thumb.append(v)
                elif v.suffix.lower() in EXTENSIONES_IMAGEN:
                    pm = QPixmap(str(v))
                    if not pm.isNull():
                        thumb_item.setIcon(self._make_youtube_thumb_icon(pm, self.tabla.iconSize()))
            self.tabla.setItem(r, 0, thumb_item)

            # Col 1: nombre
            nombre_rel = str(v.relative_to(self.carpeta_actual)).replace("\\", "/")
            name_item = QTableWidgetItem(nombre_rel)
            name_item.setToolTip(str(v))
            self.tabla.setItem(r, 1, name_item)

            # Col 2: vistas (numérico para ordenar)
            item_v = QTableWidgetItem()
            item_v.setData(Qt.ItemDataRole.DisplayRole, stats['reproducciones'])
            self.tabla.setItem(r, 2, item_v)

            # Col 3: tiempo acumulado
            item_t = QTableWidgetItem()
            item_t.setData(Qt.ItemDataRole.DisplayRole, stats['tiempo_visto_seg'] // 60)
            item_t.setText(f"{stats['tiempo_visto_seg'] // 60}m")
            self.tabla.setItem(r, 3, item_t)

            # Col 4: peso
            item_p = QTableWidgetItem()
            item_p.setData(Qt.ItemDataRole.DisplayRole, int(peso_bytes))
            if peso_mb >= 1024:
                item_p.setText(f"{peso_mb/1024:.1f} GB")
            else:
                item_p.setText(f"{peso_mb:.0f} MB")
            self.tabla.setItem(r, 4, item_p)

            # Col 5: visto
            self.tabla.setItem(r, 5, QTableWidgetItem(visto))

            # Col 6: hash (consultado en lote arriba)
            tiene = (v.suffix.lower() in EXTENSIONES_VIDEO) and (ruta_norm in hash_set)
            hash_item = QTableWidgetItem("🟢" if tiene else "")
            hash_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            self.tabla.setItem(r, 6, hash_item)

            if ruta_norm in self.duplicate_paths:
                rojo = QColor("#d11a2a")
                for c in range(1, 7):
                    it = self.tabla.item(r, c)
                    if it:
                        it.setForeground(rojo)
            elif visto == "✓":
                verde = QColor("#27ae60")
                for c in range(1, 7):
                    it = self.tabla.item(r, c)
                    if it:
                        it.setForeground(verde)

        self.tabla.setSortingEnabled(True)
        self._apply_table_filter()
        self._refresh_visual_recommendations(items, stats_map, thumbs_map)
        if sin_thumb and self.ffprobe_path:
            self._gen_thumbs(sin_thumb)
        # La sincronización de metadatos se ejecuta automáticamente junto al hash en background.

    def _launch_meta_check(self, videos: list):
        """Synchronize reviewed videos by max values between DB and embedded metadata."""
        if self._meta_check_stop:
            return
        if self._meta_check_thread and self._meta_check_thread.is_alive():
            return  # already running; new result will come naturally on next refresh

        videos = [v for v in videos if v.exists() and self._is_video_revisado(v)]
        total = len(videos)
        self._meta_check_total = total
        self._meta_check_done = 0
        self._meta_check_aligned = 0
        self._update_meta_check_button_text()
        if total == 0:
            self.lbl_progreso.setVisible(False)
            self.lbl_progreso.setText("")
            self.statusBar().showMessage("Comprobación: no hay videos para revisar", 1800)
            return
        self.lbl_progreso.setVisible(True)
        self.lbl_progreso.setText(f"Comprobación 0/{total}…")
        self.statusBar().showMessage(f"Comprobación iniciada: 0/{total}", 1800)
        LOGGER.info("Comprobación masiva iniciada: %d videos", total)

        def worker():
            processed = 0
            aligned_count = 0
            aborted = False
            local_db = None
            try:
                local_db = VideoDatabase()
            except Exception:
                local_db = self.db

            for idx, v in enumerate(videos, 1):
                if self._meta_check_stop:
                    aborted = True
                    break
                try:
                    ruta_str = str(v)
                    synced = self._force_align_video_stats_max(ruta_str, db_obj=local_db)
                    if synced:
                        aligned_count += 1
                        QMetaObject.invokeMethod(
                            self, "_on_meta_aligned",
                            Qt.ConnectionType.QueuedConnection,
                            Q_ARG(str, ruta_str),
                        )
                    processed = idx
                    if idx == 1 or idx % 10 == 0 or idx == total:
                        QMetaObject.invokeMethod(
                            self, "_on_meta_check_progress",
                            Qt.ConnectionType.QueuedConnection,
                            Q_ARG(int, idx), Q_ARG(int, total),
                            Q_ARG(int, aligned_count), Q_ARG(str, v.name),
                        )
                except Exception:
                    LOGGER.exception("Error en comprobación masiva: %s", v)
                    pass

            try:
                if local_db is not self.db and getattr(local_db, "conn", None):
                    local_db.conn.close()
            except Exception:
                pass

            QMetaObject.invokeMethod(
                self, "_on_meta_check_done",
                Qt.ConnectionType.QueuedConnection,
                Q_ARG(int, processed), Q_ARG(int, total),
                Q_ARG(int, aligned_count), Q_ARG(int, 1 if aborted else 0),
            )
            LOGGER.info(
                "Comprobación masiva finalizada: procesados=%d total=%d sincronizados=%d abortada=%s",
                processed, total, aligned_count, aborted,
            )

        self._meta_check_thread = threading.Thread(target=worker, daemon=True)
        self._meta_check_thread.start()

    def _launch_meta_check_for_video(self, video: Path, source=""):
        """Run a lightweight one-video max synchronization check for reviewed videos."""
        if self._meta_check_stop:
            return
        if not video or not video.exists() or not self._is_video_revisado(video):
            return
        LOGGER.info("Comprobación individual encolada (%s): %s", source or "check", video)

        def worker_one():
            local_db = None
            try:
                local_db = VideoDatabase()
                ruta_str = str(video)
                synced = self._force_align_video_stats_max(ruta_str, db_obj=local_db)
                QMetaObject.invokeMethod(
                    self, "_on_meta_single_checked",
                    Qt.ConnectionType.QueuedConnection,
                    Q_ARG(str, ruta_str),
                    Q_ARG(int, 1 if synced else 0),
                    Q_ARG(str, source or "check"),
                )
            except Exception:
                LOGGER.exception("Error en comprobación individual (%s): %s", source, video)
                pass
            finally:
                try:
                    if local_db and getattr(local_db, "conn", None):
                        local_db.conn.close()
                except Exception:
                    pass

        threading.Thread(target=worker_one, daemon=True).start()

    def _force_align_video_stats_max(self, ruta_str, db_obj=None):
        """Force DB + metadata alignment using the highest value on each counter."""
        db_obj = db_obj or self.db
        stats_db = db_obj.obtener_stats_video(ruta_str)
        meta = VideoMetadata.obtener_metadatos(ruta_str) or {}

        db_repros = max(0, int(stats_db.get('reproducciones', 0) or 0))
        db_tiempo = max(0, int(stats_db.get('tiempo_visto_seg', 0) or 0))
        meta_repros = max(0, int(meta.get('reproducciones', 0) or 0))
        meta_tiempo = max(0, int(meta.get('tiempo_visto_seg', 0) or 0))

        merged_repros = max(db_repros, meta_repros)
        merged_tiempo = max(db_tiempo, meta_tiempo)
        merged_fav = bool(stats_db.get('favorito', False) or meta.get('es_favorito', False))
        merged_visto = bool(
            stats_db.get('fue_visto', False)
            or meta.get('fue_visto', False)
            or merged_repros > 0
            or merged_tiempo > 0
        )
        merged_ultima = (
            meta.get('ultima_reproduccion')
            or stats_db.get('ultima_reproduccion')
            or datetime.now().isoformat()
        )

        db_needs_update = (
            merged_repros != db_repros
            or merged_tiempo != db_tiempo
            or merged_fav != bool(stats_db.get('favorito', False))
            or merged_visto != bool(stats_db.get('fue_visto', False))
        )
        if db_needs_update:
            LOGGER.info(
                "Alineando BD (max) %s: db=(%d,%d) meta=(%d,%d) -> merged=(%d,%d)",
                ruta_str, db_repros, db_tiempo, meta_repros, meta_tiempo, merged_repros, merged_tiempo,
            )
            db_obj.upsert_stats_max(
                ruta_str,
                reproducciones=merged_repros,
                tiempo_visto_seg=merged_tiempo,
                es_favorito=merged_fav,
                fue_visto=merged_visto,
                ultima_reproduccion=merged_ultima,
            )

        meta_needs_update = (
            int(meta.get('reproducciones', 0) or 0) != merged_repros
            or int(meta.get('tiempo_visto_seg', 0) or 0) != merged_tiempo
            or bool(meta.get('es_favorito', False)) != merged_fav
            or bool(meta.get('fue_visto', False)) != merged_visto
        )
        if meta_needs_update:
            payload = {
                'reproducciones': merged_repros,
                'tiempo_visto_seg': merged_tiempo,
                'ultima_reproduccion': merged_ultima,
                'es_favorito': merged_fav,
                'fue_visto': merged_visto,
            }
            LOGGER.info(
                "Alineando metadatos (max) %s: meta=(%d,%d) -> merged=(%d,%d)",
                ruta_str, meta_repros, meta_tiempo, merged_repros, merged_tiempo,
            )
            VideoMetadata.guardar_metadatos(ruta_str, payload)

        return (merged_repros > 0 or merged_tiempo > 0)

    @pyqtSlot(str, int, str)
    def _on_meta_single_checked(self, ruta_str, aligned_flag, source):
        if aligned_flag:
            self._on_meta_aligned(ruta_str)
            origen = source.replace("_", " ").strip() if source else "comprobación"
            LOGGER.info("Comprobación sincronizada (%s): %s", origen, ruta_str)
            self.statusBar().showMessage(
                f"Comprobación sincronizada ({origen}): {Path(ruta_str).name}",
                1400,
            )

    def _update_meta_check_button_text(self):
        if not hasattr(self, "btn_stop_meta"):
            return
        if self._meta_check_stop:
            self.btn_stop_meta.setText("▶ Comprobación")
            return
        if self._meta_check_total > 0 and self._meta_check_done < self._meta_check_total:
            self.btn_stop_meta.setText(f"⏸ Comprobación {self._meta_check_done}/{self._meta_check_total}")
        else:
            self.btn_stop_meta.setText("⏸ Comprobación")

    @pyqtSlot(int, int, int, str)
    def _on_meta_check_progress(self, actual, total, aligned_count, nombre):
        self._meta_check_done = actual
        self._meta_check_total = total
        self._meta_check_aligned = aligned_count
        self._update_meta_check_button_text()
        self.lbl_progreso.setVisible(True)
        self.lbl_progreso.setText(
            f"Comprobación {actual}/{total} | OK {aligned_count}: {nombre}"
        )
        self.statusBar().showMessage(
            f"Comprobación: {actual}/{total} | OK {aligned_count} | {nombre}",
            1200,
        )

    @pyqtSlot(int, int, int, int)
    def _on_meta_check_done(self, processed, total, aligned_count, aborted_flag):
        self._meta_check_done = processed
        self._meta_check_total = total
        self._meta_check_aligned = aligned_count
        self._update_meta_check_button_text()
        if aborted_flag:
            self.lbl_progreso.setVisible(True)
            self.lbl_progreso.setText(
                f"Comprobación pausada {processed}/{total} | OK {aligned_count}"
            )
            self.statusBar().showMessage(
                f"Comprobación pausada ({processed}/{total}, OK {aligned_count})",
                2500,
            )
            return
        self.lbl_progreso.setVisible(False)
        self.lbl_progreso.setText("")
        self.statusBar().showMessage(
            f"Comprobación completada ({processed}/{total}, OK {aligned_count})",
            2500,
        )

    @pyqtSlot(str)
    def _on_meta_aligned(self, ruta_str):
        """Color the table row green for a video whose DB stats match its embedded metadata."""
        verde = QColor("#2e7d32")
        for r in range(self.tabla.rowCount()):
            it = self.tabla.item(r, 0)
            if it and it.data(Qt.ItemDataRole.UserRole) == ruta_str:
                name_item = self.tabla.item(r, 1)
                if name_item:
                    name_item.setForeground(verde)
                break

    def _buscar_duplicados_video(self, ruta, alcance='global'):
        ruta_norm = str(ruta).replace('\\', '/')
        es_imagen = Path(ruta).suffix.lower() in EXTENSIONES_IMAGEN
        if not self.db.tiene_hash(ruta_norm):
            self._notify("Calculando hash para buscar duplicados...", 1800)
            if es_imagen:
                _calcular_hash_visual_imagen(ruta_norm)
            else:
                _calcular_hash_visual_background(ruta_norm)
            self._rebuild_duplicate_index()

        duplicados = list(self.duplicate_map.get(ruta_norm, []))
        if alcance == 'carpeta' and self.carpeta_actual:
            pref = str(self.carpeta_actual).replace('\\', '/')
            if not pref.endswith('/'):
                pref += '/'
            duplicados = [d for d in duplicados if d.startswith(pref)]

        if not duplicados:
            alcance_txt = "carpeta seleccionada" if alcance == 'carpeta' else "base de datos"
            QMessageBox.information(self, "Duplicados", f"No se encontraron duplicados en {alcance_txt}.")
            return

        alcance_txt = "carpeta" if alcance == 'carpeta' else "base de datos"
        pares = [(ruta_norm, d) for d in duplicados]
        titulo = f"Duplicado ({alcance_txt})"
        borrados_n = self._gestionar_pares_duplicados(pares, titulo=titulo)
        if borrados_n:
            self._scan()
            self._refresh_list()
            self._notify(f"Borrados {borrados_n} duplicado(s)")

    def _gen_thumbs(self, rutas):
        if self._thumbs_stop:
            return
        if self.thumb_thread and self.thumb_thread.isRunning():
            self.thumb_thread._stop_flag = True
            self.thumb_thread.wait(2000)
        self.thumb_total = len(rutas)
        self.thumb_done = 0
        self._update_thumb_progress("iniciando...")
        self.thumb_thread = ThumbnailThread(rutas, self.ffprobe_path)
        self.thumb_thread.thumbnail_ready.connect(self._on_thumb_ready)
        self.thumb_thread.finished.connect(self._on_thumb_thread_finished)
        self.thumb_thread.start()

    def _on_thumb_ready(self, ruta_str, data):
        self.db.guardar_miniatura(ruta_str, data)
        self.thumb_done += 1
        self._update_thumb_progress(Path(ruta_str).name)
        for i in range(self.tabla.rowCount()):
            item = self.tabla.item(i, 0)
            if item and item.data(Qt.ItemDataRole.UserRole) == ruta_str:
                pm = QPixmap()
                pm.loadFromData(QByteArray(data))
                if not pm.isNull():
                    ruta = Path(ruta_str)
                    badge = self._duration_badge_text(ruta) if ruta.suffix.lower() in EXTENSIONES_VIDEO else ""
                    progress_ratio = None
                    if ruta.suffix.lower() in EXTENSIONES_VIDEO:
                        dur_secs = self._duration_seconds_cached(ruta)
                        if isinstance(dur_secs, int) and dur_secs > 0 and dur_secs < 10**9:
                            stats = self.db.obtener_stats_video(ruta_str)
                            seen_secs = int(stats.get('tiempo_visto_seg', 0) or 0)
                            progress_ratio = max(0.0, min(1.0, float(seen_secs) / float(dur_secs)))
                    item.setIcon(
                        self._make_youtube_thumb_icon(
                            pm,
                            self.tabla.iconSize(),
                            badge_text=badge,
                            progress_ratio=progress_ratio,
                        )
                    )
                break

    @pyqtSlot()
    def _on_thumb_thread_finished(self):
        if self.sender() is not self.thumb_thread:
            return
        if self.thumb_total > 0:
            self._finish_thumb_progress(
                f"Miniaturas listas: {self.thumb_done}/{self.thumb_total}"
            )

    def _on_row_changed(self, row, col, prev_row, prev_col):
        if row < 0:
            return
        item = self.tabla.item(row, 0)
        if not item:
            return
        data = item.data(Qt.ItemDataRole.UserRole)
        if not data:
            return
        ruta = Path(data)
        if not ruta.exists():
            return
        self._show_detail(ruta)

    def _on_table_cell_clicked(self, row, col):
        if row < 0 or col != 0:
            return
        item = self.tabla.item(row, 0)
        if not item:
            return
        data = item.data(Qt.ItemDataRole.UserRole)
        if not data:
            return
        ruta = Path(data)
        if not ruta.exists():
            return
        self.video_elegido = ruta
        self._show_detail(ruta)
        if ruta.suffix.lower() in EXTENSIONES_VIDEO:
            self._notify("Miniatura seleccionada. Reproduciendo en panel integrado.", 1400)
            self.reproducir_video_actual()
        else:
            # Only thumbnail click opens photo preview in the right panel.
            self.forzar_guardado_tiempo_actual()
            self._mostrar_foto_en_panel(ruta)
            self._notify("Foto previsualizada. (Solo por clic en miniatura)", 1400)

    def _show_detail(self, ruta):
        self.video_elegido = ruta
        stats = self.db.obtener_stats_video(str(ruta))
        try:
            peso = os.path.getsize(ruta) / (1024 * 1024)
        except OSError:
            peso = 0
        m, s = divmod(stats['tiempo_visto_seg'], 60)
        dur = self._duration_text(ruta) if ruta.suffix.lower() in EXTENSIONES_VIDEO else "Foto"
        fav = "  ⭐" if ruta.name.lower().startswith("top ") else ""
        carpeta_txt = str(ruta.parent)
        if self.ruta_raiz:
            try:
                carpeta_txt = str(ruta.parent.relative_to(self.ruta_raiz)).replace("\\", "/")
                if not carpeta_txt:
                    carpeta_txt = "."
            except Exception:
                carpeta_txt = str(ruta.parent)
        self.lbl_detail.setText(
            f"{ruta.name}{fav}   —   {stats['reproducciones']}x  ·  {dur}  ·  acum {m}m{s}s  ·  {peso:.1f}MB\n"
            f"📁 {carpeta_txt}"
        )
        self.btn_fav.setText("💔 Quitar Top" if ruta.name.lower().startswith("top ") else "⭐ Favorito")
        fsw = getattr(self, "_video_only_fs_window", None)
        if fsw is not None:
            fsw.lbl_title.setText(ruta.name)
            fsw.btn_fav.setText("★ Quitar Top" if ruta.name.lower().startswith("top ") else "★ Favorito")

    # ── Filtros y modos ──

    def iniciar_modo(self, modo):
        self.modo_actual = modo
        self._set_active_mode_button(modo)
        self.carpeta_fijada = None
        self.forzar_guardado_tiempo_actual()
        if self.incluir_fotos_gestion:
            try:
                base = self.carpeta_actual or self.ruta_raiz
                lista_base = [
                    f for f in (base.rglob("*") if base else [])
                    if f.suffix.lower() in EXTENSIONES_IMAGEN and f.is_file()
                ] if base else []
            except (PermissionError, OSError):
                lista_base = []
        else:
            lista_base = self.videos_base
        if modo == '8':
            if not self.carpeta_actual:
                QMessageBox.information(self, "Sin Revisar", "Selecciona una carpeta en el panel izquierdo.")
                self.lista_actual = []
                return
            try:
                if self.incluir_fotos_gestion:
                    lista_base = [
                        f for f in self.carpeta_actual.rglob("*")
                        if f.suffix.lower() in EXTENSIONES_IMAGEN and f.is_file()
                    ]
                else:
                    lista_base = [
                        v for v in self.carpeta_actual.rglob("*")
                        if v.suffix.lower() in EXTENSIONES_VIDEO and v.is_file()
                    ]
            except (PermissionError, OSError):
                lista_base = []
        self.lista_actual = self.aplicar_filtros(lista_base, modo)
        if self.lista_actual:
            self._notify(f"Modo activo: {self.mode_buttons.get(modo).text() if modo in self.mode_buttons else modo}")
            self.proximo_video()
        else:
            QMessageBox.warning(self, "Aviso", "No hay elementos para este filtro.")

    def _recommendation_source_videos(self):
        """Base list for recommendation buttons (folder-aware when applicable)."""
        try:
            if self.carpeta_actual:
                return [
                    v for v in self.videos_base
                    if self.carpeta_actual in v.parents and v.suffix.lower() in EXTENSIONES_VIDEO and v.exists()
                ]
            return [v for v in self.videos_base if v.suffix.lower() in EXTENSIONES_VIDEO and v.exists()]
        except Exception:
            return []

    def _duration_seconds_cached(self, ruta: Path):
        txt = self._duration_text(ruta)
        m = re.search(r"^(\d+)m\s+(\d+)s$", str(txt).strip())
        if m:
            return int(m.group(1)) * 60 + int(m.group(2))
        m2 = re.search(r"^(\d+)s$", str(txt).strip())
        if m2:
            return int(m2.group(1))
        return 10**9

    def _recommendation_candidates(self, kind: str, limit: int, stats_map: dict):
        pool = self._recommendation_source_videos()
        if not pool:
            return []

        def _stats(v):
            return stats_map.get(str(v).replace('\\', '/'), {})

        if kind == "favorites":
            favs = [v for v in pool if v.name.lower().startswith("top ")]
            favs.sort(key=lambda v: _stats(v).get('reproducciones', 0), reverse=True)
            return favs[:limit]

        if kind == "most_viewed":
            top = sorted(pool, key=lambda v: _stats(v).get('reproducciones', 0), reverse=True)
            return top[:limit]

        if kind == "shorts":
            shorts = sorted(pool, key=lambda v: self._duration_seconds_cached(v))
            shorts = [v for v in shorts if self._duration_seconds_cached(v) <= 95] or shorts[: max(limit * 2, 24)]
            return shorts[:limit]

        unseen = [v for v in pool if not self._is_video_revisado(v)]
        base = unseen or pool
        base = sorted(base, key=lambda v: _stats(v).get('reproducciones', 0))
        return base[:limit]

    def _make_thumb_icon_for_video(self, ruta: Path, thumbs_map: dict, size: QSize, watched_seconds: int | None = None):
        ruta_norm = str(ruta).replace('\\', '/')
        data = thumbs_map.get(ruta_norm)
        progress_ratio = None
        if watched_seconds is not None:
            dur_secs = self._duration_seconds_cached(ruta)
            if isinstance(dur_secs, int) and dur_secs > 0 and dur_secs < 10**9:
                progress_ratio = max(0.0, min(1.0, float(watched_seconds) / float(dur_secs)))
        if data:
            pm = QPixmap()
            pm.loadFromData(QByteArray(data))
            if not pm.isNull():
                return self._make_youtube_thumb_icon(
                    pm,
                    size,
                    badge_text=self._duration_badge_text(ruta),
                    progress_ratio=progress_ratio,
                )
        return self.style().standardIcon(QStyle.StandardPixmap.SP_FileIcon)

    def _make_youtube_thumb_icon(
        self,
        source_pm: QPixmap,
        size: QSize,
        radius: int = 12,
        badge_text: str = "",
        progress_ratio: float | None = None,
    ):
        """Render a centered 16:9 thumb with YouTube-like rounded corners."""
        if source_pm.isNull() or size.width() <= 0 or size.height() <= 0:
            return self.style().standardIcon(QStyle.StandardPixmap.SP_FileIcon)

        dpr = 2.0
        w = int(size.width() * dpr)
        h = int(size.height() * dpr)
        r = int(radius * dpr)

        canvas = QPixmap(w, h)
        canvas.fill(Qt.GlobalColor.transparent)

        painter = QPainter(canvas)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, True)

        clip_path = QPainterPath()
        clip_path.addRoundedRect(0.0, 0.0, float(w), float(h), float(r), float(r))
        painter.setClipPath(clip_path)

        # Fondo oscuro de vídeo para pillarbox/letterbox, similar a YouTube.
        painter.fillRect(0, 0, w, h, QColor("#0f0f0f"))

        scaled = source_pm.scaled(
            w,
            h,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        x = max(0, (w - scaled.width()) // 2)
        y = max(0, (h - scaled.height()) // 2)
        painter.drawPixmap(x, y, scaled)

        # Duration badge in the bottom-right corner, similar to YouTube.
        if badge_text:
            font = painter.font()
            font.setBold(True)
            font.setPixelSize(max(12, int(13 * dpr)))
            painter.setFont(font)
            fm = painter.fontMetrics()
            text_w = fm.horizontalAdvance(badge_text)
            text_h = fm.height()
            pad_x = int(6 * dpr)
            pad_y = int(2 * dpr)
            badge_w = text_w + pad_x * 2
            badge_h = text_h + pad_y
            margin = int(6 * dpr)
            bx = max(0, w - badge_w - margin)
            by = max(0, h - badge_h - margin)

            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QColor(0, 0, 0, 190))
            painter.drawRoundedRect(bx, by, badge_w, badge_h, int(4 * dpr), int(4 * dpr))
            painter.setPen(QColor("#ffffff"))
            painter.drawText(QRect(bx, by, badge_w, badge_h), Qt.AlignmentFlag.AlignCenter, badge_text)

        # Progress bar at the bottom, like YouTube watched progress.
        if progress_ratio is not None:
            p = max(0.0, min(1.0, float(progress_ratio)))
            if p > 0.0:
                track_h = max(3, int(5 * dpr))
                track_y = max(0, h - track_h)
                painter.setPen(Qt.PenStyle.NoPen)
                painter.setBrush(QColor(0, 0, 0, 110))
                painter.drawRect(0, track_y, w, track_h)
                fill_w = max(2, int(round(w * p)))
                painter.setBrush(QColor("#ff0033"))
                painter.drawRect(0, track_y, min(w, fill_w), track_h)
        painter.end()

        canvas.setDevicePixelRatio(dpr)
        return QIcon(canvas)

    def _populate_reco_list(self, widget: QListWidget, videos: list, thumbs_map: dict, *, shorts=False):
        if not hasattr(self, "reco_list"):
            return
        widget.blockSignals(True)
        widget.clear()
        icon_size = widget.iconSize()
        for v in videos:
            item = QListWidgetItem()
            item.setData(Qt.ItemDataRole.UserRole, str(v))
            item.setIcon(self._make_thumb_icon_for_video(v, thumbs_map, icon_size))
            if shorts:
                item.setText(f"{v.stem[:18]}\n{self._duration_text(v)}")
            else:
                item.setText(v.stem[:26])
            item.setToolTip(str(v))
            widget.addItem(item)
        widget.blockSignals(False)

    def _show_home_dashboard(self):
        self._dashboard_home_active = True
        if hasattr(self, "reco_panel"):
            self.reco_panel.setVisible(False)
        if hasattr(self, "tabla"):
            self.tabla.setVisible(False)
        if hasattr(self, "dashboard_scroll"):
            self.dashboard_scroll.setVisible(True)
        # El dashboard ocupa todo el ancho: ocultamos la columna central (reproductor).
        center_w = getattr(self, "_center_widget", None)
        if center_w is not None:
            center_w.setVisible(False)

    def _go_to_dashboard(self):
        """Volver al dashboard de inicio (clic en logo de YouTube)."""
        self.carpeta_actual = None
        tree = getattr(self, "tree", None)
        if tree is not None:
            try:
                tree.setCurrentItem(None)
            except Exception:
                pass
        self._show_home_dashboard()
        self._refresh_home_dashboard()
        if hasattr(self, "lbl_folder"):
            self.lbl_folder.setText("Inicio")
        if hasattr(self, "_refresh_channel_bar"):
            self._refresh_channel_bar(None, 0)
        self._notify("Inicio", 1200)

    def _show_folder_lists(self):
        self._dashboard_home_active = False
        if hasattr(self, "dashboard_scroll"):
            self.dashboard_scroll.setVisible(False)
        if hasattr(self, "reco_panel"):
            self.reco_panel.setVisible(True)
        if hasattr(self, "tabla"):
            self.tabla.setVisible(True)
        center_w = getattr(self, "_center_widget", None)
        if center_w is not None:
            center_w.setVisible(True)

    def _scroll_dashboard_carousel(self, widget: QListWidget, direction: int):
        if widget is None:
            return
        sb = widget.horizontalScrollBar()
        if sb is None:
            return
        card_w = max(1, int(widget.gridSize().width()))
        step = max(card_w * 2, int(widget.viewport().width() * 0.85))
        sb.setValue(sb.value() + (step if direction > 0 else -step))

    def _sync_dashboard_nav_buttons(self, widget: QListWidget, btn_left: QPushButton, btn_right: QPushButton):
        if widget is None:
            return
        sb = widget.horizontalScrollBar()
        if sb is None:
            return
        lo = int(sb.minimum())
        hi = int(sb.maximum())
        val = int(sb.value())
        has_scroll = hi > lo
        if btn_left is not None:
            btn_left.setEnabled(has_scroll and val > lo)
            btn_left.setVisible(has_scroll)
        if btn_right is not None:
            btn_right.setEnabled(has_scroll and val < hi)
            btn_right.setVisible(has_scroll)

    def _clear_dashboard_block_queue(self):
        self._dashboard_block_queue = []
        self._dashboard_block_queue_name = ""
        self._dashboard_prewarm_target = ""

    def _collect_dashboard_block_videos(self, widget: QListWidget):
        if widget is None:
            return []
        out = []
        seen = set()
        channel_video_map = getattr(self, "_dashboard_channel_video_map", {}) or {}

        def _push_candidate(p: Path):
            try:
                p = Path(p)
            except Exception:
                return
            if not p.exists() or p.suffix.lower() not in EXTENSIONES_VIDEO:
                return
            key = str(p).replace('\\', '/')
            if key in seen:
                return
            seen.add(key)
            out.append(p)

        for i in range(widget.count()):
            item = widget.item(i)
            if item is None:
                continue
            ruta = item.data(Qt.ItemDataRole.UserRole)
            if not ruta:
                continue
            payload_type = item.data(Qt.ItemDataRole.UserRole + 1)
            if payload_type == "channel":
                ch = Path(ruta)
                ch_key = str(ch).replace('\\', '/')
                candidates = list(channel_video_map.get(ch_key, []))
                if not candidates:
                    candidates = [
                        v for v in getattr(self, "videos_base", [])
                        if v.exists() and v.suffix.lower() in EXTENSIONES_VIDEO and v.parent == ch
                    ]
                for v in candidates:
                    _push_candidate(v)
            else:
                _push_candidate(Path(ruta))
        return out

    def _play_dashboard_block(self, widget: QListWidget, block_title: str = "Bloque"):
        self._stop_10s_preview()
        videos = self._collect_dashboard_block_videos(widget)
        if not videos:
            self._notify("Este bloque no tiene videos disponibles", 1800)
            return

        self._dashboard_block_queue = list(videos[1:])
        self._dashboard_block_queue_name = str(block_title or "Bloque")
        self._prewarm_next_dashboard_video()

        first = Path(videos[0])
        self.forzar_guardado_tiempo_actual()
        self._play_dashboard_video_fast(first)
        self._notify(f"Play {self._dashboard_block_queue_name}: {first.name}", 1800)

    def _play_dashboard_video_fast(self, ruta: Path) -> bool:
        """Play a dashboard queue video without forcing a full folder refresh."""
        p = Path(ruta)
        if not p.exists() or p.suffix.lower() not in EXTENSIONES_VIDEO:
            return False
        self.carpeta_actual = p.parent
        self._show_folder_lists()
        if hasattr(self, "lbl_folder"):
            self.lbl_folder.setText(f"{p.parent.name}  —  reproducción rápida")
        self.video_elegido = p
        # Only selects if the row is already present in the current table model.
        self._select_table_row_for_path(p)
        self._reproducir_elegido()
        return True

    def _prewarm_video_file(self, ruta: Path, max_bytes: int = 4 * 1024 * 1024):
        """Best-effort sequential read to warm OS file cache for faster next open."""
        try:
            p = Path(ruta)
            if not p.exists() or not p.is_file():
                return
            remaining = max(256 * 1024, int(max_bytes or 0))
            with open(p, "rb") as f:
                while remaining > 0:
                    chunk = f.read(min(512 * 1024, remaining))
                    if not chunk:
                        break
                    remaining -= len(chunk)
        except Exception:
            return

    def _prewarm_next_dashboard_video(self):
        """Warm up the next queued dashboard video in a background thread."""
        next_path = None
        for cand in self._dashboard_block_queue:
            p = Path(cand)
            if p.exists() and p.suffix.lower() in EXTENSIONES_VIDEO:
                next_path = p
                break
        if next_path is None:
            self._dashboard_prewarm_target = ""
            return

        target_str = str(next_path)
        if self._dashboard_prewarm_target == target_str:
            return
        running = getattr(self, "_dashboard_prewarm_thread", None)
        if running is not None and running.is_alive():
            return

        self._dashboard_prewarm_target = target_str
        t = threading.Thread(target=self._prewarm_video_file, args=(next_path,), daemon=True)
        self._dashboard_prewarm_thread = t
        t.start()

    def _play_next_from_dashboard_block_queue(self):
        while self._dashboard_block_queue:
            cand = Path(self._dashboard_block_queue.pop(0))
            if not cand.exists() or cand.suffix.lower() not in EXTENSIONES_VIDEO:
                continue
            if self._play_dashboard_video_fast(cand):
                self._prewarm_next_dashboard_video()
                return True
        self._clear_dashboard_block_queue()
        return False

    def _autoplay_next_video(self):
        if self._play_next_from_dashboard_block_queue():
            return
        self.proximo_video()

    # ------------------------------------------------------------------ 10s preview
    def _stop_10s_preview(self):
        """Cancel any running 10s-preview session."""
        self._preview_10s_active = False
        if self._preview_10s_timer is not None:
            try:
                self._preview_10s_timer.stop()
                self._preview_10s_timer.deleteLater()
            except Exception:
                pass
            self._preview_10s_timer = None
        self._preview_10s_queue = []
        self._preview_10s_block_title = None

    def _cancel_10s_preview_shortcut(self):
        """Stop the active 10s preview from the keyboard without interrupting current playback."""
        if not self._preview_10s_active:
            return
        self._stop_10s_preview()
        self._notify("Preview 10s desactivado (Ctrl+Shift+X)", 1800)

    def _play_block_10s_preview(self, widget: "QListWidget", block_title: str = "Bloque"):
        """Play 10 seconds from a random position for every video in the block."""
        if self._preview_10s_active and self._preview_10s_block_title == block_title:
            self._stop_10s_preview()
            self._notify("Preview 10s desactivado", 1800)
            return
        videos = self._collect_dashboard_block_videos(widget)
        if not videos:
            self._notify("Este bloque no tiene vídeos disponibles", 1800)
            return
        self._stop_10s_preview()
        self._stop_player_autonext(notify=False)
        self._clear_dashboard_block_queue()
        random.shuffle(videos)
        self._preview_10s_queue = list(videos)
        self._preview_10s_active = True
        self._preview_10s_block_title = block_title
        self._notify(f"▶ 10s · {block_title} ({len(videos)} vídeos)", 2200)
        self._advance_10s_preview()

    def _advance_10s_preview(self):
        """Load next video in the 10s-preview queue and schedule advance."""
        if not self._preview_10s_active:
            return
        while self._preview_10s_queue:
            ruta = Path(self._preview_10s_queue.pop(0))
            if not ruta.exists() or ruta.suffix.lower() not in EXTENSIONES_VIDEO:
                continue
            # Determine a random seek position leaving 12s before end.
            dur_secs = self._duration_seconds_cached(ruta)
            if isinstance(dur_secs, int) and dur_secs > 15 and dur_secs < 10 ** 9:
                max_start = max(0, dur_secs - 12)
                seek_s = random.randint(0, max_start)
            else:
                seek_s = 0
            seek_ms = seek_s * 1000
            self._preview_10s_seek_ms = seek_ms
            # Start playback.
            self.carpeta_actual = ruta.parent
            self._show_folder_lists()
            if hasattr(self, "lbl_folder"):
                self.lbl_folder.setText(f"{ruta.parent.name}  —  preview 10s")
            self.video_elegido = ruta
            self._select_table_row_for_path(ruta)
            self._reproducir_elegido()
            # Seek to random position ~600 ms after play starts.
            if seek_ms > 0:
                QTimer.singleShot(
                    600,
                    lambda ms=seek_ms: (
                        self.media_player.setPosition(ms) if self._preview_10s_active else None
                    ),
                )
            # Schedule advance to next video after 10.6 s.
            self._preview_10s_timer = QTimer(self)
            self._preview_10s_timer.setSingleShot(True)
            self._preview_10s_timer.timeout.connect(self._advance_10s_preview)
            self._preview_10s_timer.start(10_600)
            return
        # Queue exhausted.
        self._stop_10s_preview()
        self._notify("Preview 10s finalizado", 1800)
    # ------------------------------------------------------------------ /10s preview

    def _populate_dashboard_strip(self, widget: QListWidget, videos: list, thumbs_map: dict, stats_map: dict, text_mode: str):
        if widget is None:
            return
        widget.blockSignals(True)
        widget.clear()
        icon_size = widget.iconSize()
        for v in videos:
            stats = stats_map.get(str(v).replace('\\', '/'), {}) if stats_map else {}
            repros = int(stats.get('reproducciones', 0) or 0)
            seen_secs = int(stats.get('tiempo_visto_seg', 0) or 0)
            mins = seen_secs // 60
            item = QListWidgetItem()
            item.setData(Qt.ItemDataRole.UserRole, str(v))
            item.setIcon(self._make_thumb_icon_for_video(v, thumbs_map, icon_size, watched_seconds=seen_secs))
            if text_mode == "last_seen":
                item.setText(f"{v.name[:30]}\\n{mins} min vistos · {repros} vistas")
            elif text_mode == "discovery":
                ch = v.parent.name if v.parent else "Canal"
                item.setText(f"{v.name[:24]}\\n{ch} · por descubrir ({mins} min, {repros} vistas)")
            elif text_mode == "channels":
                ch = v.parent.name if v.parent else "Canal"
                item.setText(f"{v.name[:24]}\\n{ch} · {repros} vistas")
            elif text_mode == "channels_least":
                ch = v.parent.name if v.parent else "Canal"
                item.setText(f"{v.name[:24]}\\n{ch} · poco visto ({repros})")
            elif text_mode == "unreviewed":
                item.setText(f"{v.name[:30]}\\nSin revisar")
            else:
                item.setText(f"{v.name[:30]}\\n{mins} min · {repros} vistas")
            item.setSizeHint(QSize(386, 304))
            item.setToolTip(str(v))
            widget.addItem(item)
        widget.blockSignals(False)

    def _make_dashboard_channel_icon(self, carpeta: Path, target_size: QSize, channel_videos: list, thumbs_map: dict):
        img = self._find_folder_image(carpeta)
        if img:
            pm = QPixmap(str(img))
            if not pm.isNull():
                return self._make_premium_folder_channel_icon(pm, target_size)
        if channel_videos:
            return self._make_thumb_icon_for_video(channel_videos[0], thumbs_map, target_size)
        return QIcon()

    def _populate_dashboard_channels(self, widget: QListWidget, channels: list, stats_map: dict, channel_video_map: dict, thumbs_map: dict):
        if widget is None:
            return
        widget.blockSignals(True)
        widget.clear()
        icon_size = widget.iconSize()
        for ch in channels:
            ch_key = str(ch).replace('\\', '/')
            vids = list(channel_video_map.get(ch_key, []))
            total_views = 0
            for v in vids:
                total_views += int(stats_map.get(str(v).replace('\\', '/'), {}).get('reproducciones', 0) or 0)
            item = QListWidgetItem()
            item.setData(Qt.ItemDataRole.UserRole, str(ch))
            item.setData(Qt.ItemDataRole.UserRole + 1, "channel")
            item.setIcon(self._make_dashboard_channel_icon(ch, icon_size, vids, thumbs_map))
            item.setText(f"{ch.name[:28]}\\n{len(vids)} videos · {total_views} vistas")
            item.setSizeHint(QSize(386, 304))
            item.setToolTip(str(ch))
            widget.addItem(item)
        widget.blockSignals(False)

    def _build_channels_mix(self, pool: list, stats_map: dict, limit: int = 18, least: bool = False):
        per_channel = defaultdict(list)
        channel_score = defaultdict(int)
        for v in pool:
            key = str(v).replace('\\', '/')
            st = stats_map.get(key, {})
            views = int(st.get('reproducciones', 0) or 0)
            secs = int(st.get('tiempo_visto_seg', 0) or 0)
            ch = v.parent
            per_channel[ch].append(v)
            channel_score[ch] += (views * 100 + secs // 10)

        top_channels = sorted(
            channel_score.keys(),
            key=lambda c: (channel_score[c], c.name.lower()),
            reverse=not least,
        )[:8]
        for ch in top_channels:
            if least:
                per_channel[ch].sort(
                    key=lambda v: (
                        int(stats_map.get(str(v).replace('\\', '/'), {}).get('reproducciones', 0) or 0),
                        int(stats_map.get(str(v).replace('\\', '/'), {}).get('tiempo_visto_seg', 0) or 0),
                        v.name.lower(),
                    )
                )
            else:
                per_channel[ch].sort(
                    key=lambda v: (
                        0 if v.name.lower().startswith("top ") else 1,
                        -int(stats_map.get(str(v).replace('\\', '/'), {}).get('reproducciones', 0) or 0),
                        -int(stats_map.get(str(v).replace('\\', '/'), {}).get('tiempo_visto_seg', 0) or 0),
                        v.name.lower(),
                    )
                )

        out = []
        idx = 0
        while len(out) < limit and top_channels:
            any_added = False
            for ch in list(top_channels):
                vids = per_channel.get(ch, [])
                if idx < len(vids):
                    out.append(vids[idx])
                    any_added = True
                    if len(out) >= limit:
                        break
            if not any_added:
                break
            idx += 1
        return out

    def _build_discovery_videos(self, channel_video_map: dict, channel_score: dict, stats_map: dict, limit: int = 18):
        # "Por descubrir": videos de canales muy vistos que aún no vimos o vimos poco.
        high_channels = sorted(
            channel_score.keys(),
            key=lambda c: (channel_score[c], c.name.lower()),
            reverse=True,
        )[:8]
        if not high_channels:
            return []

        low_seen_by_channel = {}
        for ch in high_channels:
            ch_key = str(ch).replace('\\', '/')
            candidates = []
            for v in channel_video_map.get(ch_key, []):
                key = str(v).replace('\\', '/')
                st = stats_map.get(key, {})
                views = int(st.get('reproducciones', 0) or 0)
                secs = int(st.get('tiempo_visto_seg', 0) or 0)
                has_last_play = bool(st.get('ultima_reproduccion'))
                # Nunca visto, o visto muy poco (<= 5 min o <= 1 reproducción).
                is_unseen = (not has_last_play) and views == 0 and secs == 0
                is_low_seen = secs < 300 or views <= 1
                if is_unseen or is_low_seen:
                    candidates.append(v)

            candidates.sort(
                key=lambda v: (
                    0 if ((not bool(stats_map.get(str(v).replace('\\', '/'), {}).get('ultima_reproduccion')))
                          and int(stats_map.get(str(v).replace('\\', '/'), {}).get('reproducciones', 0) or 0) == 0
                          and int(stats_map.get(str(v).replace('\\', '/'), {}).get('tiempo_visto_seg', 0) or 0) == 0) else 1,
                    int(stats_map.get(str(v).replace('\\', '/'), {}).get('tiempo_visto_seg', 0) or 0),
                    int(stats_map.get(str(v).replace('\\', '/'), {}).get('reproducciones', 0) or 0),
                    v.name.lower(),
                )
            )
            low_seen_by_channel[ch_key] = candidates

        out = []
        idx = 0
        while len(out) < limit and high_channels:
            any_added = False
            for ch in high_channels:
                ch_key = str(ch).replace('\\', '/')
                vids = low_seen_by_channel.get(ch_key, [])
                if idx < len(vids):
                    out.append(vids[idx])
                    any_added = True
                    if len(out) >= limit:
                        break
            if not any_added:
                break
            idx += 1
        return out

    def _build_short_videos(self, pool: list, stats_map: dict, limit: int = 18) -> list:
        """Return up to *limit* videos with duration < 90 seconds, shuffled per refresh."""
        SHORT_SECS = 90
        MAX_PROBE = 120  # probe at most this many uncached videos to avoid slowdown

        def _cached_secs(ruta_str: str) -> int | None:
            txt = self.duration_cache.get(ruta_str)
            if not txt:
                return None
            m = re.search(r"^(\d+)m\s+(\d+)s$", str(txt).strip())
            if m:
                return int(m.group(1)) * 60 + int(m.group(2))
            m2 = re.search(r"^(\d+)s$", str(txt).strip())
            if m2:
                return int(m2.group(1))
            return None

        out = []
        candidates = list(pool)
        random.shuffle(candidates)
        probed = 0
        for v in candidates:
            ruta_str = str(v)
            secs = _cached_secs(ruta_str)
            if secs is None and self.ffprobe_path and probed < MAX_PROBE:
                try:
                    cmd = [
                        self.ffprobe_path,
                        '-v', 'error',
                        '-show_entries', 'format=duration',
                        '-of', 'default=noprint_wrappers=1:nokey=1',
                        ruta_str,
                    ]
                    raw = float(subprocess.check_output(cmd, timeout=3).decode().strip())
                    secs = max(0, int(raw))
                    dm, ds = divmod(secs, 60)
                    self.duration_cache[ruta_str] = f"{dm}m {ds}s"
                except Exception:
                    secs = None
                probed += 1
            if secs is not None and secs < SHORT_SECS:
                out.append(v)
            if len(out) >= limit:
                break
        random.shuffle(out)
        return out

    def _refresh_home_dashboard(self):
        if not hasattr(self, "dash_recommended_list"):
            return
        if not self.ruta_raiz or self.incluir_fotos_gestion:
            for w in (
                self.dash_recommended_list,
                self.dash_last_seen_list,
                self.dash_channels_mix_list,
                self.dash_unreviewed_list,
                self.dash_channels_least_list,
                self.dash_discovery_list,
                self.dash_short_list,
            ):
                w.clear()
            return

        pool = [v for v in self.videos_base if v.suffix.lower() in EXTENSIONES_VIDEO and v.exists()]
        if not pool:
            for w in (
                self.dash_recommended_list,
                self.dash_last_seen_list,
                self.dash_channels_mix_list,
                self.dash_unreviewed_list,
                self.dash_channels_least_list,
                self.dash_discovery_list,
                self.dash_short_list,
            ):
                w.clear()
            return

        stats_map = self.db.obtener_stats_batch([str(v) for v in pool])

        channel_video_map = defaultdict(list)
        channel_score = defaultdict(int)
        for v in pool:
            ch = v.parent
            ch_key = str(ch).replace('\\', '/')
            channel_video_map[ch_key].append(v)
            st = stats_map.get(str(v).replace('\\', '/'), {})
            views = int(st.get('reproducciones', 0) or 0)
            secs = int(st.get('tiempo_visto_seg', 0) or 0)
            channel_score[ch] += (views * 100 + secs // 10)

        channels_top = sorted(
            channel_score.keys(),
            key=lambda c: (channel_score[c], c.name.lower()),
            reverse=True,
        )[:18]
        self._dashboard_channel_video_map = {
            k: list(vs) for k, vs in channel_video_map.items()
        }

        recommended_pool = sorted(
            pool,
            key=lambda v: (
                0 if v.name.lower().startswith("top ") else 1,
                -int(stats_map.get(str(v).replace('\\', '/'), {}).get('tiempo_visto_seg', 0) or 0),
                -int(stats_map.get(str(v).replace('\\', '/'), {}).get('reproducciones', 0) or 0),
                v.name.lower(),
            ),
        )
        # Bloque 1: variación constante para evitar ver siempre los mismos.
        # Tomamos una bolsa amplia de candidatos "buenos" y elegimos 18 al azar.
        reco_candidates = recommended_pool[: max(18, min(len(recommended_pool), 120))]
        if len(reco_candidates) <= 18:
            recommended = list(reco_candidates)
            random.shuffle(recommended)
        else:
            recommended = random.sample(reco_candidates, 18)

        last_seen = sorted(
            [
                v
                for v in pool
                if stats_map.get(str(v).replace('\\', '/'), {}).get('ultima_reproduccion')
                and int(stats_map.get(str(v).replace('\\', '/'), {}).get('tiempo_visto_seg', 0) or 0) >= 60
            ],
            key=lambda v: str(stats_map.get(str(v).replace('\\', '/'), {}).get('ultima_reproduccion') or ""),
            reverse=True,
        )[:18]

        channels_mix = self._build_channels_mix(pool, stats_map, limit=18)
        channels_least = self._build_channels_mix(pool, stats_map, limit=18, least=True)

        unseen = [v for v in pool if not self._is_video_revisado(v)]
        random.shuffle(unseen)
        unreviewed = unseen[:18]

        discovery = self._build_discovery_videos(channel_video_map, channel_score, stats_map, limit=18)

        stats_map_dash = dict(stats_map)

        channel_representatives = []
        for ch in channels_top:
            ch_key = str(ch).replace('\\', '/')
            vids = channel_video_map.get(ch_key, [])
            if vids:
                channel_representatives.append(vids[0])

        short_videos = self._build_short_videos(pool, stats_map, limit=18)

        selected = recommended + last_seen + channels_mix + unreviewed + channels_least + discovery + channel_representatives + short_videos
        want_paths = [str(v) for v in selected]
        thumbs_map = self.db.obtener_miniaturas_batch(want_paths) if want_paths else {}
        self._dashboard_thumb_cache = dict(thumbs_map)

        self._populate_dashboard_strip(self.dash_recommended_list, recommended, thumbs_map, stats_map_dash, "recommended")
        self._populate_dashboard_strip(self.dash_last_seen_list, last_seen, thumbs_map, stats_map_dash, "last_seen")
        self._populate_dashboard_channels(self.dash_channels_mix_list, channels_top, stats_map_dash, self._dashboard_channel_video_map, thumbs_map)
        self._populate_dashboard_strip(self.dash_unreviewed_list, unreviewed, thumbs_map, stats_map_dash, "unreviewed")
        self._populate_dashboard_strip(self.dash_channels_least_list, channels_least, thumbs_map, stats_map_dash, "channels_least")
        self._populate_dashboard_strip(self.dash_discovery_list, discovery, thumbs_map, stats_map_dash, "discovery")
        self._populate_dashboard_strip(self.dash_short_list, short_videos, thumbs_map, stats_map_dash, "short")

    def _on_dashboard_item_clicked(self, item: QListWidgetItem):
        if not item:
            return
        ruta = item.data(Qt.ItemDataRole.UserRole)
        if not ruta:
            return
        self._stop_10s_preview()
        self._clear_dashboard_block_queue()
        payload_type = item.data(Qt.ItemDataRole.UserRole + 1)
        # Build a continuation queue from the same block widget, starting after the
        # clicked item. This way pressing Siguiente after a card click still follows
        # the block's playlist (e.g. "Videos cortos").
        src_widget = self.sender()
        if isinstance(src_widget, QListWidget) and payload_type != "channel":
            clicked_idx = src_widget.row(item) if src_widget else -1
            if clicked_idx >= 0:
                queue_candidates = []
                for i in range(clicked_idx + 1, src_widget.count()):
                    it = src_widget.item(i)
                    if it is None:
                        continue
                    r = it.data(Qt.ItemDataRole.UserRole)
                    if not r:
                        continue
                    if it.data(Qt.ItemDataRole.UserRole + 1) == "channel":
                        continue
                    cand_path = Path(r)
                    if cand_path.exists() and cand_path.suffix.lower() in EXTENSIONES_VIDEO:
                        queue_candidates.append(cand_path)
                if queue_candidates:
                    self._dashboard_block_queue = queue_candidates
                    self._dashboard_block_queue_name = src_widget.objectName() or "Bloque"
                    self._prewarm_next_dashboard_video()
        if payload_type == "channel":
            ch = Path(ruta)
            ch_key = str(ch).replace('\\', '/')
            channel_video_map = getattr(self, "_dashboard_channel_video_map", {}) or {}
            candidates = [
                v for v in channel_video_map.get(ch_key, [])
                if v.exists() and v.suffix.lower() in EXTENSIONES_VIDEO
            ]
            if not candidates:
                candidates = [
                    v for v in getattr(self, "videos_base", [])
                    if v.exists() and v.suffix.lower() in EXTENSIONES_VIDEO and v.parent == ch
                ]
            if not candidates:
                self._notify("Este canal no tiene videos disponibles", 1800)
                return
            p = random.choice(candidates)
        else:
            p = Path(ruta)
        if not p.exists():
            self._notify("El vídeo del dashboard ya no existe", 1800)
            self._refresh_home_dashboard()
            return
        self.forzar_guardado_tiempo_actual()
        self.carpeta_actual = p.parent
        self._show_folder_lists()
        self._refresh_list()
        self.video_elegido = p
        self._select_table_row_for_path(p)
        self._reproducir_elegido()

    def _refresh_visual_recommendations(self, items: list, stats_map: dict, thumbs_map: dict):
        if not hasattr(self, "reco_list"):
            return
        if self.incluir_fotos_gestion:
            self.reco_list.clear()
            self._reco_top_pool = []
            self._reco_top_cursor = 0
            self._reco_extra_pool = []
            self._reco_extra_cursor = 0
            return

        pool = self._recommendation_source_videos()
        if not pool:
            self.reco_list.clear()
            self._reco_top_pool = []
            self._reco_top_cursor = 0
            self._reco_extra_pool = []
            self._reco_extra_cursor = 0
            return

        full_stats = stats_map or self.db.obtener_stats_batch([str(v) for v in pool])
        top_pool = [v for v in pool if v.name.lower().startswith("top ")]
        extra_pool = [v for v in pool if not v.name.lower().startswith("top ")]
        top_pool.sort(
            key=lambda v: (
                -int(full_stats.get(str(v).replace('\\', '/'), {}).get('reproducciones', 0)),
                v.name.lower(),
            )
        )
        extra_pool.sort(
            key=lambda v: (
                -int(full_stats.get(str(v).replace('\\', '/'), {}).get('tiempo_visto_seg', 0)),
                -int(full_stats.get(str(v).replace('\\', '/'), {}).get('reproducciones', 0)),
                v.name.lower(),
            )
        )

        self._reco_top_pool = top_pool
        self._reco_top_cursor = 0
        self._reco_extra_pool = extra_pool
        self._reco_extra_cursor = 0
        self._reco_thumb_cache = dict(thumbs_map or {})
        self._reco_stats_map = dict(full_stats or {})
        self.reco_list.clear()
        self._append_reco_top_chunk()
        # Si no hay favoritos (o ya se agotaron en el primer lote), mostrar 10 extras.
        if self._reco_top_cursor >= len(self._reco_top_pool):
            self._append_reco_top_chunk()

    def _append_reco_top_chunk(self):
        if self._reco_loading_chunk:
            return
        using_extras = self._reco_top_cursor >= len(self._reco_top_pool)
        if using_extras:
            if not self._reco_extra_pool or self._reco_extra_cursor >= len(self._reco_extra_pool):
                return
        else:
            if not self._reco_top_pool:
                return

        self._reco_loading_chunk = True
        try:
            if using_extras:
                start = self._reco_extra_cursor
                end = min(start + self._reco_extra_chunk_size, len(self._reco_extra_pool))
                chunk = self._reco_extra_pool[start:end]
            else:
                start = self._reco_top_cursor
                end = min(start + self._reco_top_chunk_size, len(self._reco_top_pool))
                chunk = self._reco_top_pool[start:end]
            needed = [str(v) for v in chunk]
            missing = [r for r in needed if r.replace('\\', '/') not in self._reco_thumb_cache]
            if missing:
                try:
                    self._reco_thumb_cache.update(self.db.obtener_miniaturas_batch(missing))
                except Exception:
                    pass

            self.reco_list.blockSignals(True)
            icon_size = self.reco_list.iconSize()
            for v in chunk:
                stats = self._reco_stats_map.get(str(v).replace('\\', '/'), {})
                repros = int(
                    stats.get('reproducciones', 0)
                )
                seen_secs = int(stats.get('tiempo_visto_seg', 0) or 0)
                seen_mins = max(0, seen_secs // 60)
                item = QListWidgetItem()
                item.setData(Qt.ItemDataRole.UserRole, str(v))
                item.setIcon(self._make_thumb_icon_for_video(v, self._reco_thumb_cache, icon_size, watched_seconds=seen_secs))
                if using_extras:
                    item.setText(f"{v.name[:34]}\n{seen_mins} min vistos · {repros} vistas")
                else:
                    item.setText(f"{v.name[:34]}\n{repros} vistas")
                item.setSizeHint(QSize(198, 148))
                item.setToolTip(str(v))
                self.reco_list.addItem(item)
            self.reco_list.blockSignals(False)
            if using_extras:
                self._reco_extra_cursor = end
            else:
                self._reco_top_cursor = end
        finally:
            self._reco_loading_chunk = False

    def _on_reco_scroll_changed(self, value: int):
        bar = self.reco_list.horizontalScrollBar() if hasattr(self, "reco_list") else None
        if not bar:
            return
        if value >= bar.maximum() - 4:
            self._append_reco_top_chunk()

    def _on_reco_item_clicked(self, item: QListWidgetItem):
        if not item:
            return
        ruta = item.data(Qt.ItemDataRole.UserRole)
        if not ruta:
            return
        self._clear_dashboard_block_queue()
        p = Path(ruta)
        if not p.exists():
            self._notify("El vídeo recomendado ya no existe", 1800)
            self._refresh_list()
            return
        self.forzar_guardado_tiempo_actual()
        self.video_elegido = p
        self._select_table_row_for_path(p)
        self._reproducir_elegido()

    def _select_table_row_for_path(self, ruta: Path):
        target = str(ruta)
        for r in range(self.tabla.rowCount()):
            it = self.tabla.item(r, 0)
            if it and it.data(Qt.ItemDataRole.UserRole) == target:
                self.tabla.setCurrentCell(r, 0)
                self.tabla.scrollToItem(it)
                return True
        return False

    def _play_recommendation(self, kind: str):
        pool = self._recommendation_source_videos()
        if not pool:
            QMessageBox.information(self, "Recomendados", "No hay vídeos disponibles para recomendar.")
            return

        stats_map = self.db.obtener_stats_batch([str(v) for v in pool])

        def _stats(v):
            return stats_map.get(str(v).replace('\\', '/'), {})

        choice = None

        if kind == "favorites":
            favs = [v for v in pool if v.name.lower().startswith("top ")]
            if favs:
                favs.sort(key=lambda v: _stats(v).get('reproducciones', 0), reverse=True)
                choice = random.choice(favs[: min(len(favs), 20)])

        elif kind == "most_viewed":
            top = sorted(pool, key=lambda v: _stats(v).get('reproducciones', 0), reverse=True)
            if top:
                choice = random.choice(top[: min(len(top), 25)])

        elif kind == "shorts":
            shorts = sorted(pool, key=lambda v: self._duration_seconds_cached(v))
            shorts = [v for v in shorts if self._duration_seconds_cached(v) <= 95] or shorts[:40]
            if shorts:
                choice = random.choice(shorts[: min(len(shorts), 30)])

        else:  # for_you
            unseen = [v for v in pool if not self._is_video_revisado(v)]
            base = unseen or pool
            base.sort(key=lambda v: _stats(v).get('reproducciones', 0))
            choice = random.choice(base[: min(len(base), 35)]) if base else None

        if not choice:
            QMessageBox.information(self, "Recomendados", "No se pudo obtener una recomendación ahora mismo.")
            return

        self.forzar_guardado_tiempo_actual()
        self.video_elegido = Path(choice)
        self._reproducir_elegido()
        self._notify(f"Recomendado: {self.video_elegido.name}", 1800)

    def aplicar_filtros(self, lista, modo):
        if modo == '8':
            sin_revisar = [v for v in lista if not self._is_video_revisado(v)]
            random.shuffle(sin_revisar)
            return sin_revisar
        if modo == '2':
            fav = [v for v in lista if v.name.lower().startswith("top ")]
            random.shuffle(fav)
            return fav[:100]
        if modo == '6':
            pesados = sorted(lista, key=lambda v: (os.path.getsize(v) if v.exists() else 0), reverse=True)[:100]
            random.shuffle(pesados)
            return pesados
        if modo == '7':
            stats_map = self.db.obtener_stats_batch([str(v) for v in lista])
            por_t = sorted(
                lista,
                key=lambda v: stats_map.get(str(v).replace('\\', '/'), {}).get('tiempo_visto_seg', 0),
                reverse=True,
            )[:100]
            random.shuffle(por_t)
            return por_t
        if modo in ['3', '4']:
            stats_map = self.db.obtener_stats_batch([str(v) for v in lista])
            grupos = defaultdict(list)
            for v in lista:
                repros = stats_map.get(str(v).replace('\\', '/'), {}).get('reproducciones', 0)
                grupos[repros].append(v)
            res = []
            for n in sorted(grupos.keys(), reverse=(modo == '4')):
                vids = grupos[n]
                random.shuffle(vids)
                res.extend(vids)
                if len(res) >= 100:
                    break
            top = res[:100]
            random.shuffle(top)
            return top
        return lista

    # ── Reproducción secuencial ──

    def _remember_played_video(self, ruta: Path):
        """Añade el vídeo actual al historial de reproducción."""
        if not ruta:
            return
        try:
            ruta = Path(ruta)
        except Exception:
            return
        if 0 <= self._play_history_idx < len(self._play_history):
            if self._play_history[self._play_history_idx] == ruta:
                return
        if self._play_history_idx < len(self._play_history) - 1:
            self._play_history = self._play_history[: self._play_history_idx + 1]
        self._play_history.append(ruta)
        if len(self._play_history) > 2000:
            self._play_history = self._play_history[-2000:]
        self._play_history_idx = len(self._play_history) - 1

    def video_anterior(self):
        print("\n⏮ Anterior pulsado")
        self.forzar_guardado_tiempo_actual()
        if not self._play_history:
            QMessageBox.information(self, "Anterior", "No hay historial de reproducción.")
            return
        idx = self._play_history_idx if self._play_history_idx >= 0 else len(self._play_history)
        idx -= 1
        while idx >= 0:
            cand = Path(self._play_history[idx])
            if cand.exists():
                self._play_history_idx = idx
                self.video_elegido = cand
                self._reproducir_elegido(push_history=False)
                return
            idx -= 1
        QMessageBox.information(self, "Anterior", "No hay un vídeo anterior disponible.")

    def proximo_video(self):
        print("\n⏭ Siguiente pulsado")
        self.forzar_guardado_tiempo_actual()
        if self._play_next_from_dashboard_block_queue():
            return
        if not self.lista_actual:
            self._scan()
            if self.incluir_fotos_gestion:
                try:
                    base = self.carpeta_actual or self.ruta_raiz
                    lista_base = [
                        f for f in (base.rglob("*") if base else [])
                        if f.suffix.lower() in EXTENSIONES_IMAGEN and f.is_file()
                    ] if base else []
                except (PermissionError, OSError):
                    lista_base = []
            else:
                lista_base = self.videos_base
                if self.modo_actual == '8' and self.carpeta_actual:
                    try:
                        lista_base = [
                            v for v in self.carpeta_actual.rglob("*")
                            if v.suffix.lower() in EXTENSIONES_VIDEO and v.is_file()
                        ]
                    except (PermissionError, OSError):
                        lista_base = []
            self.lista_actual = self.aplicar_filtros(lista_base, self.modo_actual)
            if self.carpeta_fijada:
                self.lista_actual = [v for v in self.lista_actual if self.carpeta_fijada in v.parents]
        if not self.lista_actual:
            QMessageBox.information(self, "Fin", "No quedan vídeos en la lista.")
            self._notify("Lista terminada", 2000)
            return
        if self.modo_actual in ['2', '3', '4', '6', '7', '8'] and not self.carpeta_fijada:
            self.video_elegido = self.lista_actual.pop(0)
        else:
            stats_map = self.db.obtener_stats_batch([str(v) for v in self.lista_actual])
            def _repros(v):
                return stats_map.get(str(v).replace('\\', '/'), {}).get('reproducciones', 0)
            self.lista_actual.sort(key=_repros)
            min_r = _repros(self.lista_actual[0])
            candidatos = [v for v in self.lista_actual if _repros(v) == min_r]
            self.video_elegido = random.choice(candidatos)
            if self.video_elegido in self.lista_actual:
                self.lista_actual.remove(self.video_elegido)
        self._reproducir_elegido()

    def forzar_guardado_tiempo_actual(self):
        self._cancel_photo_autonext()
        checkpoint_folder = None
        if self.player_thread and self.player_thread.isRunning():
            video_anterior = self.video_elegido
            inicio = getattr(self.player_thread, '_inicio', None)
            print(f"⏹ Cerrando reproductor: {video_anterior.name if video_anterior else '?'}")
            try:
                self.player_thread.finished_signal.disconnect(self._on_play_done)
            except:
                pass
            self.player_thread.detener_reproductor()
            self.player_thread.wait()  # esperar sin timeout; el kill hace que sea rápido
            self.player_thread = None
            self.media_player.stop()
            self._active_ruta_reproduccion = None
            if video_anterior and video_anterior.exists():
                t_sesion = int(time.time() - inicio) if inicio else 0
                ruta_anterior_str = str(video_anterior)
                datos = self.db.obtener_stats_video(ruta_anterior_str)
                primera = datos['reproducciones'] == 0
                self.db.registrar_visualizacion(ruta_anterior_str, t_sesion)
                if not self._is_video_revisado(video_anterior):
                    self._queue_rwd_startup_only(video_anterior, "deferred_play_forzado")
                ruta_final = str(self.video_elegido if self.video_elegido else video_anterior)
                stats = self.db.obtener_stats_video(ruta_final)
                self._queue_metadata_sync_path(ruta_final)
                self._launch_meta_check_for_video(Path(ruta_final), source="play_forzado")
                print(f"💾 Guardado (forzado): {video_anterior.name} → {Path(ruta_final).name} (+{t_sesion}s, repro={stats['reproducciones']}, primera={primera})")
                checkpoint_folder = (self.video_elegido if self.video_elegido else video_anterior).parent
            else:
                print(f"⚠ Video anterior no existe, no se guardó nada")
        elif self.media_player.playbackState() != QMediaPlayer.PlaybackState.StoppedState and self.video_elegido and self.video_elegido.exists():
            video_anterior = self.video_elegido
            inicio = getattr(self.player_thread, '_inicio', None)
            print(f"⏹ Cerrando reproductor integrado: {video_anterior.name if video_anterior else '?'}")
            self.media_player.stop()
            self._active_ruta_reproduccion = None
            if video_anterior and video_anterior.exists():
                t_sesion = int(time.time() - inicio) if inicio else 0
                ruta_anterior_str = str(video_anterior)
                datos = self.db.obtener_stats_video(ruta_anterior_str)
                primera = datos['reproducciones'] == 0
                self.db.registrar_visualizacion(ruta_anterior_str, t_sesion)
                if not self._is_video_revisado(video_anterior):
                    self._queue_rwd_startup_only(video_anterior, "deferred_play_forzado")
                ruta_final = str(self.video_elegido if self.video_elegido else video_anterior)
                stats = self.db.obtener_stats_video(ruta_final)
                self._queue_metadata_sync_path(ruta_final)
                self._launch_meta_check_for_video(Path(ruta_final), source="play_forzado")
                print(f"💾 Guardado (forzado): {video_anterior.name} → {Path(ruta_final).name} (+{t_sesion}s, repro={stats['reproducciones']}, primera={primera})")
                checkpoint_folder = (self.video_elegido if self.video_elegido else video_anterior).parent

        if checkpoint_folder:
            self._log_playback_checkpoint(checkpoint_folder, "playback_forzado")

    # ── Reproducción ──

    def reproducir_video_actual(self):
        if self._privacy_locked:
            return
        if not self.video_elegido or not self.video_elegido.exists():
            QMessageBox.warning(self, "Error", "No hay elemento seleccionado")
            return
        self._clear_dashboard_block_queue()
        if self.video_elegido.suffix.lower() in EXTENSIONES_IMAGEN:
            self._reproducir_foto(self.video_elegido)
            return
        print(f"\n▶ Play manual: {self.video_elegido.name}")
        self.forzar_guardado_tiempo_actual()
        self._reproducir_elegido()

    def _reproducir_elegido(self, push_history=True):
        if not self.video_elegido or not self.video_elegido.exists():
            return
        if self.video_elegido.suffix.lower() in EXTENSIONES_IMAGEN:
            self._reproducir_foto(self.video_elegido, push_history=push_history)
            return
        self._cancel_photo_autonext()
        if push_history:
            self._remember_played_video(self.video_elegido)
        self._start_folder_view_log(self.video_elegido.parent)
        ruta_str = str(self.video_elegido)
        stats = self.db.obtener_stats_video(ruta_str)
        try:
            peso = os.path.getsize(self.video_elegido) / (1024 * 1024)
        except OSError:
            peso = 0
        m, s = divmod(stats['tiempo_visto_seg'], 60)
        dur = self._duration_text(self.video_elegido)
        fav = "  ⭐" if self.video_elegido.name.lower().startswith("top ") else ""
        carpeta_txt = str(self.video_elegido.parent)
        if self.ruta_raiz:
            try:
                carpeta_txt = str(self.video_elegido.parent.relative_to(self.ruta_raiz)).replace("\\", "/")
                if not carpeta_txt:
                    carpeta_txt = "."
            except Exception:
                carpeta_txt = str(self.video_elegido.parent)
        self.lbl_detail.setText(
            f"▶ {self.video_elegido.name}{fav}   —   {stats['reproducciones']}x  ·  {dur}  ·  acum {m}m{s}s  ·  {peso:.1f}MB\n"
            f"📁 {carpeta_txt}"
        )
        self.btn_fav.setText("💔 Quitar Top" if self.video_elegido.name.lower().startswith("top ") else "⭐ Favorito")
        # seleccionar en la tabla si existe
        for i in range(self.tabla.rowCount()):
            it = self.tabla.item(i, 0)
            if it and Path(it.data(Qt.ItemDataRole.UserRole)) == self.video_elegido:
                self.tabla.selectRow(i)
                break

        if self.player_thread and self.player_thread.isRunning():
            self.player_thread.detener_reproductor()
            self.player_thread.wait()  # esperar sin timeout; el kill hace que sea rápido
            self.player_thread = None
        self.media_player.stop()
        self.preview_stack.setCurrentWidget(self.video_widget)
        print(f"🎬 Abriendo video: {self.video_elegido.name} (repro={stats['reproducciones']}, acum={m}m{s}s, {peso:.1f}MB)")
        self._active_ruta_reproduccion = ruta_str
        self.player_thread = _EmbeddedPlaybackState(ruta_str)
        self.media_player.setSource(QUrl.fromLocalFile(ruta_str))
        self.media_player.play()
        self._restart_player_autonext_timer()
        self._notify(f"Reproduciendo: {self.video_elegido.name}")
        # Hashear en background si aún no tiene hash
        threading.Thread(
            target=self._hash_video_and_postcheck_worker,
            args=(ruta_str, "hash_reproduccion"),
            daemon=True
        ).start()

    def _reproducir_foto(self, ruta: Path, push_history=True):
        """Muestra una foto en el panel y la registra como vista/revisada."""
        self._cancel_photo_autonext()
        ruta = Path(ruta)
        if not ruta.exists():
            return
        if push_history:
            self._remember_played_video(ruta)
        self._start_folder_view_log(ruta.parent)
        self._mostrar_foto_en_panel(ruta)
        ruta_str = str(ruta)
        self.db.registrar_visualizacion(ruta_str, 0)
        stats = self.db.obtener_stats_video(ruta_str)
        try:
            peso = os.path.getsize(ruta) / (1024 * 1024)
        except OSError:
            peso = 0
        fav = "  ⭐" if ruta.name.lower().startswith("top ") else ""
        carpeta_txt = str(ruta.parent)
        if self.ruta_raiz:
            try:
                carpeta_txt = str(ruta.parent.relative_to(self.ruta_raiz)).replace("\\", "/")
                if not carpeta_txt:
                    carpeta_txt = "."
            except Exception:
                carpeta_txt = str(ruta.parent)
        self.lbl_detail.setText(
            f"📷 {ruta.name}{fav}   —   {stats['reproducciones']}x  ·  Foto  ·  {peso:.1f}MB\n"
            f"📁 {carpeta_txt}"
        )
        self.btn_fav.setText("💔 Quitar Top" if ruta.name.lower().startswith("top ") else "⭐ Favorito")
        for i in range(self.tabla.rowCount()):
            it = self.tabla.item(i, 0)
            if it and Path(it.data(Qt.ItemDataRole.UserRole)) == ruta:
                self.tabla.selectRow(i)
                break
        # Marcar como revisada añadiendo "rwd " al nombre
        if not self._is_video_revisado(ruta):
            self._rename_rwd(ruta)
            ruta = self.video_elegido  # puede haber cambiado tras renombrar
        self._refresh_list()
        if ruta and Path(ruta).exists():
            self.video_elegido = Path(ruta)
            for i in range(self.tabla.rowCount()):
                it = self.tabla.item(i, 0)
                if it and Path(it.data(Qt.ItemDataRole.UserRole)) == self.video_elegido:
                    self.tabla.selectRow(i)
                    break
            self._show_detail(self.video_elegido)
        self._notify(f"Foto: {Path(ruta).name}")
        self._schedule_photo_autonext(ruta, delay_ms=5000)

    def _cancel_photo_autonext(self):
        # Invalidates any pending callback previously scheduled for photos.
        self._photo_autonext_token += 1

    def _schedule_photo_autonext(self, ruta: Path, delay_ms=5000):
        token = self._photo_autonext_token
        ruta_str = str(Path(ruta))
        QTimer.singleShot(max(1, int(delay_ms)), lambda: self._fire_photo_autonext(token, ruta_str))

    def _fire_photo_autonext(self, token, ruta_str):
        if token != self._photo_autonext_token:
            return
        if self._privacy_locked:
            return
        actual = self.video_elegido
        if not actual or not Path(actual).exists():
            return
        if Path(actual).suffix.lower() not in EXTENSIONES_IMAGEN:
            return
        if str(Path(actual)) != str(Path(ruta_str)):
            return
        self._cancel_photo_autonext()
        self.proximo_video()

    def _on_play_done(self, ruta_str, t_sesion):
        ruta_reproducida = Path(ruta_str) if ruta_str else None
        if not ruta_reproducida:
            return
        self.video_elegido = ruta_reproducida
        self._active_ruta_reproduccion = None
        nombre = ruta_reproducida.name
        print(f"⏹ Video cerrado: {nombre} (+{t_sesion}s)")
        ruta_video_str = str(ruta_reproducida)
        datos = self.db.obtener_stats_video(ruta_video_str)
        primera = datos['reproducciones'] == 0
        self.db.registrar_visualizacion(ruta_video_str, t_sesion)
        if not self._is_video_revisado(ruta_reproducida):
            self._queue_rwd_startup_only(ruta_reproducida, "deferred_play_end")
        ruta_final = str(self.video_elegido)
        nombre_final = self.video_elegido.name  # capturar antes de que _refresh_list lo anule
        stats = self.db.obtener_stats_video(ruta_final)
        self._queue_metadata_sync_path(ruta_final)
        self._launch_meta_check_for_video(Path(ruta_final), source="play_end")
        self._log_playback_checkpoint(self.video_elegido.parent, "playback_end")
        self._refresh_list()
        self.player_thread = None
        print(f"💾 Guardado: {nombre} → {nombre_final} (+{t_sesion}s, repro={stats['reproducciones']}, primera={primera})")

    def _release_main_video_file_handle(self):
        """Ensure the integrated player releases file handle before rename/delete on Windows."""
        try:
            self.media_player.stop()
            self.media_player.setSource(QUrl())
            QApplication.processEvents()
            time.sleep(0.08)
            QApplication.processEvents()
        except Exception:
            pass

    def _release_photo_preview_file_handle(self):
        """Release any pixmap-based handle to current photo before delete/rename on Windows."""
        try:
            self._photo_preview_pm = None
            if hasattr(self, "photo_preview_label") and self.photo_preview_label is not None:
                self.photo_preview_label.clear()
            QApplication.processEvents()
        except Exception:
            pass

    def _rename_with_retry(self, old_path: Path, new_path: Path, attempts: int = 6, delay: float = 0.12):
        """Rename with short retries to avoid transient WinError 32 file-lock races."""
        last_error = None
        for i in range(max(1, attempts)):
            try:
                Path(old_path).rename(Path(new_path))
                return
            except OSError as e:
                last_error = e
                if getattr(e, "winerror", None) in (32, 5) and i < attempts - 1:
                    QApplication.processEvents()
                    time.sleep(delay)
                    continue
                raise
        if last_error:
            raise last_error

    def _queue_rwd_retry(self, ruta: Path, error_msg: str = ""):
        try:
            ruta = Path(ruta)
        except Exception:
            return
        ruta_str = str(ruta)
        self._pending_rwd_renames.add(ruta_str)
        # Persistir en BD para reintentar al iniciar la app
        try:
            self.db.add_pending_rename(ruta_str, error_msg)
        except Exception:
            pass
        if not self._pending_rwd_timer.isActive():
            self._pending_rwd_timer.start()

    def _queue_rwd_startup_only(self, ruta: Path, error_msg: str = ""):
        """Persist a pending _rwd rename for next app startup without retrying in-session."""
        try:
            ruta = Path(ruta)
        except Exception:
            return
        try:
            self.db.add_pending_rename(str(ruta), error_msg or "deferred_photo_rename")
        except Exception:
            pass
        self._refresh_pending_ops_badge()

    def _retry_pending_rwd_renames(self):
        if not self._pending_rwd_renames:
            self._pending_rwd_timer.stop()
            return
        pendientes = list(self._pending_rwd_renames)
        self._pending_rwd_renames.clear()
        for ruta_str in pendientes:
            intentos = self._pending_rwd_attempts.get(ruta_str, 0) + 1
            self._pending_rwd_attempts[ruta_str] = intentos
            ok = self._rename_rwd(Path(ruta_str), queue_on_fail=False, silent=True)
            if ok:
                self._pending_rwd_attempts.pop(ruta_str, None)
                try:
                    self.db.eliminar_pending_rename(ruta_str)
                except Exception:
                    pass
                continue
            # Si todavía está bloqueado y no superamos el máximo, reencolar
            if intentos < self._pending_rwd_max_session_attempts:
                self._pending_rwd_renames.add(ruta_str)
            # Si lo superamos: queda sólo en BD, se reintentará al iniciar la app
        if not self._pending_rwd_renames:
            self._pending_rwd_timer.stop()

    def _retry_persisted_rwd_renames_on_startup(self):
        """Al iniciar la app, intenta una vez los renombrados que quedaron pendientes."""
        try:
            pendientes = self.db.obtener_pending_renames()
        except Exception:
            return
        if not pendientes:
            return
        recuperados = 0
        for ruta_str in pendientes:
            try:
                p = Path(ruta_str)
                if not p.exists():
                    # Si el archivo ya no existe (renombrado externamente o borrado),
                    # limpiar la entrada pendiente.
                    self.db.eliminar_pending_rename(ruta_str)
                    continue
                if self._is_video_revisado(p):
                    self.db.eliminar_pending_rename(ruta_str)
                    continue
                ok = self._rename_rwd(p, queue_on_fail=True, silent=True, defer_runtime_rename=False)
                if ok:
                    recuperados += 1
            except Exception:
                pass
        if recuperados:
            print(f"♻️  Renombrados _rwd recuperados al iniciar: {recuperados}")
        self._refresh_pending_ops_badge()

    def _obtener_pending_photo_fav_startup(self):
        try:
            raw = self.db.obtener_setting("pending_photo_fav_renames_startup", "[]")
            data = json.loads(raw) if raw else []
            rows = []
            if isinstance(data, list):
                for row in data:
                    if not isinstance(row, dict):
                        continue
                    old = str(row.get("old", "")).replace('\\', '/')
                    new = str(row.get("new", "")).replace('\\', '/')
                    if old and new:
                        rows.append({"old": old, "new": new})
            return rows
        except Exception:
            return []

    def _guardar_pending_photo_fav_startup(self, rows):
        try:
            self.db.guardar_setting(
                "pending_photo_fav_renames_startup",
                json.dumps(rows, ensure_ascii=False),
            )
        except Exception:
            pass

    def _queue_photo_fav_rename_startup(self, old_path: Path, new_path: Path):
        old_norm = str(Path(old_path)).replace('\\', '/')
        new_norm = str(Path(new_path)).replace('\\', '/')
        rows = self._obtener_pending_photo_fav_startup()
        for row in rows:
            if row.get("old") == old_norm and row.get("new") == new_norm:
                return
        rows.append({"old": old_norm, "new": new_norm})
        self._guardar_pending_photo_fav_startup(rows)
        self._refresh_pending_ops_badge()

    def _retry_photo_fav_renames_on_startup(self):
        """Process deferred photo favorite renames only once at app startup."""
        rows = self._obtener_pending_photo_fav_startup()
        if not rows:
            LOGGER.info("Startup deferred photo-fav: no pending entries")
            return
        remaining = []
        done = 0
        for row in rows:
            old_p = Path(row.get("old", ""))
            new_p = Path(row.get("new", ""))
            try:
                if not old_p.exists():
                    LOGGER.info("Startup deferred photo-fav skip missing: %s", old_p)
                    continue
                if new_p.exists():
                    LOGGER.info("Startup deferred photo-fav skip target exists: %s", new_p)
                    continue
                self._rename_with_retry(old_p, new_p, attempts=3, delay=0.12)
                self.db.renombrar_ruta(old_p, new_p)
                new_norm = str(new_p).replace('\\', '/')
                self.db.marcar_favorito(new_norm, new_p.name.lower().startswith("top "))
                self.duration_cache.pop(str(old_p), None)
                done += 1
                LOGGER.info("Startup deferred photo-fav applied: %s -> %s", old_p, new_p)
            except Exception:
                LOGGER.exception("Startup deferred photo-fav failed: %s -> %s", old_p, new_p)
                remaining.append({"old": str(old_p).replace('\\', '/'), "new": str(new_p).replace('\\', '/')})
        self._guardar_pending_photo_fav_startup(remaining)
        if done:
            print(f"♻️  Favoritos de fotos aplicados al iniciar: {done}")
        LOGGER.info(
            "Startup deferred photo-fav summary: applied=%d remaining=%d",
            done, len(remaining),
        )
        self._refresh_pending_ops_badge()

    def _run_startup_deferred_renames(self, trigger="startup"):
        """Run all deferred photo/file rename operations scheduled for startup."""
        c0 = self._pending_ops_counts()
        LOGGER.info(
            "Deferred ops begin (%s): rwd=%d borrar=%d top=%d meta=%d",
            trigger,
            c0["rwd"], c0["borrar"], c0["top"], c0["meta"],
        )
        self._retry_persisted_rwd_renames_on_startup()
        self._retry_photo_fav_renames_on_startup()
        self._retry_pending_file_ops(trigger=trigger)
        self._process_pending_metadata_sync_on_startup()
        self._refresh_pending_ops_badge()
        c1 = self._pending_ops_counts()
        LOGGER.info(
            "Deferred ops end (%s): rwd=%d borrar=%d top=%d meta=%d",
            trigger,
            c1["rwd"], c1["borrar"], c1["top"], c1["meta"],
        )

    def _run_pending_ops_now(self):
        """Run startup-deferred operations immediately from UI."""
        c0 = self._pending_ops_counts()
        total0 = c0["rwd"] + c0["borrar"] + c0["top"] + c0["meta"]
        if total0 <= 0:
            self._refresh_pending_ops_badge()
            self._notify("No hay tareas pendientes", 1800)
            return

        self._notify("Ejecutando tareas pendientes...", 1800)
        QApplication.processEvents()
        self._run_startup_deferred_renames(trigger="manual_button")
        c1 = self._pending_ops_counts()
        total1 = c1["rwd"] + c1["borrar"] + c1["top"] + c1["meta"]
        aplicadas = max(0, total0 - total1)
        self._notify(f"Pendientes ejecutadas: {aplicadas} | restantes: {total1}", 3200)

    def _pending_ops_counts(self):
        """Return counts of deferred ops scheduled for next startup."""
        try:
            n_rwd = len(self.db.obtener_pending_renames())
        except Exception:
            n_rwd = 0
        n_del = len(getattr(self, "_pending_delete_paths", set()) or set())
        n_fav = len(getattr(self, "_pending_fav_renames", {}) or {})
        n_meta = len(self._get_pending_metadata_sync_paths())
        return {
            "rwd": int(n_rwd),
            "borrar": int(n_del),
            "top": int(n_fav),
            "meta": int(n_meta),
        }

    def _refresh_pending_ops_badge(self):
        if not hasattr(self, "lbl_pending_ops"):
            return
        c = self._pending_ops_counts()
        total = c["rwd"] + c["borrar"] + c["top"] + c["meta"]
        if hasattr(self, "btn_run_pending_ops"):
            self.btn_run_pending_ops.setEnabled(total > 0)
        if total <= 0:
            self.lbl_pending_ops.setText("Pendientes inicio: 0")
            self.lbl_pending_ops.setToolTip("No hay operaciones diferidas")
            return
        self.lbl_pending_ops.setText(f"Pendientes inicio: {total}")
        self.lbl_pending_ops.setToolTip(
            f"rwd: {c['rwd']}  |  borrar: {c['borrar']}  |  top: {c['top']}  |  metadatos: {c['meta']}"
        )

    def _load_pending_file_ops(self):
        """Load deferred favorite/delete operations from DB settings."""
        try:
            raw_del = self.db.obtener_setting("pending_delete_paths", "[]")
            data_del = json.loads(raw_del) if raw_del else []
            if isinstance(data_del, list):
                self._pending_delete_paths = {
                    str(p).replace('\\', '/') for p in data_del if p
                }
        except Exception:
            self._pending_delete_paths = set()

        try:
            raw_fav = self.db.obtener_setting("pending_fav_renames", "[]")
            data_fav = json.loads(raw_fav) if raw_fav else []
            self._pending_fav_renames = {}
            if isinstance(data_fav, list):
                for row in data_fav:
                    if not isinstance(row, dict):
                        continue
                    old = str(row.get("old", "")).replace('\\', '/')
                    new = str(row.get("new", "")).replace('\\', '/')
                    if old and new:
                        self._pending_fav_renames[old] = new
        except Exception:
            self._pending_fav_renames = {}

        # Startup-only execution: do not retry in-session.
        self._refresh_pending_ops_badge()

    def _save_pending_file_ops(self):
        """Persist deferred favorite/delete operations to DB settings."""
        try:
            self.db.guardar_setting(
                "pending_delete_paths",
                json.dumps(sorted(self._pending_delete_paths), ensure_ascii=False),
            )
            rows = [
                {"old": old, "new": new}
                for old, new in sorted(self._pending_fav_renames.items())
            ]
            self.db.guardar_setting(
                "pending_fav_renames",
                json.dumps(rows, ensure_ascii=False),
            )
        except Exception:
            pass

    def _get_pending_metadata_sync_paths(self):
        try:
            raw = self.db.obtener_setting("pending_metadata_sync_paths", "[]")
            data = json.loads(raw) if raw else []
            if not isinstance(data, list):
                return []
            return [str(p).replace('\\', '/') for p in data if p]
        except Exception:
            return []

    def _save_pending_metadata_sync_paths(self, paths):
        try:
            unique = sorted({str(p).replace('\\', '/') for p in (paths or []) if p})
            self.db.guardar_setting(
                "pending_metadata_sync_paths",
                json.dumps(unique, ensure_ascii=False),
            )
        except Exception:
            pass

    def _queue_metadata_sync_path(self, ruta):
        try:
            ruta_norm = str(Path(ruta)).replace('\\', '/')
        except Exception:
            return
        paths = self._get_pending_metadata_sync_paths()
        if ruta_norm not in paths:
            paths.append(ruta_norm)
            self._save_pending_metadata_sync_paths(paths)
        self._refresh_pending_ops_badge()

    def _drop_metadata_sync_path(self, ruta):
        try:
            ruta_norm = str(Path(ruta)).replace('\\', '/')
        except Exception:
            return
        paths = [p for p in self._get_pending_metadata_sync_paths() if p != ruta_norm]
        self._save_pending_metadata_sync_paths(paths)
        self._refresh_pending_ops_badge()

    def _process_pending_metadata_sync_on_startup(self):
        """Apply delayed metadata writes at startup (after deferred rename/delete ops)."""
        paths = self._get_pending_metadata_sync_paths()
        if not paths:
            LOGGER.info("Startup deferred metadata: no pending entries")
            return
        remaining = []
        done = 0
        skipped_missing = 0
        skipped_non_video = 0
        for ruta_str in paths:
            try:
                p = Path(ruta_str)
                if not p.exists():
                    skipped_missing += 1
                    LOGGER.info("Startup deferred metadata skip missing: %s", ruta_str)
                    continue
                if p.suffix.lower() not in EXTENSIONES_VIDEO:
                    skipped_non_video += 1
                    LOGGER.info("Startup deferred metadata skip non-video: %s", ruta_str)
                    continue
                stats = self.db.obtener_stats_video(ruta_str)
                VideoMetadata.guardar_metadatos(ruta_str, stats)
                done += 1
                LOGGER.info("Startup deferred metadata applied: %s", ruta_str)
            except Exception:
                LOGGER.exception("Startup deferred metadata failed: %s", ruta_str)
                remaining.append(ruta_str)
        self._save_pending_metadata_sync_paths(remaining)
        if done:
            print(f"♻️  Metadatos sincronizados al iniciar: {done}")
        LOGGER.info(
            "Startup deferred metadata summary: applied=%d remaining=%d skipped_missing=%d skipped_non_video=%d",
            done, len(remaining), skipped_missing, skipped_non_video,
        )
        self._refresh_pending_ops_badge()

    def _queue_delete_retry(self, ruta: Path, error_msg: str = ""):
        try:
            ruta_norm = str(Path(ruta)).replace('\\', '/')
        except Exception:
            return
        self._pending_delete_paths.add(ruta_norm)
        self._save_pending_file_ops()
        self._refresh_pending_ops_badge()
        motivo = error_msg or "manual_queue"
        LOGGER.info("Deferred delete queued: %s (%s)", ruta_norm, motivo)

    def _queue_fav_retry(self, old_path: Path, new_path: Path, error_msg: str = ""):
        try:
            old_norm = str(Path(old_path)).replace('\\', '/')
            new_norm = str(Path(new_path)).replace('\\', '/')
        except Exception:
            return
        self._pending_fav_renames[old_norm] = new_norm
        self._save_pending_file_ops()
        self._refresh_pending_ops_badge()
        if error_msg:
            LOGGER.warning("Favorite rename queued for retry: %s -> %s (%s)", old_norm, new_norm, error_msg)

    def _retry_pending_file_ops(self, trigger="startup"):
        if not self._pending_delete_paths and not self._pending_fav_renames:
            self._pending_file_ops_timer.stop()
            LOGGER.info("Deferred file-ops (%s): no pending delete/top entries", trigger)
            return

        changed = False
        active_norm = str(self.video_elegido).replace('\\', '/') if self.video_elegido else ""
        fav_applied = 0
        fav_skipped = 0
        fav_failed = 0
        del_applied = 0
        del_skipped = 0
        del_failed = 0

        # Retry pending favorite renames
        for old_norm, new_norm in list(self._pending_fav_renames.items()):
            old_p = Path(old_norm)
            new_p = Path(new_norm)

            if active_norm and active_norm == old_norm and self._is_playing():
                fav_skipped += 1
                continue
            if not old_p.exists():
                self._pending_fav_renames.pop(old_norm, None)
                changed = True
                fav_skipped += 1
                LOGGER.info("Startup deferred fav skip missing source: %s", old_norm)
                continue
            if new_p.exists():
                self._pending_fav_renames.pop(old_norm, None)
                changed = True
                fav_skipped += 1
                LOGGER.info("Startup deferred fav skip target exists: %s", new_norm)
                continue

            try:
                self._rename_with_retry(old_p, new_p, attempts=3, delay=0.12)
                self.db.renombrar_ruta(old_p, new_p)
                new_norm2 = str(new_p).replace('\\', '/')
                self.db.marcar_favorito(new_norm2, new_p.name.lower().startswith("top "))
                # Carry metadata-sync intent to the new path.
                self._drop_metadata_sync_path(old_p)
                self._queue_metadata_sync_path(new_p)
                self.duration_cache.pop(str(old_p), None)
                if self.video_elegido and Path(self.video_elegido) == old_p:
                    self.video_elegido = new_p
                self._pending_fav_renames.pop(old_norm, None)
                changed = True
                fav_applied += 1
                LOGGER.info("Startup deferred fav applied: %s -> %s", old_norm, new_norm)
            except OSError:
                fav_failed += 1
                LOGGER.warning("Startup deferred fav OS error: %s -> %s", old_norm, new_norm)
            except Exception:
                fav_failed += 1
                LOGGER.exception("Startup deferred fav failed: %s -> %s", old_norm, new_norm)

        # Retry pending deletes
        for ruta_norm in list(self._pending_delete_paths):
            p = Path(ruta_norm)
            if active_norm and active_norm == ruta_norm and self._is_playing():
                del_skipped += 1
                continue
            if not p.exists():
                self._pending_delete_paths.discard(ruta_norm)
                changed = True
                del_skipped += 1
                LOGGER.info("Deferred delete skipped missing (%s): %s", trigger, ruta_norm)
                continue
            try:
                p.unlink()
                if self.video_elegido and Path(self.video_elegido) == p:
                    self.video_elegido = None
                self._drop_metadata_sync_path(p)
                self.duration_cache.pop(str(p), None)
                self._pending_delete_paths.discard(ruta_norm)
                changed = True
                del_applied += 1
                LOGGER.info("Deferred delete applied (%s): %s", trigger, ruta_norm)
            except OSError:
                del_failed += 1
                LOGGER.warning("Deferred delete OS error (%s): %s", trigger, ruta_norm)
            except Exception:
                del_failed += 1
                LOGGER.exception("Deferred delete failed (%s): %s", trigger, ruta_norm)

        if changed:
            self._save_pending_file_ops()
            self._scan()
            self._refresh_list()
            self._refresh_pending_ops_badge()

        if not self._pending_delete_paths and not self._pending_fav_renames:
            self._pending_file_ops_timer.stop()
            self._refresh_pending_ops_badge()

        LOGGER.info(
            "Deferred file-ops summary (%s): fav_applied=%d fav_skipped=%d fav_failed=%d del_applied=%d del_skipped=%d del_failed=%d remaining_fav=%d remaining_del=%d",
            trigger,
            fav_applied, fav_skipped, fav_failed,
            del_applied, del_skipped, del_failed,
            len(self._pending_fav_renames), len(self._pending_delete_paths),
        )

    def _rename_rwd(self, ruta, queue_on_fail=True, silent=False, defer_runtime_rename=True):
        try:
            ruta = Path(ruta)
            if not ruta.exists():
                try:
                    self.db.eliminar_pending_rename(str(ruta))
                except Exception:
                    pass
                return True
            # Runtime behavior: never rename now, only queue for startup.
            if defer_runtime_rename:
                if self._is_video_revisado(ruta):
                    try:
                        self.db.eliminar_pending_rename(str(ruta))
                    except Exception:
                        pass
                    return True
                self._queue_rwd_startup_only(ruta, "deferred_runtime_rename")
                return True
            # Images use "rwd " prefix; videos use "_rwd" suffix
            if ruta.suffix.lower() in EXTENSIONES_IMAGEN:
                if ruta.name.lower().startswith("rwd ") or ruta.name.lower().startswith("top rwd "):
                    try:
                        self.db.eliminar_pending_rename(str(ruta))
                    except Exception:
                        pass
                    return True
                nueva = ruta.parent / f"rwd {ruta.name}"
                c = 1
                while nueva.exists():
                    stem, ext = os.path.splitext(ruta.name)
                    nueva = ruta.parent / f"rwd {stem}_{c}{ext}"
                    c += 1
            else:
                if ruta.stem.endswith('_rwd'):
                    try:
                        self.db.eliminar_pending_rename(str(ruta))
                    except Exception:
                        pass
                    return True
                nueva = ruta.parent / f"{ruta.stem}_rwd{ruta.suffix}"
                c = 1
                while nueva.exists():
                    nueva = ruta.parent / f"{ruta.stem}_rwd_{c}{ruta.suffix}"
                    c += 1
            ruta_antigua = ruta
            self._release_main_video_file_handle()
            self._rename_with_retry(ruta, nueva)
            if self.video_elegido and Path(self.video_elegido) == ruta_antigua:
                self.video_elegido = nueva
            self.db.renombrar_ruta(ruta_antigua, nueva)
            try:
                self.db.eliminar_pending_rename(str(ruta_antigua))
            except Exception:
                pass
            print(f"📝 Renombrado: {ruta.name} → {nueva.name}")
            return True
        except OSError as e:
            if not silent:
                print(f"❌ Error renombrando: {e}")
            if queue_on_fail:
                self._queue_rwd_retry(Path(ruta), str(e))
            return False
        except Exception as e:
            if not silent:
                print(f"❌ Error renombrando: {e}")
            if queue_on_fail:
                self._queue_rwd_retry(Path(ruta), str(e))
            return False

    def _tabla_context_menu(self, pos):
        item = self.tabla.itemAt(pos)
        if not item:
            return
        row = item.row()
        thumb_item = self.tabla.item(row, 0)
        if not thumb_item:
            return
        ruta_str = thumb_item.data(Qt.ItemDataRole.UserRole)
        if not ruta_str:
            return
        menu = QMenu(self.tabla)
        act_play = menu.addAction("Reproducir")
        act_check_video = menu.addAction("Comprobar video")
        act_delete = menu.addAction("Borrar")
        act_rename = menu.addAction("Renombrar")
        menu.addSeparator()
        act_dup_folder = menu.addAction("Buscar duplicados para este archivo (carpeta)")
        act_dup_global = menu.addAction("Buscar duplicados para este archivo (toda la base)")
        accion = menu.exec(self.tabla.viewport().mapToGlobal(pos))
        self.tabla.selectRow(row)
        ruta = Path(ruta_str)
        self.video_elegido = ruta
        self._show_detail(ruta)
        if accion == act_play:
            self.reproducir_video_actual()
        elif accion == act_check_video:
            self._comprobar_video(ruta)
        elif accion == act_delete:
            self.borrar_video()
        elif accion == act_rename:
            self.renombrar_video(Path(ruta_str))
        elif accion == act_dup_folder:
            self._buscar_duplicados_video(ruta, alcance='carpeta')
        elif accion == act_dup_global:
            self._buscar_duplicados_video(ruta, alcance='global')

    def _comprobar_video(self, ruta: Path):
        if not ruta or not ruta.exists() or ruta.suffix.lower() not in EXTENSIONES_VIDEO:
            QMessageBox.information(self, "Comprobar video", "El elemento seleccionado no es un video válido.")
            return
        if not self._is_video_revisado(ruta):
            QMessageBox.information(
                self,
                "Comprobar video",
                "Ese video aún no está revisado, así que no entra en la comprobación automática.",
            )
            return
        self._notify(f"Comprobando video: {ruta.name}", 1800)
        LOGGER.info("Comprobación manual de video: %s", ruta)
        self._launch_meta_check_for_video(ruta, source="manual_video")

    def renombrar_video(self, ruta=None):
        if ruta is None or ruta is False:
            ruta = self.video_elegido
        if not ruta or not Path(ruta).exists():
            QMessageBox.warning(self, "Renombrar", "Selecciona un video primero")
            return
        ruta = Path(ruta)

        if self.player_thread and self.player_thread.isRunning():
            self.player_thread.detener_reproductor()
        self._release_main_video_file_handle()

        dlg = QInputDialog(self)
        dlg.setWindowTitle("Renombrar video")
        dlg.setLabelText("Nuevo nombre (sin extensión):")
        dlg.setTextValue(ruta.stem)
        dlg.resize(1200, 300)
        for child in dlg.findChildren(QWidget):
            child.setMinimumHeight(max(child.minimumHeight(), 32))
        ok = dlg.exec()
        nuevo_stem = dlg.textValue()
        if not ok:
            return
        nuevo_stem = nuevo_stem.strip()
        if not nuevo_stem or nuevo_stem == ruta.stem:
            return
        nueva = ruta.with_name(f"{nuevo_stem}{ruta.suffix}")
        if nueva.exists():
            QMessageBox.critical(self, "Error", f"Ya existe {nueva.name}")
            return
        try:
            ruta_antigua = ruta
            self._rename_with_retry(ruta, nueva)
            self.db.renombrar_ruta(ruta_antigua, nueva)
            self.duration_cache.pop(str(ruta_antigua), None)
            self.video_elegido = nueva
            self._scan()
            self._refresh_list()
            print(f"📝 Renombrado: {ruta.name} → {nueva.name}")
            self._notify(f"Renombrado: {nueva.name}")
        except OSError as e:
            QMessageBox.critical(self, "Error", f"No se pudo renombrar:\n{e}")
        except Exception as e:
            QMessageBox.critical(self, "Error", str(e))

    # ── Favorito con renombrado ──

    def toggle_favorito_y_renombrar(self):
        if (not self.video_elegido or not self.video_elegido.exists()) and hasattr(self, "tabla"):
            row = self.tabla.currentRow()
            if row >= 0:
                it = self.tabla.item(row, 0)
                if it:
                    ruta = it.data(Qt.ItemDataRole.UserRole)
                    if ruta:
                        p = Path(ruta)
                        if p.exists():
                            self.video_elegido = p
        if not self.video_elegido or not self.video_elegido.exists():
            return
        old_path = self.video_elegido
        is_fav = old_path.name.lower().startswith("top ")

        if is_fav:
            new_name = old_path.name[4:]
        else:
            new_name = f"top {old_path.name}"
        new_path = old_path.with_name(new_name)
        if new_path.exists():
            QMessageBox.critical(self, "Error", f"Ya existe {new_name}")
            return
        # Startup-only for both photos and videos.
        self._queue_fav_retry(old_path, new_path, "deferred_favorite_rename")
        self._notify("Favorito guardado para aplicar al iniciar la app", 3500)

    # ── Fijar / Vetar ──

    def fijar_carpeta(self):
        if not self.video_elegido:
            return
        self.carpeta_fijada = self.video_elegido.parent
        self.lista_actual = [v for v in self.lista_actual if self.carpeta_fijada in v.parents]
        QMessageBox.information(self, "Fijado", f"Carpeta: {self.carpeta_fijada.name}")
        self._notify(f"Carpeta fijada: {self.carpeta_fijada.name}")

    def vetar_carpeta(self):
        default = ""
        if self.video_elegido:
            default = self.video_elegido.parent.name
        elif self.carpeta_fijada:
            default = self.carpeta_fijada.name
        nombre, ok = QInputDialog.getText(self, "Vetar", "Nombre de carpeta a vetar:", text=default)
        nombre = nombre.strip()
        if ok and nombre:
            self.carpetas_vetadas.add(nombre.lower())
            self.db.agregar_carpeta_vetada(nombre)
            self._scan()
            self._build_tree()
            if self.video_elegido and nombre.lower() in [p.lower() for p in self.video_elegido.parts]:
                self.video_elegido = None
                self.lbl_detail.setText("Selecciona un video")

    # ── Borrar ──

    def borrar_video(self):
        if not self.video_elegido:
            return
        if QMessageBox.question(self, "Borrar", f"¿Eliminar {self.video_elegido.name}?") == QMessageBox.StandardButton.Yes:
            try:
                ruta_borrada = self.video_elegido
                self._queue_delete_retry(ruta_borrada, "deferred_delete")
                self._notify("Borrado guardado: se aplicará al iniciar o con 'Ejecutar pendientes'", 4000)
            except Exception as e:
                QMessageBox.critical(self, "Error", str(e))

    # ── Hashes ──

    def cargar_hashes(self):
        if os.path.exists(HASHES_DB):
            try:
                with open(HASHES_DB, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception:
                pass
        return {}

    def guardar_hashes(self, hashes_dict):
        try:
            with open(HASHES_DB, 'w', encoding='utf-8') as f:
                json.dump(hashes_dict, f, indent=2)
            for key, hash_data in hashes_dict.items():
                if '|' in key:
                    ruta = key.rsplit('|', 1)[0]
                    try:
                        tamaño = int(key.rsplit('|', 1)[1])
                    except:
                        tamaño = 0
                else:
                    ruta = key
                    tamaño = 0
                self.db.guardar_hash_visual(ruta, tamaño, hash_data)
        except Exception as e:
            print(f"Error guardando hashes: {e}")

    def mostrar_porcentaje_hashes(self):
        if not self.ruta_raiz:
            return
        videos = [f for f in self.ruta_raiz.rglob("*") if f.suffix.lower() in EXTENSIONES_VIDEO and f.is_file()]
        imagenes = [f for f in self.ruta_raiz.rglob("*") if f.suffix.lower() in EXTENSIONES_IMAGEN and f.is_file()]
        total = len(videos) + len(imagenes)
        if not total:
            self.lbl_hash.setText("")
            return
        con_hash = sum(1 for v in videos if self.db.tiene_hash(str(v)))
        con_hash += sum(1 for img in imagenes if self.db.tiene_hash(str(img)))
        pct = (con_hash / total) * 100
        self.lbl_hash.setText(f"Hasheados: {con_hash}/{total} ({pct:.0f}%)")

    # ── Duplicados ──

    def _set_progress(self, idx, total, texto):
        self.progress_bar.setVisible(True)
        self.progress_bar.setMaximum(total)
        self.progress_bar.setValue(idx + 1)
        self.lbl_progreso.setVisible(True)
        self.lbl_progreso.setText(texto)
        QApplication.processEvents()

    def _hide_progress(self):
        self.progress_bar.setVisible(False)
        self.lbl_progreso.setVisible(False)
        self.lbl_progreso.setText("")

    def _formatear_size(self, size_bytes):
        if size_bytes < 1024**3:
            return f"{size_bytes / (1024**2):.1f} MB"
        return f"{size_bytes / (1024**3):.2f} GB"

    def _gestionar_pares_duplicados(self, pares, titulo="Duplicado"):
        """Interactive delete/open/skip flow for duplicate pairs."""
        borrados = set()
        auto_borrar_sugeridos = False
        for a, b in pares:
            a = Path(a)
            b = Path(b)
            if a in borrados or b in borrados:
                continue
            if (not a.exists()) or (not b.exists()):
                continue
            try:
                size_a = os.path.getsize(a)
                size_b = os.path.getsize(b)
            except OSError:
                continue

            sug = 1 if size_a < size_b else 2
            if auto_borrar_sugeridos:
                objetivo = a if sug == 1 else b
                try:
                    objetivo.unlink()
                    borrados.add(objetivo)
                except Exception as e:
                    QMessageBox.critical(self, "Error", str(e))
                continue

            msg = QMessageBox(self)
            msg.setWindowTitle(titulo)
            msg.setText(
                f"[1] {a.name}  ({self._formatear_size(size_a)})\n"
                f"    {a}\n\n"
                f"[2] {b.name}  ({self._formatear_size(size_b)})\n"
                f"    {b}\n\n"
                f"Sugerencia: borrar [{sug}]"
            )
            btn1 = msg.addButton("Borrar 1", QMessageBox.ButtonRole.AcceptRole)
            btn2 = msg.addButton("Borrar 2", QMessageBox.ButtonRole.AcceptRole)
            btn_sug = msg.addButton("Borrar sugerido", QMessageBox.ButtonRole.AcceptRole)
            btn_sug_all = msg.addButton("Borrar sugeridos (resto)", QMessageBox.ButtonRole.AcceptRole)
            btn_o1 = msg.addButton("Abrir 1", QMessageBox.ButtonRole.ActionRole)
            btn_o2 = msg.addButton("Abrir 2", QMessageBox.ButtonRole.ActionRole)
            msg.addButton("Saltar", QMessageBox.ButtonRole.RejectRole)

            while True:
                msg.exec()
                clicked = msg.clickedButton()
                if clicked == btn1:
                    try:
                        a.unlink()
                        borrados.add(a)
                    except Exception as e:
                        QMessageBox.critical(self, "Error", str(e))
                    break
                elif clicked == btn2:
                    try:
                        b.unlink()
                        borrados.add(b)
                    except Exception as e:
                        QMessageBox.critical(self, "Error", str(e))
                    break
                elif clicked == btn_sug:
                    objetivo = a if sug == 1 else b
                    try:
                        objetivo.unlink()
                        borrados.add(objetivo)
                    except Exception as e:
                        QMessageBox.critical(self, "Error", str(e))
                    break
                elif clicked == btn_sug_all:
                    objetivo = a if sug == 1 else b
                    try:
                        objetivo.unlink()
                        borrados.add(objetivo)
                    except Exception as e:
                        QMessageBox.critical(self, "Error", str(e))
                    auto_borrar_sugeridos = True
                    break
                elif clicked == btn_o1:
                    try:
                        os.startfile(str(a))
                    except Exception:
                        pass
                elif clicked == btn_o2:
                    try:
                        os.startfile(str(b))
                    except Exception:
                        pass
                else:
                    break

        return len(borrados)

    def borrar_duplicados_carpeta(self):
        if self.carpeta_fijada:
            carpeta = self.carpeta_fijada
        elif self.video_elegido:
            carpeta = self.video_elegido.parent
        elif self.ruta_raiz:
            carpeta = self.ruta_raiz
        else:
            QMessageBox.warning(self, "Sin carpeta", "No hay carpeta seleccionada.")
            return

        # En modo "Solo fotos", gestiona duplicados usando hashes visuales de imágenes.
        if self.incluir_fotos_gestion:
            try:
                fotos = [f for f in carpeta.rglob("*") if f.suffix.lower() in EXTENSIONES_IMAGEN and f.is_file()]
            except (PermissionError, OSError) as e:
                QMessageBox.critical(self, "Duplicados", f"No se pudo leer la carpeta:\n{e}")
                return

            if len(fotos) < 2:
                QMessageBox.information(self, "Sin duplicados", "No hay suficientes fotos.")
                return

            rutas_fotos = [str(f).replace('\\', '/') for f in fotos]
            pendientes_hash = [r for r in rutas_fotos if not self.db.tiene_hash(r)]
            total_pendientes = len(pendientes_hash)
            if total_pendientes:
                for i, ruta in enumerate(pendientes_hash):
                    self._set_progress(i, total_pendientes, f"Hasheando foto: {Path(ruta).name}")
                    _calcular_hash_archivo(ruta)
                self._hide_progress()

            self._rebuild_duplicate_index()

            pares = []
            for ruta in rutas_fotos:
                for dup in self.duplicate_map.get(ruta, []):
                    if Path(dup).suffix.lower() in EXTENSIONES_IMAGEN:
                        pares.append((Path(ruta), Path(dup)))
            # Evita pares repetidos A-B / B-A.
            pares_unicos = []
            vistos = set()
            for a, b in pares:
                k = tuple(sorted((str(a), str(b))))
                if k in vistos:
                    continue
                vistos.add(k)
                pares_unicos.append((a, b))

            if not pares_unicos:
                QMessageBox.information(self, "Sin duplicados", "No se encontraron duplicados entre fotos.")
                return

            borrados_n = self._gestionar_pares_duplicados(pares_unicos, titulo="Duplicado de foto en carpeta")
            self._scan()
            self._refresh_list()
            QMessageBox.information(self, "Listo", f"Duplicados eliminados: {borrados_n}")
            return

        modo, ok = QInputDialog.getItem(self, "Tipo de búsqueda", "Elige:",
            ["1 - Solo duración (rápido)", "2 - Tamaño+Duración (seguro)", "3 - Visual (lento, preciso)"], 1, False)
        if not ok:
            return
        modo = modo[0]

        videos = [f for f in carpeta.rglob("*") if f.suffix.lower() in EXTENSIONES_VIDEO and f.is_file()]
        if len(videos) < 2:
            QMessageBox.information(self, "Sin duplicados", "No hay suficientes videos.")
            return

        info_videos = {}
        for v in videos:
            try:
                cmd = [FFPROBE_PATH, '-v', 'error', '-show_entries', 'format=duration',
                       '-of', 'default=noprint_wrappers=1:nokey=1', str(v)]
                dur = float(subprocess.check_output(cmd).decode().strip())
                info_videos[v] = {'size': v.stat().st_size, 'dur': dur}
            except:
                continue

        duplicados = []

        if modo == '1':
            vids = sorted(info_videos.items(), key=lambda x: x[1]['dur'])
            n = len(vids)
            for i in range(n):
                vi, infoi = vids[i]
                self._set_progress(i, n, f"Comparando: {vi.name}")
                for j in range(i + 1, n):
                    vj, infoj = vids[j]
                    if abs(infoi['dur'] - infoj['dur']) < 0.1:
                        duplicados.append((vi, vj))
                    elif (infoj['dur'] - infoi['dur']) > 0.1:
                        break
            self._hide_progress()

        elif modo == '2':
            por_tam = defaultdict(list)
            for v, info in info_videos.items():
                por_tam[info['size']].append((v, info['dur']))
            grupos = [g for g in por_tam.values() if len(g) > 1]
            for idx, grupo in enumerate(grupos):
                self._set_progress(idx, len(grupos), f"Grupo {idx+1}/{len(grupos)}")
                for i in range(len(grupo)):
                    for j in range(i + 1, len(grupo)):
                        if abs(grupo[i][1] - grupo[j][1]) < 0.5:
                            duplicados.append((grupo[i][0], grupo[j][0]))
            self._hide_progress()

        elif modo == '3':
            try:
                import cv2
                import imagehash
                from PIL import Image
                import numpy as np
                from concurrent.futures import ThreadPoolExecutor, as_completed
            except ImportError:
                QMessageBox.critical(self, "Faltan dependencias", "Instala opencv-python, imagehash, Pillow y numpy.")
                return

            hashes_db = self.cargar_hashes()

            class VF:
                def __init__(self, ruta):
                    self.ruta = str(ruta)
                    self.nombre = os.path.basename(self.ruta)
                    try:
                        self.size_bytes = os.path.getsize(self.ruta)
                    except OSError:
                        self.size_bytes = 0
                    self.duracion = 0
                    self.resolucion = (0, 0)
                    self.es_valido = False
                    self.hashes = None

                def analizar(self):
                    try:
                        cap = cv2.VideoCapture(self.ruta, cv2.CAP_ANY)
                        if cap.isOpened():
                            fps = cap.get(cv2.CAP_PROP_FPS)
                            fc = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                            if fps > 0 and fc > 0:
                                self.duracion = fc / fps
                                self.resolucion = (w, h)
                                self.es_valido = True
                        cap.release()
                    except:
                        pass

                def gen_hashes(self, progress_cb=None, idx=None, total=None):
                    key = f"{self.ruta}|{self.size_bytes}"
                    if self.hashes is not None:
                        return
                    if key in hashes_db:
                        try:
                            self.hashes = np.array(hashes_db[key], dtype=bool)
                            if progress_cb and idx is not None:
                                progress_cb(idx, total, self.nombre)
                            return
                        except:
                            pass
                    lista = []
                    try:
                        cap = cv2.VideoCapture(self.ruta, cv2.CAP_ANY)
                        tf = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                        if tf > 15:
                            for p in np.linspace(tf * 0.05, tf * 0.95, 15).astype(int):
                                cap.set(cv2.CAP_PROP_POS_FRAMES, p)
                                ret, frame = cap.read()
                                if ret and frame is not None:
                                    try:
                                        frm = cv2.resize(frame, (64, 64))
                                        img = Image.fromarray(cv2.cvtColor(frm, cv2.COLOR_BGR2RGB))
                                        lista.append(imagehash.phash(img).hash.flatten())
                                    except:
                                        pass
                        cap.release()
                    except:
                        pass
                    if len(lista) >= 5:
                        self.hashes = np.array(lista)
                        hashes_db[key] = self.hashes.tolist()
                    else:
                        self.hashes = np.array([])
                    if progress_cb and idx is not None:
                        progress_cb(idx, total, self.nombre)

            def cmp_visual(a, b):
                if a.hashes is None or len(a.hashes) == 0 or b.hashes is None or len(b.hashes) == 0:
                    return False
                import numpy as np
                lim = min(len(a.hashes), len(b.hashes))
                diff = np.bitwise_xor(a.hashes[:lim], b.hashes[:lim])
                dist = np.count_nonzero(diff, axis=1)
                return (np.count_nonzero(dist <= 14) / lim) >= 0.55

            objs = []
            with ThreadPoolExecutor(max_workers=8) as ex:
                futs = {ex.submit(lambda r: (lambda v: (v.analizar(), v)[-1])(VF(r)), str(v)): v for v in info_videos}
                for f in as_completed(futs):
                    vo = f.result()
                    if vo.es_valido and vo.duracion > 0:
                        objs.append(vo)

            objs.sort(key=lambda x: x.duracion)
            n = len(objs)
            for idx, v in enumerate(objs):
                self._set_progress(idx, n, f"Hashes: {v.nombre}")
                v.gen_hashes(progress_cb=lambda i, t, nm: self._set_progress(i, t, f"Hashes: {nm}"), idx=idx, total=n)
            self.guardar_hashes(hashes_db)

            for i in range(n):
                self._set_progress(i, n, f"Comparando: {objs[i].nombre}")
                for j in range(i + 1, n):
                    if (objs[j].duracion - objs[i].duracion) > (objs[i].duracion * 0.05 + 5):
                        break
                    if cmp_visual(objs[i], objs[j]):
                        duplicados.append((Path(objs[i].ruta), Path(objs[j].ruta)))
            self._hide_progress()

        if not duplicados:
            QMessageBox.information(self, "Sin duplicados", "No se encontraron duplicados.")
            return

        borrados_n = self._gestionar_pares_duplicados(duplicados, titulo="Duplicado en carpeta")
        self._scan()
        self._refresh_list()
        QMessageBox.information(self, "Listo", f"Duplicados eliminados: {borrados_n}")


def main():
    LOGGER.info("Arranque abrearch_premium con args=%s", sys.argv[1:])
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    w = VideoBrowserApp()
    w.showMaximized()
    # Ejecuta renombrados diferidos (fotos _rwd y favoritos) al iniciar.
    QTimer.singleShot(2000, w._run_startup_deferred_renames)
    # Carpeta inicial por argumento: python abrearch_premium.py "D:\videos"
    args = [a for a in sys.argv[1:] if not a.startswith('-')]
    if args:
        ruta_arg = Path(args[0]).expanduser()
        if ruta_arg.is_dir():
            def _load_initial_root():
                w.statusBar().showMessage("Cargando estructura inicial...", 0)
                w.ruta_raiz = ruta_arg
                w.lbl_ruta.setText(str(ruta_arg))
                w._scan()
                w._build_tree()
                # Carpeta cargada, pero mostramos el dashboard de inicio.
                w._go_to_dashboard()
                w.mostrar_porcentaje_hashes()
            # Let the window render first; heavy disk scan starts right after.
            QTimer.singleShot(100, _load_initial_root)
        else:
            LOGGER.warning("Ruta no válida al arrancar: %s", ruta_arg)
            print(f"⚠ Ruta no válida: {ruta_arg}")
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
