"""
Módulo para gestionar metadatos internos de videos usando ffmpeg/ffprobe
Guarda estadísticas en el tag 'comment' del contenedor de video
"""

import subprocess
import json
import os
import shutil
from pathlib import Path

FFPROBE_PATH = r"D:\projects\FINANZAS\tools\ffprobe.exe"
FFMPEG_PATH = r"D:\projects\FINANZAS\tools\ffmpeg.exe"
FFMPEG_METADATA_TIMEOUT = 300


def _safe_unlink(path_obj):
    """Best-effort temp file removal that never raises."""
    try:
        if path_obj and path_obj.exists():
            path_obj.unlink()
    except (FileNotFoundError, PermissionError, OSError):
        pass

class VideoMetadata:

    @staticmethod
    def _resolve_ffprobe_path(ffprobe_path=None):
        """Return a usable ffprobe path from preferred path, configured default, or PATH."""
        if ffprobe_path and os.path.exists(ffprobe_path):
            return ffprobe_path
        if os.path.exists(FFPROBE_PATH):
            return FFPROBE_PATH
        return shutil.which("ffprobe") or shutil.which("ffprobe.exe")

    @staticmethod
    def guardar_metadatos(ruta_video, datos):
        temp_output = None
        try:
            ruta_video = Path(ruta_video)
            if not ruta_video.exists() or not os.path.exists(FFMPEG_PATH):
                return False

            metadata_dict = {
                'reproducciones': datos.get('reproducciones', 0),
                'tiempo_visto_seg': datos.get('tiempo_visto_seg', 0),
                'ultima_reproduccion': datos.get('ultima_reproduccion', ''),
                'es_favorito': datos.get('es_favorito', False),
                'fue_visto': datos.get('fue_visto', False)
            }

            metadata_json = json.dumps(metadata_dict, separators=(',', ':'), ensure_ascii=False)

            temp_output = ruta_video.parent / f"{ruta_video.stem}_tmpmeta{ruta_video.suffix}"

            cmd = [
                FFMPEG_PATH,
                '-i', str(ruta_video),
                '-c', 'copy',
                '-map', '0',
                '-metadata', f'comment={metadata_json}',
                '-y',
                str(temp_output)
            ]

            result = subprocess.run(cmd, capture_output=True, timeout=FFMPEG_METADATA_TIMEOUT)

            if result.returncode == 0 and temp_output.exists():
                os.replace(str(temp_output), str(ruta_video))
                print(f"✅ Metadatos internos guardados en: {ruta_video.name}")
                return True
            else:
                _safe_unlink(temp_output)
                print(f"⚠️ ffmpeg falló: {result.stderr.decode(errors='ignore')[-200:]}")
                return False

        except subprocess.TimeoutExpired:
            _safe_unlink(temp_output)
            print(f"⚠️ Timeout guardando metadatos en {ruta_video.name}")
            return False
        except Exception as e:
            _safe_unlink(temp_output)
            print(f"⚠️ Error en guardar_metadatos: {e}")
            return False

    @staticmethod
    def obtener_metadatos(ruta_video, ffprobe_path=None):
        try:
            ruta_video = Path(ruta_video)
            probe_path = VideoMetadata._resolve_ffprobe_path(ffprobe_path)
            if not ruta_video.exists() or not probe_path:
                return None

            cmd = [
                probe_path,
                '-v', 'quiet',
                '-print_format', 'json',
                '-show_format',
                str(ruta_video)
            ]

            result = subprocess.run(cmd, capture_output=True, timeout=15)
            if result.returncode != 0:
                return None

            info = json.loads(result.stdout.decode('utf-8', errors='ignore'))
            tags = info.get('format', {}).get('tags', {})

            comment = tags.get('comment') or tags.get('COMMENT') or tags.get('Comment')
            if not comment:
                return None

            return json.loads(comment)

        except (json.JSONDecodeError, KeyError):
            return None
        except subprocess.TimeoutExpired:
            return None
        except Exception as e:
            print(f"⚠️ Error en obtener_metadatos: {e}")
            return None

    @staticmethod
    def limpiar_metadatos(ruta_video):
        try:
            ruta_video = Path(ruta_video)
            if not ruta_video.exists() or not os.path.exists(FFMPEG_PATH):
                return False

            temp_output = ruta_video.parent / f"{ruta_video.stem}_tmpmeta{ruta_video.suffix}"

            cmd = [
                FFMPEG_PATH,
                '-i', str(ruta_video),
                '-c', 'copy',
                '-map', '0',
                '-metadata', 'comment=',
                '-y',
                str(temp_output)
            ]

            result = subprocess.run(cmd, capture_output=True, timeout=FFMPEG_METADATA_TIMEOUT)
            if result.returncode == 0 and temp_output.exists():
                os.replace(str(temp_output), str(ruta_video))
                return True
            _safe_unlink(temp_output)
            return False
        except:
            return False

    @staticmethod
    def obtener_miniatura(ruta_video, tamaño_salida=(160, 90)):
        try:
            ruta_video = Path(ruta_video)
            if not ruta_video.exists() or not FFMPEG_PATH or not os.path.exists(FFMPEG_PATH):
                return None

            thumb_file = Path(ruta_video.parent) / f".thumb_{ruta_video.stem}.jpg"

            if thumb_file.exists():
                import time
                if time.time() - os.path.getmtime(thumb_file) < 7 * 24 * 60 * 60:
                    return str(thumb_file)

            cmd = [
                FFMPEG_PATH,
                '-i', str(ruta_video),
                '-ss', '00:00:05',
                '-vf', f'scale={tamaño_salida[0]}:{tamaño_salida[1]}',
                '-vframes', '1',
                '-y',
                str(thumb_file)
            ]

            result = subprocess.run(cmd, capture_output=True, timeout=10)

            if result.returncode == 0 and thumb_file.exists():
                return str(thumb_file)

            return None

        except subprocess.TimeoutExpired:
            return None
        except Exception as e:
            print(f"⚠️ Error extrayendo miniatura: {e}")
            return None

    @staticmethod
    def sincronizar_metadatos_a_bd(ruta_video, db):
        try:
            metadata = VideoMetadata.obtener_metadatos(ruta_video)
            if metadata:
                for _ in range(metadata.get('reproducciones', 0)):
                    db.registrar_visualizacion(ruta_video, metadata.get('tiempo_visto_seg', 0) // max(1, metadata.get('reproducciones', 1)))

                if metadata.get('es_favorito'):
                    db.marcar_favorito(ruta_video, True)

                print(f"✅ Metadatos sincronizados de {Path(ruta_video).name} a BD")
                return True

            return False
        except Exception as e:
            print(f"⚠️ Error sincronizando metadatos: {e}")
            return False
