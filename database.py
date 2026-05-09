import sqlite3
import json
import os
from pathlib import Path
from datetime import datetime

class VideoDatabase:
    """Gestor de base de datos SQLite para estadísticas de videos."""
    
    DB_FILE = "videos.db"
    
    def __init__(self):
        self.conn = None
        self._db_ok = True
        self._salvaged_stats = []
        self._salvaged_hashes = []
        self.inicializar_db()
    
    def _try_recover_db(self):
        """Recover the corrupt DB. Strategy:
        1. Direct table SELECT (works when only free-list/schema pages are corrupt)
        2. sqlite3 CLI .recover (page-level, needs sqlite3.exe in PATH)
        3. Last resort: rename corrupt + fresh empty DB
        """
        import shutil
        import subprocess
        import tempfile

        db_path = Path(self.DB_FILE).resolve()
        recovered_path = db_path.with_suffix('.db.recovering')

        # Close existing connection first
        if self.conn:
            try:
                self.conn.close()
            except Exception:
                pass
            self.conn = None

        recovered = False

        # ── Strategy 1: direct table read ──────────────────────────────────
        # Even when sqlite_master is damaged, the data pages may still be fine.
        salvaged_stats = []    # list of (ruta, repros, tiempo_seg)
        salvaged_hashes = []   # list of (ruta, tamaño, hash_blob)
        try:
            corrupt_conn = sqlite3.connect(str(db_path))
            corrupt_conn.row_factory = sqlite3.Row
            try:
                rows = corrupt_conn.execute(
                    "SELECT ruta, reproducciones, tiempo_visto_seg, es_favorito, fue_visto "
                    "FROM video_stats"
                ).fetchall()
                salvaged_stats = [
                    (r['ruta'], r['reproducciones'], r['tiempo_visto_seg'],
                     r['es_favorito'], r['fue_visto'])
                    for r in rows
                ]
                print(f"✅ Lectura directa: {len(salvaged_stats)} registros de video_stats recuperados.")
            except Exception as e:
                print(f"ℹ️  video_stats no legible directamente ({e}).")
            try:
                rows = corrupt_conn.execute(
                    "SELECT ruta, tamaño_bytes, hash_visual FROM video_hashes"
                ).fetchall()
                salvaged_hashes = [
                    (r['ruta'], r['tamaño_bytes'], r['hash_visual'])
                    for r in rows
                ]
                print(f"✅ Lectura directa: {len(salvaged_hashes)} registros de video_hashes recuperados.")
            except Exception as e:
                print(f"ℹ️  video_hashes no legible directamente ({e}).")
            corrupt_conn.close()
        except Exception as e:
            print(f"⚠️ No se pudo abrir BD corrupta para lectura directa ({e}).")

        if salvaged_stats or salvaged_hashes:
            recovered = True

        # ── Strategy 2: sqlite3 CLI .recover ──────────────────────────────
        if not recovered:
            sqlite3_exe = shutil.which("sqlite3") or shutil.which("sqlite3.exe")
            if sqlite3_exe:
                try:
                    recovered_path.unlink(missing_ok=True)
                    with tempfile.NamedTemporaryFile(mode='w', suffix='.sql',
                                                     delete=False, encoding='utf-8') as tf:
                        tf.write(".recover\n")
                        script_path = tf.name
                    with open(script_path, 'r', encoding='utf-8') as script_f:
                        result = subprocess.run(
                            [sqlite3_exe, str(db_path)],
                            stdin=script_f,
                            capture_output=True,
                            timeout=60,
                        )
                    try:
                        Path(script_path).unlink(missing_ok=True)
                    except Exception:
                        pass
                    sql_dump = result.stdout.decode('utf-8', errors='replace')
                    if sql_dump.strip():
                        new_conn = sqlite3.connect(str(recovered_path))
                        errors = 0
                        for stmt in sql_dump.split(';\n'):
                            stmt = stmt.strip()
                            if not stmt:
                                continue
                            try:
                                new_conn.execute(stmt)
                            except Exception:
                                errors += 1
                        new_conn.commit()
                        new_conn.close()
                        db_path.unlink(missing_ok=True)
                        recovered_path.rename(db_path)
                        print(f"✅ BD recuperada con sqlite3 .recover ({errors} sentencias con error).")
                        self.conn = sqlite3.connect(str(db_path))
                        self.conn.row_factory = sqlite3.Row
                        self._db_ok = True
                        return
                    else:
                        print(f"⚠️ sqlite3 .recover no produjo datos.")
                except Exception as e:
                    print(f"⚠️ sqlite3 CLI .recover falló ({e}).")
                    recovered_path.unlink(missing_ok=True)
            else:
                print("ℹ️  sqlite3 CLI no encontrado en PATH.")

        # ── Rebuild from salvaged data ─────────────────────────────────────
        backup = db_path.with_suffix('.db.corrupt')
        try:
            shutil.move(str(db_path), str(backup))
            print(f"ℹ️  BD corrupta guardada como {backup.name}.")
        except Exception as e:
            print(f"⚠️ No se pudo guardar copia de BD corrupta: {e}")

        # Create a fresh DB and re-insert salvaged rows
        self.conn = sqlite3.connect(str(db_path))
        self.conn.row_factory = sqlite3.Row
        self._db_ok = True
        # inicializar_db will create the tables; insert salvaged data after
        self._salvaged_stats = salvaged_stats
        self._salvaged_hashes = salvaged_hashes

    def inicializar_db(self):
        """Crea las tablas si no existen y migra datos de JSON si es necesario."""
        try:
            self.conn = sqlite3.connect(self.DB_FILE)
            self.conn.row_factory = sqlite3.Row
            # Quick integrity check before proceeding
            ok = self.conn.execute("PRAGMA integrity_check").fetchone()[0]
            if ok != "ok":
                raise sqlite3.DatabaseError(f"integrity_check: {ok}")
        except sqlite3.DatabaseError as e:
            print(f"⚠️ BD corrupta ({e}), intentando recuperar…")
            self._try_recover_db()
        # Performance pragmas: WAL allows concurrent reads while writing,
        # and a larger cache + memory temp store reduces disk I/O on HDDs.
        try:
            self.conn.execute("PRAGMA journal_mode=WAL")
            self.conn.execute("PRAGMA synchronous=NORMAL")
            self.conn.execute("PRAGMA temp_store=MEMORY")
            self.conn.execute("PRAGMA cache_size=-65536")  # ~64 MB page cache
            self.conn.execute("PRAGMA mmap_size=268435456")  # 256 MB memory map
        except Exception:
            pass
        cursor = self.conn.cursor()
        
        # Tabla de estadísticas de videos
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS video_stats (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ruta TEXT UNIQUE NOT NULL,
                nombre_archivo TEXT,
                reproducciones INTEGER DEFAULT 0,
                tiempo_visto_seg INTEGER DEFAULT 0,
                ultima_reproduccion TEXT,
                es_favorito BOOLEAN DEFAULT 0,
                fue_visto BOOLEAN DEFAULT 0,
                miniatura BLOB,
                fecha_created TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Migrar columna miniatura si no existe (para BD existentes)
        try:
            cursor.execute("ALTER TABLE video_stats ADD COLUMN miniatura BLOB")
            self.conn.commit()
        except sqlite3.OperationalError:
            pass  # Ya existe
        
        # Tabla de hashes visuales para detección de duplicados
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS video_hashes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ruta TEXT UNIQUE NOT NULL,
                tamaño_bytes INTEGER,
                hash_visual BLOB,
                fecha_calculado TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Tabla de carpetas vetadas
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS carpetas_vetadas (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ruta_carpeta TEXT UNIQUE NOT NULL,
                fecha_agregada TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Tabla de miniaturas de carpeta (múltiples por carpeta)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS folder_thumbnails (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                carpeta TEXT NOT NULL,
                video_ruta TEXT NOT NULL,
                frame_no INTEGER NOT NULL DEFAULT 0,
                thumbnail_blob BLOB NOT NULL,
                fecha TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Tabla de miniaturas sugeridas por búsqueda automática de caras.
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS folder_thumbnail_suggestions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                carpeta TEXT NOT NULL,
                video_ruta TEXT NOT NULL,
                frame_no INTEGER NOT NULL DEFAULT 0,
                thumbnail_blob BLOB NOT NULL,
                score REAL DEFAULT 0,
                fecha TEXT DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(carpeta, video_ruta, frame_no)
            )
        """)

        # Ajustes generales persistentes de la app
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS app_settings (
                clave TEXT PRIMARY KEY,
                valor TEXT,
                fecha_updated TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Renombrados pendientes (videos que no se pudieron marcar _rwd porque
        # estaban en uso por otro proceso). Se reintenta al iniciar la app.
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS pending_renames (
                ruta_origen TEXT PRIMARY KEY,
                fecha_added TEXT DEFAULT CURRENT_TIMESTAMP,
                intentos INTEGER DEFAULT 0,
                ultimo_error TEXT
            )
        """)

        # Índices para acelerar consultas frecuentes (sobre todo en HDD).
        # Las columnas UNIQUE ya tienen índice implícito; folder_thumbnails.carpeta no.
        try:
            cursor.execute(
                "CREATE INDEX IF NOT EXISTS idx_folder_thumbs_carpeta "
                "ON folder_thumbnails(carpeta)"
            )
            cursor.execute(
                "CREATE INDEX IF NOT EXISTS idx_folder_thumb_suggest_carpeta "
                "ON folder_thumbnail_suggestions(carpeta)"
            )
            cursor.execute(
                "CREATE INDEX IF NOT EXISTS idx_video_hashes_hashnotnull "
                "ON video_hashes(ruta) WHERE hash_visual IS NOT NULL"
            )
        except sqlite3.OperationalError:
            pass

        self.conn.commit()
        
        # Re-insert salvaged rows from a previous corruption recovery
        salvaged_stats = getattr(self, '_salvaged_stats', [])
        salvaged_hashes = getattr(self, '_salvaged_hashes', [])
        if salvaged_stats or salvaged_hashes:
            cur2 = self.conn.cursor()
            for ruta, repros, tiempo, es_fav, fue_visto in salvaged_stats:
                try:
                    nombre = Path(ruta).name if ruta else ''
                    cur2.execute("""
                        INSERT OR IGNORE INTO video_stats
                        (ruta, nombre_archivo, reproducciones, tiempo_visto_seg, es_favorito, fue_visto)
                        VALUES (?, ?, ?, ?, ?, ?)
                    """, (ruta, nombre, repros, tiempo, es_fav, fue_visto))
                except Exception:
                    pass
            for ruta, tamaño, hash_blob in salvaged_hashes:
                try:
                    cur2.execute("""
                        INSERT OR IGNORE INTO video_hashes (ruta, tamaño_bytes, hash_visual)
                        VALUES (?, ?, ?)
                    """, (ruta, tamaño, hash_blob))
                except Exception:
                    pass
            self.conn.commit()
            print(f"✅ Datos recuperados reinsertados: {len(salvaged_stats)} vistas, {len(salvaged_hashes)} hashes.")
            self._salvaged_stats = []
            self._salvaged_hashes = []

        # Migrar datos de JSON si existen
        self._migrar_datos_json()
    
    def _migrar_datos_json(self):
        """Migra datos de archivos JSON a la base de datos."""
        json_stats = "video_stats.json"
        json_hashes = "video_hashes.json"
        
        # Migrar video_stats.json
        if os.path.exists(json_stats):
            try:
                with open(json_stats, 'r', encoding='utf-8') as f:
                    datos_json = json.load(f)
                
                cursor = self.conn.cursor()
                for ruta, datos in datos_json.items():
                    nombre = Path(ruta).name if ruta else "desconocido"
                    cursor.execute("""
                        INSERT OR IGNORE INTO video_stats 
                        (ruta, nombre_archivo, reproducciones, tiempo_visto_seg, ultima_reproduccion, es_favorito, fue_visto)
                        VALUES (?, ?, ?, ?, ?, ?, ?)
                    """, (
                        ruta,
                        nombre,
                        datos.get('reproducciones', 0),
                        datos.get('tiempo_visto_seg', 0),
                        datos.get('ultima_vez', ''),
                        1 if datos.get('favorito', False) else 0,
                        1  # Si estaba en JSON, fue visto
                    ))
                self.conn.commit()
                print(f"✅ Migrados datos de {json_stats} a base de datos")
                # Renombrar archivo JSON para no volver a migrar
                os.replace(json_stats, json_stats + ".bak")
            except Exception as e:
                print(f"⚠️ Error migrando {json_stats}: {e}")
        
        # Migrar video_hashes.json
        if os.path.exists(json_hashes):
            try:
                with open(json_hashes, 'r', encoding='utf-8') as f:
                    hashes_json = json.load(f)
                
                cursor = self.conn.cursor()
                for ruta_size_key, hash_data in hashes_json.items():
                    # Formato: "ruta|tamaño"
                    if '|' in ruta_size_key:
                        ruta = ruta_size_key.rsplit('|', 1)[0]
                        try:
                            tamaño = int(ruta_size_key.rsplit('|', 1)[1])
                        except:
                            tamaño = 0
                    else:
                        ruta = ruta_size_key
                        tamaño = 0
                    
                    # Guardar hash como JSON string en BLOB
                    import json as json_lib
                    hash_blob = json_lib.dumps(hash_data).encode('utf-8')
                    
                    cursor.execute("""
                        INSERT OR IGNORE INTO video_hashes 
                        (ruta, tamaño_bytes, hash_visual)
                        VALUES (?, ?, ?)
                    """, (ruta, tamaño, hash_blob))
                self.conn.commit()
                print(f"✅ Migrados datos de {json_hashes} a base de datos")
                # Renombrar archivo JSON para no volver a migrar
                os.replace(json_hashes, json_hashes + ".bak")
            except Exception as e:
                print(f"⚠️ Error migrando {json_hashes}: {e}")

    def restaurar_desde_metadatos(self, ruta_raiz, progress_callback=None, ffprobe_path=None):
        """Escanea todos los videos bajo ruta_raiz, lee el tag 'comment' con ffprobe
        y restaura reproducciones + tiempo_visto_seg en la BD.
        progress_callback(actual, total, nombre) se llama opcionalmente por cada video.
        Devuelve (restaurados, sin_datos, errores).
        """
        from video_metadata import VideoMetadata

        EXTENSIONES_VIDEO = {'.mp4', '.avi', '.mkv', '.mov', '.wmv', '.flv', '.webm', '.m4v'}
        ruta_raiz = Path(ruta_raiz)
        videos = [
            f for f in ruta_raiz.rglob("*")
            if f.is_file() and f.suffix.lower() in EXTENSIONES_VIDEO
        ]
        total = len(videos)
        restaurados = sin_datos = errores = 0
        local_conn = sqlite3.connect(self.DB_FILE)
        local_conn.row_factory = sqlite3.Row
        try:
            cursor = local_conn.cursor()
            for i, video in enumerate(videos, 1):
                if progress_callback:
                    try:
                        progress_callback(i, total, video.name)
                    except Exception:
                        pass
                try:
                    meta = VideoMetadata.obtener_metadatos(str(video), ffprobe_path=ffprobe_path)
                    if not meta:
                        sin_datos += 1
                        continue
                    repros = int(meta.get('reproducciones', 0))
                    tiempo = int(meta.get('tiempo_visto_seg', 0))
                    es_fav = bool(meta.get('es_favorito', False))
                    fue_visto = bool(meta.get('fue_visto', repros > 0))
                    ultima = meta.get('ultima_reproduccion', '')
                    if repros == 0 and tiempo == 0:
                        sin_datos += 1
                        continue
                    ruta_str = str(video).replace("\\", "/")
                    nombre = video.name
                    cursor.execute("""
                        INSERT INTO video_stats
                            (ruta, nombre_archivo, reproducciones, tiempo_visto_seg,
                             ultima_reproduccion, es_favorito, fue_visto)
                        VALUES (?, ?, ?, ?, ?, ?, ?)
                        ON CONFLICT(ruta) DO UPDATE SET
                            reproducciones   = MAX(reproducciones,   excluded.reproducciones),
                            tiempo_visto_seg = MAX(tiempo_visto_seg, excluded.tiempo_visto_seg),
                            es_favorito      = excluded.es_favorito,
                            fue_visto        = excluded.fue_visto
                    """, (ruta_str, nombre, repros, tiempo, ultima, es_fav, fue_visto))
                    restaurados += 1
                except Exception as e:
                    errores += 1
                    print(f"⚠️ Error leyendo metadatos de {video.name}: {e}")
            local_conn.commit()
        finally:
            try:
                local_conn.close()
            except Exception:
                pass
        print(f"✅ Restauración desde metadatos: {restaurados} restaurados, "
              f"{sin_datos} sin datos, {errores} errores (de {total} videos).")
        return restaurados, sin_datos, errores

    def obtener_stats_video(self, ruta):
        """Obtiene estadísticas de un video."""
        cursor = self.conn.cursor()
        ruta_str = str(ruta).replace("\\", "/")
        cursor.execute("""
            SELECT reproducciones, tiempo_visto_seg, ultima_reproduccion, es_favorito, fue_visto
            FROM video_stats
            WHERE ruta = ?
        """, (ruta_str,))
        
        resultado = cursor.fetchone()
        if resultado:
            return {
                'reproducciones': resultado['reproducciones'],
                'tiempo_visto_seg': resultado['tiempo_visto_seg'],
                'ultima_reproduccion': resultado['ultima_reproduccion'] or '',
                'favorito': bool(resultado['es_favorito']),
                'fue_visto': bool(resultado['fue_visto'])
            }
        return {
            'reproducciones': 0,
            'tiempo_visto_seg': 0,
            'ultima_reproduccion': '',
            'favorito': False,
            'fue_visto': False
        }

    def upsert_stats_max(self, ruta, reproducciones=0, tiempo_visto_seg=0,
                         es_favorito=False, fue_visto=False, ultima_reproduccion=''):
        """Alinea estadísticas de un video por máximos, sin perder valores mayores existentes.

        Devuelve el dict final con las estadísticas persistidas.
        """
        cursor = self.conn.cursor()
        ruta_str = str(ruta).replace("\\", "/")
        nombre = Path(ruta).name
        repros = max(0, int(reproducciones or 0))
        tiempo = max(0, int(tiempo_visto_seg or 0))
        es_fav_i = 1 if es_favorito else 0
        fue_visto_i = 1 if fue_visto else 0
        ultima = str(ultima_reproduccion or '')

        cursor.execute("""
            INSERT INTO video_stats
                (ruta, nombre_archivo, reproducciones, tiempo_visto_seg,
                 ultima_reproduccion, es_favorito, fue_visto)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(ruta) DO UPDATE SET
                nombre_archivo    = excluded.nombre_archivo,
                reproducciones    = MAX(video_stats.reproducciones, excluded.reproducciones),
                tiempo_visto_seg  = MAX(video_stats.tiempo_visto_seg, excluded.tiempo_visto_seg),
                ultima_reproduccion = CASE
                    WHEN excluded.ultima_reproduccion <> '' THEN excluded.ultima_reproduccion
                    ELSE video_stats.ultima_reproduccion
                END,
                es_favorito       = MAX(video_stats.es_favorito, excluded.es_favorito),
                fue_visto         = MAX(video_stats.fue_visto, excluded.fue_visto)
        """, (ruta_str, nombre, repros, tiempo, ultima, es_fav_i, fue_visto_i))

        self.conn.commit()
        return self.obtener_stats_video(ruta_str)
    
    def registrar_visualizacion(self, ruta, tiempo_segundos):
        """Registra que un video fue visto y actualiza estadísticas."""
        cursor = self.conn.cursor()
        ruta_str = str(ruta).replace("\\", "/")
        nombre = Path(ruta).name
        ahora = datetime.now().isoformat()
        
        # Insertar o actualizar
        cursor.execute("""
            INSERT INTO video_stats 
            (ruta, nombre_archivo, reproducciones, tiempo_visto_seg, ultima_reproduccion, fue_visto)
            VALUES (?, ?, 1, ?, ?, 1)
            ON CONFLICT(ruta) DO UPDATE SET
                reproducciones = reproducciones + 1,
                tiempo_visto_seg = tiempo_visto_seg + ?,
                ultima_reproduccion = ?,
                fue_visto = 1
        """, (ruta_str, nombre, tiempo_segundos, ahora, tiempo_segundos, ahora))
        
        self.conn.commit()
        
        # Retornar reproducciones actuales
        cursor.execute("SELECT reproducciones FROM video_stats WHERE ruta = ?", (ruta_str,))
        resultado = cursor.fetchone()
        return resultado['reproducciones'] if resultado else 1
    
    def marcar_favorito(self, ruta, es_favorito):
        """Marca o desmarca un video como favorito."""
        cursor = self.conn.cursor()
        ruta_str = str(ruta).replace("\\", "/")
        nombre = Path(ruta).name
        
        cursor.execute("""
            INSERT INTO video_stats (ruta, nombre_archivo, es_favorito)
            VALUES (?, ?, ?)
            ON CONFLICT(ruta) DO UPDATE SET
                es_favorito = ?
        """, (ruta_str, nombre, 1 if es_favorito else 0, 1 if es_favorito else 0))
        
        self.conn.commit()
    
    def obtener_todos_vistos(self):
        """Obtiene todos los videos que han sido vistos."""
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT ruta FROM video_stats 
            WHERE fue_visto = 1
        """)
        return [row['ruta'] for row in cursor.fetchall()]
    
    def guardar_hash_visual(self, ruta, tamaño_bytes, hash_data):
        """Guarda el hash visual de un video."""
        cursor = self.conn.cursor()
        ruta_str = str(ruta).replace("\\", "/")
        
        import json as json_lib
        hash_blob = json_lib.dumps(hash_data).encode('utf-8')
        
        cursor.execute("""
            INSERT INTO video_hashes (ruta, tamaño_bytes, hash_visual)
            VALUES (?, ?, ?)
            ON CONFLICT(ruta) DO UPDATE SET
                hash_visual = ?,
                tamaño_bytes = ?,
                fecha_calculado = CURRENT_TIMESTAMP
        """, (ruta_str, tamaño_bytes, hash_blob, hash_blob, tamaño_bytes))
        
        self.conn.commit()
    
    def obtener_hash_visual(self, ruta, tamaño_bytes):
        """Obtiene el hash visual de un video."""
        cursor = self.conn.cursor()
        ruta_str = str(ruta).replace("\\", "/")
        
        cursor.execute("""
            SELECT hash_visual FROM video_hashes 
            WHERE ruta = ? AND tamaño_bytes = ?
        """, (ruta_str, tamaño_bytes))
        
        resultado = cursor.fetchone()
        if resultado and resultado['hash_visual']:
            try:
                import json as json_lib
                return json_lib.loads(resultado['hash_visual'].decode('utf-8'))
            except:
                return None
        return None
    
    def agregar_carpeta_vetada(self, ruta_carpeta):
        """Agrega una carpeta a la lista de vetadas."""
        cursor = self.conn.cursor()
        cursor.execute("""
            INSERT OR IGNORE INTO carpetas_vetadas (ruta_carpeta)
            VALUES (?)
        """, (str(ruta_carpeta),))
        self.conn.commit()
    
    def obtener_carpetas_vetadas(self):
        """Obtiene todas las carpetas vetadas."""
        cursor = self.conn.cursor()
        cursor.execute("SELECT ruta_carpeta FROM carpetas_vetadas")
        return [row['ruta_carpeta'] for row in cursor.fetchall()]
    
    def eliminar_carpeta_vetada(self, ruta_carpeta):
        """Elimina una carpeta de la lista de vetadas."""
        cursor = self.conn.cursor()
        cursor.execute("""
            DELETE FROM carpetas_vetadas 
            WHERE ruta_carpeta = ?
        """, (str(ruta_carpeta),))
        self.conn.commit()
    
    def exportar_a_json(self, archivo_salida="video_stats.json"):
        """Exporta las estadísticas a JSON para compatibilidad."""
        cursor = self.conn.cursor()
        cursor.execute("SELECT * FROM video_stats")
        
        stats_dict = {}
        for row in cursor.fetchall():
            stats_dict[row['ruta']] = {
                'reproducciones': row['reproducciones'],
                'tiempo_visto_seg': row['tiempo_visto_seg'],
                'ultima_vez': row['ultima_reproduccion'] or '',
                'favorito': bool(row['es_favorito']),
                'fue_visto': bool(row['fue_visto'])
            }
        
        with open(archivo_salida, 'w', encoding='utf-8') as f:
            json.dump(stats_dict, f, indent=4, ensure_ascii=False)
        
        print(f"✅ Exportado a {archivo_salida}")
    
    def guardar_miniatura(self, ruta, datos_imagen):
        """Guarda la miniatura (bytes JPEG/PNG) de un video."""
        cursor = self.conn.cursor()
        ruta_str = str(ruta).replace("\\", "/")
        nombre = Path(ruta).name
        cursor.execute("""
            INSERT INTO video_stats (ruta, nombre_archivo, miniatura)
            VALUES (?, ?, ?)
            ON CONFLICT(ruta) DO UPDATE SET miniatura = ?
        """, (ruta_str, nombre, datos_imagen, datos_imagen))
        self.conn.commit()

    def obtener_miniatura(self, ruta):
        """Devuelve los bytes de la miniatura o None."""
        cursor = self.conn.cursor()
        ruta_str = str(ruta).replace("\\", "/")
        cursor.execute("SELECT miniatura FROM video_stats WHERE ruta = ?", (ruta_str,))
        row = cursor.fetchone()
        if row and row['miniatura']:
            return bytes(row['miniatura'])
        return None

    def tiene_hash(self, ruta):
        """Devuelve True si el video tiene hash visual calculado."""
        try:
            cursor = self.conn.cursor()
            ruta_str = str(ruta).replace("\\", "/")
            cursor.execute("SELECT 1 FROM video_hashes WHERE ruta = ? AND hash_visual IS NOT NULL", (ruta_str,))
            return cursor.fetchone() is not None
        except sqlite3.DatabaseError as e:
            print(f"⚠️ BD corrupta en tiene_hash ({e}), intentando recuperar…")
            self._try_recover_db()
            self.inicializar_db()
            return False

    def obtener_todos_hashes_visuales(self):
        """Devuelve todos los hashes visuales disponibles con su ruta."""
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT ruta, hash_visual
            FROM video_hashes
            WHERE hash_visual IS NOT NULL
        """)
        filas = []
        for row in cursor.fetchall():
            try:
                hash_data = json.loads(row['hash_visual'].decode('utf-8'))
                filas.append({'ruta': row['ruta'], 'hash_visual': hash_data})
            except Exception:
                continue
        return filas

    # ── Batch helpers (mucho más rápidos que llamar 1 query por video) ─────
    @staticmethod
    def _norm_ruta(ruta):
        return str(ruta).replace("\\", "/")

    def obtener_stats_batch(self, rutas):
        """Devuelve dict {ruta_normalizada: stats_dict} para una lista de rutas.

        Mucho más rápido que llamar `obtener_stats_video` en bucle: una sola
        consulta SQL en lugar de N. Las rutas que no estén en BD se omiten;
        el llamador debe usar `.get(ruta, default)` para obtener el fallback.
        """
        if not rutas:
            return {}
        cursor = self.conn.cursor()
        norm = [self._norm_ruta(r) for r in rutas]
        result = {}
        # SQLite limita los parámetros (~999); paginamos por seguridad
        chunk = 500
        for i in range(0, len(norm), chunk):
            sub = norm[i:i + chunk]
            placeholders = ",".join("?" * len(sub))
            cursor.execute(
                f"SELECT ruta, reproducciones, tiempo_visto_seg, ultima_reproduccion, "
                f"es_favorito, fue_visto FROM video_stats WHERE ruta IN ({placeholders})",
                sub,
            )
            for row in cursor.fetchall():
                result[row['ruta']] = {
                    'reproducciones': row['reproducciones'],
                    'tiempo_visto_seg': row['tiempo_visto_seg'],
                    'ultima_reproduccion': row['ultima_reproduccion'] or '',
                    'favorito': bool(row['es_favorito']),
                    'fue_visto': bool(row['fue_visto']),
                }
        return result

    def obtener_miniaturas_batch(self, rutas):
        """Devuelve dict {ruta_normalizada: bytes} para las rutas con miniatura."""
        if not rutas:
            return {}
        cursor = self.conn.cursor()
        norm = [self._norm_ruta(r) for r in rutas]
        result = {}
        chunk = 500
        for i in range(0, len(norm), chunk):
            sub = norm[i:i + chunk]
            placeholders = ",".join("?" * len(sub))
            cursor.execute(
                f"SELECT ruta, miniatura FROM video_stats "
                f"WHERE ruta IN ({placeholders}) AND miniatura IS NOT NULL",
                sub,
            )
            for row in cursor.fetchall():
                if row['miniatura']:
                    result[row['ruta']] = bytes(row['miniatura'])
        return result

    def obtener_rutas_con_hash(self, rutas=None):
        """Devuelve un set con las rutas (normalizadas) que tienen hash visual.

        Si `rutas` es None devuelve todas las rutas con hash; si se pasa una
        lista, solo se consultan esas. Pensado para sustituir bucles de
        `tiene_hash` por una única consulta.
        """
        cursor = self.conn.cursor()
        if rutas is None:
            cursor.execute(
                "SELECT ruta FROM video_hashes WHERE hash_visual IS NOT NULL"
            )
            return {row['ruta'] for row in cursor.fetchall()}
        norm = [self._norm_ruta(r) for r in rutas]
        out = set()
        chunk = 500
        for i in range(0, len(norm), chunk):
            sub = norm[i:i + chunk]
            placeholders = ",".join("?" * len(sub))
            cursor.execute(
                f"SELECT ruta FROM video_hashes "
                f"WHERE hash_visual IS NOT NULL AND ruta IN ({placeholders})",
                sub,
            )
            for row in cursor.fetchall():
                out.add(row['ruta'])
        return out

    def obtener_conteos_miniaturas_carpetas(self):
        """Devuelve dict {carpeta_normalizada: numero_de_miniaturas} en una sola query."""
        cursor = self.conn.cursor()
        try:
            cursor.execute(
                "SELECT carpeta, COUNT(*) AS n FROM folder_thumbnails GROUP BY carpeta"
            )
            return {row['carpeta']: int(row['n']) for row in cursor.fetchall()}
        except Exception:
            return {}

    def obtener_conteos_sugeridas_carpetas(self):
        """Devuelve dict {carpeta_normalizada: numero_de_sugeridas} en una sola query."""
        cursor = self.conn.cursor()
        try:
            cursor.execute(
                "SELECT carpeta, COUNT(*) AS n FROM folder_thumbnail_suggestions GROUP BY carpeta"
            )
            return {row['carpeta']: int(row['n']) for row in cursor.fetchall()}
        except Exception:
            return {}

    def renombrar_ruta(self, ruta_antigua, ruta_nueva):
        """Actualiza la ruta de un video en todas las tablas (stats + hashes).
        Usar tras renombrar un archivo en disco para no perder estadísticas/hashes."""
        antigua = str(ruta_antigua).replace("\\", "/")
        nueva = str(ruta_nueva).replace("\\", "/")
        nuevo_nombre = Path(ruta_nueva).name
        cursor = self.conn.cursor()
        try:
            cursor.execute(
                "UPDATE video_stats SET ruta = ?, nombre_archivo = ? WHERE ruta = ?",
                (nueva, nuevo_nombre, antigua)
            )
        except Exception as e:
            print(f"⚠ Error actualizando video_stats: {e}")
        try:
            # Eliminar destino huérfano si lo hubiera, para no chocar con UNIQUE
            cursor.execute("DELETE FROM video_hashes WHERE ruta = ?", (nueva,))
            cursor.execute(
                "UPDATE video_hashes SET ruta = ? WHERE ruta = ?",
                (nueva, antigua)
            )
        except Exception as e:
            print(f"⚠ Error actualizando video_hashes: {e}")
        self.conn.commit()

    def renombrar_prefijo_ruta(self, carpeta_antigua, carpeta_nueva):
        """Actualiza referencias al renombrar una carpeta completa.

        Migra rutas en todas las tablas relevantes sin perder estadísticas
        ni hashes cuando existan filas de destino previas.
        """
        old_base = str(carpeta_antigua).replace("\\", "/").rstrip("/")
        new_base = str(carpeta_nueva).replace("\\", "/").rstrip("/")
        if not old_base or old_base == new_base:
            return

        old_prefix = old_base + "/"
        old_prefix_len = len(old_prefix)
        cursor = self.conn.cursor()

        def map_path(ruta):
            r = str(ruta).replace("\\", "/")
            if r == old_base:
                return new_base
            if r.startswith(old_prefix):
                return new_base + "/" + r[old_prefix_len:]
            return r

        try:
            cursor.execute("BEGIN")

            # video_stats: merge in case destination rows already exist.
            cursor.execute(
                """
                SELECT id, ruta, nombre_archivo, reproducciones, tiempo_visto_seg,
                       ultima_reproduccion, es_favorito, fue_visto, miniatura
                FROM video_stats
                WHERE ruta = ? OR ruta LIKE ?
                """,
                (old_base, old_prefix + "%"),
            )
            stats_rows = cursor.fetchall()
            for row in stats_rows:
                old_ruta = row["ruta"]
                new_ruta = map_path(old_ruta)
                if new_ruta == old_ruta:
                    continue

                cursor.execute("SELECT * FROM video_stats WHERE ruta = ?", (new_ruta,))
                target = cursor.fetchone()
                if target:
                    reproducciones = int(target["reproducciones"] or 0) + int(row["reproducciones"] or 0)
                    tiempo = int(target["tiempo_visto_seg"] or 0) + int(row["tiempo_visto_seg"] or 0)
                    ultima = target["ultima_reproduccion"] or row["ultima_reproduccion"]
                    if (row["ultima_reproduccion"] or "") > (target["ultima_reproduccion"] or ""):
                        ultima = row["ultima_reproduccion"]
                    es_favorito = 1 if (int(target["es_favorito"] or 0) or int(row["es_favorito"] or 0)) else 0
                    fue_visto = 1 if (int(target["fue_visto"] or 0) or int(row["fue_visto"] or 0)) else 0
                    miniatura = target["miniatura"] if target["miniatura"] is not None else row["miniatura"]

                    cursor.execute(
                        """
                        UPDATE video_stats
                        SET nombre_archivo = ?,
                            reproducciones = ?,
                            tiempo_visto_seg = ?,
                            ultima_reproduccion = ?,
                            es_favorito = ?,
                            fue_visto = ?,
                            miniatura = ?
                        WHERE ruta = ?
                        """,
                        (Path(new_ruta).name, reproducciones, tiempo, ultima, es_favorito, fue_visto, miniatura, new_ruta),
                    )
                    cursor.execute("DELETE FROM video_stats WHERE id = ?", (row["id"],))
                else:
                    cursor.execute(
                        "UPDATE video_stats SET ruta = ?, nombre_archivo = ? WHERE id = ?",
                        (new_ruta, Path(new_ruta).name, row["id"]),
                    )

            # video_hashes: preserve destination hash if present; otherwise carry source hash.
            cursor.execute(
                """
                SELECT id, ruta, tamaño_bytes, hash_visual
                FROM video_hashes
                WHERE ruta = ? OR ruta LIKE ?
                """,
                (old_base, old_prefix + "%"),
            )
            hash_rows = cursor.fetchall()
            for row in hash_rows:
                old_ruta = row["ruta"]
                new_ruta = map_path(old_ruta)
                if new_ruta == old_ruta:
                    continue

                cursor.execute("SELECT id, tamaño_bytes, hash_visual FROM video_hashes WHERE ruta = ?", (new_ruta,))
                target = cursor.fetchone()
                if target:
                    needs_hash = (target["hash_visual"] is None) and (row["hash_visual"] is not None)
                    needs_size = (not int(target["tamaño_bytes"] or 0)) and int(row["tamaño_bytes"] or 0)
                    if needs_hash or needs_size:
                        cursor.execute(
                            """
                            UPDATE video_hashes
                            SET hash_visual = COALESCE(hash_visual, ?),
                                tamaño_bytes = CASE WHEN COALESCE(tamaño_bytes, 0) = 0 THEN ? ELSE tamaño_bytes END
                            WHERE id = ?
                            """,
                            (row["hash_visual"], row["tamaño_bytes"], target["id"]),
                        )
                    cursor.execute("DELETE FROM video_hashes WHERE id = ?", (row["id"],))
                else:
                    cursor.execute("UPDATE video_hashes SET ruta = ? WHERE id = ?", (new_ruta, row["id"]))

            # folder_thumbnails (carpeta)
            cursor.execute(
                "SELECT id, carpeta FROM folder_thumbnails WHERE carpeta = ? OR carpeta LIKE ?",
                (old_base, old_prefix + "%"),
            )
            for row in cursor.fetchall():
                cursor.execute(
                    "UPDATE folder_thumbnails SET carpeta = ? WHERE id = ?",
                    (map_path(row["carpeta"]), row["id"]),
                )

            # folder_thumbnails (video_ruta)
            cursor.execute(
                "SELECT id, video_ruta FROM folder_thumbnails WHERE video_ruta = ? OR video_ruta LIKE ?",
                (old_base, old_prefix + "%"),
            )
            for row in cursor.fetchall():
                cursor.execute(
                    "UPDATE folder_thumbnails SET video_ruta = ? WHERE id = ?",
                    (map_path(row["video_ruta"]), row["id"]),
                )

            # app_settings values that store absolute/relative paths.
            cursor.execute("SELECT clave, valor FROM app_settings")
            for row in cursor.fetchall():
                valor = row["valor"]
                if valor is None:
                    continue
                nuevo_valor = map_path(valor)
                if nuevo_valor != valor:
                    cursor.execute(
                        "UPDATE app_settings SET valor = ?, fecha_updated = CURRENT_TIMESTAMP WHERE clave = ?",
                        (nuevo_valor, row["clave"]),
                    )

            self.conn.commit()
        except Exception:
            self.conn.rollback()
            raise

    def stats_carpeta(self, prefijo_carpeta):
        cursor = self.conn.cursor()
        prefijo = str(prefijo_carpeta).replace("\\", "/")
        if not prefijo.endswith("/"):
            prefijo += "/"
        cursor.execute("""
            SELECT COALESCE(SUM(reproducciones), 0) as total_vistas,
                   COALESCE(SUM(tiempo_visto_seg), 0) as total_tiempo
            FROM video_stats
            WHERE ruta LIKE ? || '%'
        """, (prefijo,))
        row = cursor.fetchone()
        return {
            'total_vistas': row['total_vistas'],
            'total_tiempo': row['total_tiempo']
        }

    def guardar_miniatura_carpeta(self, carpeta, video_ruta, frame_no, thumb_bytes):
        """Guarda una miniatura asociada a una carpeta (puede haber varias)."""
        cursor = self.conn.cursor()
        carpeta_str = str(carpeta).replace("\\", "/")
        video_str = str(video_ruta).replace("\\", "/")
        cursor.execute("""
            INSERT INTO folder_thumbnails (carpeta, video_ruta, frame_no, thumbnail_blob)
            VALUES (?, ?, ?, ?)
        """, (carpeta_str, video_str, frame_no, thumb_bytes))
        self.conn.commit()
        return cursor.lastrowid

    def obtener_miniaturas_carpeta(self, carpeta):
        """Devuelve todas las miniaturas asociadas a una carpeta, ordenadas por fecha."""
        cursor = self.conn.cursor()
        carpeta_str = str(carpeta).replace("\\", "/")
        cursor.execute("""
            SELECT id, video_ruta, frame_no, thumbnail_blob, fecha
            FROM folder_thumbnails
            WHERE carpeta = ?
            ORDER BY fecha ASC
        """, (carpeta_str,))
        result = []
        for row in cursor.fetchall():
            result.append({
                'id': row['id'],
                'video_ruta': row['video_ruta'],
                'frame_no': row['frame_no'],
                'thumbnail_blob': bytes(row['thumbnail_blob']),
                'fecha': row['fecha'],
            })
        return result

    def eliminar_miniatura_carpeta(self, thumb_id):
        """Elimina una miniatura de carpeta por su id."""
        cursor = self.conn.cursor()
        cursor.execute("DELETE FROM folder_thumbnails WHERE id = ?", (thumb_id,))
        self.conn.commit()

    def obtener_ids_miniaturas_carpeta(self, carpeta):
        """Devuelve solo los ids de las miniaturas de una carpeta (sin cargar el BLOB).

        Mucho más rápido que `obtener_miniaturas_carpeta` cuando solo se necesita
        contar o elegir una al azar para luego cargarla con `obtener_miniatura_por_id`.
        """
        cursor = self.conn.cursor()
        carpeta_str = str(carpeta).replace("\\", "/")
        cursor.execute(
            "SELECT id FROM folder_thumbnails WHERE carpeta = ? ORDER BY fecha ASC",
            (carpeta_str,),
        )
        return [int(row['id']) for row in cursor.fetchall()]

    def obtener_miniatura_por_id(self, thumb_id):
        """Devuelve los bytes de una miniatura de carpeta por su id, o None."""
        cursor = self.conn.cursor()
        cursor.execute(
            "SELECT thumbnail_blob FROM folder_thumbnails WHERE id = ?", (int(thumb_id),)
        )
        row = cursor.fetchone()
        if row and row['thumbnail_blob']:
            return bytes(row['thumbnail_blob'])
        return None

    def obtener_origen_miniatura(self, thumb_id):
        """Devuelve (video_ruta, frame_no) de la miniatura indicada o None."""
        cursor = self.conn.cursor()
        cursor.execute(
            "SELECT video_ruta, frame_no FROM folder_thumbnails WHERE id = ?",
            (int(thumb_id),),
        )
        row = cursor.fetchone()
        if not row:
            return None
        return (row['video_ruta'], int(row['frame_no'] or 0))

    # ── Miniaturas sugeridas automáticas por carpeta ─────────────────────────
    def guardar_miniatura_sugerida_carpeta(self, carpeta, video_ruta, frame_no, thumb_bytes, score=0.0):
        """Guarda una miniatura sugerida de carpeta evitando duplicados exactos."""
        cursor = self.conn.cursor()
        carpeta_str = str(carpeta).replace("\\", "/")
        video_str = str(video_ruta).replace("\\", "/")
        cursor.execute(
            """
            INSERT OR IGNORE INTO folder_thumbnail_suggestions
            (carpeta, video_ruta, frame_no, thumbnail_blob, score)
            VALUES (?, ?, ?, ?, ?)
            """,
            (carpeta_str, video_str, int(frame_no), thumb_bytes, float(score or 0.0)),
        )
        self.conn.commit()
        return cursor.lastrowid

    def obtener_miniaturas_sugeridas_carpeta(self, carpeta, limit=120):
        """Devuelve sugerencias para una carpeta, mejores primero."""
        cursor = self.conn.cursor()
        carpeta_str = str(carpeta).replace("\\", "/")
        cursor.execute(
            """
            SELECT id, video_ruta, frame_no, thumbnail_blob, score, fecha
            FROM folder_thumbnail_suggestions
            WHERE carpeta = ?
            ORDER BY score DESC, fecha DESC
            LIMIT ?
            """,
            (carpeta_str, int(max(1, limit))),
        )
        out = []
        for row in cursor.fetchall():
            out.append({
                'id': row['id'],
                'video_ruta': row['video_ruta'],
                'frame_no': int(row['frame_no'] or 0),
                'thumbnail_blob': bytes(row['thumbnail_blob']) if row['thumbnail_blob'] else b'',
                'score': float(row['score'] or 0.0),
                'fecha': row['fecha'],
            })
        return out

    def contar_miniaturas_sugeridas_carpeta(self, carpeta):
        cursor = self.conn.cursor()
        carpeta_str = str(carpeta).replace("\\", "/")
        cursor.execute(
            "SELECT COUNT(*) AS n FROM folder_thumbnail_suggestions WHERE carpeta = ?",
            (carpeta_str,),
        )
        row = cursor.fetchone()
        return int(row['n'] or 0) if row else 0

    def eliminar_miniatura_sugerida_carpeta(self, thumb_id):
        cursor = self.conn.cursor()
        cursor.execute("DELETE FROM folder_thumbnail_suggestions WHERE id = ?", (int(thumb_id),))
        self.conn.commit()

    def eliminar_todas_sugeridas_carpeta(self, carpeta):
        """Elimina todas las miniaturas sugeridas de una carpeta."""
        cursor = self.conn.cursor()
        carpeta_str = str(carpeta).replace("\\", "/")
        cursor.execute("DELETE FROM folder_thumbnail_suggestions WHERE carpeta = ?", (carpeta_str,))
        self.conn.commit()

    def recortar_miniaturas_sugeridas_carpeta(self, carpeta, max_count=10):
        """Mantiene solo las `max_count` sugeridas más recientes por carpeta."""
        max_count = max(1, int(max_count))
        cursor = self.conn.cursor()
        carpeta_str = str(carpeta).replace("\\", "/")
        cursor.execute(
            """
            DELETE FROM folder_thumbnail_suggestions
            WHERE carpeta = ?
              AND id NOT IN (
                  SELECT id
                  FROM folder_thumbnail_suggestions
                  WHERE carpeta = ?
                  ORDER BY fecha DESC, id DESC
                  LIMIT ?
              )
            """,
            (carpeta_str, carpeta_str, max_count),
        )
        self.conn.commit()

    # ── Renombrados _rwd pendientes (persisten entre arranques) ────────────
    def add_pending_rename(self, ruta_origen, error_msg=None):
        """Guarda/actualiza un renombrado _rwd que no pudo completarse."""
        cursor = self.conn.cursor()
        ruta = self._norm_ruta(ruta_origen)
        cursor.execute(
            """
            INSERT INTO pending_renames (ruta_origen, intentos, ultimo_error)
            VALUES (?, 1, ?)
            ON CONFLICT(ruta_origen) DO UPDATE SET
                intentos = intentos + 1,
                ultimo_error = excluded.ultimo_error,
                fecha_added = CURRENT_TIMESTAMP
            """,
            (ruta, str(error_msg) if error_msg else None),
        )
        self.conn.commit()

    def obtener_pending_renames(self):
        """Devuelve lista de rutas (str) con renombrados _rwd pendientes."""
        cursor = self.conn.cursor()
        try:
            cursor.execute(
                "SELECT ruta_origen FROM pending_renames ORDER BY fecha_added ASC"
            )
            return [row['ruta_origen'] for row in cursor.fetchall()]
        except Exception:
            return []

    def eliminar_pending_rename(self, ruta_origen):
        cursor = self.conn.cursor()
        cursor.execute(
            "DELETE FROM pending_renames WHERE ruta_origen = ?",
            (self._norm_ruta(ruta_origen),),
        )
        self.conn.commit()

    def guardar_setting(self, clave, valor):
        """Guarda un ajuste simple clave/valor persistente."""
        cursor = self.conn.cursor()
        cursor.execute(
            """
            INSERT INTO app_settings (clave, valor, fecha_updated)
            VALUES (?, ?, CURRENT_TIMESTAMP)
            ON CONFLICT(clave) DO UPDATE SET
                valor = excluded.valor,
                fecha_updated = CURRENT_TIMESTAMP
            """,
            (str(clave), None if valor is None else str(valor)),
        )
        self.conn.commit()

    def obtener_setting(self, clave, default=None):
        """Recupera un ajuste simple clave/valor persistente."""
        cursor = self.conn.cursor()
        cursor.execute("SELECT valor FROM app_settings WHERE clave = ?", (str(clave),))
        row = cursor.fetchone()
        if row is None:
            return default
        return row['valor']

    def cerrar(self):
        """Cierra la conexión a la base de datos."""
        if self.conn:
            self.conn.close()
    
    def __del__(self):
        """Cierra la BD cuando se destruye el objeto."""
        self.cerrar()
