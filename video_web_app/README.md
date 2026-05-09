# Video Web App (nuevo proyecto)

Proyecto web nuevo para ver tus videos desde el movil, manteniendo los archivos en el PC.

## Estructura

- `backend/app/main.py`: API y servidor web
- `backend/app/services/media.py`: logica de escaneo y validacion de rutas
- `backend/app/services/state_store.py`: estados de videos en SQLite (favorito/visto)
- `backend/app/services/auth.py`: validacion de token para la API
- `backend/app/static/`: interfaz web responsive

## Requisitos

- Python 3.10+
- Red local (PC y movil en la misma WiFi)

## Arranque en Windows (PowerShell)

```powershell
cd d:\projects\funcionalidades\video_web_app\backend
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
$env:MEDIA_ROOT = "D:\\ruta\\a\\tus\\videos"
$env:ACCESS_TOKEN = "cambia-este-token"
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

Si no defines `ACCESS_TOKEN`, la API queda abierta dentro de la red local.

## Abrir desde el movil

1. En el PC, busca tu IP local:

```powershell
ipconfig
```

2. En el movil, abre:

```text
http://TU_IP_LOCAL:8000
```

Ejemplo: `http://192.168.1.20:8000`

## Notas de seguridad

- Esto esta pensado para red local.
- Si quieres acceso remoto, usa Tailscale o VPN.
- No expongas directamente el puerto en internet sin autenticacion y HTTPS.

## Funcionalidades actuales

- Listado y busqueda de videos por nombre.
- Reproduccion por streaming desde el navegador movil.
- Favoritos y vistos persistidos en SQLite (`backend/video_web.db`).
- Token opcional para proteger `/api/*`.

## Siguiente paso recomendado

- Agregar paginacion para bibliotecas muy grandes
- Mostrar miniaturas de video en la lista
- Crear usuarios (no solo token global)
