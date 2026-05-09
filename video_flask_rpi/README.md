# Abrearch Flask para Raspberry Pi

App web en Flask para dejar encendida en una Raspberry, apuntando a carpeta compartida (SMB/NFS) y accesible desde cualquier sitio de forma segura.

## Que incluye

- Login por usuario/password.
- Explorador de carpetas y videos.
- Streaming con soporte Range.
- Miniaturas de carpeta y video.
- Favoritos, vistos, contador de reproducciones y tiempo en SQLite.
- Filtros tipo app escritorio: Todos, Favoritos, Sin revisar, -Vistos, +Vistos, +Tiempo, Pesados.
- Boton de log.

## Estructura

- `app.py`: backend Flask y API.
- `templates/`: HTML de login y panel principal.
- `static/`: CSS y JS.
- `thumb_cache/`: cache de miniaturas.
- `.env.example`: variables de entorno.

## 1) Preparar Raspberry

```bash
sudo apt update
sudo apt install -y python3 python3-venv ffmpeg cifs-utils
```

## 2) Copiar proyecto y crear entorno

```bash
cd /opt
sudo mkdir -p abrearch-flask
sudo chown -R $USER:$USER /opt/abrearch-flask
# copia aqui los archivos del proyecto video_flask_rpi
cd /opt/abrearch-flask
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Edita `.env` con tus valores reales.

## 3) Montar carpeta compartida SMB (ejemplo)

Crear punto de montaje:

```bash
sudo mkdir -p /mnt/media_share
```

Guardar credenciales SMB:

```bash
sudo nano /etc/smb-credentials-media
```

Contenido:

```text
username=TU_USUARIO
password=TU_PASSWORD
```

Permisos:

```bash
sudo chmod 600 /etc/smb-credentials-media
```

Agregar en `/etc/fstab` (ejemplo):

```text
//IP_DEL_PC/NOMBRE_RECURSO /mnt/media_share cifs credentials=/etc/smb-credentials-media,uid=1000,gid=1000,iocharset=utf8,vers=3.0,nofail,x-systemd.automount 0 0
```

Montar:

```bash
sudo mount -a
```

## 4) Ejecutar manualmente

```bash
cd /opt/abrearch-flask
source .venv/bin/activate
python app.py
```

Abrir en LAN:

```text
http://IP_DE_LA_RASPBERRY:8080
```

## 5) Dejarlo 24/7 con systemd

Copia `deploy/abrearch-flask.service` a `/etc/systemd/system/abrearch-flask.service` y ajusta rutas si hace falta.

```bash
sudo cp deploy/abrearch-flask.service /etc/systemd/system/abrearch-flask.service
sudo systemctl daemon-reload
sudo systemctl enable --now abrearch-flask
sudo systemctl status abrearch-flask
```

Logs servicio:

```bash
journalctl -u abrearch-flask -f
```

## 6) Acceso desde cualquier sitio (recomendado)

### Opcion A: Tailscale (recomendada)

```bash
curl -fsSL https://tailscale.com/install.sh | sh
sudo tailscale up
```

Luego accedes por la IP/hostname de Tailscale desde movil o PC remoto.

### Opcion B: Cloudflare Tunnel

Publica la app sin abrir puertos directos en router.

## Seguridad minima recomendada

- Cambiar `APP_USERNAME` y `APP_PASSWORD`.
- Cambiar `SECRET_KEY`.
- No abrir puerto al mundo sin VPN/tunnel.
- Mantener Raspberry y dependencias actualizadas.
