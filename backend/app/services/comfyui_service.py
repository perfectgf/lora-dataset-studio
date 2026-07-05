"""
SAFE ComfyUI Service
===================
✅ Flask-safe (no process scanning)
✅ PID-based restart only
✅ Windows compatible

Lifted from the parent project's app/services/comfyui_service.py for LoRA
Dataset Studio: SRC's module-level COMFYUI_API_ADDRESS constant becomes a live
`cfg.get('comfyui.api_url')` call (config.json changes take effect without a
restart). SRC's COMFYUI_BASE_DIR/COMFYUI_BATCH_FILE imports are dropped — this
app never launches or stops ComfyUI itself, so start_comfyui_process /
stop_comfyui_process were already no-ops and stay that way.
"""

import os
import time
import socket
import threading
import logging
import json
import requests
from urllib.parse import urljoin
from typing import Optional, Tuple, Dict

from .. import config as cfg

logger = logging.getLogger(__name__)

COMFYUI_PID_FILE = os.path.join(os.path.dirname(__file__), "comfyui.pid")


class ComfyUIService:
    def __init__(self):
        self.api_host = "127.0.0.1"
        self.api_port = 8188
        self.startup_timeout = 60
        self.check_interval = 2
        self._startup_lock = threading.Lock()
        self._is_starting = False

    # ---------------- API ----------------
    def parse_api_address(self):
        addr = cfg.get('comfyui.api_url').replace("http://", "").replace("https://", "").rstrip("/")
        if ':' in addr:
            self.api_host, port = addr.split(':', 1)
            self.api_port = int(port)
        else:
            self.api_host = addr
            self.api_port = 8188

    def check_connection(self) -> bool:
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.settimeout(2)
                if s.connect_ex((self.api_host, self.api_port)) != 0:
                    return False
            r = requests.get(urljoin(cfg.get('comfyui.api_url'), "/history"), timeout=3)
            return r.status_code in (200, 404)
        except (socket.error, requests.RequestException, ConnectionError, OSError):
            return False

    # ---------------- PID (DEPRECATED) ----------------
    def _read_pid(self) -> Optional[int]:
        """Lecture PID désactivée"""
        return None

    # ---------------- Lifecycle ----------------
    def start_comfyui(self) -> Tuple[bool, str]:
        """
        Vérifie si ComfyUI est accessible.
        Ne lance plus de processus (gestion externe).
        """
        self.parse_api_address()
        if self.check_connection():
            return True, "Running (External)"

        logger.warning("⚠️ ComfyUI n'est pas accessible, mais le démarrage automatique est désactivé.")
        return False, "ComfyUI not running (External management required)"

    def ensure_comfyui_running(self) -> Tuple[bool, str]:
        """Vérifie simplement la connexion."""
        self.parse_api_address()
        if self.check_connection():
            return True, "Running"
        return False, "ComfyUI not running (Please start external supervisor)"

    def restart_comfyui_async(self, delay: int = 5):
        """
        DEPRECATED: Le redémarrage est géré par le superviseur externe ou le watchdog.
        """
        logger.warning("⚠️ Demande de redémarrage ignorée (gestion externe).")
        pass

    # ✅ API publique unifiée utilisée par queue_manager
    def stop_comfyui_process(self):
        """Arrêt désactivé."""
        logger.warning("⚠️ stop_comfyui_process ignoré.")
        return True

    def start_comfyui_process(self):
        """Démarrage désactivé."""
        logger.warning("⚠️ start_comfyui_process ignoré.")
        return self.check_connection()

    # ---------------- Prompt ----------------
    def queue_prompt(self, prompt: Dict, client_id: str):
        ok, msg = self.ensure_comfyui_running()
        if not ok:
            return None, msg
        payload = json.dumps({"prompt": prompt, "client_id": client_id})
        r = requests.post(urljoin(cfg.get('comfyui.api_url'), "/prompt"), data=payload)
        if r.status_code == 200:
            return r.json(), None
        return None, r.text


comfyui_service = ComfyUIService()

def ensure_comfyui_before_generation():
    return comfyui_service.ensure_comfyui_running()

def check_comfyui_status():
    return {
        "running": comfyui_service.check_connection(),
        "pid": None
    }
