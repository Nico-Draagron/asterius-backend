import os
import logging
from pathlib import Path
from typing import List, Optional

try:
    from google.cloud import storage
    _GCS_AVAILABLE = True
except ImportError:
    _GCS_AVAILABLE = False

logger = logging.getLogger("CloudStorageManager")
logger.setLevel(logging.INFO)

class CloudStorageManager:
    def __init__(self, bucket_name: Optional[str] = None):
        if not _GCS_AVAILABLE:
            raise ImportError("google-cloud-storage não está instalado.")
        self.project_id = os.getenv("PROJECT_ID", "default-project")
        self.bucket_name = bucket_name or os.getenv("GCS_BUCKET_NAME") or f"{self.project_id}-asterius-data"
        self.client = storage.Client()
        self.bucket = self.client.bucket(self.bucket_name)
        logger.info(f"Conectado ao bucket: {self.bucket_name}")

    def upload_file(self, local_path: str, gcs_path: str):
        try:
            local_path = Path(local_path)
            blob = self.bucket.blob(gcs_path)
            blob.upload_from_filename(str(local_path))
            logger.info(f"Upload: {local_path} -> gs://{self.bucket_name}/{gcs_path}")
        except Exception as e:
            logger.error(f"Erro ao fazer upload: {e}")
            raise

    def download_file(self, gcs_path: str, local_path: str):
        try:
            local_path = Path(local_path)
            local_path.parent.mkdir(parents=True, exist_ok=True)
            blob = self.bucket.blob(gcs_path)
            blob.download_to_filename(str(local_path))
            logger.info(f"Download: gs://{self.bucket_name}/{gcs_path} -> {local_path}")
        except Exception as e:
            logger.error(f"Erro ao fazer download: {e}")
            raise

    def list_files(self, prefix: str = "") -> List[str]:
        try:
            blobs = self.client.list_blobs(self.bucket_name, prefix=prefix)
            files = [blob.name for blob in blobs]
            logger.info(f"Listando arquivos com prefixo '{prefix}': {len(files)} encontrados")
            return files
        except Exception as e:
            logger.error(f"Erro ao listar arquivos: {e}")
            return []

    def sync_local_to_gcs(self, local_dir: str, gcs_prefix: str = ""):
        local_dir = Path(local_dir)
        for file_path in local_dir.rglob("*"):
            if file_path.is_file():
                rel_path = file_path.relative_to(local_dir)
                gcs_path = str(Path(gcs_prefix) / rel_path).replace("\\", "/")
                try:
                    self.upload_file(str(file_path), gcs_path)
                except Exception as e:
                    logger.error(f"Erro ao sincronizar {file_path}: {e}")

    def sync_gcs_to_local(self, gcs_prefix: str, local_dir: str):
        local_dir = Path(local_dir)
        files = self.list_files(gcs_prefix)
        for gcs_path in files:
            rel_path = Path(gcs_path).relative_to(gcs_prefix) if gcs_prefix else Path(gcs_path)
            local_path = local_dir / rel_path
            try:
                self.download_file(gcs_path, str(local_path))
            except Exception as e:
                logger.error(f"Erro ao baixar {gcs_path}: {e}")

    def is_available(self) -> bool:
        return _GCS_AVAILABLE

# Singleton
_storage_manager: Optional[CloudStorageManager] = None

def get_storage_manager() -> Optional[CloudStorageManager]:
    global _storage_manager
    if _storage_manager is None and _GCS_AVAILABLE:
        try:
            _storage_manager = CloudStorageManager()
        except Exception as e:
            logger.error(f"Erro ao inicializar CloudStorageManager: {e}")
            _storage_manager = None
    return _storage_manager

def upload_nomads_data(local_file: str, gcs_prefix: str = "NOMADS/dados/cache"):
    sm = get_storage_manager()
    if sm:
        gcs_path = str(Path(gcs_prefix) / Path(local_file).name).replace("\\", "/")
        sm.upload_file(local_file, gcs_path)

def download_nomads_data(filename: str, local_dir: str = "backend/NOMADS/dados/cache"):
    sm = get_storage_manager()
    if sm:
        gcs_path = f"NOMADS/dados/cache/{filename}"
        local_path = str(Path(local_dir) / filename)
        sm.download_file(gcs_path, local_path)

def sync_nomads_to_cloud():
    sm = get_storage_manager()
    if sm:
        sm.sync_local_to_gcs("backend/NOMADS/dados/cache", "NOMADS/dados/cache")

def sync_nomads_from_cloud():
    sm = get_storage_manager()
    if sm:
        sm.sync_gcs_to_local("NOMADS/dados/cache", "backend/NOMADS/dados/cache")
