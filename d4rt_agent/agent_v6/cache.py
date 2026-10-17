"""One durable archive per clip/backbone; V5's file adapters work in a temporary directory."""
import hashlib
import os
from pathlib import Path
import tarfile
import tempfile


class ClipArchive:
    def __init__(self, root, backbone):
        self.root = Path(root) / backbone
        self.root.mkdir(parents=True, exist_ok=True)
        self._temporary = tempfile.TemporaryDirectory(prefix='agent-v6-clip-')
        self.directory = Path(self._temporary.name)
        self.path = None

    def activate(self, source_key):
        import shutil
        self.flush()
        for item in self.directory.iterdir():
            if item.is_dir():
                shutil.rmtree(item)
            else:
                item.unlink()
        key = hashlib.sha256(str(source_key).encode()).hexdigest()
        self.path = self.root / f'{key}.tar.gz'
        if self.path.exists():
            with tarfile.open(self.path, 'r:gz') as archive:
                for member in archive.getmembers():
                    target = (self.directory / member.name).resolve()
                    if not target.is_relative_to(self.directory) or member.issym() or member.islnk():
                        raise ValueError('cache archive contains an unsafe member')
                archive.extractall(self.directory, filter='data')
        return self.directory

    def flush(self):
        if self.path is not None:
            temporary = self.path.with_suffix('.tmp')
            with tarfile.open(temporary, 'w:gz', dereference=True, compresslevel=1) as archive:
                for item in sorted(self.directory.iterdir()):
                    archive.add(item, arcname=item.name)
            os.replace(temporary, self.path)

    def close(self):
        try:
            self.flush()
        finally:
            self._temporary.cleanup()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
