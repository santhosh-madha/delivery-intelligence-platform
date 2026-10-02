"""Cloud image model verification and single-instance startup."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import urllib.parse
import urllib.request

from .predict import sha256


def prepare_model(directory, url=None):
    directory = Path(directory)
    manifest = json.loads((directory/'manifest.json').read_text())
    model = directory/'model.cbm'
    if not model.exists():
        if not url or urllib.parse.urlparse(url).scheme != 'https':
            raise ValueError('Supply HTTPS MODEL_URL for the published model release asset.')
        temporary = directory/'model.download'
        try:
            with urllib.request.urlopen(url, timeout=120) as response, temporary.open('wb') as output:
                if urllib.parse.urlparse(response.url).scheme != 'https':
                    raise ValueError('Model download must remain HTTPS.')
                size = 0
                while chunk := response.read(1024*1024):
                    size += len(chunk)
                    if size > 150*1024*1024:
                        raise ValueError('Model download exceeds 150 MiB.')
                    output.write(chunk)
            if sha256(temporary) != manifest['model_sha256']:
                raise ValueError('Model checksum mismatch.')
            temporary.replace(model)
        finally:
            temporary.unlink(missing_ok=True)
    if sha256(model) != manifest['model_sha256']:
        raise ValueError('Model checksum mismatch.')


def start():
    if os.environ.get('ETA_ENV') != 'cloud':
        raise RuntimeError('Cloud startup requires ETA_ENV=cloud.')
    if len(os.environ.get('ETA_API_KEY', '')) < 32:
        raise RuntimeError('Set ETA_API_KEY with at least 32 characters.')
    if not os.environ.get('DATABASE_URL'):
        raise RuntimeError('Set DATABASE_URL.')
    port = int(os.environ.get('PORT', '8000'))
    if not 1 <= port <= 65535:
        raise ValueError('Invalid PORT.')
    prepare_model(os.environ['ETA_MODEL_DIR'])
    # Free Render has no pre-deploy job. One instance runs migrations before serving.
    subprocess.run(['alembic', 'upgrade', 'head'], check=True)
    os.execvp('uvicorn', ['uvicorn', 'delivery_intelligence_platform.api:app',
                         '--host', '0.0.0.0', '--port', str(port), '--workers', '1'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['prepare-model', 'start'])
    parser.add_argument('--directory', default='/app/model')
    args = parser.parse_args()
    if args.action == 'prepare-model':
        prepare_model(args.directory, os.environ.get('MODEL_URL'))
    else:
        start()


if __name__ == '__main__':
    main()
