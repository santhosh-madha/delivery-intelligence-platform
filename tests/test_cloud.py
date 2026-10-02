import hashlib
import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from delivery_intelligence_platform import database
from delivery_intelligence_platform.cloud import prepare_model, start


def test_database_url_enforces_verified_tls(monkeypatch):
    monkeypatch.setattr(database, 'dotenv_values', lambda _: {})
    monkeypatch.setenv('ETA_ENV', 'cloud')
    monkeypatch.setenv('DATABASE_URL', 'postgresql://user:password@example.neon.tech/db?sslmode=disable')
    factory = MagicMock()
    monkeypatch.setattr(database, 'create_engine', factory)
    database.make_engine()
    options = factory.call_args.kwargs
    assert options['connect_args']['sslmode'] == 'verify-full'
    assert options['connect_args']['sslrootcert'] == 'system'
    assert factory.call_args.args[0].drivername == 'postgresql+psycopg'
    assert options['hide_parameters'] is True


def test_cloud_database_never_falls_back_to_local(monkeypatch):
    monkeypatch.setattr(database, 'dotenv_values', lambda _: {})
    monkeypatch.setenv('ETA_ENV', 'cloud')
    monkeypatch.delenv('DATABASE_URL', raising=False)
    with pytest.raises(ValueError, match='requires DATABASE_URL'):
        database.make_engine()


def test_invalid_database_url_error_does_not_echo_secret(monkeypatch):
    monkeypatch.setattr(database, 'dotenv_values', lambda _: {})
    monkeypatch.setenv('DATABASE_URL', 'sqlite:///private-secret')
    with pytest.raises(ValueError) as error:
        database.make_engine()
    assert 'private-secret' not in str(error.value)


def test_model_checksum_verified(tmp_path):
    content=b'fake test model'
    (tmp_path/'manifest.json').write_text(json.dumps({'model_sha256': hashlib.sha256(content).hexdigest()}))
    (tmp_path/'model.cbm').write_bytes(content)
    prepare_model(tmp_path)
    (tmp_path/'model.cbm').write_bytes(b'corrupted')
    with pytest.raises(ValueError, match='checksum'):
        prepare_model(tmp_path)


def test_missing_model_rejects_insecure_download(tmp_path):
    (tmp_path/'manifest.json').write_text(json.dumps({'model_sha256':'unused'}))
    with pytest.raises(ValueError, match='HTTPS'):
        prepare_model(tmp_path, 'http://example.com/model.cbm')


def test_start_rejects_missing_key_before_migration(monkeypatch):
    monkeypatch.setenv('ETA_ENV','cloud')
    monkeypatch.delenv('ETA_API_KEY', raising=False)
    with pytest.raises(RuntimeError, match='ETA_API_KEY'):
        start()
