import os

import pytest


def pytest_configure(config):
    os.environ.setdefault("DJANGO_SETTINGS_MODULE", "bikes_project.settings")
    os.environ.setdefault("DB_PATH", "/tmp/test_bikes.db")
    os.environ.setdefault("ALLEGRO_CLIENT_ID", "test_id")
    os.environ.setdefault("ALLEGRO_CLIENT_SECRET", "test_secret")
    os.environ.setdefault("ALLEGRO_AUTH_URL", "https://example.invalid/auth")
    os.environ.setdefault("ALLEGRO_API_BASE", "https://example.invalid/api")
    os.environ.setdefault("DOWNLOAD_SLEEP_SECONDS", "0")
    os.environ.setdefault("CLIP_MODEL", "ViT-B-32")
    os.environ.setdefault("CLIP_PRETRAINED", "openai")
    os.environ.setdefault("CLIP_CACHE_DIR", "/tmp/clip_cache_test")
    os.environ.setdefault("SCRAPER_SEARCH", "rower szosowy")
    os.environ.setdefault("SCRAPER_MIN_DELAY", "0")
    os.environ.setdefault("SCRAPER_MAX_DELAY", "0")


@pytest.fixture(scope="session")
def django_db_setup(django_test_environment, django_db_blocker):
    """Override to also create our unmanaged tables after Django migrates."""
    with django_db_blocker.unblock():
        from django.test.utils import setup_databases
        setup_databases(0, False)

        from django.db import connection
        from downloader.writer import _TABLES
        with connection.cursor() as cursor:
            for stmt in _TABLES:
                cursor.execute(stmt)
