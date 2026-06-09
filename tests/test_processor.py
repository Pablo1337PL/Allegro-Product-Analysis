import io
import pickle
import sqlite3
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from PIL import Image

from downloader.writer import init_db, upsert_images, upsert_offer


# ── helpers ───────────────────────────────────────────────────────────────────


def _make_jpeg_bytes() -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (64, 64), color=(100, 150, 200)).save(buf, format="JPEG")
    return buf.getvalue()


_OFFER_A = {
    "id": "pipeline-001",
    "name": "Pipeline Test Bike",
    "condition": "USED",
    "sellingMode": {"format": "BUY_NOW", "price": {"amount": "2000", "currency": "PLN"}},
    "seller": {"id": "s1", "login": "seller1"},
    "url": "https://allegro.pl/oferta/pipeline-001",
    "images": [{"url": "https://img.example.com/p001.jpg"}],
    "publication": {"endingAt": None},
    "parameters": [],
}


def _setup_file_db(path: str, offers: list[dict], images: dict[str, list[str]] | None = None) -> None:
    conn = sqlite3.connect(path)
    init_db(conn)
    for offer in offers:
        upsert_offer(conn, offer)
    if images:
        for offer_id, urls in images.items():
            upsert_images(conn, offer_id, [{"url": u} for u in urls])
    conn.commit()
    conn.close()


# ── scraper ───────────────────────────────────────────────────────────────────


def test_scrape_description_extracts_named_section():
    from processor.scraper import scrape_description

    html = "<html><body><div data-box-name='description'>Carbon road bike in great shape.</div></body></html>"
    mock_resp = MagicMock()
    mock_resp.text = html
    mock_resp.raise_for_status = MagicMock()

    with patch("processor.scraper.requests.get", return_value=mock_resp), \
         patch("processor.scraper.time.sleep"):
        result = scrape_description("https://allegro.pl/oferta/fake-001")

    assert "Carbon road bike" in result


def test_scrape_description_fallback_to_paragraphs():
    from processor.scraper import scrape_description

    html = "<html><body><main><p>Rama karbonowa.</p><p>Rozmiar M.</p></main></body></html>"
    mock_resp = MagicMock()
    mock_resp.text = html
    mock_resp.raise_for_status = MagicMock()

    with patch("processor.scraper.requests.get", return_value=mock_resp), \
         patch("processor.scraper.time.sleep"):
        result = scrape_description("https://allegro.pl/oferta/fake-002")

    assert "karbonowa" in result


def test_scrape_description_returns_empty_on_error():
    from processor.scraper import scrape_description

    with patch("processor.scraper.requests.get", side_effect=Exception("Connection refused")):
        result = scrape_description("https://allegro.pl/oferta/unreachable")

    assert result == ""


def test_scrape_description_returns_empty_on_http_error():
    from processor.scraper import scrape_description

    mock_resp = MagicMock()
    mock_resp.raise_for_status.side_effect = Exception("404")

    with patch("processor.scraper.requests.get", return_value=mock_resp):
        result = scrape_description("https://allegro.pl/oferta/missing")

    assert result == ""


# ── clip worker ───────────────────────────────────────────────────────────────


def _make_clip_worker_with_mocks():
    import torch
    from processor.clip_worker import CLIPWorker

    real_output = torch.randn(1, 512)

    mock_model = MagicMock()
    mock_model.encode_image.return_value = real_output
    mock_model.encode_text.return_value = real_output

    # preprocess must accept a PIL Image and return a tensor-like
    mock_preprocess = MagicMock(
        return_value=MagicMock(
            unsqueeze=MagicMock(
                return_value=MagicMock(to=MagicMock(return_value=torch.randn(1, 3, 224, 224)))
            )
        )
    )

    with patch("processor.clip_worker.open_clip.create_model_and_transforms",
               return_value=(mock_model, None, mock_preprocess)), \
         patch("processor.clip_worker.open_clip.get_tokenizer", return_value=MagicMock()):
        worker = CLIPWorker()

    return worker


def test_clip_worker_encode_returns_correct_shape():
    worker = _make_clip_worker_with_mocks()

    mock_resp = MagicMock()
    mock_resp.content = _make_jpeg_bytes()
    mock_resp.raise_for_status = MagicMock()

    with patch("processor.clip_worker.requests.get", return_value=mock_resp):
        result = worker.encode_image_url("https://img.example.com/bike.jpg")

    assert result is not None
    assert result.shape == (512,)
    assert result.dtype == np.float32


def test_clip_worker_encode_returns_none_on_error():
    worker = _make_clip_worker_with_mocks()

    with patch("processor.clip_worker.requests.get", side_effect=Exception("timeout")):
        result = worker.encode_image_url("https://img.example.com/broken.jpg")

    assert result is None


# ── pipeline ──────────────────────────────────────────────────────────────────


def test_pipeline_skips_already_processed(tmp_path, monkeypatch):
    from processor.pipeline import run_pipeline

    db_file = str(tmp_path / "test.db")
    monkeypatch.setenv("DB_PATH", db_file)
    _setup_file_db(db_file, [_OFFER_A], {"pipeline-001": ["https://img.example.com/p001.jpg"]})

    conn = sqlite3.connect(db_file)
    conn.execute(
        "INSERT INTO offer_features (offer_id, description_text) VALUES (?, ?)",
        ("pipeline-001", "already done"),
    )
    conn.commit()
    conn.close()

    with patch("processor.pipeline.CLIPWorker") as MockCLIP, \
         patch("processor.pipeline.scrape_description") as mock_scrape:
        count = run_pipeline()

    assert count == 0
    MockCLIP.assert_not_called()
    mock_scrape.assert_not_called()


def test_pipeline_processes_new_offer(tmp_path, monkeypatch):
    from processor.pipeline import run_pipeline

    db_file = str(tmp_path / "test.db")
    monkeypatch.setenv("DB_PATH", db_file)
    _setup_file_db(db_file, [_OFFER_A], {"pipeline-001": ["https://img.example.com/p001.jpg"]})

    fake_vector = np.ones(512, dtype=np.float32)
    mock_clip = MagicMock()
    mock_clip.encode_image_url.return_value = fake_vector

    with patch("processor.pipeline.CLIPWorker", return_value=mock_clip), \
         patch("processor.pipeline.scrape_description", return_value="Nice carbon bike"):
        count = run_pipeline()

    assert count == 1
    conn = sqlite3.connect(db_file)
    row = conn.execute(
        "SELECT description_text, clip_vector FROM offer_features WHERE offer_id='pipeline-001'"
    ).fetchone()
    conn.close()
    assert row is not None
    assert row[0] == "Nice carbon bike"
    assert pickle.loads(row[1]).shape == (512,)


def test_pipeline_handles_missing_image(tmp_path, monkeypatch):
    from processor.pipeline import run_pipeline

    offer = {**_OFFER_A, "id": "pipeline-002", "url": "https://allegro.pl/oferta/p002"}
    db_file = str(tmp_path / "test.db")
    monkeypatch.setenv("DB_PATH", db_file)
    _setup_file_db(db_file, [offer])  # no images

    mock_clip = MagicMock()
    mock_clip.encode_image_url.return_value = None

    with patch("processor.pipeline.CLIPWorker", return_value=mock_clip), \
         patch("processor.pipeline.scrape_description", return_value=""):
        count = run_pipeline()

    assert count == 1
    conn = sqlite3.connect(db_file)
    row = conn.execute(
        "SELECT clip_vector FROM offer_features WHERE offer_id='pipeline-002'"
    ).fetchone()
    conn.close()
    assert row is not None
    assert row[0] is None
