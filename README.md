# Allegro Road Bikes Analyser

A personal data-science project that scrapes road bike listings from [Allegro.pl](https://allegro.pl), stores them in a local SQLite database, and exposes a Django dashboard for browsing, filtering, and price analysis.

Allegro is protected by DataDome WAF. The scraper uses [undetected-geckodriver](https://github.com/tinysnake/undetected-geckodriver) — a patched real Firefox build that removes the `navigator.webdriver` marker — to pass DataDome automatically. All offer data is extracted from the `__listing_StoreState` JSON blob embedded in each page — never from fragile CSS classes.

---

## Features

- Paginated scraper across all Allegro road-bike listings (no API key required)
- Offer detail storage: title, price, condition, seller, images, structured parameters
- CLIP image embeddings (ViT-B-32 via `open-clip-torch`) for each listing's primary photo
- Description text scraped from individual offer pages
- Django dashboard with filtering, price histogram, and offer detail views
- HTMX-powered partial updates — no JavaScript framework
- SQLite in WAL mode — zero infrastructure, survives restarts

---

## Tech Stack

| Layer | Technology |
|---|---|
| Scraping | Selenium + undetected-geckodriver (real Firefox) |
| HTTP fallback | requests + BeautifulSoup4 / lxml |
| Feature extraction | open-clip-torch (ViT-B-32), Pillow, NumPy |
| Database | SQLite (WAL mode), raw `sqlite3` module |
| Dashboard | Django 5, HTMX, Chart.js |
| Retry logic | tenacity |
| Config | python-decouple + `.env` file |
| Tests | pytest, pytest-django, factory-boy |
| Python env | conda (`automl`) |

---

## Project Structure

```
.
├── scraper/             # Allegro listing scraper (undetected Firefox)
│   ├── browser.py       # WebDriver factory — undetected-geckodriver + symlink fix
│   ├── extractor.py     # __listing_StoreState JSON parser
│   └── scheduler.py     # Main loop: warm-up → paginate → extract → write
├── processor/           # Feature extraction (runs after scrape)
│   ├── scraper.py       # DescriptionScraper — Selenium-based offer description fetch
│   ├── clip_worker.py   # CLIP image embeddings
│   └── pipeline.py      # Iterates unprocessed offers, writes to offer_features
├── dashboard/           # Django app
│   ├── models.py        # ORM models (mirrors SQLite schema)
│   ├── views.py         # List, detail, AJAX chart/stats views
│   ├── urls.py
│   └── templates/
├── downloader/          # SQLite writer shared by scraper and processor
│   └── writer.py        # open_db, init_db, upsert_offer, upsert_parameters, upsert_images
├── ml/                  # Future: price prediction + anomaly detection (Phase 4)
├── db/                  # SQLite database lives here (gitignored)
├── bikes_project/       # Django project settings
├── manage.py
├── requirements.txt
├── .env.example
└── CLAUDE.md            # Developer instructions
```

---

## Setup

### 1. Activate the conda environment

```bash
conda activate automl
```

All commands below assume the `automl` env is active. If you need to install it from scratch:

```bash
pip install -r requirements.txt
```

### 2. Configure environment variables

```bash
cp .env.example .env
```

Open `.env` and at minimum set `DJANGO_SECRET_KEY` to a long random string. All other defaults work out of the box for local use.

```env
SCRAPE_QUERY=rower szosowy          # Allegro search query
SCRAPE_SLEEP_MIN=3                  # Min seconds between pages
SCRAPE_SLEEP_MAX=7                  # Max seconds between pages
# MAX_PAGES=5                       # Uncomment to cap pages for testing

DB_PATH=db/bikes.db

CLIP_MODEL=ViT-B-32
CLIP_PRETRAINED=openai

DJANGO_SECRET_KEY=change-me-to-a-long-random-string
DJANGO_DEBUG=True
DJANGO_ALLOWED_HOSTS=localhost,127.0.0.1
```

### 3. Initialise the database

```bash
mkdir -p db
python manage.py migrate
```

---

## Running Each Component

### Scraper

The scraper uses `undetected-geckodriver` — a patched real Firefox build that removes the `navigator.webdriver` marker — to bypass DataDome automatically. A Firefox window opens on your desktop while the scrape runs and closes when it's done.

```bash
python scraper/scheduler.py
```

Cap pages during development:

```bash
python scraper/scheduler.py --max-pages 5
```

The scraper warms up on the Allegro homepage first, then paginates through listing pages with 3–7 second delays between requests.

---

### Processor

Picks up offers not yet in `offer_features`, fetches description text, and generates CLIP image embeddings. Run after the scraper.

```bash
python processor/pipeline.py
```

On first run, CLIP downloads the ViT-B-32 model (~500 MB) to `~/.cache/open_clip`. This only happens once.

---

### Dashboard

```bash
python manage.py runserver
```

Open [http://localhost:8000](http://localhost:8000).

Features:
- Browse and filter all listings by condition, price range, and title search
- Offer detail page with images, parameters, and extracted features
- Price distribution histogram (Chart.js, updated via HTMX)

---

### Tests

```bash
pytest tests/ -v
```

---

## Database Schema

**`offers`** — one row per listing  
`id, title, price, currency, seller_id, seller_login, condition, listing_type, offer_url, thumbnail_url, end_time, fetched_at, raw_json`

**`offer_parameters`** — structured key/value pairs from each offer  
`offer_id, name, value`

**`offer_images`** — all image URLs per offer  
`offer_id, url, position`

**`offer_features`** — populated by the processor  
`offer_id, clip_vector (BLOB), description_text, extracted_groupset, extracted_frame_material, price_predicted, anomaly_score`

SQLite shell access:

```bash
sqlite3 db/bikes.db
```

---

## Typical Workflow

```
# Weekly refresh
python scraper/scheduler.py       # scrape all listings (~30–60 min for full run)
python processor/pipeline.py      # extract descriptions + CLIP embeddings
python manage.py runserver        # explore results at http://localhost:8000
```

The scraper upserts — re-running is always safe.
