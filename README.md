# Allegro Road Bikes Analyser

A personal data-science project that scrapes road bike listings from [Allegro.pl](https://allegro.pl), stores them in a local SQLite database, and exposes a Django dashboard for browsing, filtering, ML-based analysis, similarity search, and a multi-algorithm clustering lab.

If you want I can provide db/ , scraper/ , test/ directories: janp.taran@gmail.com. Just write me a reason.

---

## Features

- **Search scraper** over the road-bikes category with dynamic page-count discovery (skips paid `promoted` items)
- **Parallel thumbnail embedding**: `scraper/thumbnail_embedder.py` CLIP-embeds each listing thumbnail in a separate process running alongside the search scraper
- **`is_bike` classification**: right after the listing scrape, KMeans(k=5) on raw thumbnail CLIP embeddings splits the pool into 5 clusters — the cheapest-average cluster is labelled noise (`is_bike=0`), the other four are bikes — so the detail scrape can skip noise entirely
- **Detail scraper** (bikes only): description, full-resolution gallery, structured parameters, active/inactive status
- **Denormalised parameters**: each Allegro parameter is a wide `offers.spec_*` column (one source of truth in `scraper/spec_columns.py`), NULL where an offer lacks it — no long key/value table
- **Inline spec extraction**: during the detail scrape, a local LLM (Ollama) reads each offer's cleaned description and fills still-missing `spec_*` columns plus groupset name — extracted values are range/unit-validated (`scraper/value_validator.py`) before being written
- **Condition grading**: a local LLM (Ollama) reads the description and grades the bike's *actual* condition 1–5 (5 = like-new, 1 = broken) into `offers.condition`, and enriches the categorical `spec_stan` (mediocre - to be dropped prob)
- **CLIP image embeddings** (ViT-B-32) of each offer's primary photo, plus zero-shot image attributes (drop/flat handlebars, disc brakes, pedals)
- **Visual segmentation**: YOLOv8 detects the bike, SAM masks it, the crop is composited on white and embedded into a second "visual" CLIP vector
- **`parsed_offers`**: an ML-ready denormalised table built incrementally — numeric `spec_*` values copied straight across, categorical values mapped through a persisted, append-only `IncrementalCategoryEncoder` — the single feature source for similarity and clustering
- **Dual similarity engines**: FAISS top-5 neighbours per offer — *"Similar specs"* (from `parsed_offers`) and *"Similar looking"* (segmented visual embedding)
- **Clustering lab**: multiple algorithms (KMeans, MiniBatch, Birch, DBSCAN, HDBSCAN, AffinityPropagation) × two feature sets (tabular, visual), each a "run"; PCA/UMAP/t-SNE 2D projections for plotting; `tabular_kmeans` is the primary run that drives the inline dashboard UI
- **`run_pipeline.py`**: one command runs the whole thing overnight (scrape → detail → process → parsed_offers → similarity/clusters), with preflight checks for Ollama/DB
- **Django dashboard**: listings (filter/search, thumbnails), offer detail (gallery, params, ML panel, two similar-bike panels), cluster-lab run explorer, stats — HTMX partials, Chart.js histograms, Plotly cluster scatter

---

## Tech Stack

| Layer | Technology |
|---|---|
| Scraping | -- |
| Deciding if offer is bike | CLIP thumbnail embbedings + KMean(k=5) |
| Image embeddings | open-clip-torch (ViT-B-32), Pillow, NumPy (**pinned `<2`**) |
| Visual segmentation | ultralytics YOLOv8 + SAM2 |
| Text → structured | local Ollama (`qwen2.5:3b`) for inline spec extraction and condition grading |
| Similarity index | faiss-cpu (IndexFlatIP) |
| Clustering | kmeans, minibatch_kmeans, dbscan, hdbscan, birch, affinity_propagation |
| Database | SQLite (WAL mode), raw `sqlite3` module |
| Dashboard | Django 5, HTMX, Chart.js, Plotly |
| Retry logic | tenacity |
| Config | python-decouple + `.env` file |
| Tests | pytest, pytest-django, factory_boy |

---

## Project Structure

```
.
├── scraper/                  # 
│   ├── browser.py            # 
│   ├── extractor.py          # __listing_StoreState JSON parser for search pages
│   ├── scheduler.py          # Search loop: warm-up → paginate → extract → write
│   ├── thumbnail_embedder.py # CLIP-embeds listing thumbnails, run parallel to scheduler.py
│   ├── bike_classifier.py    # KMeans(k=5) is_bike decision on thumbnail CLIP embeddings, run after listing scrape
│   ├── detail_extractor.py   # facade box / itemprop parser for individual offer pages
│   ├── detail_scheduler.py   # Detail loop (bikes only): per-offer URL → extract → write, session cycling
│   ├── description_preprocessor.py # Clean offer description HTML before Ollama sees it
│   ├── value_validator.py    # Range/unit checks + fixes for spec_* values (structural + Ollama)
│   ├── ollama_common.py      # Shared low-level Ollama plumbing (availability check, generate, debug CSV)
│   ├── ollama_extractor.py   # Inline spec_*/groupset extraction during the detail scrape
│   ├── parsed_offers_builder.py # Builds parsed_offers + category_encoders (ML-ready table)
│   ├── writer.py             # open_db, init_db, migrate_db, upsert_* — shared DB layer
│   └── spec_columns.py       # Canonical param-name → offers.spec_* column registry
├── processor/                # Lazy feature extraction (DB-only, no browser)
│   ├── clip_worker.py        # CLIP embeddings + zero-shot image attributes
│   ├── condition_worker.py   # Ollama condition grade (1-5) + spec_stan enrich
│   ├── segmentation_worker.py  # YOLOv8 + SAM bike segmentation → visual embedding
│   └── pipeline.py           # 4-pass runner (clip→zero-shot→condition→segmentation)
├── ml/
│   ├── similarity.py         # Feature vectors (from parsed_offers) + FAISS top-5 (specs + visual)
│   ├── cluster_lab.py        # Multi-algorithm clustering lab (PCA/UMAP/t-SNE × N algos)
│   ├── build_index.py        # CLI: build similarity + clustering lab
│   ├── build_showcase.py     # CLI: visual image-processing showcase
│   └── build_spec_showcase.py # CLI: spec-extraction showcase
├── dashboard/                # Django app (models managed=False, mirrors raw schema)
│   ├── models.py · views.py · urls.py
│   └── templates/dashboard/
├── bikes_project/            # Django project settings
├── run_pipeline.py           # One command: scrape → detail → process → parsed_offers → ml
├── scripts/migrate_specs.py  # One-time: pivot offer_parameters → offers.spec_* (already run)
├── db/bikes.db               # SQLite database (gitignored)
├── tests/                    # pytest + pytest-django
├── manage.py · requirements.txt · .env.example
```

---

## Setup

### 1. Create environment and download requirements

```bash
pip install -r requirements.txt
```

### 2. Configure environment variables

```bash
cp .env.example .env
```

Edit `.env` — at minimum set `DJANGO_SECRET_KEY`. All other defaults work for local use (category URL, scrape delays, model names, paths). See `.env.example` for the full annotated template.

### 3. Initialise the database

```bash
mkdir -p db
python manage.py migrate
```

`scraper/writer.py` owns the raw schema (the Django models are `managed = False`). `init_db` + `migrate_db` are called from every scraper/processor entry point and are safe to re-run — new columns and tables are added idempotently.

---

## Pipeline: From Scrape to Dashboard

The full journey of an offer, from a listing on Allegro to a card on the dashboard. Each stage only touches rows that are missing its data, and every write is an upsert — so re-running any stage is always safe, and you can stop and resume at any point.

```
                                Allegro.pl
                                      │
        ┌─────────────────────────────┴─────────────────────────────┐
        │  1. SEARCH SCRAPE        scraper/scheduler.py              │
        │     paginate the category pages → parse                   │
        │     __listing_StoreState JSON → write offers (title,       │
        │     price, seller, thumbnail, listing params). is_bike     │
        │     stays NULL — decided next, from the thumbnail.         │
        │     scraper/thumbnail_embedder.py --watch runs in a        │
        │     SEPARATE PROCESS alongside this, CLIP-embedding each   │
        │     page's thumbnails as they're written.                  │
        └─────────────────────────────┬─────────────────────────────┘
                                      │  offers (listing-level) + clip_vector_thumb
        ┌─────────────────────────────┴─────────────────────────────┐
        │  1b. BIKE CLASSIFICATION scraper/bike_classifier.py        │
        │     KMeans(k=5) on raw thumbnail CLIP embeddings over      │
        │     every offer with one → the single cheapest-average     │
        │     cluster is is_bike=0 (noise), the other four are       │
        │     is_bike=1 (bikes). Re-runs on the full pool every       │
        │     pipeline run. No detail data, no Ollama, needed.       │
        └─────────────────────────────┬─────────────────────────────┘
                                      │  offers.is_bike decided
        ┌─────────────────────────────┴─────────────────────────────┐
        │  2. DETAIL SCRAPE (bikes only) scraper/detail_scheduler.py │
        │     visit /oferta/<id> → parse the `facade` JSON box →     │
        │     description, full-res gallery (offer_images), raw      │
        │     spec_* TEXT params, sold/stock, is_active. Hands the   │
        │     cleaned description to a background Ollama worker that │
        │     fills still-missing spec_* + groupset name (validated  │
        │     before writing). Session-cycles every ~200 offers to   │
        │     dodge DataDome's per-session limit. Skips is_bike=0.   │
        └─────────────────────────────┬─────────────────────────────┘
                                      │  bike offers fully populated
        ┌─────────────────────────────┴─────────────────────────────┐
        │  3. FEATURE PROCESSOR    processor/pipeline.py  (4 passes) │
        │     a. clip          → clip_vector  (ViT-B-32 embedding)   │
        │     b. zero-shot      → has_drop/flat/disc/pedals, …       │
        │     c. condition      → grade 1-5 + enrich spec_stan (LLM) │
        │     d. segmentation   → YOLO+SAM crop → clip_vector_visual │
        └─────────────────────────────┬─────────────────────────────┘
                                      │  offer_features populated
        ┌─────────────────────────────┴─────────────────────────────┐
        │  4. PARSED OFFERS    scraper/parsed_offers_builder.py      │
        │     numeric spec_* copied across; categorical values       │
        │     mapped through a persisted IncrementalCategoryEncoder  │
        │     → parsed_offers (the single ML feature source)         │
        └─────────────────────────────┬─────────────────────────────┘
                                      │  parsed_offers ready
        ┌─────────────────────────────┴─────────────────────────────┐
        │  5. SIMILARITY + CLUSTERS    ml/build_index.py             │
        │     specs vector  → FAISS → offer_similarity ('specs')     │
        │     visual vector → FAISS → offer_similarity ('visual')    │
        │     cluster lab: KMeans/Birch/DBSCAN/HDBSCAN/… × tabular   │
        │     & visual → clusters + offer_cluster_assignments        │
        │     (PCA/UMAP/t-SNE coords). tabular_kmeans = primary.     │
        └─────────────────────────────┬─────────────────────────────┘
                                      │  db/bikes.db ready
        ┌─────────────────────────────┴─────────────────────────────┐
        │  6. DASHBOARD            manage.py runserver               │
        │     listings · offer detail (params, ML panel, "Similar    │
        │     specs" + "Similar looking") · cluster lab · stats      │
        └────────────────────────────────────────────────────────────┘
```

### One-shot run

```bash
python run_pipeline.py
```

Runs all stages above in order (1 → 1b → 2 → 3 → 4 → 5) with preflight checks
(Ollama reachable + model pulled, DB writable). Useful flags: `--skip-scrape`,
`--skip-details`, `--skip-ml`, `--max-pages N` (cap the listing scrape for
testing), `--continue-on-error`, `--skip-preflight`.

Then explore:

```bash
python manage.py runserver   # http://localhost:8000
```

---

## Stage Reference

### 1. Search scraper (+ parallel thumbnail embedding)

Scrapes all listing pages in the road-bikes category and populates `offers`
with title, price, seller, thumbnail, and listing-level data. Stores **only
the first image** (small thumbnail, `is_thumbnail=1`). A real Firefox window
opens during the run.

`scraper/thumbnail_embedder.py --watch` runs as a **separate process
alongside** the scraper, CLIP-embedding each page's thumbnails
(`offer_features.clip_vector_thumb`) as they land rather than waiting for the
whole listing scrape to finish — `run_pipeline.py` starts/stops it for you.
`is_bike` itself is left `NULL` here — it's decided next (Stage 1b).

```bash
python scraper/scheduler.py
# Cap pages for testing (or set SCRAPER_MAX_PAGES in .env)
python scraper/scheduler.py --max-pages 5

# Run alongside the above in another terminal (or let run_pipeline.py do it):
python scraper/thumbnail_embedder.py --watch
# One-shot / standalone (no --watch):
python scraper/thumbnail_embedder.py
```

### 1b. Bike classification

`scraper/bike_classifier.py` decides `offers.is_bike` via **KMeans(k=5)** on
raw thumbnail CLIP embeddings (`clip_vector_thumb`) — no detail data, no
Ollama call, needed. It re-runs over the *entire* pool of offers with an
embedding every time (not just new ones), so it self-corrects as the dataset
grows; the single cluster with the lowest average price is labelled noise
(`is_bike=0`), the other four are bikes (`is_bike=1`). Before clustering it
self-drains any thumbnail the parallel embedder missed. An offer flipping
from `is_bike=1` to `0` has its `parsed_offers` row deleted immediately.
Skipped entirely when fewer than `MIN_OFFERS` (25, `BIKE_MIN_OFFERS` env)
offers have an embedding.

```bash
python scraper/bike_classifier.py
```

### 2. Detail scraper (bikes only)

Visits each `is_bike=1` (or not-yet-classified) offer's canonical page
(`/oferta/<id>`), extracts the description, **full-resolution** gallery
(written to `offer_images` with `is_thumbnail=0`), structured `spec_*`
parameters, and active/inactive status. Restarts the browser session every
~200 offers to stay under DataDome's rate limit. `is_bike=0` (noise) offers
are excluded from the queue entirely — see Stage 1b above.

`offers.spec_*` only ever stores the raw seller TEXT — there's no
unit-converted/typed companion column. Right after each offer's structured
data is written, the cleaned description
(`scraper/description_preprocessor.py`) is handed to a background
`OllamaExtractorWorker` (`scraper/ollama_extractor.py`) that fills whichever
`spec_*` columns are still NULL, plus `groupset_name`,
validating every extracted value through `scraper/value_validator.py` before
writing — and never overwriting an already-populated column. Degrades
gracefully (leaves columns NULL) if Ollama is unreachable.

```bash
python scraper/detail_scheduler.py
python scraper/detail_scheduler.py --max-offers 50   # testing cap
python scraper/detail_scheduler.py --recheck-days 3  # re-check stale active offers
```

### 3. Feature processor

Four idempotent passes (`--passes clip,zero-shot,condition,segmentation`),
each only touching offers missing that data. (Groupset extraction and
`spec_*` backfill from description text happen inline during the detail
scrape — see above — not here.)

1. **clip** — CLIP ViT-B-32 embedding of each bike's primary photo → `clip_vector`
2. **zero-shot** — binary image attributes (drop/flat handlebars, disc brakes, pedals) from the embedding
3. **condition** — a local LLM grades the bike's actual condition 1–5 → `offers.condition`, and enriches `spec_stan` (fill NULL / mark broken). Needs Ollama
4. **segmentation** — YOLOv8 + SAM crop composited on white → `clip_vector_visual`

```bash
python processor/pipeline.py                       # all passes
python processor/pipeline.py --passes clip
python processor/pipeline.py --passes condition
python processor/pipeline.py --passes segmentation
```

First run downloads CLIP (~500 MB) to `~/.cache/`, plus YOLOv8n (~6 MB) and
SAM2-base (~150 MB) for the segmentation pass. The **condition** pass (and
the detail scraper's inline spec extractor) use a local
[Ollama](https://ollama.com) server (`ollama pull qwen2.5:3b`) — both degrade
gracefully without it. Bike classification (stage 2b) does not use Ollama.

### 4. Build `parsed_offers`

Builds/updates the ML-ready `parsed_offers` table — must run after the
feature processor (it copies CLIP zero-shot columns) and before the
similarity/clustering build.

```bash
python scraper/parsed_offers_builder.py                  # incremental (default)
python scraper/parsed_offers_builder.py --full           # rebuild rows, keep encoders
python scraper/parsed_offers_builder.py --reset-encoders # wipe + rebuild from scratch
python scraper/parsed_offers_builder.py --show-encoders  # print current mappings
```

### 5. Similarity index + clustering lab

Builds two FAISS similarity engines (both reading from `parsed_offers`) and
the multi-algorithm clustering lab. Required before the dashboard can show
similar bikes or cluster pages.

```bash
python ml/build_index.py                    # everything
python ml/build_index.py --only-similarity
python ml/build_index.py --only-clusters

# Run just the clustering lab (both feature sets, or filter)
python ml/cluster_lab.py
python ml/cluster_lab.py --feature-set tabular
python ml/cluster_lab.py --algorithm hdbscan
```

Optionally build the **image-processing showcase** shown at the bottom of the
visual cluster-lab pages — 3 random bikes rendered across the segmentation
pipeline (original → YOLO detection → SAM mask → white-bg crop):

```bash
python ml/build_showcase.py            # 3 random bikes
python ml/build_showcase.py --count 4 --seed 7
```

And the tabular analogue — the **spec-extraction showcase** at the bottom of the
tabular cluster-lab pages, showing how 3 offers' free-text descriptions are read by
the same Ollama extractor used during the detail scrape, and validated, into
`spec_*` columns. Needs a local Ollama server — offers with too few extractable
fields are skipped:

```bash
python ml/build_spec_showcase.py
python ml/build_spec_showcase.py --count 4 --seed 7
```

### 6. Dashboard

```bash
python manage.py runserver
```

Open [http://localhost:8000](http://localhost:8000).

- **Listings** (`/`) — filter by condition, price, title, bike classification, or primary-run cluster; HTMX-paginated rows with thumbnails
- **Offer detail** (`/offer/<id>/`) — full-resolution gallery, `spec_*` parameters, ML analysis panel (cluster, price vs avg, attribute pills, groupset, condition grade), and **two** similar-bike panels: *"Similar specs"* and *"Similar looking"* (HTMX)
- **Cluster lab** (`/clusters/lab/`) — pick any `feature_set × algorithm` run; each run page shows a Plotly 2D scatter with a PCA/UMAP/t-SNE toggle and cluster summary cards. Hovering a dot previews the bike's thumbnail; clicking opens the offer. Visual runs render the image-processing showcase at the bottom (`ml/build_showcase.py`); tabular runs render the spec-extraction showcase (`ml/build_spec_showcase.py`). `/clusters/` redirects here.
- **Stats** (`/stats/`) — price histogram and aggregate charts

### 7. Tests

```bash
pytest tests/ -v
```

---

## Database Schema

SQLite, WAL mode. Schema owned by `scraper/writer.py`; the Django models are `managed = False`.

| Table | Purpose |
|---|---|
| **`offers`** | One row per listing. Core fields + `is_bike` (set by `bike_classifier.py`'s KMeans(k=5) pass on thumbnail CLIP embeddings, right after the listing scrape), `is_active`, scrape timestamps, `raw_json`/`detail_raw_json`, and the wide **`spec_*`** parameter columns (raw seller TEXT only — numeric conversion happens in `parsed_offers`). |
| **`offer_images`** | All image URLs. `is_thumbnail=1` = small list thumbnail; `is_thumbnail=0` = full-resolution gallery. |
| **`offer_features`** | Lazily populated: `clip_vector_thumb` (thumbnail embedding, `thumbnail_embedder.py`, drives `is_bike`), `clip_vector`/`clip_vector_visual` (gallery/segmented embeddings, the processor), segmentation status, zero-shot attributes, description extraction (groupset name). |
| **`parsed_offers`** | ML-ready denormalised row per bike offer — numeric `spec_*` copied across, categorical values encoded via a persisted `IncrementalCategoryEncoder`. The single feature source for similarity and clustering. |
| **`category_encoders`** | Backing store for `IncrementalCategoryEncoder`: `(column_name, raw_value) → encoded_int`, stable once assigned. |
| **`offer_similarity`** | Precomputed top-5 neighbours per offer **per engine** (`comparison_type` = `'specs'` \| `'visual'`). |
| **`clusters`** | Per-cluster summary, one row per (`run_id`, `cluster_id`). A run = `feature_set × algorithm`; `tabular_kmeans` is flagged `is_primary`. |
| **`offer_cluster_assignments`** | Per-offer membership for every run, plus PCA/UMAP/t-SNE 2D coords for plotting. |
| **`scrape_log`** | One row per scrape run. |

The `spec_*` columns replace a former `offer_parameters` long table (pivoted and dropped by the one-time `scripts/migrate_specs.py`). `scraper/spec_columns.py` is the single source of truth for the param-name → column mapping and generates the schema, the writer's pivot, and the dashboard labels.

```bash
sqlite3 db/bikes.db
```


# Main views

TODO