"""Multi-algorithm clustering lab — Phase C.

Runs several clustering algorithms over two independent feature sets and stores
every result so the dashboard can compare them:

  * **tabular** — the structured specs + description vector built from
    `parsed_offers` by `ml/similarity.py` (always available)
  * **visual**  — `offer_features.clip_vector_visual` (skipped until the visual
    pipeline / Phase B populates it)

For each (feature_set × algorithm) "run" it writes:
  * `clusters`                    — one summary row per cluster (label, avg price,
                                    count, silhouette, k, is_primary)
  * `offer_cluster_assignments`   — one row per offer (cluster_id + PCA/UMAP/t-SNE
                                    2D coords for plotting)

Clustering input is a PCA-reduced matrix (≤10D); the three 2D projections are for
plotting ONLY (t-SNE in particular is never used as clustering input). Parametric
algorithms auto-select k via silhouette score.

The **primary run** (`tabular` + `kmeans`) is flagged `is_primary=1` — the
dashboard list filter, offer-detail cluster panel and stats use it.
"""
import logging
import sqlite3
from collections import Counter

import numpy as np

log = logging.getLogger(__name__)

PRIMARY_RUN = ("tabular", "kmeans")
K_RANGE = range(3, 13)            # candidate cluster counts for parametric algos
MIN_OFFERS = 10                  # need at least this many to bother clustering
CLUSTER_INPUT_DIMS = 10          # PCA target dims for the clustering input matrix


def run_id_for(feature_set: str, algorithm: str) -> str:
    return f"{feature_set}_{algorithm}"


# ── algorithms ────────────────────────────────────────────────────────────────
def _build_clusterers() -> dict[str, tuple[bool, object]]:
    """{name: (needs_k, fn(X, k) -> labels)}. hdbscan is optional."""
    from sklearn.cluster import (
        AffinityPropagation, Birch, DBSCAN, KMeans, MiniBatchKMeans,
    )

    clusterers: dict[str, tuple[bool, object]] = {
        "kmeans": (True, lambda X, k: KMeans(
            n_clusters=k, random_state=42, n_init=10).fit_predict(X)),
        "minibatch_kmeans": (True, lambda X, k: MiniBatchKMeans(
            n_clusters=k, random_state=42, n_init=10).fit_predict(X)),
        "birch": (True, lambda X, k: Birch(n_clusters=k).fit_predict(X)),
        "dbscan": (False, lambda X, _k: DBSCAN(
            eps=_auto_eps(X), min_samples=5).fit_predict(X)),
        "affinity_propagation": (False, lambda X, _k: AffinityPropagation(
            random_state=42).fit_predict(X)),
    }
    try:
        from hdbscan import HDBSCAN
        clusterers["hdbscan"] = (False, lambda X, _k: HDBSCAN(
            min_cluster_size=max(5, len(X) // 30)).fit_predict(X))
    except ImportError:
        log.warning("hdbscan not installed — skipping that algorithm")
    return clusterers


def _auto_eps(X: np.ndarray, k: int = 5) -> float:
    """k-distance elbow heuristic: median distance to the k-th nearest neighbour."""
    from sklearn.neighbors import NearestNeighbors
    k = min(k, len(X) - 1)
    nbrs = NearestNeighbors(n_neighbors=k).fit(X)
    distances, _ = nbrs.kneighbors(X)
    return max(float(np.median(distances[:, -1])), 1e-6)


# ── dimensionality reduction ──────────────────────────────────────────────────
def _pca_reduce(X: np.ndarray, n: int) -> np.ndarray:
    from sklearn.decomposition import PCA
    n = min(n, X.shape[1], len(X) - 1)
    if n < 2:
        return X
    return PCA(n_components=n, random_state=42).fit_transform(X)


def _project_2d(X: np.ndarray) -> dict[str, np.ndarray | None]:
    """PCA/UMAP/t-SNE 2D projections for plotting. Optional ones degrade to None."""
    n = len(X)
    out: dict[str, np.ndarray | None] = {"pca": None, "umap": None, "tsne": None}

    try:
        from sklearn.decomposition import PCA
        out["pca"] = PCA(n_components=2, random_state=42).fit_transform(X)
    except Exception as exc:                       # pragma: no cover - defensive
        log.warning("PCA 2D failed: %s", exc)

    try:
        import umap
        out["umap"] = umap.UMAP(
            n_components=2, random_state=42, metric="cosine",
            n_neighbors=min(15, n - 1), min_dist=0.1,
        ).fit_transform(X)
    except Exception as exc:
        log.warning("UMAP 2D unavailable/failed: %s", exc)

    try:
        from sklearn.manifold import TSNE
        out["tsne"] = TSNE(
            n_components=2, random_state=42, init="pca",
            perplexity=min(30, max(5, n - 1)),
        ).fit_transform(X)
    except Exception as exc:
        log.warning("t-SNE 2D failed: %s", exc)

    return out


# ── label / silhouette selection ──────────────────────────────────────────────
def auto_select_k(X: np.ndarray, fn) -> tuple[int | None, np.ndarray | None, float | None]:
    """Try K_RANGE, return (best_k, best_labels, best_silhouette) by silhouette."""
    from sklearn.metrics import silhouette_score
    best = (None, None, -1.0)
    for k in K_RANGE:
        if k >= len(X):
            break
        labels = fn(X, k)
        if len(set(labels)) < 2:
            continue
        score = silhouette_score(X, labels, metric="euclidean")
        if score > best[2]:
            best = (k, labels, float(score))
    return best if best[1] is not None else (None, None, None)


def run_no_k_clusterer(X: np.ndarray, fn) -> tuple[np.ndarray, float | None]:
    """Run a parameter-free clusterer; silhouette excludes noise (-1) points."""
    from sklearn.metrics import silhouette_score
    labels = fn(X, None)
    mask = labels != -1
    if mask.sum() < 2 or len(set(labels[mask])) < 2:
        return labels, None
    return labels, float(silhouette_score(X[mask], labels[mask], metric="euclidean"))


def _most_common(values: list) -> str | None:
    filtered = [v for v in values if v]
    return Counter(filtered).most_common(1)[0][0] if filtered else None


def generate_tabular_label(members: list[dict]) -> str:
    """Name a tabular cluster from its average price, most common frame
    material, and average weight — the same three figures shown in its
    quick-stats card. No tier/groupset wording (that concept is gone)."""
    material = _most_common([m.get("params", {}).get("Materiał ramy") for m in members])
    prices = [m["price"] for m in members if m.get("price")]
    avg = int(sum(prices) / len(prices)) if prices else 0
    weights = [
        m.get("parsed", {}).get("weight_kg") for m in members
        if m.get("parsed", {}).get("weight_kg")
    ]
    avg_weight = sum(weights) / len(weights) if weights else None

    parts = [p for p in (material and material.capitalize(),) if p]
    parts.append(f"~{avg:,} PLN".replace(",", " "))
    if avg_weight:
        parts.append(f"{avg_weight:.1f}kg")
    return " · ".join(parts)


def cluster_quick_stats(members: list[dict]) -> tuple[float | None, float | None, str | None]:
    """(price_stddev, avg_weight_kg, frame_material_breakdown_json) for one
    cluster's members. Weight/material data only exists for the tabular
    feature set (members carry `offer["parsed"]`/`offer["params"]`); visual
    members are bare {id, price} dicts, so this returns (stddev, None, None)
    for them."""
    import json
    import statistics

    prices = [m["price"] for m in members if m.get("price")]
    price_stddev = statistics.pstdev(prices) if len(prices) >= 2 else None

    weights = [
        m.get("parsed", {}).get("weight_kg") for m in members
        if m.get("parsed", {}).get("weight_kg")
    ]
    avg_weight_kg = sum(weights) / len(weights) if weights else None

    materials = [m.get("params", {}).get("Materiał ramy") for m in members if m.get("params")]
    materials = [mat.capitalize() for mat in materials if mat]
    breakdown = dict(Counter(materials)) if materials else None
    breakdown_json = json.dumps(breakdown, ensure_ascii=False) if breakdown else None

    return price_stddev, avg_weight_kg, breakdown_json


def generate_visual_label(members: list[dict], cluster_id: int) -> str:
    prices = [m["price"] for m in members if m.get("price")]
    avg = int(sum(prices) / len(prices)) if prices else 0
    price_str = f"~{avg:,} PLN".replace(",", " ")
    if cluster_id == -1:
        return f"Unclustered (visual outliers) · {price_str}"
    return f"Visual group {cluster_id} · {len(members)} bikes · {price_str}"


# ── persistence ───────────────────────────────────────────────────────────────
def _write_run(
    conn: sqlite3.Connection,
    feature_set: str,
    algorithm: str,
    offers: list[dict],
    labels: np.ndarray,
    coords: dict[str, np.ndarray | None],
    silhouette: float | None,
    k_selected: int | None,
) -> None:
    from scraper.writer import upsert_cluster, upsert_cluster_assignment

    rid = run_id_for(feature_set, algorithm)
    is_primary = (feature_set, algorithm) == PRIMARY_RUN

    def _xy(name: str, i: int) -> tuple[float | None, float | None]:
        arr = coords.get(name)
        if arr is None:
            return None, None
        return float(arr[i][0]), float(arr[i][1])

    for i, offer in enumerate(offers):
        px, py = _xy("pca", i)
        ux, uy = _xy("umap", i)
        tx, ty = _xy("tsne", i)
        upsert_cluster_assignment(
            conn, offer["id"], rid, int(labels[i]),
            {"pca_x": px, "pca_y": py, "umap_x": ux, "umap_y": uy,
             "tsne_x": tx, "tsne_y": ty},
        )

    for cid in sorted(set(int(x) for x in labels)):
        members = [offers[i] for i in range(len(offers)) if int(labels[i]) == cid]
        prices = [m["price"] for m in members if m.get("price")]
        avg_price = sum(prices) / len(prices) if prices else None
        label = (generate_tabular_label(members) if feature_set == "tabular"
                 else generate_visual_label(members, cid))
        price_stddev, avg_weight_kg, frame_material_breakdown = cluster_quick_stats(members)
        upsert_cluster(
            conn, rid, feature_set, algorithm, cid, label, avg_price,
            len(members), silhouette, k_selected, is_primary,
            price_stddev=price_stddev, avg_weight_kg=avg_weight_kg,
            frame_material_breakdown=frame_material_breakdown,
        )


# ── orchestration ─────────────────────────────────────────────────────────────
def _load_feature_set(conn: sqlite3.Connection, feature_set: str):
    from ml.similarity import load_vectors_for_clustering, load_visual_for_clustering
    if feature_set == "tabular":
        return load_vectors_for_clustering(conn)
    if feature_set == "visual":
        return load_visual_for_clustering(conn)
    raise ValueError(f"unknown feature_set: {feature_set}")


def run_all_clustering(
    conn: sqlite3.Connection,
    feature_sets: list[str] | None = None,
    algorithms: list[str] | None = None,
) -> dict:
    """Rebuild all clustering runs. Returns {run_id: n_clusters} for what ran."""
    from scraper.writer import clear_clusters

    feature_sets = feature_sets or ["tabular", "visual"]
    clusterers = _build_clusterers()
    clear_clusters(conn)

    summary: dict[str, int] = {}
    for fs in feature_sets:
        offers, X = _load_feature_set(conn, fs)
        if X is None or len(offers) < MIN_OFFERS:
            log.info("Feature set '%s': %d offers — skipped (need ≥%d, or no data)",
                     fs, len(offers), MIN_OFFERS)
            continue

        log.info("Feature set '%s': %d offers, %d dims", fs, len(offers), X.shape[1])
        coords = _project_2d(X)
        Xc = _pca_reduce(X, CLUSTER_INPUT_DIMS)

        for algo, (needs_k, fn) in clusterers.items():
            if algorithms and algo not in algorithms:
                continue
            try:
                if needs_k:
                    k, labels, sil = auto_select_k(Xc, fn)
                else:
                    labels, sil = run_no_k_clusterer(Xc, fn)
                    k = None
            except Exception as exc:
                log.warning("  %s/%s failed: %s", fs, algo, exc)
                continue
            if labels is None:
                log.warning("  %s/%s produced no usable clustering", fs, algo)
                continue

            _write_run(conn, fs, algo, offers, labels, coords, sil, k)
            n_clusters = len([c for c in set(int(x) for x in labels) if c != -1])
            summary[run_id_for(fs, algo)] = n_clusters
            log.info("  %s/%s: %d clusters, silhouette=%s",
                     fs, algo, n_clusters, f"{sil:.3f}" if sil is not None else "n/a")

    conn.commit()
    log.info("Clustering lab done — %d runs written", len(summary))
    return summary


def list_available_runs(conn: sqlite3.Connection) -> list[dict]:
    """Distinct runs with metadata for the dashboard's run picker."""
    rows = conn.execute(
        """
        SELECT run_id, feature_set, algorithm,
               MAX(silhouette_score)                      AS silhouette,
               MAX(k_selected)                            AS k_selected,
               MAX(is_primary)                            AS is_primary,
               SUM(CASE WHEN cluster_id >= 0 THEN 1 ELSE 0 END) AS n_clusters,
               SUM(CASE WHEN cluster_id = -1 THEN count ELSE 0 END) AS noise_count,
               SUM(count)                                 AS total
        FROM clusters
        GROUP BY run_id, feature_set, algorithm
        ORDER BY feature_set, silhouette DESC
        """
    ).fetchall()
    cols = ("run_id", "feature_set", "algorithm", "silhouette", "k_selected",
            "is_primary", "n_clusters", "noise_count", "total")
    return [dict(zip(cols, r)) for r in rows]


def main() -> None:
    import argparse
    import logging as _logging
    import sys

    from decouple import config

    from scraper.writer import migrate_db, open_db

    _logging.basicConfig(
        level=_logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[_logging.StreamHandler(sys.stdout)],
    )

    parser = argparse.ArgumentParser(description="Multi-algorithm clustering lab")
    parser.add_argument("--feature-set", choices=["tabular", "visual"],
                        help="Limit to one feature set (default: both)")
    parser.add_argument("--algorithm", help="Limit to one algorithm")
    args = parser.parse_args()

    conn = open_db(config("DB_PATH"))
    migrate_db(conn)
    run_all_clustering(
        conn,
        feature_sets=[args.feature_set] if args.feature_set else None,
        algorithms=[args.algorithm] if args.algorithm else None,
    )


if __name__ == "__main__":
    main()
