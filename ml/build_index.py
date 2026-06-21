"""CLI: build FAISS similarity index + UMAP clusters.

Usage:
    python ml/build_index.py                     # both similarity + clusters
    python ml/build_index.py --only-similarity
    python ml/build_index.py --only-clusters
    python ml/build_index.py --n-clusters 6
"""
import argparse
import logging
import sys

from decouple import config

from scraper.writer import migrate_db, open_db

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build similarity (specs + visual) + the clustering lab"
    )
    parser.add_argument("--only-similarity", action="store_true")
    parser.add_argument("--only-clusters",   action="store_true")
    args = parser.parse_args()

    conn = open_db(config("DB_PATH"))
    migrate_db(conn)

    run_sim     = not args.only_clusters
    run_cluster = not args.only_similarity

    if run_sim:
        from ml.similarity import build_similarity_index, build_visual_similarity
        n = build_similarity_index(conn)
        log.info("Specs similarity: %d offers indexed", n)
        nv = build_visual_similarity(conn)
        log.info("Visual similarity: %d offers indexed", nv)

    if run_cluster:
        from ml.cluster_lab import run_all_clustering
        runs = run_all_clustering(conn)
        log.info("Clustering lab: %d runs (%s)", len(runs), ", ".join(runs) or "none")

    conn.close()


if __name__ == "__main__":
    main()
