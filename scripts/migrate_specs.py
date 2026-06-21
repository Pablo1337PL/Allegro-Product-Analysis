"""One-time migration: pivot offer_parameters into wide offers.spec_* columns,
then drop the offer_parameters long table.

Idempotent:
  * If offer_parameters no longer exists, the backfill/drop steps are skipped.
  * migrate_db only adds columns that are missing.

Run once against the live DB (back it up first — this DROPs a populated table):

    cp db/bikes.db db/bikes.db.bak
    ~/miniconda3/envs/automl/bin/python scripts/migrate_specs.py
"""
from __future__ import annotations

import os
import sqlite3
import sys

# Allow running as a script from the repo root.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scraper import spec_columns
from scraper.writer import migrate_db, open_db


def _table_exists(conn: sqlite3.Connection, name: str) -> bool:
    return conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (name,)
    ).fetchone() is not None


def backfill(conn: sqlite3.Connection) -> int:
    """Pivot every offer_parameters row into the offers spec_* columns.

    Returns the number of offers touched. Loads the long table once, groups by
    offer, and reuses the same pivot logic as the writer so typed columns and
    unknown-name slugging behave identically.
    """
    rows = conn.execute(
        "SELECT offer_id, name, value FROM offer_parameters"
    ).fetchall()

    by_offer: dict[str, list[dict]] = {}
    for offer_id, name, value in rows:
        by_offer.setdefault(offer_id, []).append({"name": name, "value": value})

    known = {r[1] for r in conn.execute("PRAGMA table_info(offers)")}
    touched = 0
    for offer_id, params in by_offer.items():
        values = spec_columns.pivot(params)
        if not values:
            continue
        for col in values:
            if col not in known:
                try:
                    conn.execute(f"ALTER TABLE offers ADD COLUMN {col} TEXT")
                except sqlite3.OperationalError:
                    pass
                known.add(col)
        set_clause = ", ".join(f"{col} = ?" for col in values)
        conn.execute(
            f"UPDATE offers SET {set_clause} WHERE id = ?",
            (*values.values(), offer_id),
        )
        touched += 1
    conn.commit()
    return touched


def main() -> None:
    db_path = os.environ.get("DB_PATH", "db/bikes.db")
    print(f"Database: {db_path}")
    conn = open_db(db_path)

    # 1. Ensure the spec_* columns exist.
    migrate_db(conn)
    print("Schema migrated (spec_* columns ensured).")

    if not _table_exists(conn, "offer_parameters"):
        print("offer_parameters already dropped — nothing to backfill. Done.")
        conn.close()
        return

    # 2. Backfill from the long table.
    param_rows = conn.execute("SELECT COUNT(*) FROM offer_parameters").fetchone()[0]
    offers_with_params = conn.execute(
        "SELECT COUNT(DISTINCT offer_id) FROM offer_parameters"
    ).fetchone()[0]
    touched = backfill(conn)
    print(
        f"Backfilled {param_rows} parameter rows across "
        f"{offers_with_params} offers → {touched} offers updated."
    )

    # 3. Sanity check: every offer that had params now has ≥1 non-null spec col.
    spec_cols = [c for c, _ in spec_columns.all_columns()]
    non_null = " OR ".join(f"{c} IS NOT NULL" for c in spec_cols)
    populated = conn.execute(
        f"SELECT COUNT(*) FROM offers WHERE {non_null}"
    ).fetchone()[0]
    print(f"Offers with at least one spec_* value: {populated}")
    if populated < touched:
        print("WARNING: fewer populated offers than touched — NOT dropping table.")
        conn.close()
        return

    # 4. Drop the long table.
    conn.execute("DROP TABLE offer_parameters")
    conn.commit()
    print("Dropped offer_parameters.")

    conn.close()
    print("Migration complete.")


if __name__ == "__main__":
    main()
