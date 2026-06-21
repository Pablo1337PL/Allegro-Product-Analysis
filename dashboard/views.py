import json

from django.conf import settings
from django.db import connection
from django.db.models import Avg, Count, F, Min, Max
from django.http import JsonResponse
from django.shortcuts import get_object_or_404, render
from django.views import View
from django.views.generic import DetailView, ListView, TemplateView

from scraper import spec_columns

from .models import Offer, OfferFeature, OfferImage, ParsedOffer

# The "primary" clustering run that drives the inline cluster UI (list filter,
# offer-detail panel, stats). It is the tabular KMeans run, flagged is_primary=1
# by ml/cluster_lab.py. The cluster lab pages expose all runs for comparison.
PRIMARY_RUN_ID = "tabular_kmeans"


# ── raw-SQL helpers for the run-based cluster tables ──────────────────────────
def _fetch_dicts(sql: str, params: list) -> list[dict]:
    with connection.cursor() as cursor:
        cursor.execute(sql, params)
        cols = [c[0] for c in cursor.description]
        return [dict(zip(cols, row)) for row in cursor.fetchall()]


def _decode_categories(enc_column: str) -> dict[int, str]:
    """encoded_int -> raw_value map for one parsed_offers `*_enc` column, read
    from `category_encoders` (the live source of truth, not the stale
    db/encoder_mappings.json export). Empty dict if nothing's been encoded yet."""
    rows = _fetch_dicts(
        "SELECT raw_value, encoded_int FROM category_encoders WHERE column_name = %s",
        [enc_column],
    )
    return {r["encoded_int"]: r["raw_value"] for r in rows}


def _clusters_for_run(run_id: str) -> list[dict]:
    """Per-cluster summaries for a run (noise cluster -1 last).

    `frame_material_breakdown` is decoded from its stored JSON into a
    `{material: count}` dict (tabular runs only; None for visual runs).
    """
    rows = _fetch_dicts(
        """
        SELECT cluster_id AS id, label, avg_price, price_stddev,
               avg_weight_kg, frame_material_breakdown, count
        FROM clusters
        WHERE run_id = %s AND count > 0
        ORDER BY (cluster_id = -1), cluster_id
        """,
        [run_id],
    )
    for r in rows:
        breakdown = r.get("frame_material_breakdown")
        r["frame_material_breakdown"] = json.loads(breakdown) if breakdown else None
    return rows


def _cluster_offer_ids(run_id: str, cluster_id: int) -> list[str]:
    return [
        r["offer_id"]
        for r in _fetch_dicts(
            "SELECT offer_id FROM offer_cluster_assignments "
            "WHERE run_id = %s AND cluster_id = %s",
            [run_id, cluster_id],
        )
    ]


def _offer_cluster(run_id: str, offer_id: str) -> dict | None:
    rows = _fetch_dicts(
        """
        SELECT a.cluster_id AS id, c.label, c.avg_price, c.count
        FROM offer_cluster_assignments a
        JOIN clusters c ON c.run_id = a.run_id AND c.cluster_id = a.cluster_id
        WHERE a.run_id = %s AND a.offer_id = %s
        """,
        [run_id, offer_id],
    )
    return rows[0] if rows else None


def _offer_cluster_avg_map(run_id: str) -> dict[str, float]:
    """offer_id -> the average price of the cluster it belongs to, for one run.

    Used to sort the listing by each offer's price *relative to its cluster*
    (a rough deal signal). Only clusters with a usable avg_price are included.
    """
    return {
        r["offer_id"]: r["avg_price"]
        for r in _fetch_dicts(
            """
            SELECT a.offer_id, c.avg_price
            FROM offer_cluster_assignments a
            JOIN clusters c ON c.run_id = a.run_id AND c.cluster_id = a.cluster_id
            WHERE a.run_id = %s AND c.avg_price IS NOT NULL AND c.avg_price > 0
            """,
            [run_id],
        )
    }


def _enc_options(enc_column: str) -> list[dict]:
    """Decoded {value, label} choices for one parsed_offers `*_enc` column,
    sorted by label, plus an explicit 'Unknown' (-1) bucket at the end."""
    decode = _decode_categories(enc_column)
    options = sorted(
        ({"value": k, "label": v} for k, v in decode.items()),
        key=lambda o: (o["label"] or "").lower(),
    )
    options.append({"value": -1, "label": "Unknown"})
    return options


def _list_runs() -> list[dict]:
    """All clustering runs with metadata, for the lab's run picker."""
    return _fetch_dicts(
        """
        SELECT run_id, feature_set, algorithm,
               MAX(silhouette_score) AS silhouette,
               MAX(k_selected)       AS k_selected,
               MAX(is_primary)       AS is_primary,
               SUM(CASE WHEN cluster_id >= 0 THEN 1 ELSE 0 END)      AS n_clusters,
               SUM(CASE WHEN cluster_id = -1 THEN count ELSE 0 END)  AS noise_count,
               SUM(count)            AS total
        FROM clusters
        GROUP BY run_id, feature_set, algorithm
        ORDER BY feature_set, silhouette DESC
        """,
        [],
    )


# ── Listing filter registry ───────────────────────────────────────────────────
# Every parsed_offers column is exposed as a filter. Numeric columns get a
# min/max range pair; categorical `*_enc` columns get a decoded dropdown; the
# boolean image attributes (which live on the Offer row itself) get a tri-state
# Any/Yes/No dropdown. `price` keeps its dedicated min_price/max_price params
# (it filters the Offer row, so non-bike listings stay reachable).
PARSED_NUMERIC_FIELDS: list[tuple[str, str]] = [
    ("frame_size_cm", "Frame size (cm)"),
    ("wheel_size_inch", 'Wheel size (")'),
    ("tire_width_mm", "Tire width (mm)"),
    ("gear_count", "Gear count"),
    ("weight_kg", "Weight (kg)"),
    ("max_load_kg", "Max load (kg)"),
    ("production_year", "Production year"),
    ("motor_power_w", "Motor power (W)"),
    ("range_km", "Range (km)"),
    ("charge_time_h", "Charge time (h)"),
    ("height_min_cm", "Rider height min (cm)"),
    ("height_max_cm", "Rider height max (cm)"),
    ("zero_shot_confidence", "Zero-shot confidence"),
]

PARSED_ENC_FIELDS: list[tuple[str, str]] = [
    ("frame_material_enc", "Frame material"),
    ("brake_type_enc", "Brake type"),
    ("gender_enc", "Gender"),
    ("brand_enc", "Brand"),
    ("color_enc", "Color"),
    ("bike_type_enc", "Bike type"),
    ("condition_enc", "Condition"),
]

# These columns are mirrored onto the Offer row, so filter them there directly
# (no join through `parsed`, so listings without a parsed_offers row stay
# reachable when the attribute is set).
BOOL_FIELDS: list[tuple[str, str]] = [
    ("has_drop_handlebars", "Drop handlebars"),
    ("has_flat_handlebars", "Flat handlebars"),
    ("has_disc_brakes", "Disc brakes"),
    ("has_pedals", "Pedals"),
]

# Columns kept light for the in-Python cluster-avg sort path (avoids loading the
# big raw_json / detail_raw_json blobs for every filtered row).
_LIST_ONLY_FIELDS = (
    "id", "title", "price", "currency", "is_bike", "listing_type",
    "seller_login", "fetched_at", "thumbnail_url",
)


class OfferListView(ListView):
    model = Offer
    paginate_by = 50
    context_object_name = "offers"

    def get_template_names(self) -> list[str]:
        if self.request.htmx:
            return ["dashboard/_offer_rows.html"]
        return ["dashboard/offer_list.html"]

    def get_queryset(self):
        qs = Offer.objects.all()
        p = self.request.GET

        search = p.get("search", "").strip()
        if search:
            qs = qs.filter(title__icontains=search)

        bike = p.get("is_bike", "").strip()
        if bike == "1":
            qs = qs.filter(is_bike=1)
        elif bike == "0":
            qs = qs.filter(is_bike=0)
        elif bike == "null":
            qs = qs.filter(is_bike__isnull=True)

        try:
            qs = qs.filter(price__gte=float(p["min_price"]))
        except (KeyError, ValueError):
            pass
        try:
            qs = qs.filter(price__lte=float(p["max_price"]))
        except (KeyError, ValueError):
            pass

        cluster = p.get("cluster", "").strip()
        if cluster.lstrip("-").isdigit():
            # Filter by membership in the primary run's cluster.
            ids = _cluster_offer_ids(PRIMARY_RUN_ID, int(cluster))
            qs = qs.filter(id__in=ids)

        # Numeric parsed_offers columns → min/max range, joined through `parsed`.
        for field, _label in PARSED_NUMERIC_FIELDS:
            lo = p.get(f"{field}_min", "").strip()
            if lo:
                try:
                    qs = qs.filter(**{f"parsed__{field}__gte": float(lo)})
                except ValueError:
                    pass
            hi = p.get(f"{field}_max", "").strip()
            if hi:
                try:
                    qs = qs.filter(**{f"parsed__{field}__lte": float(hi)})
                except ValueError:
                    pass

        # Categorical *_enc columns → exact encoded-int match.
        for field, _label in PARSED_ENC_FIELDS:
            val = p.get(field, "").strip()
            if val.lstrip("-").isdigit():
                qs = qs.filter(**{f"parsed__{field}": int(val)})

        # Boolean image attributes (on the Offer row) → tri-state Any/Yes/No.
        for field, _label in BOOL_FIELDS:
            val = p.get(field, "").strip()
            if val == "1":
                qs = qs.filter(**{field: 1})
            elif val == "0":
                qs = qs.filter(**{field: 0})

        qs = qs.distinct()

        sort = p.get("sort", "").strip()
        if sort in ("cluster_asc", "cluster_desc"):
            return self._sorted_by_cluster_avg(qs, descending=sort == "cluster_desc")
        if sort == "price_asc":
            return qs.order_by(F("price").asc(nulls_last=True))
        if sort == "price_desc":
            return qs.order_by(F("price").desc(nulls_last=True))
        return qs  # default: Offer.Meta ordering (-fetched_at)

    def _sorted_by_cluster_avg(self, qs, *, descending: bool) -> list:
        """Order offers by price ÷ their primary-cluster average price.

        Done in Python (the cluster tables aren't ORM-mapped) and returned as a
        list — Django's Paginator handles a list just like a queryset. Offers
        with no price or no cluster average are appended last in default order.
        Each offer carries a `vs_avg_*` annotation for the row template.
        """
        avg_map = _offer_cluster_avg_map(PRIMARY_RUN_ID)
        offers = list(qs.only(*_LIST_ONLY_FIELDS))
        for o in offers:
            avg = avg_map.get(o.id)
            if o.price and avg:
                pct = round((o.price / avg - 1) * 100)
                o.vs_avg_pct = pct
                o.vs_avg_label = f"{pct:+d}% vs cluster"
                o.vs_avg_cheap = pct < 0
            else:
                o.vs_avg_pct = None

        rated = [o for o in offers if o.vs_avg_pct is not None]
        unrated = [o for o in offers if o.vs_avg_pct is None]
        rated.sort(key=lambda o: o.vs_avg_pct, reverse=descending)
        return rated + unrated

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        p = self.request.GET
        context["clusters"] = _clusters_for_run(PRIMARY_RUN_ID)
        context["current_sort"] = p.get("sort", "")
        context["numeric_filters"] = [
            {"field": f, "label": label,
             "min": p.get(f"{f}_min", ""), "max": p.get(f"{f}_max", "")}
            for f, label in PARSED_NUMERIC_FIELDS
        ]
        context["enc_filters"] = [
            {"field": f, "label": label, "current": p.get(f, ""),
             "options": _enc_options(f)}
            for f, label in PARSED_ENC_FIELDS
        ]
        context["bool_filters"] = [
            {"field": f, "label": label, "current": p.get(f, "")}
            for f, label in BOOL_FIELDS
        ]
        return context


def _llm_evidence(offer_id: str) -> dict[str, str]:
    """column_name -> evidence span for spec values the LLM extractor wrote
    (action filled/overwritten). Drives the per-parameter 'AI · found in: …'
    annotation on the detail page (§3 — evidence span only, no explanation)."""
    return {
        r["column_name"]: r["evidence"]
        for r in _fetch_dicts(
            "SELECT column_name, evidence FROM offer_spec_extractions "
            "WHERE offer_id = %s AND action IN ('filled', 'overwritten') "
            "AND evidence IS NOT NULL AND evidence != ''",
            [offer_id],
        )
    }


def _offer_parameters(offer) -> list[dict]:
    """Build the detail-page parameter rows from the offer's spec_* columns,
    annotating any value the LLM extractor sourced with its evidence span."""
    evidence = _llm_evidence(offer.id)
    rows = []
    for col, label in spec_columns.column_labels().items():
        value = getattr(offer, col, None)
        if value not in (None, ""):
            rows.append({"name": label, "value": value, "ai_evidence": evidence.get(col)})
    return rows


class OfferDetailView(DetailView):
    model = Offer
    template_name = "dashboard/offer_detail.html"
    context_object_name = "offer"

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        offer = self.object
        # Full-definition gallery images only (is_thumbnail=0).
        context["images"] = OfferImage.objects.filter(offer=offer, is_thumbnail=0)
        context["parameters"] = _offer_parameters(offer)

        features = None
        try:
            features = offer.features
        except OfferFeature.DoesNotExist:
            pass
        context["features"] = features

        cluster = _offer_cluster(PRIMARY_RUN_ID, offer.id)
        if cluster:
            context["cluster"] = cluster
            if offer.price and cluster["avg_price"]:
                diff_pct = (offer.price / cluster["avg_price"] - 1) * 100
                context["price_vs_avg"] = round(diff_pct, 1)

        return context


class SimilarOffersView(View):
    """HTMX partial: top-5 similar offers for one comparison engine.

    ?type=specs (default) → 'Similar specs'; ?type=visual → 'Similar looking'.
    Raw SQL because offer_similarity has a composite PK.
    """

    def get(self, request, pk):
        get_object_or_404(Offer, pk=pk)
        ctype = request.GET.get("type", "specs")
        if ctype not in ("specs", "visual"):
            ctype = "specs"

        similarities = _fetch_dicts(
            """
            SELECT s.similar_id, s.score, s.rank,
                   o.title, o.price, o.currency, o.thumbnail_url
            FROM offer_similarity s
            JOIN offers o ON o.id = s.similar_id
            WHERE s.offer_id = %s AND s.comparison_type = %s
            ORDER BY s.rank
            LIMIT 5
            """,
            [pk, ctype],
        )
        return render(
            request,
            "dashboard/_similar_panel.html",
            {"similarities": similarities, "comparison_type": ctype},
        )


class ClusterLabIndexView(TemplateView):
    """Run picker — all clustering runs grouped by feature set."""

    template_name = "dashboard/cluster_lab_index.html"

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        runs = _list_runs()
        context["tabular_runs"] = [r for r in runs if r["feature_set"] == "tabular"]
        context["visual_runs"] = [r for r in runs if r["feature_set"] == "visual"]
        return context


class ClusterLabRunView(TemplateView):
    """Single run — 2D scatter (reducer toggle) + cluster cards."""

    template_name = "dashboard/cluster_lab_run.html"

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        run_id = kwargs["run_id"]
        meta = next((r for r in _list_runs() if r["run_id"] == run_id), None)
        context["run_id"] = run_id
        context["meta"] = meta
        context["clusters"] = _clusters_for_run(run_id)
        return context


class ShowcasesView(TemplateView):
    """How visual and tabular features are built, on their own page (not
    bundled into a specific cluster run)."""

    template_name = "dashboard/showcases.html"

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        context["showcase"] = _load_manifest("manifest.json")
        context["spec_showcase"] = _load_manifest("specs_manifest.json")
        return context


def _load_manifest(name: str) -> list[dict] | None:
    """A precomputed showcase manifest under MEDIA_ROOT/showcase/, or None if it
    hasn't been generated yet (ml/build_showcase.py / ml/build_spec_showcase.py)."""
    manifest = settings.MEDIA_ROOT / "showcase" / name
    if not manifest.exists():
        return None
    try:
        return json.loads(manifest.read_text())
    except (ValueError, OSError):
        return None


class ClusterLabDataView(View):
    """JSON for a run's scatter plot: points carry all three 2D projections."""

    def get(self, request, run_id):
        rows = _fetch_dicts(
            """
            SELECT a.offer_id, o.title, o.price, o.thumbnail_url, a.cluster_id,
                   a.pca_x, a.pca_y, a.umap_x, a.umap_y, a.tsne_x, a.tsne_y
            FROM offer_cluster_assignments a
            JOIN offers o ON o.id = a.offer_id
            WHERE a.run_id = %s
            """,
            [run_id],
        )
        points = [
            {
                "id": r["offer_id"],
                "title": (r["title"] or "")[:60],
                "price": r["price"],
                "thumb": r["thumbnail_url"],
                "cluster": r["cluster_id"],
                "pca": [r["pca_x"], r["pca_y"]],
                "umap": [r["umap_x"], r["umap_y"]],
                "tsne": [r["tsne_x"], r["tsne_y"]],
            }
            for r in rows
        ]
        labels = {
            c["id"]: c["label"] or f"Cluster {c['id']}"
            for c in _clusters_for_run(run_id)
        }
        return JsonResponse({"points": points, "cluster_labels": labels})


class StatsView(TemplateView):
    template_name = "dashboard/stats.html"

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)

        agg = Offer.objects.filter(price__isnull=False, price__gt=0).aggregate(
            total=Count("id"),
            avg_price=Avg("price"),
            min_price=Min("price"),
            max_price=Max("price"),
        )
        context["agg"] = {k: round(v, 2) if v else 0 for k, v in agg.items()}

        by_condition = list(
            Offer.objects.values("condition")
            .annotate(count=Count("id"), avg=Avg("price"))
        )
        # Numeric grades ascending first (1..5), non-numeric (NULL/garbage) last.
        by_condition.sort(key=lambda r: (
            (0, int(r["condition"])) if r["condition"] and str(r["condition"]).isdigit()
            else (1, 0)
        ))
        context["by_condition"] = by_condition

        context["by_classification"] = list(
            Offer.objects.values("is_bike")
            .annotate(count=Count("id"))
            .order_by("is_bike")
        )

        context["clusters"] = _clusters_for_run(PRIMARY_RUN_ID)

        # Categorical *_enc breakdowns, decoded to human-readable labels.
        context["by_bike_type_enc"] = _enc_breakdown("bike_type_enc")
        context["by_gender_enc"] = _enc_breakdown("gender_enc")
        context["color_by_gender"] = _color_by_gender_breakdown()

        return context


def _enc_breakdown(enc_column: str) -> list[dict]:
    """Count of ParsedOffer rows per decoded label of one `*_enc` column."""
    decode = _decode_categories(enc_column)
    rows = (
        ParsedOffer.objects.values(enc_column)
        .annotate(count=Count("offer_id"))
        .order_by("-count")
    )
    return [
        {"label": decode.get(r[enc_column], "Unknown") if r[enc_column] != -1 else "Unknown",
         "count": r["count"]}
        for r in rows
    ]


def _color_by_gender_breakdown() -> dict:
    """color_enc x gender_enc counts, pivoted for a stacked bar chart:
    one bar per color, one stacked segment per gender."""
    color_decode = _decode_categories("color_enc")
    gender_decode = _decode_categories("gender_enc")
    rows = (
        ParsedOffer.objects.values("color_enc", "gender_enc")
        .annotate(count=Count("offer_id"))
    )
    genders = sorted({r["gender_enc"] for r in rows})
    colors = sorted({r["color_enc"] for r in rows})
    pivot = {(r["color_enc"], r["gender_enc"]): r["count"] for r in rows}
    return {
        "colors": [color_decode.get(c, "Unknown") if c != -1 else "Unknown" for c in colors],
        "genders": [gender_decode.get(g, "Unknown") if g != -1 else "Unknown" for g in genders],
        "series": [
            [pivot.get((c, g), 0) for c in colors]
            for g in genders
        ],
    }


def price_chart_data(request):
    prices = list(
        Offer.objects.filter(price__isnull=False, price__gt=0).values_list("price", flat=True)
    )
    if not prices:
        return JsonResponse({"labels": [], "data": []})

    bin_width = 500
    cap = 25000
    bins = cap // bin_width
    counts = [0] * (bins + 1)  # last slot is the "cap+" overflow bucket
    for p in prices:
        if p >= cap:
            counts[bins] += 1
        else:
            counts[int(p // bin_width)] += 1

    labels = [f"{i * bin_width}–{(i + 1) * bin_width}" for i in range(bins)]
    labels.append(f"{cap}+")
    return JsonResponse({"labels": labels, "data": counts})


def stats_partial(request):
    stats = Offer.objects.aggregate(total=Count("id"), avg_price=Avg("price"))
    total = stats["total"] or 0
    avg_price = round(stats["avg_price"] or 0)
    return JsonResponse({"total": total, "avg_price": avg_price})


# ── New /stats/ visualisations: histograms, scatter, boxplot, heatmap ────────

HISTOGRAM_FIELDS: dict[str, tuple[str, float]] = {
    # slug -> (ParsedOffer field, bin width)
    "frame_size_cm": ("frame_size_cm", 2),
    "wheel_size_inch": ("wheel_size_inch", 1),
    "tire_width_mm": ("tire_width_mm", 5),
    "production_year": ("production_year", 1),
}


def field_histogram_data(request, field):
    spec = HISTOGRAM_FIELDS.get(field)
    if spec is None:
        return JsonResponse({"error": "unknown field"}, status=404)
    model_field, bin_width = spec

    values = list(
        ParsedOffer.objects.filter(**{f"{model_field}__isnull": False}).values_list(model_field, flat=True)
    )
    if not values:
        return JsonResponse({"labels": [], "data": []})

    min_v, max_v = min(values), max(values)
    if min_v == max_v:
        return JsonResponse({"labels": [str(min_v)], "data": [len(values)]})

    bins = max(1, int((max_v - min_v) / bin_width) + 1)
    counts = [0] * bins
    for v in values:
        idx = min(int((v - min_v) / bin_width), bins - 1)
        counts[idx] += 1

    if bin_width == 1:
        labels = [str(int(min_v + i)) for i in range(bins)]
    else:
        labels = [f"{min_v + i * bin_width:g}–{min_v + (i + 1) * bin_width:g}" for i in range(bins)]
    return JsonResponse({"labels": labels, "data": counts})


SCATTER_PAIRS: dict[str, tuple[str, str | None]] = {
    # slug -> (x field, y field); y=None means "derived" (handled specially below)
    "price_vs_gear_count": ("price", "gear_count"),
    "price_vs_frame_size_cm": ("price", "frame_size_cm"),
    "price_vs_weight_kg": ("price", "weight_kg"),
    "frame_size_vs_height_min": ("frame_size_cm", "height_min_cm"),
    "frame_size_vs_height_mid": ("frame_size_cm", None),
    "frame_size_vs_height_max": ("frame_size_cm", "height_max_cm"),
}


def scatter_chart_data(request, pair_id):
    spec = SCATTER_PAIRS.get(pair_id)
    if spec is None:
        return JsonResponse({"error": "unknown pair"}, status=404)
    x_field, y_field = spec

    if pair_id == "frame_size_vs_height_mid":
        rows = ParsedOffer.objects.filter(
            frame_size_cm__isnull=False, height_min_cm__isnull=False, height_max_cm__isnull=False,
        ).values_list("frame_size_cm", "height_min_cm", "height_max_cm")
        points = [{"x": x, "y": (lo + hi) / 2} for x, lo, hi in rows]
    else:
        rows = ParsedOffer.objects.filter(
            **{f"{x_field}__isnull": False, f"{y_field}__isnull": False}
        ).values_list(x_field, y_field)
        points = [{"x": x, "y": y} for x, y in rows]

    return JsonResponse({"points": points})


BOXPLOT_FIELDS: dict[str, str] = {
    "frame_material_enc": "Frame material",
    "brake_type_enc": "Brake type",
}


def boxplot_chart_data(request, field):
    if field not in BOXPLOT_FIELDS:
        return JsonResponse({"error": "unknown field"}, status=404)
    decode = _decode_categories(field)

    rows = ParsedOffer.objects.filter(price__isnull=False).values_list(field, "price")
    groups: dict[int, list[float]] = {}
    for enc, price in rows:
        groups.setdefault(enc, []).append(price)

    enc_values = sorted(groups, key=lambda e: -len(groups[e]))
    labels = [decode.get(e, "Unknown") if e != -1 else "Unknown" for e in enc_values]
    data = [groups[e] for e in enc_values]
    return JsonResponse({"labels": labels, "data": data})


def pedals_handlebars_heatmap_data(request):
    rows = (
        ParsedOffer.objects.values("has_pedals", "has_drop_handlebars")
        .annotate(count=Count("offer_id"))
    )
    cells = [
        {"x": "Pedals" if r["has_pedals"] else "No pedals",
         "y": "Drop bars" if r["has_drop_handlebars"] else "No drop bars",
         "v": r["count"]}
        for r in rows
    ]
    return JsonResponse({"cells": cells})


CORRELATION_FIELDS = [
    "price", "frame_size_cm", "wheel_size_inch", "tire_width_mm", "gear_count",
    "weight_kg", "height_min_cm", "height_max_cm", "production_year",
    "frame_material_enc", "brake_type_enc", "gender_enc", "color_enc",
    "bike_type_enc", "condition_enc",
]


def correlation_heatmap_data(request):
    if ParsedOffer.objects.count() < 2:
        return JsonResponse({"columns": [], "matrix": []})

    import pandas as pd

    rows = list(ParsedOffer.objects.values(*CORRELATION_FIELDS))
    df = pd.DataFrame(rows, columns=CORRELATION_FIELDS).apply(pd.to_numeric, errors="coerce")
    corr = df.corr(min_periods=2)
    matrix = corr.fillna(0).values.tolist()
    return JsonResponse({"columns": CORRELATION_FIELDS, "matrix": matrix})
