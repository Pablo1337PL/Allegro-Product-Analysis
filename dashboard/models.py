from django.db import models


class Offer(models.Model):
    id = models.TextField(primary_key=True)
    title = models.TextField(null=True, blank=True)
    price = models.FloatField(null=True, blank=True)
    currency = models.TextField(null=True, blank=True)
    seller_id = models.TextField(null=True, blank=True)
    seller_login = models.TextField(null=True, blank=True)
    seller_rating = models.FloatField(null=True, blank=True)
    seller_feedback_count = models.IntegerField(null=True, blank=True)
    condition = models.TextField(null=True, blank=True)
    listing_type = models.TextField(null=True, blank=True)
    offer_url = models.TextField(null=True, blank=True)
    thumbnail_url = models.TextField(null=True, blank=True)
    description = models.TextField(null=True, blank=True)
    views_count = models.IntegerField(null=True, blank=True)
    sold_count = models.IntegerField(null=True, blank=True)
    is_active = models.IntegerField(default=1)
    is_bike = models.IntegerField(null=True, blank=True)
    end_time = models.TextField(null=True, blank=True)
    fetched_at = models.TextField()
    detail_scraped_at = models.TextField(null=True, blank=True)
    detail_checked_at = models.TextField(null=True, blank=True)
    raw_json = models.TextField()
    detail_raw_json = models.TextField(null=True, blank=True)

    # ── Denormalised detail-page parameters (offers.spec_* columns) ───────────
    # One column per Allegro parameter name; see scraper/spec_columns.py for the
    # canonical name→column mapping. Raw seller text only — no typed/converted
    # companions; numeric conversion happens in parsed_offers (ParsedOffer below).
    spec_marka = models.TextField(null=True, blank=True)
    spec_model = models.TextField(null=True, blank=True)
    spec_rozmiar_ramy = models.TextField(null=True, blank=True)
    spec_rozmiar_kola = models.TextField(null=True, blank=True)
    spec_material_ramy = models.TextField(null=True, blank=True)
    spec_kolor = models.TextField(null=True, blank=True)
    spec_stan = models.TextField(null=True, blank=True)
    spec_plec = models.TextField(null=True, blank=True)
    spec_hamulce = models.TextField(null=True, blank=True)
    spec_rodzaj_przerzutki = models.TextField(null=True, blank=True)
    spec_korba = models.TextField(null=True, blank=True)
    spec_liczba_biegow = models.TextField(null=True, blank=True)
    spec_amortyzacja = models.TextField(null=True, blank=True)
    spec_pedaly = models.TextField(null=True, blank=True)
    spec_grupa_osprzetu = models.TextField(null=True, blank=True)
    spec_waga = models.TextField(null=True, blank=True)
    spec_rok_produkcji = models.TextField(null=True, blank=True)
    spec_wzrost_uzytkownika = models.TextField(null=True, blank=True)
    spec_stopien_zlozenia = models.TextField(null=True, blank=True)
    spec_szerokosc_opony = models.TextField(null=True, blank=True)
    spec_szerokosc_cale = models.TextField(null=True, blank=True)
    spec_wyposazenie_dodatkowe = models.TextField(null=True, blank=True)
    spec_rodzaj = models.TextField(null=True, blank=True)
    spec_stan_opakowania = models.TextField(null=True, blank=True)
    spec_waga_produktu_z_opakowaniem_jednostkowym = models.TextField(null=True, blank=True)
    spec_faktura = models.TextField(null=True, blank=True)
    spec_kod_producenta = models.TextField(null=True, blank=True)
    spec_ean_gtin = models.TextField(null=True, blank=True)
    spec_kod_taryfy_celnej = models.TextField(null=True, blank=True)
    spec_obciazenie_maksymalne = models.TextField(null=True, blank=True)
    spec_certyfikaty_zgodnosci = models.TextField(null=True, blank=True)
    spec_zawiera_baterie = models.TextField(null=True, blank=True)
    spec_moc_silnika = models.TextField(null=True, blank=True)
    spec_maksymalna_moc_znamionowa = models.TextField(null=True, blank=True)
    spec_akumulator = models.TextField(null=True, blank=True)
    spec_predkosc_maksymalna = models.TextField(null=True, blank=True)
    spec_umiejscowienie_silnika = models.TextField(null=True, blank=True)
    spec_zasieg = models.TextField(null=True, blank=True)
    spec_czas_ladowania = models.TextField(null=True, blank=True)
    spec_producent_silnika = models.TextField(null=True, blank=True)

    # ── Zero-shot image attributes (CLIP, Phase 2; written by the zero-shot pass) ─
    has_drop_handlebars = models.IntegerField(null=True, blank=True)
    has_flat_handlebars = models.IntegerField(null=True, blank=True)
    has_disc_brakes = models.IntegerField(null=True, blank=True)
    has_pedals = models.IntegerField(null=True, blank=True)
    zero_shot_confidence = models.FloatField(null=True, blank=True)

    class Meta:
        managed = False
        db_table = "offers"
        ordering = ["-fetched_at"]


class OfferImage(models.Model):
    id = models.AutoField(primary_key=True)
    offer = models.ForeignKey(Offer, on_delete=models.CASCADE, related_name="images", db_column="offer_id")
    url = models.TextField()
    position = models.IntegerField(default=0)
    is_thumbnail = models.IntegerField(default=0)

    class Meta:
        managed = False
        db_table = "offer_images"
        ordering = ["position"]
        unique_together = [("offer", "url")]


class OfferFeature(models.Model):
    offer = models.OneToOneField(
        Offer,
        on_delete=models.CASCADE,
        primary_key=True,
        related_name="features",
        db_column="offer_id",
    )
    clip_vector = models.BinaryField(null=True, blank=True)
    description_text = models.TextField(null=True, blank=True)
    extracted_groupset = models.TextField(null=True, blank=True)
    price_predicted = models.FloatField(null=True, blank=True)
    anomaly_score = models.FloatField(null=True, blank=True)
    # Zero-shot image attributes moved to the Offer model (offers table).
    # Visual embedding (Phase B — segmented detail image; NULL until built)
    clip_vector_visual = models.BinaryField(null=True, blank=True)
    # Description extraction (Phase 2b)
    desc_extracted_at = models.TextField(null=True, blank=True)

    class Meta:
        managed = False
        db_table = "offer_features"


# offer_similarity (now keyed by offer_id + comparison_type + rank) and the
# run-based clusters / offer_cluster_assignments tables use composite primary
# keys, so the dashboard reads them via raw SQL (see dashboard/views.py) rather
# than the Django ORM, which can't model composite PKs cleanly.


class ParsedOffer(models.Model):
    """Clustering/similarity-ready row built by
    `scraper/parsed_offers_builder.py`. Categorical `*_enc` columns are stable
    ints from `category_encoders`, not human-readable — decode them via that
    table (or `db/encoder_mappings.json`) for display."""

    offer = models.OneToOneField(
        Offer,
        on_delete=models.CASCADE,
        primary_key=True,
        related_name="parsed",
        db_column="id",
    )

    price = models.FloatField(null=True, blank=True)
    frame_size_cm = models.FloatField(null=True, blank=True)
    wheel_size_inch = models.FloatField(null=True, blank=True)
    tire_width_mm = models.FloatField(null=True, blank=True)
    gear_count = models.IntegerField(null=True, blank=True)
    weight_kg = models.FloatField(null=True, blank=True)
    max_load_kg = models.FloatField(null=True, blank=True)
    production_year = models.IntegerField(null=True, blank=True)
    motor_power_w = models.FloatField(null=True, blank=True)
    range_km = models.FloatField(null=True, blank=True)
    charge_time_h = models.FloatField(null=True, blank=True)
    height_min_cm = models.IntegerField(null=True, blank=True)
    height_max_cm = models.IntegerField(null=True, blank=True)

    frame_material_enc = models.IntegerField(default=-1)
    brake_type_enc = models.IntegerField(default=-1)
    gender_enc = models.IntegerField(default=-1)
    brand_enc = models.IntegerField(default=-1)
    color_enc = models.IntegerField(default=-1)
    bike_type_enc = models.IntegerField(default=-1)
    condition_enc = models.IntegerField(default=-1)

    has_drop_handlebars = models.IntegerField(null=True, blank=True)
    has_flat_handlebars = models.IntegerField(null=True, blank=True)
    has_disc_brakes = models.IntegerField(null=True, blank=True)
    has_pedals = models.IntegerField(null=True, blank=True)
    zero_shot_confidence = models.FloatField(null=True, blank=True)

    parsed_at = models.TextField(null=True, blank=True)

    class Meta:
        managed = False
        db_table = "parsed_offers"
