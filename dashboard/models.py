from django.db import models


class Offer(models.Model):
    id = models.TextField(primary_key=True)
    title = models.TextField(null=True, blank=True)
    price = models.FloatField(null=True, blank=True)
    currency = models.TextField(null=True, blank=True)
    seller_id = models.TextField(null=True, blank=True)
    seller_login = models.TextField(null=True, blank=True)
    condition = models.TextField(null=True, blank=True)
    listing_type = models.TextField(null=True, blank=True)
    offer_url = models.TextField(null=True, blank=True)
    thumbnail_url = models.TextField(null=True, blank=True)
    end_time = models.TextField(null=True, blank=True)
    fetched_at = models.TextField()
    raw_json = models.TextField()

    class Meta:
        managed = False
        db_table = "offers"
        ordering = ["-fetched_at"]


class OfferParameter(models.Model):
    id = models.AutoField(primary_key=True)
    offer = models.ForeignKey(Offer, on_delete=models.CASCADE, related_name="parameters", db_column="offer_id")
    name = models.TextField()
    value = models.TextField(null=True, blank=True)

    class Meta:
        managed = False
        db_table = "offer_parameters"
        unique_together = [("offer", "name")]


class OfferImage(models.Model):
    id = models.AutoField(primary_key=True)
    offer = models.ForeignKey(Offer, on_delete=models.CASCADE, related_name="images", db_column="offer_id")
    url = models.TextField()
    position = models.IntegerField(default=0)

    class Meta:
        managed = False
        db_table = "offer_images"
        ordering = ["position"]
        unique_together = [("offer", "position")]


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
    extracted_frame_material = models.TextField(null=True, blank=True)
    price_predicted = models.FloatField(null=True, blank=True)
    anomaly_score = models.FloatField(null=True, blank=True)

    class Meta:
        managed = False
        db_table = "offer_features"
