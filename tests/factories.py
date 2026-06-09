import factory
from factory.django import DjangoModelFactory

from dashboard.models import Offer, OfferImage, OfferParameter


class OfferFactory(DjangoModelFactory):
    class Meta:
        model = Offer

    id = factory.Sequence(lambda n: f"offer-test-{n:06d}")
    title = factory.Faker("sentence", nb_words=5)
    price = factory.Faker("pyfloat", min_value=500.0, max_value=12000.0, right_digits=2)
    currency = "PLN"
    seller_id = factory.Sequence(lambda n: f"seller-{n}")
    seller_login = factory.Sequence(lambda n: f"seller_login_{n}")
    condition = "USED"
    listing_type = "BUY_NOW"
    offer_url = factory.LazyAttribute(lambda o: f"https://allegro.pl/oferta/{o.id}")
    thumbnail_url = factory.LazyAttribute(lambda o: f"https://img.allegro.pl/{o.id}.jpg")
    fetched_at = "2024-06-01T10:00:00+00:00"
    raw_json = "{}"


class OfferParameterFactory(DjangoModelFactory):
    class Meta:
        model = OfferParameter

    offer = factory.SubFactory(OfferFactory)
    name = factory.Sequence(lambda n: f"param-{n}")
    value = factory.Faker("word")


class OfferImageFactory(DjangoModelFactory):
    class Meta:
        model = OfferImage

    offer = factory.SubFactory(OfferFactory)
    url = factory.Faker("image_url")
    position = factory.Sequence(lambda n: n)
