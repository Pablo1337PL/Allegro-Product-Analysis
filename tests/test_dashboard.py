import pytest

from tests.factories import OfferFactory, OfferImageFactory, OfferParameterFactory


@pytest.mark.django_db
def test_offer_list_returns_200(client):
    response = client.get("/")
    assert response.status_code == 200


@pytest.mark.django_db
def test_offer_list_shows_titles(client):
    OfferFactory(id="offer-abc", title="Carbon Racer 2024", price=3000.0)
    response = client.get("/")
    assert b"Carbon Racer 2024" in response.content


@pytest.mark.django_db
def test_offer_list_filter_condition(client):
    OfferFactory(id="offer-new-01", condition="NEW", price=2000.0)
    OfferFactory(id="offer-used-01", condition="USED", price=1500.0)
    response = client.get("/?condition=NEW")
    assert response.status_code == 200
    content = response.content.decode()
    assert "offer-new-01" in content
    assert "offer-used-01" not in content


@pytest.mark.django_db
def test_offer_list_filter_price(client):
    OfferFactory(id="offer-cheap-01", price=800.0)
    OfferFactory(id="offer-pricey-01", price=9000.0)
    response = client.get("/?min_price=5000")
    assert response.status_code == 200
    content = response.content.decode()
    assert "offer-pricey-01" in content
    assert "offer-cheap-01" not in content


@pytest.mark.django_db
def test_offer_list_search(client):
    OfferFactory(id="offer-giant-01", title="Giant TCR Advanced 2023")
    OfferFactory(id="offer-trek-01", title="Trek Emonda SL6")
    response = client.get("/?search=Giant")
    assert response.status_code == 200
    content = response.content.decode()
    assert "Giant TCR Advanced" in content
    assert "Trek Emonda" not in content


@pytest.mark.django_db
def test_offer_list_htmx_returns_partial(client):
    OfferFactory.create_batch(3)
    response = client.get("/", HTTP_HX_REQUEST="true")
    assert response.status_code == 200
    content = response.content.decode()
    assert "<html" not in content.lower()
    assert "<table" in content.lower()


@pytest.mark.django_db
def test_offer_list_empty_state(client):
    response = client.get("/?condition=NONEXISTENT_CONDITION")
    assert response.status_code == 200
    assert b"No offers found" in response.content


@pytest.mark.django_db
def test_offer_detail_returns_200(client):
    offer = OfferFactory(id="offer-detail-01", title="Specialized Tarmac SL7")
    response = client.get(f"/offer/{offer.id}/")
    assert response.status_code == 200
    assert b"Specialized Tarmac SL7" in response.content


@pytest.mark.django_db
def test_offer_detail_shows_parameters(client):
    offer = OfferFactory(id="offer-params-01")
    OfferParameterFactory(offer=offer, name="Rozmiar ramy", value="M")
    response = client.get(f"/offer/{offer.id}/")
    assert b"Rozmiar ramy" in response.content
    assert b"</td>" in response.content


@pytest.mark.django_db
def test_offer_detail_shows_images(client):
    offer = OfferFactory(id="offer-imgs-01")
    OfferImageFactory(offer=offer, url="https://img.example.com/test.jpg", position=0)
    response = client.get(f"/offer/{offer.id}/")
    assert b"test.jpg" in response.content


@pytest.mark.django_db
def test_offer_detail_404_for_missing(client):
    response = client.get("/offer/nonexistent-id-xyz/")
    assert response.status_code == 404


@pytest.mark.django_db
def test_price_chart_data_empty(client):
    response = client.get("/api/chart/prices/")
    assert response.status_code == 200
    data = response.json()
    assert data == {"labels": [], "data": []}


@pytest.mark.django_db
def test_price_chart_data_returns_histogram(client):
    for price in [1000, 2000, 3000, 4000, 5000]:
        OfferFactory(price=float(price))
    response = client.get("/api/chart/prices/")
    assert response.status_code == 200
    data = response.json()
    assert len(data["labels"]) == 20
    assert len(data["data"]) == 20
    assert sum(data["data"]) == 5


@pytest.mark.django_db
def test_stats_partial_returns_counts(client):
    OfferFactory.create_batch(4, price=2000.0)
    response = client.get("/api/stats/")
    assert response.status_code == 200
    data = response.json()
    assert data["total"] >= 4
    assert data["avg_price"] > 0
