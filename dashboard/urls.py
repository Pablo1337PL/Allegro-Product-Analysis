from django.urls import path

from . import views

urlpatterns = [
    path("", views.OfferListView.as_view(), name="offer_list"),
    path("offer/<str:pk>/", views.OfferDetailView.as_view(), name="offer_detail"),
    path("api/chart/prices/", views.price_chart_data, name="price_chart_data"),
    path("api/stats/", views.stats_partial, name="stats_partial"),
]
