from django.urls import path
from django.views.generic import RedirectView

from . import views

urlpatterns = [
    path("", views.OfferListView.as_view(), name="offer_list"),
    path("offer/<str:pk>/", views.OfferDetailView.as_view(), name="offer_detail"),
    path("offer/<str:pk>/similar/", views.SimilarOffersView.as_view(), name="similar_offers"),
    # Cluster lab (multi-algorithm). The old single-run /clusters/ redirects here.
    path("clusters/", RedirectView.as_view(pattern_name="cluster_lab_index", permanent=False)),
    path("clusters/lab/", views.ClusterLabIndexView.as_view(), name="cluster_lab_index"),
    path("clusters/lab/<str:run_id>/", views.ClusterLabRunView.as_view(), name="cluster_lab_run"),
    path("showcases/", views.ShowcasesView.as_view(), name="showcases"),
    path("stats/", views.StatsView.as_view(), name="stats"),
    path("api/chart/prices/", views.price_chart_data, name="price_chart_data"),
    path("api/chart/histogram/<str:field>/", views.field_histogram_data, name="field_histogram_data"),
    path("api/chart/scatter/<str:pair_id>/", views.scatter_chart_data, name="scatter_chart_data"),
    path("api/chart/boxplot/<str:field>/", views.boxplot_chart_data, name="boxplot_chart_data"),
    path("api/chart/heatmap/pedals-handlebars/", views.pedals_handlebars_heatmap_data, name="pedals_handlebars_heatmap_data"),
    path("api/chart/heatmap/correlation/", views.correlation_heatmap_data, name="correlation_heatmap_data"),
    path("api/stats/", views.stats_partial, name="stats_partial"),
    path("api/cluster-lab/<str:run_id>/", views.ClusterLabDataView.as_view(), name="cluster_lab_data"),
]
