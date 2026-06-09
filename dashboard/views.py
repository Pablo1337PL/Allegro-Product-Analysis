from django.db.models import Avg, Count
from django.http import HttpResponse, JsonResponse
from django.views.generic import DetailView, ListView

from .models import Offer, OfferFeature, OfferImage, OfferParameter


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

        condition = p.get("condition", "").strip()
        if condition:
            qs = qs.filter(condition=condition)

        search = p.get("search", "").strip()
        if search:
            qs = qs.filter(title__icontains=search)

        try:
            qs = qs.filter(price__gte=float(p["min_price"]))
        except (KeyError, ValueError):
            pass

        try:
            qs = qs.filter(price__lte=float(p["max_price"]))
        except (KeyError, ValueError):
            pass

        return qs

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        context["conditions"] = (
            Offer.objects.values_list("condition", flat=True)
            .distinct()
            .order_by("condition")
        )
        return context


class OfferDetailView(DetailView):
    model = Offer
    template_name = "dashboard/offer_detail.html"
    context_object_name = "offer"

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        offer = self.object
        context["images"] = OfferImage.objects.filter(offer=offer)
        context["parameters"] = OfferParameter.objects.filter(offer=offer).order_by("name")
        try:
            context["features"] = offer.features
        except OfferFeature.DoesNotExist:
            context["features"] = None
        return context


def price_chart_data(request):
    prices = list(
        Offer.objects.filter(price__isnull=False, price__gt=0).values_list("price", flat=True)
    )
    if not prices:
        return JsonResponse({"labels": [], "data": []})

    min_p, max_p = min(prices), max(prices)
    if min_p == max_p:
        return JsonResponse({"labels": [str(int(min_p))], "data": [len(prices)]})

    bins = 20
    width = (max_p - min_p) / bins
    counts = [0] * bins
    for p in prices:
        idx = min(int((p - min_p) / width), bins - 1)
        counts[idx] += 1

    labels = [f"{int(min_p + i * width)}–{int(min_p + (i + 1) * width)}" for i in range(bins)]
    return JsonResponse({"labels": labels, "data": counts})


def stats_partial(request):
    stats = Offer.objects.aggregate(total=Count("id"), avg_price=Avg("price"))
    total = stats["total"] or 0
    avg_price = round(stats["avg_price"] or 0)
    return JsonResponse({"total": total, "avg_price": avg_price})
