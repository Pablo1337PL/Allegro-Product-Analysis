import logging
import os
from io import BytesIO

import numpy as np
import open_clip
import requests
import torch
from PIL import Image

log = logging.getLogger(__name__)

# Prompt pairs for binary zero-shot classification.
# Each entry: attr_name → (positive_prompt, negative_prompt)
ZERO_SHOT_ATTRS: dict[str, tuple[str, str]] = {
    "has_drop_handlebars": (
        "a racing bicycle with drop handlebars curving downward",
        "a bicycle with flat or riser handlebars",
    ),
    "has_flat_handlebars": (
        "a bicycle with flat straight handlebar",
        "a bicycle with curved drop or riser handlebars",
    ),
    "has_disc_brakes": (
        "a bicycle with disc brake rotors on the wheels",
        "a bicycle with rim brakes, no disc rotors",
    ),
    "has_pedals": (
        "a bicycle with pedals attached to the crank arms",
        "a bicycle with bare crank arms and no pedals fitted",
    ),
}


class CLIPWorker:
    def __init__(self) -> None:
        model_name = os.environ.get("CLIP_MODEL", "ViT-B-32")
        pretrained = os.environ.get("CLIP_PRETRAINED", "openai")
        cache_dir = os.path.expanduser(
            os.environ.get("CLIP_CACHE_DIR", "~/.cache/open_clip")
        )

        self.device = "cpu"
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            model_name,
            pretrained=pretrained,
            cache_dir=cache_dir,
            device=self.device,
        )
        self.tokenizer = open_clip.get_tokenizer(model_name)
        self.model.eval()
        log.info("CLIP %s loaded on %s", model_name, self.device)

        # Pre-compute all text embeddings once — they're constant across offers.
        self._attr_embeddings: dict[str, tuple[np.ndarray, np.ndarray]] = {
            attr: (self.encode_text(pos), self.encode_text(neg))
            for attr, (pos, neg) in ZERO_SHOT_ATTRS.items()
        }

    def encode_image(self, img: "Image.Image") -> np.ndarray:
        """Embed a PIL image into an L2-normalised CLIP vector."""
        tensor = self.preprocess(img.convert("RGB")).unsqueeze(0).to(self.device)
        with torch.no_grad():
            feat = self.model.encode_image(tensor)
            feat = feat / feat.norm(dim=-1, keepdim=True)
        return feat.squeeze(0).cpu().numpy().astype(np.float32)

    def encode_image_url(self, url: str) -> np.ndarray | None:
        try:
            resp = requests.get(url, timeout=10)
            resp.raise_for_status()
            img = Image.open(BytesIO(resp.content)).convert("RGB")
            return self.encode_image(img)
        except Exception as exc:
            log.warning("CLIP encode failed for %s: %s", url, exc)
            return None

    def encode_text(self, text: str) -> np.ndarray:
        tokens = self.tokenizer([text]).to(self.device)
        with torch.no_grad():
            feat = self.model.encode_text(tokens)
            feat = feat / feat.norm(dim=-1, keepdim=True)
        return feat.squeeze(0).cpu().numpy().astype(np.float32)

    def run_zero_shot(self, image_embedding: np.ndarray) -> dict:
        """Return zero-shot attribute predictions for a single L2-normed image vector.

        Returns a dict with integer 0/1 values for each binary attribute plus a
        float zero_shot_confidence (mean positive−negative similarity margin).
        """
        attrs: dict = {}
        confidences: list[float] = []

        for attr, (pos_emb, neg_emb) in self._attr_embeddings.items():
            pos_sim = float(np.dot(image_embedding, pos_emb))
            neg_sim = float(np.dot(image_embedding, neg_emb))
            attrs[attr] = 1 if pos_sim > neg_sim else 0
            confidences.append(abs(pos_sim - neg_sim))

        attrs["zero_shot_confidence"] = float(np.mean(confidences))
        return attrs
