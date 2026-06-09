import logging
import os
from io import BytesIO

import numpy as np
import open_clip
import requests
import torch
from PIL import Image

log = logging.getLogger(__name__)


class CLIPWorker:
    def __init__(self) -> None:
        model_name = os.environ.get("CLIP_MODEL", "ViT-B-32")
        pretrained = os.environ.get("CLIP_PRETRAINED", "openai")
        cache_dir = os.environ.get("CLIP_CACHE_DIR", "/app/clip_cache")

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

    def encode_image_url(self, url: str) -> np.ndarray | None:
        try:
            resp = requests.get(url, timeout=10)
            resp.raise_for_status()
            img = Image.open(BytesIO(resp.content)).convert("RGB")
            tensor = self.preprocess(img).unsqueeze(0).to(self.device)
            with torch.no_grad():
                feat = self.model.encode_image(tensor)
                feat = feat / feat.norm(dim=-1, keepdim=True)
            return feat.squeeze(0).cpu().numpy().astype(np.float32)
        except Exception as exc:
            log.warning("CLIP encode failed for %s: %s", url, exc)
            return None

    def encode_text(self, text: str) -> np.ndarray:
        tokens = self.tokenizer([text]).to(self.device)
        with torch.no_grad():
            feat = self.model.encode_text(tokens)
            feat = feat / feat.norm(dim=-1, keepdim=True)
        return feat.squeeze(0).cpu().numpy().astype(np.float32)
