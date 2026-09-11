from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple
import hashlib

import numpy as np
from PIL import Image
import torch
import open_clip
from transformers import AutoImageProcessor, AutoModel

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".tif", ".tiff", ".bmp"}


def list_images(folder: Path) -> List[Path]:
    if not folder.exists():
        return []
    return sorted([p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS])


def file_sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def dedupe_paths(paths: List[Path]) -> Tuple[List[Path], List[dict]]:
    """Keep one canonical file per exact image payload.

    Originals are never deleted. The returned duplicate map is useful for reporting
    and guarantees duplicate references do not create duplicate matching categories.
    """
    unique: List[Path] = []
    duplicates: List[dict] = []
    seen: Dict[str, Path] = {}
    for p in paths:
        digest = file_sha256(p)
        if digest in seen:
            duplicates.append({"duplicate": str(p), "canonical": str(seen[digest]), "sha256": digest})
        else:
            seen[digest] = p
            unique.append(p)
    return unique, duplicates


def safe_open(path: Path) -> Image.Image:
    return Image.open(path).convert("RGB")


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(np.float32)
    b = b.astype(np.float32)
    an = np.linalg.norm(a)
    bn = np.linalg.norm(b)
    if an == 0 or bn == 0:
        return 0.0
    return float(np.dot(a, b) / (an * bn))


def make_regions(im: Image.Image) -> Dict[str, Image.Image]:
    w, h = im.size
    def crop(y0: float, y1: float, x0: float = 0.08, x1: float = 0.92):
        return im.crop((int(w*x0), int(h*y0), int(w*x1), int(h*y1)))
    return {
        "full": im,
        "upper": crop(0.12, 0.58),
        "lower": crop(0.42, 0.88),
        "feet": crop(0.72, 1.00, 0.04, 0.96),
        "center": crop(0.20, 0.82, 0.16, 0.84),
    }


@dataclass
class MatchConfig:
    person_threshold: float = 0.42
    person_margin: float = 0.035
    garment_threshold: float = 0.47
    garment_margin: float = 0.028
    max_garments: int = 5
    clip_weight: float = 0.44
    dino_weight: float = 0.56


class VisionMatcher:
    def __init__(self, clip_model: str = "ViT-B-32", clip_pretrained: str = "laion2b_s34b_b79k", dino_model: str = "facebook/dinov2-small"):
        if torch.backends.mps.is_available():
            self.device = torch.device("mps")
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")
        self.clip_model, _, self.clip_preprocess = open_clip.create_model_and_transforms(clip_model, pretrained=clip_pretrained, device=self.device)
        self.clip_model.eval()
        self.dino_processor = AutoImageProcessor.from_pretrained(dino_model)
        self.dino_model = AutoModel.from_pretrained(dino_model).to(self.device)
        self.dino_model.eval()

    @torch.inference_mode()
    def embed_clip(self, image: Image.Image) -> np.ndarray:
        x = self.clip_preprocess(image).unsqueeze(0).to(self.device)
        feat = self.clip_model.encode_image(x)
        feat = feat / feat.norm(dim=-1, keepdim=True)
        return feat[0].float().cpu().numpy()

    @torch.inference_mode()
    def embed_dino(self, image: Image.Image) -> np.ndarray:
        inputs = self.dino_processor(images=image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        out = self.dino_model(**inputs)
        feat = out.last_hidden_state[:, 0]
        feat = feat / feat.norm(dim=-1, keepdim=True)
        return feat[0].float().cpu().numpy()

    def embed(self, image: Image.Image) -> Tuple[np.ndarray, np.ndarray]:
        return self.embed_clip(image), self.embed_dino(image)

    def build_reference_index(self, paths: List[Path]) -> Dict[str, dict]:
        unique_paths, _ = dedupe_paths(paths)
        index = {}
        for p in unique_paths:
            im = safe_open(p)
            clip, dino = self.embed(im)
            name = p.stem
            if name in index:
                suffix = 2
                while f"{name}_{suffix}" in index:
                    suffix += 1
                name = f"{name}_{suffix}"
            index[name] = {"path": str(p), "clip": clip, "dino": dino}
        return index

    def score_pair(self, clip_a, dino_a, ref, cfg: MatchConfig) -> float:
        return cfg.clip_weight * cosine(clip_a, ref["clip"]) + cfg.dino_weight * cosine(dino_a, ref["dino"])

    def rank(self, clip_a, dino_a, refs: Dict[str, dict], cfg: MatchConfig) -> List[Tuple[str, float]]:
        scores = [(name, self.score_pair(clip_a, dino_a, ref, cfg)) for name, ref in refs.items()]
        return sorted(scores, key=lambda x: x[1], reverse=True)

    def match_person(self, image: Image.Image, refs: Dict[str, dict], cfg: MatchConfig):
        regions = make_regions(image)
        all_scores: Dict[str, List[float]] = {k: [] for k in refs}
        for region_name in ["full", "upper", "center"]:
            clip, dino = self.embed(regions[region_name])
            for name, score in self.rank(clip, dino, refs, cfg):
                all_scores[name].append(score)
        ranked = sorted(((name, float(np.mean(vals))) for name, vals in all_scores.items()), key=lambda x: x[1], reverse=True)
        if not ranked:
            return None, [], True
        best = ranked[0]
        second = ranked[1] if len(ranked) > 1 else (None, -1.0)
        ambiguous = best[1] < cfg.person_threshold or (best[1] - second[1]) < cfg.person_margin
        return best[0], ranked[:5], ambiguous

    def match_garments(self, image: Image.Image, refs: Dict[str, dict], cfg: MatchConfig):
        regions = make_regions(image)
        region_candidates = []
        region_rankings = {}
        for region_name in ["upper", "lower", "feet", "center", "full"]:
            clip, dino = self.embed(regions[region_name])
            ranked = self.rank(clip, dino, refs, cfg)
            region_rankings[region_name] = ranked[:6]
            if not ranked:
                continue
            best = ranked[0]
            second = ranked[1] if len(ranked) > 1 else (None, -1.0)
            margin = best[1] - second[1]
            if best[1] >= cfg.garment_threshold and margin >= cfg.garment_margin:
                region_candidates.append((best[0], best[1], region_name, margin))

        dedup: Dict[str, Tuple[float, str, float]] = {}
        for name, score, region, margin in region_candidates:
            if name not in dedup or score > dedup[name][0]:
                dedup[name] = (score, region, margin)

        selected = [
            {"name": name, "score": round(vals[0], 4), "region": vals[1], "margin": round(vals[2], 4)}
            for name, vals in sorted(dedup.items(), key=lambda kv: kv[1][0], reverse=True)[: cfg.max_garments]
        ]
        if not selected and region_rankings.get("full"):
            name, score = region_rankings["full"][0]
            selected = [{"name": name, "score": round(score, 4), "region": "full", "margin": 0.0}]

        ambiguous = not selected or any(x["score"] < cfg.garment_threshold or x["margin"] < cfg.garment_margin for x in selected)
        alternatives = {region: [{"name": n, "score": round(s, 4)} for n, s in ranked] for region, ranked in region_rankings.items()}
        return selected, alternatives, ambiguous

    def match(self, image_path: Path, people_refs, gear_refs, cfg: MatchConfig):
        image = safe_open(image_path)
        person, people_top, person_ambiguous = self.match_person(image, people_refs, cfg)
        garments, garment_alts, garment_ambiguous = self.match_garments(image, gear_refs, cfg)
        return {
            "image": image_path.name,
            "image_path": str(image_path),
            "person": person,
            "person_top": [{"name": n, "score": round(s, 4)} for n, s in people_top],
            "garments": garments,
            "garment_alternatives": garment_alts,
            "ambiguous": bool(person_ambiguous or garment_ambiguous),
        }
