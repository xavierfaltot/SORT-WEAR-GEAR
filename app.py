from __future__ import annotations

from pathlib import Path
import csv
import json
import shutil
from typing import List

import gradio as gr

from matcher import MatchConfig, VisionMatcher, list_images, dedupe_paths, file_sha256

APP_NAME = "SORT WEAR GEAR"
VERSION = "v0.4 MULTI-MATCH + REVIEW"
MODEL = None

CSS = """
:root{--pink:#ff13a8;--cyan:#16f3e6;--violet:#873cff;--ink:#050505;--panel:#0d0d0d;--line:#2a2a2a;--text:#f6f6f6}
body,.gradio-container{background:var(--ink)!important;color:var(--text)!important}
.gradio-container{max-width:1380px!important}
.logo{display:grid;grid-template-columns:repeat(4,1fr);width:330px;gap:7px;margin:22px auto 28px}
.logo span{aspect-ratio:1/1;background:#151515;border:1px solid #2c2c2c;display:flex;align-items:center;justify-content:center;font:900 30px/1 Arial,sans-serif;box-shadow:inset 0 0 0 1px #080808}
.logo .p{background:var(--pink);color:#050505}.logo .c{background:var(--cyan);color:#050505}.logo .v{background:var(--violet);color:#050505}
.zone{border:2px dashed #303030!important;background:#0c0c0c!important;min-height:190px!important}
#people-zone{box-shadow:inset 0 0 0 2px rgba(255,19,168,.18)}
#gear-zone{box-shadow:inset 0 0 0 2px rgba(22,243,230,.18)}
#assemblies-zone{box-shadow:inset 0 0 0 2px rgba(135,60,255,.20)}
.primary{background:linear-gradient(90deg,var(--pink),var(--cyan),var(--violet))!important;color:#050505!important;font-weight:900!important;border:0!important;min-height:58px!important}
.secondary{background:#181818!important;border:1px solid #3b3b3b!important;color:#fff!important}
.reviewbox{border:1px solid #292929!important;background:#0b0b0b!important}
"""

LOGO = """
<div class="logo" aria-label="SORT WEAR GEAR">
<span class="p">S</span><span>O</span><span>R</span><span>T</span>
<span>W</span><span class="c">E</span><span>A</span><span>R</span>
<span>G</span><span>E</span><span class="v">A</span><span>R</span>
</div>
"""


def file_path(item):
    if item is None: return None
    if isinstance(item, str): return item
    if hasattr(item, "name"): return item.name
    return str(item)


def target_contains_hash(target_dir: Path, digest: str) -> bool:
    if not target_dir.exists(): return False
    for existing in target_dir.iterdir():
        if existing.is_file():
            try:
                if file_sha256(existing) == digest:
                    return True
            except Exception:
                pass
    return False


def safe_copy(src_path, target_dir):
    """Copy without ever creating a duplicate payload in the target directory."""
    src = Path(src_path)
    target = Path(target_dir)
    target.mkdir(parents=True, exist_ok=True)
    digest = file_sha256(src)
    if target_contains_hash(target, digest):
        return None
    out = target / src.name
    i = 1
    while out.exists():
        out = target / f"{src.stem}_{i}{src.suffix}"
        i += 1
    shutil.copy2(src, out)
    return str(out)


def copy_many(items, target_dir):
    copied = []
    candidates = []
    for item in items or []:
        p = file_path(item)
        if p and Path(p).is_file(): candidates.append(Path(p))
    unique, _ = dedupe_paths(candidates)
    for p in unique:
        result = safe_copy(p, target_dir)
        if result: copied.append(result)
    return copied


def library_paths(root: Path):
    return {
        "people": root / "REFERENCES" / "PEOPLE",
        "gear": root / "REFERENCES" / "GEAR",
        "to_sort": root / "TO_SORT",
        "by_people": root / "SORTED" / "BY_PEOPLE",
        "by_gear": root / "SORTED" / "BY_GEAR",
        "review": root / "REVIEW",
    }


def ensure_library(root: Path):
    paths = library_paths(root)
    for p in paths.values(): p.mkdir(parents=True, exist_ok=True)
    return paths


def ingest(people_files, gear_files, assembly_files, destination):
    if not destination or not destination.strip(): raise gr.Error("Choisis le dossier de destination.")
    root = Path(destination).expanduser().resolve()
    paths = ensure_library(root)
    copy_many(people_files, paths["people"])
    copy_many(gear_files, paths["gear"])
    copy_many(assembly_files, paths["to_sort"])
    return root, paths


def load_existing(destination):
    if not destination or not destination.strip(): raise gr.Error("Choisis une bibliothèque existante.")
    root = Path(destination).expanduser().resolve()
    paths = ensure_library(root)
    people, people_dupes = dedupe_paths(list_images(paths["people"]))
    gear, gear_dupes = dedupe_paths(list_images(paths["gear"]))
    assemblies, assembly_dupes = dedupe_paths(list_images(paths["to_sort"]))
    return (
        f"LIBRARY OPENED\nPEOPLE {len(people)} · GEAR {len(gear)} · ASSEMBLAGES {len(assemblies)}\n"
        f"DUPLICATES IGNORED {len(people_dupes)+len(gear_dupes)+len(assembly_dupes)}",
        str(root),
    )


def get_model():
    global MODEL
    if MODEL is None: MODEL = VisionMatcher()
    return MODEL


def save_json(path: Path, data):
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def load_json(path: Path, default):
    if not path.exists(): return default
    try: return json.loads(path.read_text(encoding="utf-8"))
    except Exception: return default


def unique_names(items):
    return list(dict.fromkeys(items))


def build_indexes(root: Path, predictions: List[dict]):
    paths = ensure_library(root)
    for generated_root in [paths["by_people"], paths["by_gear"], paths["review"]]:
        if generated_root.exists():
            for child in generated_root.iterdir():
                if child.is_dir(): shutil.rmtree(child)
                elif child.is_file(): child.unlink()

    rows = []
    exported_hashes = {"people": {}, "gear": {}, "review": set()}

    for pred in predictions:
        src = Path(pred["image_path"])
        src_hash = file_sha256(src)
        person = pred.get("person") or "UNKNOWN_PERSON"
        garments = unique_names([g["name"] if isinstance(g, dict) else g for g in pred.get("garments", [])])

        person_seen = exported_hashes["people"].setdefault(person, set())
        if src_hash not in person_seen:
            safe_copy(src, paths["by_people"] / person)
            person_seen.add(src_hash)

        for garment in garments:
            garment_seen = exported_hashes["gear"].setdefault(garment, set())
            if src_hash not in garment_seen:
                safe_copy(src, paths["by_gear"] / garment)
                garment_seen.add(src_hash)

        if pred.get("ambiguous") and src_hash not in exported_hashes["review"]:
            safe_copy(src, paths["review"])
            exported_hashes["review"].add(src_hash)

        person_score = pred.get("person_top", [{}])[0].get("score", "") if pred.get("person_top") else ""
        rows.append({
            "image": pred["image"],
            "person": person,
            "person_score": person_score,
            "garments": " | ".join(garments),
            "garment_scores": " | ".join(str(g.get("score", "")) for g in pred.get("garments", []) if isinstance(g, dict)),
            "ambiguous": pred.get("ambiguous", False),
        })

    with (root / "sorting_report.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["image","person","person_score","garments","garment_scores","ambiguous"])
        writer.writeheader(); writer.writerows(rows)


def sort_library(people_files, gear_files, assembly_files, destination, person_threshold, garment_threshold, max_garments):
    root, paths = ingest(people_files, gear_files, assembly_files, destination)
    people, people_dupes = dedupe_paths(list_images(paths["people"]))
    gear, gear_dupes = dedupe_paths(list_images(paths["gear"]))
    assemblies, assembly_dupes = dedupe_paths(list_images(paths["to_sort"]))
    if not people: raise gr.Error("Aucune référence PEOPLE.")
    if not gear: raise gr.Error("Aucune référence GEAR.")
    if not assemblies: raise gr.Error("Aucun ASSEMBLAGE à classer.")

    duplicate_report = {
        "people": people_dupes,
        "gear": gear_dupes,
        "assemblages": assembly_dupes,
        "total_ignored": len(people_dupes) + len(gear_dupes) + len(assembly_dupes),
    }
    save_json(root / "duplicates_report.json", duplicate_report)

    model = get_model()
    cfg = MatchConfig(person_threshold=float(person_threshold), garment_threshold=float(garment_threshold), max_garments=int(max_garments))
    people_index = model.build_reference_index(people)
    gear_index = model.build_reference_index(gear)

    corrections = load_json(root / "corrections.json", {})
    predictions = []
    for img in assemblies:
        pred = model.match(img, people_index, gear_index, cfg)
        saved = corrections.get(img.name)
        if saved:
            pred["person"] = saved.get("person", pred["person"])
            pred["garments"] = [{"name": g, "score": 1.0, "region": "human", "margin": 1.0} for g in unique_names(saved.get("garments", []))]
            pred["ambiguous"] = False
            pred["human_corrected"] = True
        predictions.append(pred)

    save_json(root / "predictions.json", predictions)
    build_indexes(root, predictions)

    manifest = {
        "app": APP_NAME,
        "version": VERSION,
        "root": str(root),
        "models": {"semantic": "OpenCLIP ViT-B-32", "instance": "facebook/dinov2-small"},
        "counts": {"people": len(people), "gear": len(gear), "assemblages": len(assemblies), "review": sum(bool(p["ambiguous"]) for p in predictions)},
        "duplicates_ignored": duplicate_report["total_ignored"],
        "workflow": ["SCAN","DEDUP REFERENCES","MATCH PEOPLE","MATCH GEAR","MULTI-GARMENT MATCH","REVIEW AMBIGUOUS","CORRECT / VALIDATE","BUILD LIBRARY"],
    }
    save_json(root / "library.json", manifest)
    ambiguous = [p["image"] for p in predictions if p.get("ambiguous")]
    status = (
        f"DONE · {len(assemblies)} UNIQUE ASSEMBLAGES\n"
        f"PEOPLE {len(people)} · GEAR {len(gear)}\n"
        f"DUPLICATES IGNORED {duplicate_report['total_ignored']}\n"
        f"AMBIGUOUS {len(ambiguous)} → REVIEW\n"
        f"EXPORT: NO DUPLICATE FILES\n"
        f"LIBRARY → {root}"
    )
    return status, gr.Dropdown(choices=ambiguous, value=(ambiguous[0] if ambiguous else None)), str(root)


def current_refs(root_s: str):
    root = Path(root_s).expanduser()
    paths = ensure_library(root)
    people, _ = dedupe_paths(list_images(paths["people"]))
    gear, _ = dedupe_paths(list_images(paths["gear"]))
    return unique_names([p.stem for p in people]), unique_names([p.stem for p in gear])


def load_review_item(image_name, root_s):
    if not image_name or not root_s: return None, None, [], "No ambiguous item selected."
    root = Path(root_s)
    predictions = load_json(root / "predictions.json", [])
    pred = next((p for p in predictions if p.get("image") == image_name), None)
    if not pred: return None, None, [], "Prediction not found."
    people, gear = current_refs(root_s)
    garments = unique_names([g["name"] if isinstance(g, dict) else g for g in pred.get("garments", [])])
    detail = json.dumps({"person_top": pred.get("person_top", []), "garment_alternatives": pred.get("garment_alternatives", {})}, indent=2, ensure_ascii=False)
    return pred.get("image_path"), gr.Dropdown(choices=people, value=pred.get("person")), gr.CheckboxGroup(choices=gear, value=garments), detail


def save_review(image_name, person, garments, root_s):
    if not image_name or not root_s: raise gr.Error("Aucun item chargé.")
    root = Path(root_s)
    garments = unique_names(list(garments or []))
    corrections = load_json(root / "corrections.json", {})
    corrections[image_name] = {"person": person, "garments": garments}
    save_json(root / "corrections.json", corrections)

    predictions = load_json(root / "predictions.json", [])
    for p in predictions:
        if p.get("image") == image_name:
            p["person"] = person
            p["garments"] = [{"name": g, "score": 1.0, "region": "human", "margin": 1.0} for g in garments]
            p["ambiguous"] = False
            p["human_corrected"] = True
            break
    save_json(root / "predictions.json", predictions)
    build_indexes(root, predictions)
    remaining = [p["image"] for p in predictions if p.get("ambiguous")]
    return f"VALIDATED · {image_name} → {person} + {', '.join(garments)}", gr.Dropdown(choices=remaining, value=(remaining[0] if remaining else None))


with gr.Blocks(title=APP_NAME, css=CSS) as demo:
    gr.HTML(LOGO)
    gr.Markdown("# SORT WEAR GEAR")
    gr.Markdown(f"**{VERSION} · OpenCLIP + DINOv2 · MULTI-GARMENT · DUPLICATE-FREE EXPORT**")
    root_state = gr.Textbox(visible=False)

    with gr.Tab("SORT"):
        with gr.Row():
            people = gr.File(label="PEOPLE", file_count="multiple", file_types=["image"], elem_classes=["zone"], elem_id="people-zone")
            gear = gr.File(label="GEAR", file_count="multiple", file_types=["image"], elem_classes=["zone"], elem_id="gear-zone")
        assemblies = gr.File(label="ASSEMBLAGES", file_count="multiple", file_types=["image"], elem_classes=["zone"], elem_id="assemblies-zone")
        destination = gr.Textbox(label="LIBRARY / DESTINATION", placeholder="/Users/xavier/Pictures/SORT_WEAR_GEAR_LIBRARY")
        with gr.Row():
            open_btn = gr.Button("REOPEN LIBRARY", elem_classes=["secondary"])
            run = gr.Button("SCAN → DEDUP → MULTI-MATCH → BUILD", elem_classes=["primary"])
        with gr.Accordion("MATCH SETTINGS", open=False):
            person_threshold = gr.Slider(0.20, 0.90, value=0.42, step=0.01, label="PERSON MIN SCORE")
            garment_threshold = gr.Slider(0.20, 0.90, value=0.47, step=0.01, label="GARMENT MIN SCORE")
            max_garments = gr.Slider(1, 8, value=5, step=1, label="MAX GARMENTS / IMAGE")
        status = gr.Textbox(label="STATUS", lines=7, interactive=False)

    with gr.Tab("REVIEW AMBIGUOUS"):
        review_select = gr.Dropdown(label="AMBIGUOUS IMAGE", choices=[])
        with gr.Row():
            review_image = gr.Image(label="IMAGE", type="filepath", height=560, elem_classes=["reviewbox"])
            with gr.Column():
                review_person = gr.Dropdown(label="PERSON — correct / validate", choices=[])
                review_gear = gr.CheckboxGroup(label="GARMENTS — add / remove / validate", choices=[])
                review_detail = gr.Code(label="TOP CANDIDATES / SCORES", language="json", lines=20)
                validate = gr.Button("VALIDATE + REMEMBER", elem_classes=["primary"])
                review_status = gr.Textbox(label="REVIEW STATUS", interactive=False)

    run.click(sort_library, inputs=[people, gear, assemblies, destination, person_threshold, garment_threshold, max_garments], outputs=[status, review_select, root_state])
    open_btn.click(load_existing, inputs=destination, outputs=[status, root_state])
    review_select.change(load_review_item, inputs=[review_select, root_state], outputs=[review_image, review_person, review_gear, review_detail])
    validate.click(save_review, inputs=[review_select, review_person, review_gear, root_state], outputs=[review_status, review_select])

if __name__ == "__main__":
    demo.queue().launch(inbrowser=True)
