from pathlib import Path
import shutil
import json
import gradio as gr

APP_NAME = "SORT WEAR GEAR v0.2"

CSS = """
:root{
  --pink:#ff13a8;
  --cyan:#16f3e6;
  --violet:#873cff;
  --ink:#080808;
  --panel:#111111;
  --text:#f5f5f5;
}
body, .gradio-container{
  background:#050505 !important;
  color:var(--text) !important;
}
.gradio-container{ max-width:1200px !important; }
#logo img{
  border-radius:18px !important;
  box-shadow:0 0 40px rgba(255,19,168,.18);
}
.zone{
  border-radius:20px !important;
  border:2px dashed #303030 !important;
  background:#0d0d0d !important;
  min-height:210px !important;
}
#people-zone{ box-shadow:inset 0 0 0 2px rgba(255,19,168,.18); }
#clothes-zone{ box-shadow:inset 0 0 0 2px rgba(22,243,230,.18); }
#assemblies-zone{ box-shadow:inset 0 0 0 2px rgba(135,60,255,.20); }
#dest{ box-shadow:inset 0 0 0 2px rgba(135,60,255,.18); }
#run{
  background:linear-gradient(90deg,var(--pink),var(--cyan),var(--violet)) !important;
  color:#050505 !important;
  font-weight:900 !important;
  border:none !important;
  min-height:56px !important;
}
"""

def file_path(item):
    if item is None:
        return None
    if isinstance(item, str):
        return item
    if hasattr(item, "name"):
        return item.name
    return str(item)

def safe_copy(src_path, target_dir):
    src = Path(src_path)
    target = Path(target_dir)
    target.mkdir(parents=True, exist_ok=True)
    out = target / src.name
    i = 1
    while out.exists():
        out = target / f"{src.stem}_{i}{src.suffix}"
        i += 1
    shutil.copy2(src, out)
    return str(out)

def copy_many(items, target_dir):
    copied = []
    for item in items or []:
        p = file_path(item)
        if p and Path(p).is_file():
            copied.append(safe_copy(p, target_dir))
    return copied

def build_library(people_files, clothes_files, assembly_files, destination):
    if not destination or not destination.strip():
        raise gr.Error("Choisis le dossier de destination.")

    root = Path(destination).expanduser()
    root.mkdir(parents=True, exist_ok=True)

    refs_people = root / "REFERENCES" / "PEOPLE"
    refs_gear = root / "REFERENCES" / "GEAR"
    incoming = root / "TO_SORT"
    sorted_people = root / "SORTED" / "BY_PEOPLE"
    sorted_gear = root / "SORTED" / "BY_GEAR"
    review = root / "REVIEW"

    for d in [refs_people, refs_gear, incoming, sorted_people, sorted_gear, review]:
        d.mkdir(parents=True, exist_ok=True)

    people = copy_many(people_files, refs_people)
    clothes = copy_many(clothes_files, refs_gear)
    assemblies = copy_many(assembly_files, incoming)

    manifest = {
        "app": APP_NAME,
        "workflow": {
            "people_references": "REFERENCES/PEOPLE",
            "gear_references": "REFERENCES/GEAR",
            "assemblies_to_sort": "TO_SORT",
            "sorted_by_people": "SORTED/BY_PEOPLE",
            "sorted_by_gear": "SORTED/BY_GEAR",
            "review": "REVIEW",
        },
        "counts": {
            "people_references": len(people),
            "gear_references": len(clothes),
            "assemblies_to_sort": len(assemblies),
        },
        "people_references": people,
        "gear_references": clothes,
        "assemblies_to_sort": assemblies,
        "status": "INGESTED_NOT_CLUSTERED",
    }
    (root / "library.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8"
    )

    return (
        f"✓ SOURCES GENS: {len(people)}\n"
        f"✓ SOURCES FRINGUES: {len(clothes)}\n"
        f"✓ IMAGES À RANGER: {len(assemblies)}\n"
        f"✓ DESTINATION: {root}\n\n"
        f"Les assemblages ont été placés dans TO_SORT.\n"
        f"Les catégories finales seront définies par tes sources GENS + FRINGUES."
    )

with gr.Blocks(title=APP_NAME) as demo:
    gr.Image(
        value="assets/sort_wear_gear_logo.png",
        show_label=False,
        interactive=False,
        elem_id="logo",
        height=300,
    )

    gr.Markdown("## SORT · WEAR · GEAR")
    gr.Markdown(
        "**1. Donne les sources GENS** · "
        "**2. Donne les sources FRINGUES** · "
        "**3. Drop les ASSEMBLAGES À RANGER** · "
        "**4. Choisis la DESTINATION**"
    )

    with gr.Row():
        people = gr.File(
            label="1 — SOURCES GENS / PEOPLE",
            file_count="multiple",
            file_types=["image"],
            elem_classes=["zone"],
            elem_id="people-zone",
        )
        clothes = gr.File(
            label="2 — SOURCES FRINGUES / GEAR",
            file_count="multiple",
            file_types=["image"],
            elem_classes=["zone"],
            elem_id="clothes-zone",
        )

    assemblies = gr.File(
        label="3 — IMAGES À RANGER / ASSEMBLAGES",
        file_count="multiple",
        file_types=["image"],
        elem_classes=["zone"],
        elem_id="assemblies-zone",
    )

    destination = gr.Textbox(
        label="4 — DOSSIER DE DESTINATION",
        placeholder="/Users/tonnom/Pictures/SORT_WEAR_GEAR_LIBRARY",
        elem_id="dest",
    )

    run = gr.Button("INGEST → PREPARE SORTING", elem_id="run")
    status = gr.Textbox(label="STATUS", lines=7, interactive=False)

    run.click(
        fn=build_library,
        inputs=[people, clothes, assemblies, destination],
        outputs=status,
    )

if __name__ == "__main__":
    demo.queue().launch(inbrowser=True, css=CSS)
