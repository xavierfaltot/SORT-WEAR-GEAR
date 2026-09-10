from pathlib import Path
import shutil
import json
import gradio as gr

APP_NAME = "SORT WEAR GEAR v0.1"

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
.gradio-container{
  max-width:1200px !important;
}
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
#people-zone{
  box-shadow:inset 0 0 0 2px rgba(255,19,168,.15);
}
#clothes-zone{
  box-shadow:inset 0 0 0 2px rgba(22,243,230,.15);
}
#dest{
  box-shadow:inset 0 0 0 2px rgba(135,60,255,.15);
}
#run{
  background:linear-gradient(90deg,var(--pink),var(--cyan),var(--violet)) !important;
  color:#050505 !important;
  font-weight:900 !important;
  border:none !important;
  min-height:56px !important;
}
h1,h2,h3,p,label,span{
  color:var(--text);
}
"""

def copy_files(files, target_dir):
    target = Path(target_dir)
    target.mkdir(parents=True, exist_ok=True)
    copied = []
    for item in files or []:
        src = Path(item)
        if src.exists() and src.is_file():
            out = target / src.name
            i = 1
            while out.exists():
                out = target / f"{src.stem}_{i}{src.suffix}"
                i += 1
            shutil.copy2(src, out)
            copied.append(str(out))
    return copied

def build_library(people_files, clothes_files, destination):
    if not destination:
        raise gr.Error("Choisis un dossier de destination.")

    root = Path(destination).expanduser()
    root.mkdir(parents=True, exist_ok=True)

    people_dir = root / "PEOPLE"
    clothes_dir = root / "GEAR"
    unsorted_dir = root / "UNSORTED"
    people_dir.mkdir(exist_ok=True)
    clothes_dir.mkdir(exist_ok=True)
    unsorted_dir.mkdir(exist_ok=True)

    people = copy_files(people_files, people_dir)
    clothes = copy_files(clothes_files, clothes_dir)

    manifest = {
        "app": APP_NAME,
        "people_count": len(people),
        "gear_count": len(clothes),
        "people": people,
        "gear": clothes,
    }
    (root / "library.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

    return (
        f"✓ PEOPLE: {len(people)}\n"
        f"✓ GEAR: {len(clothes)}\n"
        f"✓ DESTINATION: {root}\n"
        f"✓ library.json created"
    )

with gr.Blocks(title=APP_NAME) as demo:
    gr.Image(
        value="assets/sort_wear_gear_logo.png",
        show_label=False,
        interactive=False,
        elem_id="logo",
        height=320,
    )

    gr.Markdown("## SORT · WEAR · GEAR")
    gr.Markdown("Drop your people. Drop your clothes. Build your visual wardrobe library.")

    with gr.Row():
        people = gr.File(
            label="GENS / PEOPLE",
            file_count="multiple",
            file_types=["image"],
            elem_classes=["zone"],
            elem_id="people-zone",
        )
        clothes = gr.File(
            label="FRINGUES / GEAR",
            file_count="multiple",
            file_types=["image"],
            elem_classes=["zone"],
            elem_id="clothes-zone",
        )

    destination = gr.Textbox(
        label="DOSSIER DE DESTINATION",
        placeholder="/Users/tonnom/Pictures/SORT_WEAR_GEAR_LIBRARY",
        elem_id="dest",
    )

    run = gr.Button("BUILD LIBRARY", elem_id="run")
    status = gr.Textbox(label="STATUS", lines=5, interactive=False)

    run.click(
        fn=build_library,
        inputs=[people, clothes, destination],
        outputs=status,
    )

if __name__ == "__main__":
    demo.queue().launch(inbrowser=True, css=CSS)
