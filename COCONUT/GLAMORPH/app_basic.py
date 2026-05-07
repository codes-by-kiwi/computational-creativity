# app_json_ui.py
import json
from pathlib import Path
import gradio as gr
from PIL import Image
from pipeline_adapter import sample_candidates, run_generate_with_parents
import os
os.environ["NO_PROXY"] = "localhost,127.0.0.1"
os.environ.pop("HTTP_PROXY", None)
os.environ.pop("HTTPS_PROXY", None)
os.environ.pop("ALL_PROXY", None)

DATA_DIR = Path("data")
ITEMS_PATH = DATA_DIR / "items.json"

def load_candidates():
    cands = sample_candidates(4)
    gallery = []
    for it in cands:
        img = Image.open(it["image_path"]).convert("RGB")
        gallery.append((img, it["item_id"]))
    return gallery, cands

def on_pick(evt: gr.SelectData, candidates, picked):
    idx = evt.index
    if idx is None:
        return picked
    picked = picked + [candidates[idx]["item_id"]]
    # keep only last two
    return picked[-2:]

def on_generate(picked, alpha, mode):
    mode = "diffusion"
    if len(picked) != 2:
        return None, "Please select exactly two images."
    out, log, _ = run_generate_with_parents(
        parentA_id=picked[0],
        parentB_id=picked[1],
        alpha=alpha,
        mode=mode,
    )
    return out, log

with gr.Blocks(title="GLAMORPH — Pick 2 of 4") as demo:
    gr.Markdown("# GLAMORPH\n### Step 1: Pick 2 of 4 candidates → Step 2: Generate")

    btn_sample = gr.Button("🎲 Refresh")
    gallery = gr.Gallery(columns=4, height=240)
    candidates_state = gr.State([])
    picked_state = gr.State([])

    picked_box = gr.Textbox(label="Picked item_ids (order = A then B)", interactive=False)

    with gr.Row():
        alpha = gr.Slider(0, 1, value=0.5, step=0.01, label="Alpha")
        # mode = gr.Dropdown(
        #     ["diffusion", "inpaint_mutation", "inpaint_crossover"],
        #     value="diffusion",
        #     label="Mode"
        # )
        btn_gen = gr.Button("Generate", variant="primary")

    out_img = gr.Image(label="Output", type="pil")
    log = gr.Textbox(label="Logs", lines=6)

    btn_sample.click(load_candidates, outputs=[gallery, candidates_state])
    gallery.select(on_pick, inputs=[candidates_state, picked_state], outputs=[picked_state])
    picked_state.change(lambda p: ", ".join(p), inputs=picked_state, outputs=picked_box)

    btn_gen.click(on_generate, inputs=[picked_state, alpha], outputs=[out_img, log])

demo.launch()