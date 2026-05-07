import gradio as gr
from PIL import Image

from pipeline_adapter import list_candidates, run_generate_with_parents

MODES = ["diffusion", "inpaint_mutation", "inpaint_crossover"]

def build_gallery(category, style):
    cands = list_candidates(category=category, style=style, limit=60)
    # gallery items: list of (image, caption)
    gallery = []
    for label, item_id, img_path in cands:
        try:
            img = Image.open(img_path).convert("RGB")
            gallery.append((img, str(item_id)))
        except:
            continue
    return gallery, cands  # keep raw candidates in state

def on_pick(evt: gr.SelectData, candidates, pick_target, parentA, parentB):
    # evt.index gives the index in gallery
    idx = evt.index
    if candidates is None or idx is None:
        return parentA, parentB, parentA, parentB

    item_id = candidates[idx][1]
    img_path = candidates[idx][2]
    img = Image.open(img_path).convert("RGB")

    if pick_target == "Pick A":
        return item_id, parentB, img, None
    else:
        return parentA, item_id, None, img

def on_generate(parentA_id, parentB_id, alpha, mode):
    if not parentA_id or not parentB_id:
        return None, "Please pick both Parent A and Parent B."
    out, log, _ = run_generate_with_parents(parentA_id, parentB_id, alpha, mode)
    return out, log

with gr.Blocks(title="GLAMORPH — Select Parents") as demo:
    gr.Markdown("# GLAMORPH — Pick two parents → Generate")

    with gr.Row():
        category = gr.Textbox(value="dress", label="Category filter (optional)")
        style = gr.Textbox(value="casual", label="Style filter (optional, based on style_tags)")
        btn_load = gr.Button("Load gallery")

        pick_target = gr.Radio(["Pick A", "Pick B"], value="Pick A", label="Click mode")

    gallery = gr.Gallery(label="Dataset Gallery (click to select)", columns=6, height=360)
    candidates_state = gr.State([])

    with gr.Row():
        parentA_id = gr.Textbox(label="Parent A ID", interactive=False)
        parentB_id = gr.Textbox(label="Parent B ID", interactive=False)

    with gr.Row():
        previewA = gr.Image(label="Parent A Preview", type="pil")
        previewB = gr.Image(label="Parent B Preview", type="pil")

    with gr.Row():
        alpha = gr.Slider(0, 1, value=0.5, step=0.01, label="Alpha")
        mode = gr.Dropdown(MODES, value="diffusion", label="Mode")
        btn_gen = gr.Button("Generate", variant="primary")

    out = gr.Image(label="Output", type="pil")
    log = gr.Textbox(label="Logs", lines=6)

    btn_load.click(build_gallery, inputs=[category, style], outputs=[gallery, candidates_state])
    gallery.select(
        fn=on_pick,
        inputs=[candidates_state, pick_target, parentA_id, parentB_id],
        outputs=[parentA_id, parentB_id, previewA, previewB]
    )
    btn_gen.click(on_generate, inputs=[parentA_id, parentB_id, alpha, mode], outputs=[out, log])

demo.launch()
