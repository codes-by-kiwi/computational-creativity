# GLAMORPH – Co-Creativity Fashion Fusion System

GLAMORPH is a co-creative fashion generation system that combines **retrieval-based embedding fusion** and **diffusion-based image generation**.  
The project explores how users and generative models can collaboratively create novel garment designs by selecting existing fashion items and blending their visual and stylistic characteristics.

This repository contains a runnable prototype with a Gradio-based user interface.

---


## 📁 Project Structure

├── app_basic.py # Minimal Gradio demo (auto-pick parents)

├── app_select_parents.py # Gradio UI with user-selected parent items

├── pipeline_adapter.py # High-level pipeline interface between UI and core logic

├── fusion_engine.py # Core fusion logic (retrieval + diffusion)

├── diffusion_generator.py # Stable Diffusion img2img generation utilities

├── GLAMORPH.ipynb # Diffusion model implementation (notebook)

├── merge_zalando_local.py # Dataset preprocessing / merging script

├── data/ # Local dataset (JSON, images, embeddings)

├── outputs/ # Generated images

└── README.md

## 🧠 Core Pipeline

1. **Load fashion items** from JSON dataset  
2. **Build or load embeddings** for each item  
3. **Select parent items**
   - Automatically (basic demo), or
   - Explicitly via UI selection
4. **Fuse parent representations**
   - Embedding interpolation (α-weighted)
5. **Generate output**
   - Retrieval mode (nearest neighbor)
   - Diffusion mode (Stable Diffusion img2img)
  
## 🚀 Getting Started

### 1. Environment Setup

Create and activate a Conda environment:

```bash
conda create -n glamorph python=3.10
conda activate glamorph
```
Install dependencies:
```
pip install torch torchvision pillow
pip install diffusers transformers accelerate safetensors
pip install gradio
```
⚠️ Stable Diffusion models are downloaded from Hugging Face and cached locally.

### 2. Prepare Data
Place your dataset files under:
data/
├── items.json

├── embeddings.pt   # auto-generated if not present

└── zalando_images/

### 3. Run the Demo
```
python app_basic.py
```
The interface will be available at:
http://127.0.0.1:7860

