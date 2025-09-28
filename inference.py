import os
import argparse
from typing import List, Optional
from PIL import Image

import torch
from diffusers import StableDiffusionPipeline, DDIMScheduler, AutoencoderKL

import open_clip
from transformers import PretrainedConfig
from rshf.taxabind import TaxaBind

# import your improved class (the one we discussed that supports model_type='bioclip'
# and accepts bioclip tokens / taxa_texts)
from ip_adapter import IPAdapter
from tqdm import tqdm
import json


# ---------- Utilities ----------

def make_scheduler() -> DDIMScheduler:
    return DDIMScheduler(
        num_train_timesteps=1000,
        beta_start=0.00085,
        beta_end=0.012,
        beta_schedule="scaled_linear",
        clip_sample=False,
        set_alpha_to_one=False,
        steps_offset=1,
    )

def make_pipe(base_model: str, vae_model: Optional[str], device: str) -> StableDiffusionPipeline:
    vae = AutoencoderKL.from_pretrained(vae_model).to(dtype=torch.float16) if vae_model else None
    pipe = StableDiffusionPipeline.from_pretrained(
        base_model,
        torch_dtype=torch.float16,
        scheduler=make_scheduler(),
        vae=vae,
        feature_extractor=None,
        safety_checker=None,
    ).to(device)
    return pipe

def image_grid(imgs: List[Image.Image], rows: int, cols: int) -> Image.Image:
    assert len(imgs) == rows * cols
    w, h = imgs[0].size
    grid = Image.new("RGB", size=(cols * w, rows * h))
    for i, im in enumerate(imgs):
        grid.paste(im, box=((i % cols) * w, (i // cols) * h))
    return grid

def save_images(images: List[Image.Image], out_dir: str, grid_name: str = "grid.png", cols: int = 4):
    os.makedirs(out_dir, exist_ok=True)
    for i, im in enumerate(images):
        im.save(os.path.join(out_dir, f"img_{i:02d}.png"))
    rows = (len(images) + cols - 1) // cols
    grid = image_grid(images, rows, cols)
    grid.save(os.path.join(out_dir, grid_name))

def save_images_per_class(images, save_dir, class_tag):
    os.makedirs(save_dir, exist_ok=True)
    for i, im in enumerate(images):
        im.save(os.path.join(save_dir, f"{class_tag}_sample{i:02d}.png"))


def class_dir_from_image_path(rel_path: str) -> str:
    """
    Given e.g.:
      train/04486_Animalia_..._immutabilis/f9f0....jpg
    return:
      04486_Animalia_..._immutabilis
    """
    return os.path.basename(os.path.dirname(rel_path))

# ---------- CLI ----------

def parse_args():
    p = argparse.ArgumentParser("IP-Adapter Inference (BioCLIP)")
    p.add_argument("--base_model", default="runwayml/stable-diffusion-v1-5")
    p.add_argument("--vae_model", default="stabilityai/sd-vae-ft-mse")
    p.add_argument("--ip_ckpt", required=True, help="Path to ip_adapter.bin or ip_adapter.safetensors")

    # BioCLIP inputs
    # p.add_argument("--taxa", nargs="+", required=True,help="One or more taxonomy strings, e.g. 'Animalia Chordata Aves ...'")

    # Generation params
    p.add_argument("--steps", type=int, default=50)
    p.add_argument("--guidance_scale", type=float, default=3.5, help="guidance scale")
    p.add_argument("--num_samples", type=int, default=4, help="images per prompt")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out_dir", default="outputs_bioclip")
    p.add_argument("--num_tokens", type=int, default=4, help="must match training")

    p.add_argument("--json_file", required=True, help="Path to JSON list of dicts.")
    p.add_argument("--model_type", default="bioclip", choices=["bioclip", "taxabind", "location"], help="Which model type was used during IP-Adapter training?")
    return p.parse_args()


# ---------- Main ----------

@torch.inference_mode()
def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 1) SD pipeline
    pipe = make_pipe(args.base_model, args.vae_model, device)

    # 2) BioCLIP model + tokenizer
    bioclip_model, _, _ = open_clip.create_model_and_transforms("hf-hub:imageomics/bioclip-2")
    bioclip_tok = open_clip.get_tokenizer("hf-hub:imageomics/bioclip-2")

    config = PretrainedConfig.from_pretrained("MVRL/taxabind-config")
    taxabind = TaxaBind(config)
    location_encoder = taxabind.get_location_encoder()
    taxabind_image_text_model   = taxabind.get_image_text_encoder()  # open_clip model
    taxabind_tokenizer = taxabind.get_tokenizer()   

    # 3) Load JSON
    with open(args.json_file, "r") as f:
        items = json.load(f)
    os.makedirs(args.out_dir, exist_ok=True)

    # --- NEW: de-duplicate by taxonomic_name (keep the FIRST full dict) ---
    seen = set()
    unique_items = []
    for ex in items:
        tax = ex["taxonomic_name"]
        if tax not in seen:
            seen.add(tax)
            unique_items.append(ex)

    print(f"Found {len(unique_items)} unique taxonomic names (from {len(items)} rows).")



    # 3) IP-Adapter wrapper (BioCLIP mode)
    ip_model = IPAdapter(
        pipe,
        image_encoder_path=bioclip_model,   # pass the *instance*
        ip_ckpt=args.ip_ckpt,
        device=device,
        model_type="bioclip"
    )

    # 4) Loop over entries and generate+save into per-class folder
    for idx, entry in tqdm(enumerate(unique_items, start=1), total=len(unique_items)):
        tax_name = entry["taxonomic_name"]
        rel_img_path = entry["image_file"]  # not used for conditioning; used to name the class dir
        class_dir = class_dir_from_image_path(rel_img_path)
        save_dir = os.path.join(args.out_dir, class_dir)

        # BioCLIP text tokens
        tokens = bioclip_tok(tax_name).to(device)

        # Generate (NOTE: call with bioclip_tokens; do NOT pass tokens to pil_image)
        images = ip_model.generate(
            pil_image=tokens,
            num_samples=args.num_samples,
            num_inference_steps=args.steps,
            seed=args.seed,
            guidance_scale=args.guidance_scale,
        )

        # Save per-class
        save_images_per_class(images, save_dir, class_tag=class_dir)
        print(f"[{idx+1}/{len(items)}] Saved {len(images)} images -> {save_dir}")

    print("Done.")


if __name__ == "__main__":
    main()
