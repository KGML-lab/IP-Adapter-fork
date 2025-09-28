import os
from typing import List

import torch
from diffusers import StableDiffusionPipeline
from diffusers.pipelines.controlnet import MultiControlNetModel
from PIL import Image
from safetensors import safe_open
from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection

from .utils import is_torch2_available, get_generator

if is_torch2_available():
    from .attention_processor import (
        AttnProcessor2_0 as AttnProcessor,
    )
    from .attention_processor import (
        CNAttnProcessor2_0 as CNAttnProcessor,
    )
    from .attention_processor import (
        IPAttnProcessor2_0 as IPAttnProcessor,
    )
else:
    from .attention_processor import AttnProcessor, CNAttnProcessor, IPAttnProcessor
from .resampler import Resampler


class ImageProjModel(torch.nn.Module):
    """Projection Model"""

    def __init__(self, cross_attention_dim=1024, clip_embeddings_dim=1024, clip_extra_context_tokens=4):
        super().__init__()

        self.generator = None
        self.cross_attention_dim = cross_attention_dim
        self.clip_extra_context_tokens = clip_extra_context_tokens
        self.proj = torch.nn.Linear(clip_embeddings_dim, self.clip_extra_context_tokens * cross_attention_dim)
        self.norm = torch.nn.LayerNorm(cross_attention_dim)

    def forward(self, image_embeds):
        embeds = image_embeds
        clip_extra_context_tokens = self.proj(embeds).reshape(
            -1, self.clip_extra_context_tokens, self.cross_attention_dim
        )
        clip_extra_context_tokens = self.norm(clip_extra_context_tokens)
        return clip_extra_context_tokens


class MLPProjModel(torch.nn.Module):
    """SD model with image prompt"""
    def __init__(self, cross_attention_dim=1024, clip_embeddings_dim=1024):
        super().__init__()
        
        self.proj = torch.nn.Sequential(
            torch.nn.Linear(clip_embeddings_dim, clip_embeddings_dim),
            torch.nn.GELU(),
            torch.nn.Linear(clip_embeddings_dim, cross_attention_dim),
            torch.nn.LayerNorm(cross_attention_dim)
        )
        
    def forward(self, image_embeds):
        clip_extra_context_tokens = self.proj(image_embeds)
        return clip_extra_context_tokens


class IPAdapter:
    def __init__(self, sd_pipe, image_encoder_path, ip_ckpt, device, num_tokens=4, model_type='clip', model_type_secondary=None, image_proj_secondary=False, image_encoder_path_secondary=None):
        self.device = device
        self.image_encoder_path = image_encoder_path
        self.ip_ckpt = ip_ckpt
        self.num_tokens = num_tokens
        self.model_type = model_type
        self.model_type_secondary = model_type_secondary
        self.image_proj_secondary = image_proj_secondary
        self.image_encoder_path_secondary = image_encoder_path_secondary

        self.pipe = sd_pipe.to(self.device)
        self.set_ip_adapter()

        # load image encoder
        if model_type == 'clip':
            self.image_encoder = CLIPVisionModelWithProjection.from_pretrained(self.image_encoder_path).to(
                self.device, dtype=torch.float16
            )
        elif model_type == 'bioclip' or  model_type == 'taxabind':
            self.image_encoder = image_encoder_path.to(self.device, dtype=torch.float16)
        elif model_type == 'location':
            self.image_encoder = image_encoder_path.to(self.device)

        if model_type_secondary == 'clip':
            self.image_encoder_secondary = CLIPVisionModelWithProjection.from_pretrained(self.image_encoder_path_secondary).to(
                self.device, dtype=torch.float16
            )
        elif model_type_secondary == 'bioclip' or  model_type_secondary == 'taxabind':
            self.image_encoder_secondary = image_encoder_path_secondary.to(self.device, dtype=torch.float16)    
        elif model_type_secondary == 'location':
            self.image_encoder_secondary = image_encoder_path_secondary.to(self.device)

        self.clip_image_processor = CLIPImageProcessor()
        # image proj model
        self.image_proj_model, self.image_proj_model_secondary = self.init_proj()

        self.load_ip_adapter()

    def init_proj(self):
        if self.model_type == "clip":
            image_encoder_dim = self.image_encoder.config.projection_dim
        elif self.model_type == "bioclip":
            # image_encoder_dim = bioclip.text_projection.shape[1]
            image_encoder_dim = 768
        elif self.model_type == "taxabind":
            # image_encoder_dim = location_encoder.config.hidden_size
            image_encoder_dim = 512
        elif self.model_type == "location":
            # image_encoder_dim = location_encoder.config.hidden_size
            image_encoder_dim = 512

        if self.model_type_secondary != self.model_type and self.model_type_secondary is not None:
            if self.model_type_secondary == "clip":
                if self.image_proj_secondary:
                    image_encoder2_dim = CLIPVisionModelWithProjection.from_pretrained(
                        "openai/clip-vit-large-patch14"
                    ).config.projection_dim
                else:
                    image_encoder_dim += CLIPVisionModelWithProjection.from_pretrained(
                    "openai/clip-vit-large-patch14"
                ).config.projection_dim
            elif self.model_type_secondary == "bioclip":
                if self.image_proj_secondary:
                    image_encoder2_dim = 768
                else:
                    image_encoder_dim += 768
            elif self.model_type_secondary == "taxabind":
                if self.image_proj_secondary:
                    image_encoder2_dim = 512
                else:
                    image_encoder_dim += 512
            elif self.model_type_secondary == "location":
                if self.image_proj_secondary:
                    image_encoder2_dim = 512
                else:
                    image_encoder_dim += 512
            print(f"Using two image encoders: {self.model_type} + {self.model_type_secondary}, with proj model: {self.image_proj_secondary}")
            print(f"Image encoder dims: {image_encoder_dim}")

        image_proj_model = ImageProjModel(
            cross_attention_dim=self.pipe.unet.config.cross_attention_dim,
            clip_embeddings_dim=image_encoder_dim,
            clip_extra_context_tokens=self.num_tokens,
        ).to(self.device, dtype=torch.float16)
        image_proj_model2 = None
        if self.model_type_secondary != self.model_type and self.model_type_secondary is not None and self.image_proj_secondary:
            image_proj_model2 = ImageProjModel(
                cross_attention_dim=self.pipe.unet.config.cross_attention_dim,
                clip_embeddings_dim=image_encoder2_dim,
                clip_extra_context_tokens=self.num_tokens,
            ).to(self.device, dtype=torch.float16)
            # combine two proj models
            # image_proj_model = torch.nn.ModuleList([image_proj_model, image_proj_model2])
        return image_proj_model, image_proj_model2 if image_proj_model2 else None

    def set_ip_adapter(self):
        unet = self.pipe.unet
        attn_procs = {}
        for name in unet.attn_processors.keys():
            cross_attention_dim = None if name.endswith("attn1.processor") else unet.config.cross_attention_dim
            if name.startswith("mid_block"):
                hidden_size = unet.config.block_out_channels[-1]
            elif name.startswith("up_blocks"):
                block_id = int(name[len("up_blocks.")])
                hidden_size = list(reversed(unet.config.block_out_channels))[block_id]
            elif name.startswith("down_blocks"):
                block_id = int(name[len("down_blocks.")])
                hidden_size = unet.config.block_out_channels[block_id]
            if cross_attention_dim is None:
                attn_procs[name] = AttnProcessor()
            else:
                attn_procs[name] = IPAttnProcessor(
                    hidden_size=hidden_size,
                    cross_attention_dim=cross_attention_dim,
                    scale=1.0,
                    num_tokens=self.num_tokens,
                ).to(self.device, dtype=torch.float16)
        unet.set_attn_processor(attn_procs)
        if hasattr(self.pipe, "controlnet"):
            if isinstance(self.pipe.controlnet, MultiControlNetModel):
                for controlnet in self.pipe.controlnet.nets:
                    controlnet.set_attn_processor(CNAttnProcessor(num_tokens=self.num_tokens))
            else:
                self.pipe.controlnet.set_attn_processor(CNAttnProcessor(num_tokens=self.num_tokens))

    def load_ip_adapter(self):
        if os.path.splitext(self.ip_ckpt)[-1] == ".safetensors":
            state_dict = {"image_proj": {}, "ip_adapter": {}}
            with safe_open(self.ip_ckpt, framework="pt", device="cpu") as f:
                for key in f.keys():
                    if key.startswith("image_proj_model_secondary."):
                        state_dict["image_proj_secondary"][key.split(".", 1)[1]] = f.get_tensor(key)
                    if key.startswith("image_proj."):
                        state_dict["image_proj"][key.replace("image_proj.", "")] = f.get_tensor(key)
                    elif key.startswith("ip_adapter."):
                        state_dict["ip_adapter"][key.replace("ip_adapter.", "")] = f.get_tensor(key)
        else:
            state_dict = torch.load(self.ip_ckpt, map_location="cpu")
        self.image_proj_model.load_state_dict(state_dict["image_proj"])
        ip_layers = torch.nn.ModuleList(self.pipe.unet.attn_processors.values())
        ip_layers.load_state_dict(state_dict["ip_adapter"])

    def get_image_embeds_(self, pil_image=None, clip_image_embeds=None, model_type=None, encoder=None):
        if encoder is None:
            encoder = self.image_encoder
        if model_type is None:
            model_type = self.model_type

        if model_type == 'clip':
                if pil_image is not None:
                    if isinstance(pil_image, Image.Image):
                        pil_image = [pil_image]
                    clip_image = self.clip_image_processor(images=pil_image, return_tensors="pt").pixel_values
                    clip_image_embeds = encoder(clip_image.to(self.device, dtype=torch.float16)).image_embeds
                else:
                    clip_image_embeds = clip_image_embeds.to(self.device, dtype=torch.float16)
                text_embeds = clip_image_embeds
        elif model_type == 'bioclip':
            text_emb = encoder.encode_text(pil_image)
        elif model_type == 'taxabind':
            text_emb = encoder.encode_text(pil_image)
        elif model_type == 'location':
            location_emb = encoder(pil_image).to(dtype=torch.float16)
            text_emb = location_emb
        return text_emb
    
    @torch.inference_mode()
    def get_image_embeds(self, pil_image=None, clip_image_embeds=None, pil_image2=None, clip_image_embeds2=None):

        text_emb = self.get_image_embeds_(pil_image=pil_image, clip_image_embeds=clip_image_embeds, encoder=self.image_encoder, model_type=self.model_type)  
        if pil_image2 is not None:  
            text_emb2 = self.get_image_embeds_(pil_image=pil_image2, clip_image_embeds=clip_image_embeds2, encoder=self.image_encoder_secondary, model_type=self.model_type_secondary)  
        
        if not self.model_type_secondary or self.model_type_secondary == self.model_type:
            image_prompt_embeds = self.image_proj_model(text_emb)
            uncond_image_prompt_embeds = self.image_proj_model(torch.zeros_like(text_emb))
            # standard route
            return image_prompt_embeds, uncond_image_prompt_embeds
        elif self.image_proj_secondary and self.model_type_secondary != self.model_type and self.model_type_secondary is not None:
            ### two seprate projections
            image_prompt_embeds = self.image_proj_model(text_emb)
            uncond_image_prompt_embeds = self.image_proj_model(torch.zeros_like(text_emb))
            print("text_emb2.shape:", text_emb2.shape)
            image_prompt_embeds2 = self.image_proj_model_secondary(text_emb2)
            uncond_image_prompt_embeds2 = self.image_proj_model_secondary(torch.zeros_like(text_emb2))
            image_prompt_embeds = torch.cat([image_prompt_embeds, image_prompt_embeds2], dim=1)
            uncond_image_prompt_embeds = torch.cat([uncond_image_prompt_embeds, uncond_image_prompt_embeds2], dim=1)
        elif not self.image_proj_secondary and self.model_type_secondary != self.model_type and self.model_type_secondary is not None:
            # concate before proj
            text_emb = torch.cat([text_emb, text_emb2], dim=1)
            image_prompt_embeds = self.image_proj_model(text_emb)
            uncond_image_prompt_embeds = self.image_proj_model(torch.zeros_like(text_emb))
        else:
            raise NotImplementedError

        return image_prompt_embeds, uncond_image_prompt_embeds

    def set_scale(self, scale):
        for attn_processor in self.pipe.unet.attn_processors.values():
            if isinstance(attn_processor, IPAttnProcessor):
                attn_processor.scale = scale

    def generate(
        self,
        pil_image=None,
        clip_image_embeds=None,
        prompt=None,
        negative_prompt=None,
        scale=1.0,
        num_samples=4,
        seed=None,
        guidance_scale=7.5,
        num_inference_steps=30,
        pil_image2=None, clip_image_embeds2=None,
        **kwargs,
    ):
        self.set_scale(scale)

        if pil_image is not None:
            num_prompts = 1 if isinstance(pil_image, Image.Image) else len(pil_image)
        else:
            num_prompts = clip_image_embeds.size(0)

        if prompt is None:
            prompt = "best quality, high quality"
        if negative_prompt is None:
            negative_prompt = "monochrome, lowres, bad anatomy, worst quality, low quality"

        if not isinstance(prompt, List):
            prompt = [prompt] * num_prompts
        if not isinstance(negative_prompt, List):
            negative_prompt = [negative_prompt] * num_prompts
        # image_prompt_embeds, uncond_image_prompt_embeds = self.get_image_embeds(
        #     pil_image=pil_image, clip_image_embeds=clip_image_embeds
        # )

        image_prompt_embeds, uncond_image_prompt_embeds = self.get_image_embeds(
            pil_image=pil_image,
            clip_image_embeds=clip_image_embeds,
            pil_image2=pil_image2,
            clip_image_embeds2=clip_image_embeds2,
        )
        bs_embed, seq_len, _ = image_prompt_embeds.shape
        image_prompt_embeds = image_prompt_embeds.repeat(1, num_samples, 1)
        image_prompt_embeds = image_prompt_embeds.view(bs_embed * num_samples, seq_len, -1)
        uncond_image_prompt_embeds = uncond_image_prompt_embeds.repeat(1, num_samples, 1)
        uncond_image_prompt_embeds = uncond_image_prompt_embeds.view(bs_embed * num_samples, seq_len, -1)

        with torch.inference_mode():
            prompt_embeds_, negative_prompt_embeds_ = self.pipe.encode_prompt(
                prompt,
                device=self.device,
                num_images_per_prompt=num_samples,
                do_classifier_free_guidance=True,
                negative_prompt=negative_prompt,
            )
            prompt_embeds = torch.cat([prompt_embeds_, image_prompt_embeds], dim=1)
            negative_prompt_embeds = torch.cat([negative_prompt_embeds_, uncond_image_prompt_embeds], dim=1)

        generator = get_generator(seed, self.device)

        images = self.pipe(
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            generator=generator,
            **kwargs,
        ).images

        return images


class IPAdapterXL(IPAdapter):
    """SDXL"""

    def generate(
        self,
        pil_image,
        prompt=None,
        negative_prompt=None,
        scale=1.0,
        num_samples=4,
        seed=None,
        num_inference_steps=30,
        **kwargs,
    ):
        self.set_scale(scale)

        num_prompts = 1 if isinstance(pil_image, Image.Image) else len(pil_image)

        if prompt is None:
            prompt = "best quality, high quality"
        if negative_prompt is None:
            negative_prompt = "monochrome, lowres, bad anatomy, worst quality, low quality"

        if not isinstance(prompt, List):
            prompt = [prompt] * num_prompts
        if not isinstance(negative_prompt, List):
            negative_prompt = [negative_prompt] * num_prompts

        image_prompt_embeds, uncond_image_prompt_embeds = self.get_image_embeds(pil_image)
        bs_embed, seq_len, _ = image_prompt_embeds.shape
        image_prompt_embeds = image_prompt_embeds.repeat(1, num_samples, 1)
        image_prompt_embeds = image_prompt_embeds.view(bs_embed * num_samples, seq_len, -1)
        uncond_image_prompt_embeds = uncond_image_prompt_embeds.repeat(1, num_samples, 1)
        uncond_image_prompt_embeds = uncond_image_prompt_embeds.view(bs_embed * num_samples, seq_len, -1)

        with torch.inference_mode():
            (
                prompt_embeds,
                negative_prompt_embeds,
                pooled_prompt_embeds,
                negative_pooled_prompt_embeds,
            ) = self.pipe.encode_prompt(
                prompt,
                num_images_per_prompt=num_samples,
                do_classifier_free_guidance=True,
                negative_prompt=negative_prompt,
            )
            prompt_embeds = torch.cat([prompt_embeds, image_prompt_embeds], dim=1)
            negative_prompt_embeds = torch.cat([negative_prompt_embeds, uncond_image_prompt_embeds], dim=1)

        self.generator = get_generator(seed, self.device)
        
        images = self.pipe(
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
            num_inference_steps=num_inference_steps,
            generator=self.generator,
            **kwargs,
        ).images

        return images


class IPAdapterPlus(IPAdapter):
    """IP-Adapter with fine-grained features"""

    def init_proj(self):
        image_proj_model = Resampler(
            dim=self.pipe.unet.config.cross_attention_dim,
            depth=4,
            dim_head=64,
            heads=12,
            num_queries=self.num_tokens,
            embedding_dim=self.image_encoder.config.hidden_size,
            output_dim=self.pipe.unet.config.cross_attention_dim,
            ff_mult=4,
        ).to(self.device, dtype=torch.float16)
        return image_proj_model

    @torch.inference_mode()
    def get_image_embeds(self, pil_image=None, clip_image_embeds=None):
        if isinstance(pil_image, Image.Image):
            pil_image = [pil_image]
        clip_image = self.clip_image_processor(images=pil_image, return_tensors="pt").pixel_values
        clip_image = clip_image.to(self.device, dtype=torch.float16)
        clip_image_embeds = self.image_encoder(clip_image, output_hidden_states=True).hidden_states[-2]
        image_prompt_embeds = self.image_proj_model(clip_image_embeds)
        uncond_clip_image_embeds = self.image_encoder(
            torch.zeros_like(clip_image), output_hidden_states=True
        ).hidden_states[-2]
        uncond_image_prompt_embeds = self.image_proj_model(uncond_clip_image_embeds)
        return image_prompt_embeds, uncond_image_prompt_embeds


class IPAdapterFull(IPAdapterPlus):
    """IP-Adapter with full features"""

    def init_proj(self):
        image_proj_model = MLPProjModel(
            cross_attention_dim=self.pipe.unet.config.cross_attention_dim,
            clip_embeddings_dim=self.image_encoder.config.hidden_size,
        ).to(self.device, dtype=torch.float16)
        return image_proj_model


class IPAdapterPlusXL(IPAdapter):
    """SDXL"""

    def init_proj(self):
        image_proj_model = Resampler(
            dim=1280,
            depth=4,
            dim_head=64,
            heads=20,
            num_queries=self.num_tokens,
            embedding_dim=self.image_encoder.config.hidden_size,
            output_dim=self.pipe.unet.config.cross_attention_dim,
            ff_mult=4,
        ).to(self.device, dtype=torch.float16)
        return image_proj_model

    @torch.inference_mode()
    def get_image_embeds(self, pil_image):
        if isinstance(pil_image, Image.Image):
            pil_image = [pil_image]
        clip_image = self.clip_image_processor(images=pil_image, return_tensors="pt").pixel_values
        clip_image = clip_image.to(self.device, dtype=torch.float16)
        clip_image_embeds = self.image_encoder(clip_image, output_hidden_states=True).hidden_states[-2]
        image_prompt_embeds = self.image_proj_model(clip_image_embeds)
        uncond_clip_image_embeds = self.image_encoder(
            torch.zeros_like(clip_image), output_hidden_states=True
        ).hidden_states[-2]
        uncond_image_prompt_embeds = self.image_proj_model(uncond_clip_image_embeds)
        return image_prompt_embeds, uncond_image_prompt_embeds

    def generate(
        self,
        pil_image,
        prompt=None,
        negative_prompt=None,
        scale=1.0,
        num_samples=4,
        seed=None,
        num_inference_steps=30,
        **kwargs,
    ):
        self.set_scale(scale)

        num_prompts = 1 if isinstance(pil_image, Image.Image) else len(pil_image)

        if prompt is None:
            prompt = "best quality, high quality"
        if negative_prompt is None:
            negative_prompt = "monochrome, lowres, bad anatomy, worst quality, low quality"

        if not isinstance(prompt, List):
            prompt = [prompt] * num_prompts
        if not isinstance(negative_prompt, List):
            negative_prompt = [negative_prompt] * num_prompts

        image_prompt_embeds, uncond_image_prompt_embeds = self.get_image_embeds(pil_image)
        bs_embed, seq_len, _ = image_prompt_embeds.shape
        image_prompt_embeds = image_prompt_embeds.repeat(1, num_samples, 1)
        image_prompt_embeds = image_prompt_embeds.view(bs_embed * num_samples, seq_len, -1)
        uncond_image_prompt_embeds = uncond_image_prompt_embeds.repeat(1, num_samples, 1)
        uncond_image_prompt_embeds = uncond_image_prompt_embeds.view(bs_embed * num_samples, seq_len, -1)

        with torch.inference_mode():
            (
                prompt_embeds,
                negative_prompt_embeds,
                pooled_prompt_embeds,
                negative_pooled_prompt_embeds,
            ) = self.pipe.encode_prompt(
                prompt,
                num_images_per_prompt=num_samples,
                do_classifier_free_guidance=True,
                negative_prompt=negative_prompt,
            )
            prompt_embeds = torch.cat([prompt_embeds, image_prompt_embeds], dim=1)
            negative_prompt_embeds = torch.cat([negative_prompt_embeds, uncond_image_prompt_embeds], dim=1)

        generator = get_generator(seed, self.device)

        images = self.pipe(
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
            num_inference_steps=num_inference_steps,
            generator=generator,
            **kwargs,
        ).images

        return images
