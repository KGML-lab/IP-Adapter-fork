import os
import random
import argparse
from pathlib import Path
import json
import itertools
import time
import numpy as np

import torch
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.cuda.amp import GradScaler
from torchvision import transforms
from PIL import Image
from transformers import CLIPImageProcessor
from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel
from transformers import CLIPTextModel, CLIPTokenizer, CLIPVisionModelWithProjection, CLIPTextModelWithProjection

from ip_adapter.ip_adapter import ImageProjModel
from ip_adapter.utils import is_torch2_available
if is_torch2_available():
    from ip_adapter.attention_processor import IPAttnProcessor2_0 as IPAttnProcessor, AttnProcessor2_0 as AttnProcessor
else:
    from ip_adapter.attention_processor import IPAttnProcessor, AttnProcessor

import open_clip
from transformers import PretrainedConfig
from rshf.taxabind import TaxaBind


# Dataset
class MyDataset(torch.utils.data.Dataset):

    def __init__(self, json_file, tokenizer, size=512, t_drop_rate=0.05, i_drop_rate=0.05, ti_drop_rate=0.05, image_root_path="", bioclip_tokenizer=None,  taxabind_tokenizer=None, model_type="bioclip"):
        super().__init__()

        self.tokenizer = tokenizer
        self.size = size
        self.i_drop_rate = i_drop_rate
        self.bioclip_tokenizer = bioclip_tokenizer
        self.t_drop_rate = t_drop_rate
        self.ti_drop_rate = ti_drop_rate
        self.image_root_path = image_root_path
        self.taxabind_tokenizer = taxabind_tokenizer
        self.model_type = model_type

        self.data = json.load(open(json_file)) # list of dict: [{"image_file": "1.png", "text": "A dog"}]

        self.transform = transforms.Compose([
            transforms.Resize(self.size, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(self.size),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])
        self.clip_image_processor = CLIPImageProcessor()
        
    def __getitem__(self, idx):
        item = self.data[idx] 
        text = item["text"]
        image_file = item["image_file"]
        taxonomic_name = item["taxonomic_name"]
        latitude = item['latitude']
        longitude = item['longitude']
        location = [latitude, longitude]

        # read image
        raw_image = Image.open(os.path.join(self.image_root_path, image_file))
        image = self.transform(raw_image.convert("RGB"))
        clip_image = self.clip_image_processor(images=raw_image, return_tensors="pt").pixel_values
        
        # drop
        drop_image_embed = 0
        rand_num = random.random()
        if rand_num < self.i_drop_rate:
            drop_image_embed = 1
        elif rand_num < (self.i_drop_rate + self.t_drop_rate):
            text = ""
        elif rand_num < (self.i_drop_rate + self.t_drop_rate + self.ti_drop_rate):
            text = ""
            drop_image_embed = 1
        # get text and tokenize
        text_input_ids = self.tokenizer(
            text,
            max_length=self.tokenizer.model_max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        ).input_ids

        if self.model_type == "bioclip":
            taxa_tokenized = self.bioclip_tokenizer(taxonomic_name)
        elif self.model_type == "clip":
            taxa_tokenized = self.bioclip_tokenizer(
                taxonomic_name,
                max_length=self.tokenizer.model_max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt"
            ).input_ids

        taxabind_tokenized = self.taxabind_tokenizer(taxonomic_name)

        return {
            "image": image,
            "text_input_ids": text_input_ids,
            "clip_image": clip_image,
            "drop_image_embed": drop_image_embed,
            "taxa_tokenized": taxa_tokenized,
            "location": torch.tensor(location),
            "taxabind_tokenized": taxabind_tokenized
        }

    def __len__(self):
        return len(self.data)
    

def collate_fn(data):
    images = torch.stack([example["image"] for example in data])
    text_input_ids = torch.cat([example["text_input_ids"] for example in data], dim=0)
    taxa_tokenized = torch.cat([example["taxa_tokenized"] for example in data], dim=0)
    clip_images = torch.cat([example["clip_image"] for example in data], dim=0)
    drop_image_embeds = [example["drop_image_embed"] for example in data]
    location = torch.stack([example["location"] for example in data], dim=0)
    taxabind_tokenized = torch.cat([example["taxabind_tokenized"] for example in data], dim=0)

    return {
        "images": images,
        "text_input_ids": text_input_ids,
        "clip_images": clip_images,
        "drop_image_embeds": drop_image_embeds,
        "taxa_tokenized": taxa_tokenized,
        "location": location,
        "taxabind_tokenized": taxabind_tokenized
    }
    

class IPAdapter(torch.nn.Module):
    """IP-Adapter"""
    def __init__(self, unet, image_proj_model, adapter_modules, ckpt_path=None):
        super().__init__()
        self.unet = unet
        self.image_proj_model = image_proj_model
        self.adapter_modules = adapter_modules

        if ckpt_path is not None:
            self.load_from_checkpoint(ckpt_path)

    def forward(self, noisy_latents, timesteps, encoder_hidden_states, image_embeds, projection_flag=True):
        if not projection_flag:
            ip_tokens = image_embeds.unsqueeze(1) # [B, 1, 768]
        else:
            ip_tokens = self.image_proj_model(image_embeds) # [B, 4, 768]
        encoder_hidden_states = torch.cat([encoder_hidden_states, ip_tokens], dim=1) # [B, 81, 768]
        # Predict the noise residual
        noise_pred = self.unet(noisy_latents, timesteps, encoder_hidden_states).sample
        return noise_pred

    def load_from_checkpoint(self, ckpt_path: str):
        # Calculate original checksums
        orig_ip_proj_sum = torch.sum(torch.stack([torch.sum(p) for p in self.image_proj_model.parameters()]))
        orig_adapter_sum = torch.sum(torch.stack([torch.sum(p) for p in self.adapter_modules.parameters()]))

        state_dict = torch.load(ckpt_path, map_location="cpu")

        # Load state dict for image_proj_model and adapter_modules
        self.image_proj_model.load_state_dict(state_dict["image_proj"], strict=True)
        self.adapter_modules.load_state_dict(state_dict["ip_adapter"], strict=True)

        # Calculate new checksums
        new_ip_proj_sum = torch.sum(torch.stack([torch.sum(p) for p in self.image_proj_model.parameters()]))
        new_adapter_sum = torch.sum(torch.stack([torch.sum(p) for p in self.adapter_modules.parameters()]))

        # Verify if the weights have changed
        assert orig_ip_proj_sum != new_ip_proj_sum, "Weights of image_proj_model did not change!"
        assert orig_adapter_sum != new_adapter_sum, "Weights of adapter_modules did not change!"

        print(f"Successfully loaded weights from checkpoint {ckpt_path}")

    
    
def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default=None,
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--pretrained_ip_adapter_path",
        type=str,
        default=None,
        help="Path to pretrained ip adapter model. If not specified weights are initialized randomly.",
    )
    parser.add_argument(
        "--data_json_file",
        type=str,
        default=None,
        required=True,
        help="Training data",
    )
    parser.add_argument(
        "--data_root_path",
        type=str,
        default="",
        required=True,
        help="Training data root path",
    )
    parser.add_argument(
        "--image_encoder_path",
        type=str,
        default=None,
        required=True,
        help="Path to CLIP image encoder",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="sd-ip_adapter",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help=(
            "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
            " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."
        ),
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=512,
        help=(
            "The resolution for input images"
        ),
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="Learning rate to use.",
    )
    parser.add_argument("--weight_decay", type=float, default=1e-2, help="Weight decay to use.")
    parser.add_argument("--num_train_epochs", type=int, default=100)
    parser.add_argument(
        "--train_batch_size", type=int, default=8, help="Batch size (per device) for the training dataloader."
    )
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=0,
        help=(
            "Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process."
        ),
    )
    parser.add_argument(
        "--save_steps",
        type=int,
        default=2000,
        help=(
            "Save a checkpoint of the training state every X updates"
        ),
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default=None,
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
            " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
        ),
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="tensorboard",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`'
            ' (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.'
        ),
    )
    parser.add_argument(
        "--clip_extra_context_tokens",
        type=int,
        default=4,
        help=(
            "Number of extra context tokens to use for the CLIP model"
        ),
    )

    parser.add_argument(
        "--no_projection_layer",
        action="store_false",
        default=True,
        help=(
            "Disable projection layer when this flag is present"
        ),
    )

    parser.add_argument(
        "--image_encoder_embeddings_dim",
        type=int,
        default=1024,
        help=(
            "The dimension of the CLIP Image Vision embeddings"
        ),
    )


    parser.add_argument(
        "--image_encoder",
        type=str,
        default="bioclip",
        help=(
            "The type of image encoder to use: clip or bioclip"
        ),
    )
    
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Enable gradient checkpointing to save memory at the cost of slower backward pass."
    )
    
    parser.add_argument(
        "--max_grad_norm",
        type=float,
        default=1.0,
        help="Max gradient norm for gradient clipping."
    )
    
    parser.add_argument(
        "--disable_torch_compile",
        action="store_true",
        help="Disable torch.compile() optimization (PyTorch 2.0+). Use if compilation causes issues."
    )
    
    # parser.add_argument(
    #     "--lr_scheduler",
    #     type=str,
    #     default="constant",
    #     choices=["constant", "linear", "cosine", "cosine_with_restarts", "polynomial", "constant_with_warmup"],
    #     help="The scheduler type to use (commented out - original IP-Adapter doesn't use scheduler)."
    # )
    # 
    # parser.add_argument(
    #     "--lr_warmup_steps",
    #     type=int,
    #     default=500,
    #     help="Number of steps for the warmup in the lr scheduler (commented out)."
    # )

    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help="Path to checkpoint directory to resume training from (e.g., /path/to/checkpoint-10000)."
    )
    
    args = parser.parse_args()
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    return args


def setup_distributed():
    """
    Initialize distributed training for both single-node and multi-node setups.
    Supports torchrun, torch.distributed.launch, and SLURM.
    """
    # torchrun sets RANK, WORLD_SIZE, LOCAL_RANK
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ['LOCAL_RANK'])
        
    # SLURM environment
    elif 'SLURM_PROCID' in os.environ:
        rank = int(os.environ['SLURM_PROCID'])
        world_size = int(os.environ['SLURM_NTASKS'])
        
        # CRITICAL FIX for multi-node: local_rank is rank within the node
        if 'SLURM_LOCALID' in os.environ:
            local_rank = int(os.environ['SLURM_LOCALID'])
        else:
            # Fallback: calculate from node and tasks per node
            gpus_per_node = int(os.environ.get('SLURM_GPUS_PER_NODE', torch.cuda.device_count()))
            local_rank = rank % gpus_per_node
        
        # SLURM doesn't set MASTER_ADDR/MASTER_PORT by default
        if 'MASTER_ADDR' not in os.environ:
            # Get the first node in the job's node list
            import subprocess
            cmd = 'scontrol show hostnames ' + os.environ['SLURM_JOB_NODELIST']
            master_addr = subprocess.check_output(cmd.split()).decode().strip().split('\n')[0]
            os.environ['MASTER_ADDR'] = master_addr
        
        if 'MASTER_PORT' not in os.environ:
            os.environ['MASTER_PORT'] = '29500'
    
    else:
        print("Not using distributed mode")
        return 0, 1, 0
    
    # Set the device
    torch.cuda.set_device(local_rank)
    
    # Initialize process group
    dist.init_process_group(backend='nccl', init_method='env://', world_size=world_size, rank=rank)
    
    # Synchronize all processes
    dist.barrier()
    
    # Print info from rank 0
    if rank == 0:
        print(f"Distributed training setup:")
        print(f"  World size: {world_size}")
        print(f"  Rank: {rank}")
        print(f"  Local rank: {local_rank}")
        print(f"  Master addr: {os.environ.get('MASTER_ADDR', 'N/A')}")
        print(f"  Master port: {os.environ.get('MASTER_PORT', 'N/A')}")
        if 'SLURM_JOB_NODELIST' in os.environ:
            print(f"  Nodes: {os.environ['SLURM_JOB_NODELIST']}")
            print(f"  Tasks per node: {os.environ.get('SLURM_TASKS_PER_NODE', 'N/A')}")
    
    return rank, world_size, local_rank


def cleanup_distributed():
    """Cleanup distributed training"""
    if dist.is_initialized():
        dist.destroy_process_group()


def setup_logging(rank, args):
    """Setup logging for main process only"""
    if rank == 0:
        # Setup tensorboard or wandb
        if args.report_to == "tensorboard":
            from torch.utils.tensorboard import SummaryWriter
            logging_dir = Path(args.output_dir, args.logging_dir)
            os.makedirs(logging_dir, exist_ok=True)
            writer = SummaryWriter(logging_dir)
            return writer
        elif args.report_to == "wandb":
            import wandb
            wandb.init(project="ip-adapter", dir=args.output_dir, config=vars(args))
            return wandb
    return None

    
def main():
    args = parse_args()
    
    # Setup distributed training
    rank, world_size, local_rank = setup_distributed()
    
    if rank == 0:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)
    
    # Setup logging
    logger = setup_logging(rank, args)

    # Set device
    device = torch.device(f'cuda:{local_rank}')
    
    # Load scheduler, tokenizer and models.
    noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")
    tokenizer = CLIPTokenizer.from_pretrained(args.pretrained_model_name_or_path, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="text_encoder")
    vae = AutoencoderKL.from_pretrained(args.pretrained_model_name_or_path, subfolder="vae")
    unet = UNet2DConditionModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="unet")
    image_encoder = CLIPVisionModelWithProjection.from_pretrained(args.image_encoder_path)

    bioclip, preprocess_train, preprocess_val = open_clip.create_model_and_transforms('hf-hub:imageomics/bioclip-2')
    bioclip_tokenizer = open_clip.get_tokenizer('hf-hub:imageomics/bioclip-2')

    config = PretrainedConfig.from_pretrained("MVRL/taxabind-config")
    taxabind = TaxaBind(config)
    location_encoder = taxabind.get_location_encoder()
    taxabind_image_text_model   = taxabind.get_image_text_encoder()  # open_clip model
    taxabind_tokenizer = taxabind.get_tokenizer()   

    if args.image_encoder == "clip":
        clip_ckpt = "openai/clip-vit-large-patch14"   # matches SD 1.x
        clip_text_with_proj_tokenizer  = CLIPTokenizer.from_pretrained(clip_ckpt)
        clip_text_with_proj = CLIPTextModelWithProjection.from_pretrained(clip_ckpt).eval()
    

    # freeze parameters of models to save more memory
    unet.requires_grad_(False)
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    image_encoder.requires_grad_(False)
    
    bioclip.requires_grad_(False)
    location_encoder.requires_grad_(False).eval()
    taxabind_image_text_model.requires_grad_(False).eval()
    if args.image_encoder == "clip":
        clip_text_with_proj.requires_grad_(False)


    if args.image_encoder == "image":
        image_encoder_dim = image_encoder.config.projection_dim
    elif args.image_encoder == "bioclip":
        # image_encoder_dim = bioclip.text_projection.shape[1]
        image_encoder_dim = 768
    elif args.image_encoder == "taxabind":
        # image_encoder_dim = location_encoder.config.hidden_size
        image_encoder_dim = 512
    elif args.image_encoder == "location":
        # image_encoder_dim = location_encoder.config.hidden_size
        image_encoder_dim = 512
    elif args.image_encoder == "clip":
        image_encoder_dim = clip_text_with_proj.config.projection_dim

    if rank == 0:
        print('Training the IP-Adapter with image encoder: ', args.image_encoder)
    
    #ip-adapter
    image_proj_model = ImageProjModel(
        cross_attention_dim=unet.config.cross_attention_dim,
        # clip_embeddings_dim=image_encoder.config.projection_dim, # TODO: change this for bioclip
        clip_embeddings_dim=image_encoder_dim,
        clip_extra_context_tokens=args.clip_extra_context_tokens,
    )
    # init adapter modules
    attn_procs = {}
    unet_sd = unet.state_dict()
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
            layer_name = name.split(".processor")[0]
            weights = {
                "to_k_ip.weight": unet_sd[layer_name + ".to_k.weight"],
                "to_v_ip.weight": unet_sd[layer_name + ".to_v.weight"],
            }
            attn_procs[name] = IPAttnProcessor(hidden_size=hidden_size, cross_attention_dim=cross_attention_dim)
            attn_procs[name].load_state_dict(weights)
    unet.set_attn_processor(attn_procs)
    adapter_modules = torch.nn.ModuleList(unet.attn_processors.values())
    
    ip_adapter = IPAdapter(unet, image_proj_model, adapter_modules, args.pretrained_ip_adapter_path)
    
    # Enable gradient checkpointing if requested
    if args.gradient_checkpointing:
        unet.enable_gradient_checkpointing()
        if rank == 0:
            print("Gradient checkpointing enabled for UNet")
    
    # Enable torch.compile() for faster training (PyTorch 2.0+)
    # This provides 30-50% speedup with minimal overhead
    # For PyTorch 2.3.1: excellent compile support with reduce-overhead mode
    if not args.disable_torch_compile and hasattr(torch, 'compile') and torch.__version__ >= "2.0.0":
        if rank == 0:
            print(f"PyTorch {torch.__version__} detected. Compiling UNet with torch.compile()...")
            print("Note: First iteration will be slower (~30-60s) due to compilation, then much faster!")
            print("To disable: add --disable_torch_compile flag")
        
        # PyTorch 2.3.1 supports these modes:
        # - "reduce-overhead": Best for training (optimizes memory and speed)
        # - "max-autotune": Slower compile but potentially faster training
        # - "default": Balanced
        ip_adapter.unet = torch.compile(ip_adapter.unet, mode="reduce-overhead")
        
        if rank == 0:
            print("✓ UNet compiled successfully! Training will be 30-50% faster after first iteration.")
    elif args.disable_torch_compile and rank == 0:
        print("torch.compile() disabled via --disable_torch_compile flag")
    
    # Determine weight dtype
    weight_dtype = torch.float32
    if args.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif args.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16
    
    # Move models to device
    unet.to(device, dtype=weight_dtype)
    vae.to(device, dtype=weight_dtype)
    text_encoder.to(device, dtype=weight_dtype)
    image_encoder.to(device, dtype=weight_dtype)
    bioclip.to(device, dtype=weight_dtype)
    location_encoder.to(device, dtype=torch.float32) # location encoder in fp32 as it is small
    taxabind_image_text_model.to(device, dtype=torch.float32)

    if args.image_encoder == "clip":
        clip_text_with_proj.to(device, dtype=weight_dtype)
    
    # Move ip_adapter to device and wrap with DDP
    ip_adapter = ip_adapter.to(device)
    
    # DDP with performance optimizations
    # - broadcast_buffers=False: No buffers to sync (image_proj and adapters have no buffers)
    # - gradient_as_bucket_view=True: Avoids extra gradient copy, saves memory and time
    # - static_graph=False: Set True if graph doesn't change (doesn't help much here)
    ip_adapter = DDP(
        ip_adapter, 
        device_ids=[local_rank], 
        output_device=local_rank, 
        find_unused_parameters=False,
        broadcast_buffers=False,  # No buffers to sync
        gradient_as_bucket_view=True  # PyTorch 1.7+, avoids copy
    )
    
    # Optimizer - trains both image_proj_model AND adapter_modules (decoupled cross-attention)
    # adapter_modules contains the to_k_ip and to_v_ip parameters in UNet's cross-attention layers
    params_to_opt = itertools.chain(ip_adapter.module.image_proj_model.parameters(),  
                                    ip_adapter.module.adapter_modules.parameters())
    optimizer = torch.optim.AdamW(params_to_opt, lr=args.learning_rate, weight_decay=args.weight_decay)
    
    # Print trainable parameter count
    if rank == 0:
        num_params_image_proj = sum(p.numel() for p in ip_adapter.module.image_proj_model.parameters() if p.requires_grad)
        num_params_adapter = sum(p.numel() for p in ip_adapter.module.adapter_modules.parameters() if p.requires_grad)
        print(f"Trainable parameters:")
        print(f"  Image projection model: {num_params_image_proj / 1e6:.2f}M")
        print(f"  Adapter modules (cross-attention): {num_params_adapter / 1e6:.2f}M")
        print(f"  Total: {(num_params_image_proj + num_params_adapter) / 1e6:.2f}M")
        
        # Verify optimizer has all trainable params
        optimizer_params = sum(p.numel() for group in optimizer.param_groups for p in group['params'])
        print(f"  Optimizer tracking: {optimizer_params / 1e6:.2f}M")
        assert optimizer_params == num_params_image_proj + num_params_adapter, \
            "Optimizer parameter count mismatch! Some trainable params not being optimized."
    
    # LR Scheduler (commented out - original IP-Adapter doesn't use it)
    # Uncomment below to enable learning rate scheduling with warmup
    # from diffusers.optimization import get_scheduler
    # total_steps = args.num_train_epochs * len(train_dataset) // (args.train_batch_size * world_size)
    # lr_scheduler = get_scheduler(
    #     args.lr_scheduler,
    #     optimizer=optimizer,
    #     num_warmup_steps=args.lr_warmup_steps,
    #     num_training_steps=total_steps,
    # )
    # if rank == 0:
    #     print(f"LR Scheduler: {args.lr_scheduler}, Warmup steps: {args.lr_warmup_steps}, Total steps: {total_steps}")
    
    # Setup GradScaler for mixed precision (only for fp16, not bf16)
    # PyTorch 2.3.1 compatibility: explicitly set enabled=True
    scaler = GradScaler(enabled=True) if args.mixed_precision == "fp16" else None
    
    # Determine autocast dtype
    if args.mixed_precision == "fp16":
        autocast_dtype = torch.float16
    elif args.mixed_precision == "bf16":
        autocast_dtype = torch.bfloat16
    else:
        autocast_dtype = torch.float32
    
    # dataloader with DistributedSampler
    if args.image_encoder == "clip":
        train_dataset = MyDataset(args.data_json_file, tokenizer=tokenizer, size=args.resolution, image_root_path=args.data_root_path, bioclip_tokenizer=clip_text_with_proj_tokenizer, taxabind_tokenizer=taxabind_tokenizer, model_type="clip")
    else:
        train_dataset = MyDataset(args.data_json_file, tokenizer=tokenizer, size=args.resolution, image_root_path=args.data_root_path, bioclip_tokenizer=bioclip_tokenizer, taxabind_tokenizer=taxabind_tokenizer)
    
    # Use DistributedSampler
    train_sampler = DistributedSampler(
        train_dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True,
        drop_last=False  # Following Accelerate behavior (doesn't drop last batch)
    )
    
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        sampler=train_sampler,
        collate_fn=collate_fn,
        batch_size=args.train_batch_size,  # Per-GPU batch size
        num_workers=args.dataloader_num_workers,
        pin_memory=True,
        persistent_workers=True if args.dataloader_num_workers > 0 else False,  # Keep workers alive
    )
    
    if rank == 0:
        print(f"Total dataset size: {len(train_dataset)}")
        print(f"Per-GPU batch size: {args.train_batch_size}")
        print(f"Total batch size (all GPUs): {args.train_batch_size * world_size}")
        print(f"Number of GPUs: {world_size}")
        print(f"Number of training steps per epoch: {len(train_dataloader)}")
        print(f"Gradient checkpointing: {args.gradient_checkpointing}")
        print(f"Gradient clipping max norm: {args.max_grad_norm}")
    
    # Setup progress bar (only on main process)
    from tqdm.auto import tqdm
    total_steps = args.num_train_epochs * len(train_dataloader)
    progress_bar = tqdm(range(total_steps), disable=(rank != 0))
    progress_bar.set_description("Training")
    
    # Initialize optimizer gradients to zero before training
    optimizer.zero_grad()
    
    # Resume from checkpoint if specified
    global_step = 0
    starting_epoch = 0
    if args.resume_from_checkpoint is not None:
        if rank == 0:
            print(f"\nResuming from checkpoint: {args.resume_from_checkpoint}")
        
        # Load model state (ip_adapter.bin format)
        model_path = os.path.join(args.resume_from_checkpoint, "ip_adapter.bin")
        if os.path.exists(model_path):
            state_dict = torch.load(model_path, map_location="cpu")
            ip_adapter.module.image_proj_model.load_state_dict(state_dict["image_proj"], strict=True)
            ip_adapter.module.adapter_modules.load_state_dict(state_dict["ip_adapter"], strict=True)
            if rank == 0:
                print("  ✓ Loaded model weights from ip_adapter.bin")
        
        # Load optimizer state
        optimizer_path = os.path.join(args.resume_from_checkpoint, "optimizer.bin")
        if os.path.exists(optimizer_path):
            opt_state = torch.load(optimizer_path, map_location="cpu")
            optimizer.load_state_dict(opt_state["optimizer"])
            global_step = opt_state.get("global_step", 0)
            starting_epoch = opt_state.get("epoch", 0)
            if rank == 0:
                print(f"  ✓ Loaded optimizer state (global_step={global_step}, epoch={starting_epoch})")
        
        # Load GradScaler state (if using FP16)
        scaler_path = os.path.join(args.resume_from_checkpoint, "scaler.pt")
        if scaler is not None and os.path.exists(scaler_path):
            scaler.load_state_dict(torch.load(scaler_path, map_location="cpu"))
            if rank == 0:
                print("  ✓ Loaded GradScaler state")
        
        # Load random states for reproducibility
        random_states_path = os.path.join(args.resume_from_checkpoint, f"random_states_{rank}.pkl")
        if os.path.exists(random_states_path):
            rng_state = torch.load(random_states_path, map_location="cpu")
            random.setstate(rng_state['python'])
            np.random.set_state(rng_state['numpy'])
            torch.random.set_rng_state(rng_state['cpu'])
            if torch.cuda.is_available():
                torch.cuda.set_rng_state_all(rng_state['cuda'])
            if rank == 0:
                print(f"  ✓ Loaded random states for rank {rank}")
        
        # Update progress bar to reflect resumed state
        if rank == 0:
            progress_bar.update(global_step)
            print(f"  → Resuming training from epoch {starting_epoch}, step {global_step}\n")
    
    for epoch in range(starting_epoch, args.num_train_epochs):
        # Set epoch for DistributedSampler (important for proper shuffling)
        train_sampler.set_epoch(epoch)
        
        ip_adapter.train()
        begin = time.perf_counter()
        
        for step, batch in enumerate(train_dataloader):
            load_data_time = time.perf_counter() - begin
            
            # Move batch to device
            images = batch["images"].to(device, dtype=weight_dtype)
            
            # Convert images to latent space
            with torch.no_grad():
                latents = vae.encode(images).latent_dist.sample()
                latents = latents * vae.config.scaling_factor

            # Sample noise that we'll add to the latents
            noise = torch.randn_like(latents)
            bsz = latents.shape[0]
            # Sample a random timestep for each image
            timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=device)
            timesteps = timesteps.long()

            # Add noise to the latents according to the noise magnitude at each timestep
            # (this is the forward diffusion process)
            noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
        
            with torch.no_grad():
                if args.image_encoder == "image":
                    image_embeds = image_encoder(batch["clip_images"].to(device, dtype=weight_dtype)).image_embeds
                elif args.image_encoder == "bioclip":
                    image_embeds = bioclip.encode_text(batch["taxa_tokenized"].to(device))
                elif args.image_encoder == "location":
                    image_embeds = location_encoder(batch["location"].to(device))  
                elif args.image_encoder == "taxabind":
                    image_embeds = taxabind_image_text_model.encode_text(batch["taxabind_tokenized"].to(device))
                elif args.image_encoder == "clip":
                    image_embeds = clip_text_with_proj(batch["taxa_tokenized"].to(device))[0]

            image_embeds_ = []
            for image_embed, drop_image_embed in zip(image_embeds, batch["drop_image_embeds"]):
                if drop_image_embed == 1:
                    image_embeds_.append(torch.zeros_like(image_embed))
                else:
                    image_embeds_.append(image_embed)
            image_embeds = torch.stack(image_embeds_)
        
            with torch.no_grad():
                encoder_hidden_states = text_encoder(batch["text_input_ids"].to(device))[0]
            
            # Forward pass with mixed precision
            # Note: PyTorch < 2.4 doesn't support device_type parameter, so we use torch.cuda.amp.autocast directly
            with torch.cuda.amp.autocast(enabled=(args.mixed_precision != "no"), dtype=autocast_dtype):
                noise_pred = ip_adapter(noisy_latents, timesteps, encoder_hidden_states, image_embeds, projection_flag=args.no_projection_layer)
                loss = F.mse_loss(noise_pred.float(), noise.float(), reduction="mean")
            
            # Backward pass with gradient clipping
            # Clips gradients for BOTH image_proj_model AND adapter_modules (decoupled cross-attention)
            if scaler is not None:
                # FP16 training with GradScaler
                scaler.scale(loss).backward()
                
                # Unscale gradients before clipping (required for accurate clipping)
                # This must be called before clip_grad_norm_
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    itertools.chain(ip_adapter.module.image_proj_model.parameters(), 
                                  ip_adapter.module.adapter_modules.parameters()),
                    args.max_grad_norm
                )
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
            else:
                # FP32 or BF16 training (no GradScaler needed for BF16)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    itertools.chain(ip_adapter.module.image_proj_model.parameters(), 
                                  ip_adapter.module.adapter_modules.parameters()),
                    args.max_grad_norm
                )
                optimizer.step()
                optimizer.zero_grad()
            
            # LR Scheduler step (uncomment if using scheduler)
            # lr_scheduler.step()
            
            # Gather the losses across all processes for logging
            if world_size > 1:
                loss_tensor = loss.detach().clone()
                dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
                avg_loss = loss_tensor.item() / world_size
            else:
                avg_loss = loss.item()
            
            # Update progress bar and log
            if rank == 0:
                step_time = time.perf_counter() - begin
                
                # Update progress bar
                progress_bar.update(1)
                logs = {
                    "loss": f"{avg_loss:.4f}",
                    "lr": f"{optimizer.param_groups[0]['lr']:.2e}",
                    "epoch": epoch,
                }
                progress_bar.set_postfix(**logs)
                
                # Periodic detailed logging (every 10 steps)
                if step % 10 == 0:
                    print(f"\nEpoch {epoch}, step {step}/{len(train_dataloader)}, "
                          f"data_time: {load_data_time:.4f}s, step_time: {step_time:.4f}s, "
                          f"loss: {avg_loss:.6f}, lr: {optimizer.param_groups[0]['lr']:.2e}")
                
                # Log to tensorboard/wandb
                if logger is not None:
                    if args.report_to == "tensorboard":
                        logger.add_scalar("train/loss", avg_loss, global_step)
                        logger.add_scalar("train/lr", optimizer.param_groups[0]['lr'], global_step)
                    elif args.report_to == "wandb":
                        logger.log({
                            "train/loss": avg_loss,
                            "train/lr": optimizer.param_groups[0]['lr'],
                            "train/epoch": epoch,
                            "train/step": step,
                        }, step=global_step)
            
            global_step += 1
            
            # Save checkpoint only from rank 0
            if global_step % args.save_steps == 0 and rank == 0:
                save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                os.makedirs(save_path, exist_ok=True)
                
                # Save model state (ip_adapter.bin format - compatible with inference)
                torch.save({
                    'image_proj': ip_adapter.module.image_proj_model.state_dict(),
                    'ip_adapter': ip_adapter.module.adapter_modules.state_dict(),
                }, os.path.join(save_path, "ip_adapter.bin"))
                
                # Save model state (model.safetensors format - Accelerate compatible)
                from safetensors.torch import save_file
                full_state_dict = {}
                # Add image_proj_model parameters
                for name, param in ip_adapter.module.image_proj_model.named_parameters():
                    full_state_dict[f"image_proj_model.{name}"] = param
                # Add unet attention processor parameters
                for name, param in ip_adapter.module.adapter_modules.named_parameters():
                    full_state_dict[f"unet.{name}"] = param
                save_file(full_state_dict, os.path.join(save_path, "model.safetensors"))
                
                # Save optimizer state
                torch.save({
                    'optimizer': optimizer.state_dict(),
                    'global_step': global_step,
                    'epoch': epoch,
                }, os.path.join(save_path, "optimizer.bin"))
                
                # Save GradScaler state (if using FP16)
                if scaler is not None:
                    torch.save(scaler.state_dict(), os.path.join(save_path, "scaler.pt"))
                
                # Save random states for reproducibility
                rng_state = {
                    'python': random.getstate(),
                    'numpy': np.random.get_state(),
                    'cpu': torch.random.get_rng_state(),
                    'cuda': torch.cuda.get_rng_state_all(),
                }
                torch.save(rng_state, os.path.join(save_path, f"random_states_{rank}.pkl"))
                
                # Save training args for reference
                torch.save(vars(args), os.path.join(save_path, "training_args.bin"))
                
                print(f"Saved checkpoint to {save_path}")
            
            # NOTE: Removed dist.barrier() here - it was causing unnecessary synchronization
            # DDP already handles gradient synchronization internally during backward pass
            # Additional barriers slow down the training loop significantly
            
            begin = time.perf_counter()
    
    # Close progress bar
    if rank == 0:
        progress_bar.close()
    
    # Cleanup
    if rank == 0 and logger is not None:
        if args.report_to == "tensorboard":
            logger.close()
        elif args.report_to == "wandb":
            logger.finish()
    
    cleanup_distributed()
                
if __name__ == "__main__":
    main()    

# Usage with torchrun (single node, 4 GPUs):
# torchrun --nproc_per_node=4 tutorial_train_ddp.py \
#     --pretrained_model_name_or_path="runwayml/stable-diffusion-v1-5" \
#     --image_encoder_path="/users/PAS2136/mridul/scratchpad/taxabind/IP-Adapter/models/image_encoder" \
#     --data_json_file="/fs/ess/PAS2136/bio_diffusion/data/inat/images/train_reformatted_WITH_COORDS.json" \
#     --data_root_path="/fs/ess/PAS2136/bio_diffusion/data/inat/images" \
#     --mixed_precision="bf16" \
#     --resolution=512 \
#     --train_batch_size=80 \
#     --dataloader_num_workers=8 \
#     --learning_rate=1e-04 \
#     --weight_decay=0.01 \
#     --output_dir="/fs/ess/PAS2136/bio_diffusion/ip-adapter_runs/bioclip/bioclip_full_inat_4gpus_ddp" \
#     --save_steps=10000 \
#     --report_to="wandb" \
#     --clip_extra_context_tokens=4 \
#     --image_encoder='bioclip' \
#     --gradient_checkpointing \
#     --max_grad_norm=1.0

# Usage with SLURM srun (multi-node):
# srun --ntasks=8 --gpus-per-task=1 python tutorial_train_ddp.py [args...]

# Usage with torchrun (single node, 4 GPUs):
# torchrun --nproc_per_node=1 tutorial_train_ddp.py \
#     --pretrained_model_name_or_path="runwayml/stable-diffusion-v1-5" \
#     --image_encoder_path="/users/PAS2136/mridul/scratchpad/taxabind/IP-Adapter/models/image_encoder" \
#     --data_json_file="/fs/ess/PAS2136/bio_diffusion/data/inat/images/train_reformatted_WITH_COORDS.json" \
#     --data_root_path="/fs/ess/PAS2136/bio_diffusion/data/inat/images" \
#     --mixed_precision="fp16" \
#     --resolution=512 \
#     --train_batch_size=64 \
#     --dataloader_num_workers=8 \
#     --learning_rate=1e-04 \
#     --weight_decay=0.01 \
#     --output_dir="/fs/scratch/PAS2136/bio_diffusion_scratch/ip-adapter_runs/bioclip/test_ddp_1gpu" \
#     --save_steps=100 \
#     --report_to="wandb" \
#     --clip_extra_context_tokens=4 \
#     --image_encoder='bioclip' \
#     --gradient_checkpointing
