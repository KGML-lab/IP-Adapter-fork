#!/bin/bash
#SBATCH --job-name=ipadapter_bioclip
#SBATCH --account=PAS2136
#SBATCH --time=1-00:00:00

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=16
#SBATCH --gres=gpu:2




### change 5-digit MASTER_PORT as you wish, slurm will raise Error if duplicated with others
### change WORLD_SIZE as gpus/node * num_nodes
export MASTER_PORT=12355
export WORLD_SIZE2=14

### get the first node name as master address - customized for vgg slurm
### e.g. master(gnodee[2-5],gnoded1) == gnodee2
echo "NODELIST="${SLURM_NODELIST}
echo "SLURM_NTASKS="${SLURM_NTASKS}
echo "SLURM_PROCID="${SLURM_PROCID}
master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR=$master_addr
echo "MASTER_ADDR="$MASTER_ADDR

# module spider cuda
# module load cuda/12.3.0
module load miniconda3
conda info --envs
conda activate taxa_bind

echo "Starting accelerate..."

## context 4

# accelerate launch --num_processes 4 --multi_gpu tutorial_train.py \
# --pretrained_model_name_or_path="runwayml/stable-diffusion-v1-5" \
# --image_encoder_path="/users/PAS2136/mridul/scratchpad/taxabind/IP-Adapter/models/image_encoder" \
# --data_json_file="/fs/ess/PAS2136/bio_diffusion/data/inat/images/train_mini_birds.json" \
# --data_root_path="/fs/ess/PAS2136/bio_diffusion/data/inat/images" \
# --mixed_precision="fp16" \
# --resolution=512 \
# --train_batch_size=64 \
# --dataloader_num_workers=4 \
# --learning_rate=1e-04 \
# --weight_decay=0.01 \
# --output_dir="/fs/ess/PAS2136/bio_diffusion/ip-adapter_runs/bioclip/extra_context4" \
# --save_steps=2000 \
# --report_to="wandb" \
# --clip_extra_context_tokens=4 \

## context 1

# accelerate launch --num_processes 4 --multi_gpu tutorial_train.py \
# --pretrained_model_name_or_path="runwayml/stable-diffusion-v1-5" \
# --image_encoder_path="/users/PAS2136/mridul/scratchpad/taxabind/IP-Adapter/models/image_encoder" \
# --data_json_file="/fs/ess/PAS2136/bio_diffusion/data/inat/images/train_mini_birds.json" \
# --data_root_path="/fs/ess/PAS2136/bio_diffusion/data/inat/images" \
# --mixed_precision="fp16" \
# --resolution=512 \
# --train_batch_size=64 \
# --dataloader_num_workers=4 \
# --learning_rate=1e-04 \
# --weight_decay=0.01 \
# --output_dir="/fs/ess/PAS2136/bio_diffusion/ip-adapter_runs/bioclip/extra_context1" \
# --save_steps=2000 \
# --report_to="wandb" \
# --clip_extra_context_tokens=1 \

### no projection layer


accelerate launch --num_processes 4 --multi_gpu tutorial_train.py \
--pretrained_model_name_or_path="runwayml/stable-diffusion-v1-5" \
--image_encoder_path="/users/PAS2136/mridul/scratchpad/taxabind/IP-Adapter/models/image_encoder" \
--data_json_file="/fs/ess/PAS2136/bio_diffusion/data/inat/images/train_mini_birds.json" \
--data_root_path="/fs/ess/PAS2136/bio_diffusion/data/inat/images" \
--mixed_precision="fp16" \
--resolution=512 \
--train_batch_size=64 \
--dataloader_num_workers=4 \
--learning_rate=1e-04 \
--weight_decay=0.01 \
--output_dir="/fs/ess/PAS2136/bio_diffusion/ip-adapter_runs/bioclip/extra_context1" \
--save_steps=2000 \
--report_to="wandb" \
--clip_extra_context_tokens=1 \
--no_projection_layer