#!/bin/bash

#SBATCH -J tx+loc_concat
#SBATCH --cpus-per-task=12
#SBATCH --time=40:00:00 
#SBATCH --gres=gpu:4
#SBATCH --partition=h200_normal_q
#SBATCH --account=memtrack
#SBATCH --output=/scratch/bio_diffusion/slurm_logs/%j-%x.out


########## SBATCH --qos=tc_a100_normal_short

### change 5-digit MASTER_PORT as you wish, slurm will raise Error if duplicated with others
### change WORLD_SIZE as gpus/node * num_nodes
export MASTER_PORT=15890
export WORLD_SIZE2=14


# Threads = cpus-per-task
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
### get the first node name as master address - customized for vgg slurm
### e.g. master(gnodee[2-5],gnoded1) == gnodee2
echo "NODELIST="${SLURM_NODELIST}
echo "SLURM_NTASKS="${SLURM_NTASKS}
echo "SLURM_PROCID="${SLURM_PROCID}
master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR=$master_addr
echo "MASTER_ADDR="$MASTER_ADDR

# Load required modules and activate Conda environment
module reset
module load Miniconda3/24.7.1-0
conda init
source ~/.bashrc
source activate bio_up
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib/


echo "Starting accelerate..."

## context 4

# accelerate launch --num_processes 4 --multi_gpu tutorial_train.py \
# --pretrained_model_name_or_path="runwayml/stable-diffusion-v1-5" \
# --image_encoder_path="/home/mridul/IP-Adapter-fork/sdxl_models/image_encoder" \
# --data_json_file="/projects/ml4science/DATASETS/iNaturalist/train_mini_birds_arc.json" \
# --data_root_path="/projects/ml4science/DATASETS/iNaturalist/images" \
# --mixed_precision="fp16" \
# --resolution=512 \
# --train_batch_size=64 \
# --dataloader_num_workers=4 \
# --learning_rate=1e-04 \
# --weight_decay=0.01 \
# --output_dir="/scratch/bio_diffusion/ip-adapter_runs/bioclip/sdxl_extra_context4" \
# --save_steps=2000 \
# --report_to="wandb" \
# --clip_extra_context_tokens=4 

## context 1

# accelerate launch --num_processes 4 --multi_gpu --main_process_port ${MASTER_PORT} tutorial_train.py \
# --pretrained_model_name_or_path="runwayml/stable-diffusion-v1-5" \
# --image_encoder_path="/home/medha/Projects/bio-diffusion/IP-Adapter-fork/sdxl_models/image_encoder" \
# --data_json_file="/projects/ml4science/DATASETS/iNaturalist/train_mini_birds_arc.json" \
# --data_root_path="/projects/ml4science/DATASETS/iNaturalist/images" \
# --mixed_precision="fp16" \
# --resolution=512 \
# --train_batch_size=64 \
# --dataloader_num_workers=4 \
# --learning_rate=1e-04 \
# --weight_decay=0.01 \
# --output_dir="/scratch/bio_diffusion/ip-adapter_runs/bioclip/sdxl_extra_context1" \
# --save_steps=2000 \
# --report_to="wandb" \
# --clip_extra_context_tokens=1 

### no projection layer


# accelerate launch --num_processes 2 --multi_gpu --main_process_port ${MASTER_PORT} tutorial_train.py \
# --pretrained_model_name_or_path="runwayml/stable-diffusion-v1-5" \
# --image_encoder_path="/home/medha/Projects/bio-diffusion/IP-Adapter-fork/models/image_encoder" \
# --data_json_file="/projects/ml4science/DATASETS/iNaturalist/train_mini_birds_arc.json" \
# --data_root_path="/projects/ml4science/DATASETS/iNaturalist/images" \
# --mixed_precision="fp16" \
# --resolution=512 \
# --train_batch_size=64 \
# --dataloader_num_workers=4 \
# --learning_rate=1e-04 \
# --weight_decay=0.01 \
# --output_dir="/scratch/bio_diffusion/ip-adapter_runs/bioclip/no_proj" \
# --save_steps=2000 \
# --report_to="wandb" \
# --clip_extra_context_tokens=1 \
# --no_projection_layer






#########################
# asic ip adapter debug run
# accelerate launch --num_processes 4 --multi_gpu tutorial_train.py \
# --pretrained_model_name_or_path="runwayml/stable-diffusion-v1-5" \
# --image_encoder_path="/home/medha/Projects/bio-diffusion/IP-Adapter/models/image_encoder" \
# --data_json_file="/projects/ml4science/DATASETS/iNaturalist/train_mini_birds_arc.json" \
# --data_root_path="/projects/ml4science/DATASETS/iNaturalist/images" \
# --mixed_precision="fp16" \
# --resolution=512 \
# --train_batch_size=64 \
# --dataloader_num_workers=4 \
# --learning_rate=1e-04 \
# --weight_decay=0.01 \
# --output_dir="/scratch/bio_diffusion/ip-adapter_runs/bioclip/debug_save" \
# --save_steps=2000 \
# --report_to="wandb" 








##############################################

### combined   bioclip + location adapter runs
# accelerate launch --num_processes 4 --multi_gpu --main_process_port ${MASTER_PORT} tutorial_train_location.py \
# --pretrained_model_name_or_path="runwayml/stable-diffusion-v1-5" \
# --image_encoder_path="/home/medha/Projects/bio-diffusion/IP-Adapter-fork/models/image_encoder" \
# --data_json_file="/projects/ml4science/DATASETS/iNaturalist/train_mini_birds_subset_arc.json" \
# --data_root_path="/projects/ml4science/DATASETS/iNaturalist/images" \
# --mixed_precision="fp16" \
# --resolution=512 \
# --train_batch_size=64 \
# --dataloader_num_workers=4 \
# --learning_rate=1e-04 \
# --weight_decay=0.01 \
# --output_dir="/scratch/bio_diffusion/ip-adapter_runs/bioclip/bioclip_location_context4" \
# --save_steps=2000 \
# --report_to="wandb" \
# --clip_extra_context_tokens=4 \
# --image_encoder='bioclip' \
# --image_encoder_secondary='location' \
# --image_proj_secondary 


# ##### combined  taxabind + location double projection layers adapter runs 
# accelerate launch --num_processes 4 --multi_gpu --main_process_port ${MASTER_PORT} tutorial_train_location.py \
# --pretrained_model_name_or_path="runwayml/stable-diffusion-v1-5" \
# --image_encoder_path="/home/medha/Projects/bio-diffusion/IP-Adapter-fork/models/image_encoder" \
# --data_json_file="/projects/ml4science/DATASETS/iNaturalist/train_mini_birds_subset_arc.json" \
# --data_root_path="/projects/ml4science/DATASETS/iNaturalist/images" \
# --mixed_precision="fp16" \
# --resolution=512 \
# --train_batch_size=64 \
# --dataloader_num_workers=4 \
# --learning_rate=1e-04 \
# --weight_decay=0.01 \
# --output_dir="/scratch/bio_diffusion/ip-adapter_runs/bioclip/taxabind_location_context4_doubleproj" \
# --save_steps=2000 \
# --report_to="wandb" \
# --clip_extra_context_tokens=4 \
# --image_encoder='taxabind' \
# --image_encoder_secondary='location' \
# --image_proj_secondary 

##### combined  taxabind + location single projection layer conct embeds adapter runs
accelerate launch --num_processes 4 --multi_gpu --main_process_port ${MASTER_PORT} tutorial_train_location.py \
--pretrained_model_name_or_path="runwayml/stable-diffusion-v1-5" \
--image_encoder_path="/home/medha/Projects/bio-diffusion/IP-Adapter-fork/models/image_encoder" \
--data_json_file="/projects/ml4science/DATASETS/iNaturalist/train_mini_birds_subset_arc.json" \
--data_root_path="/projects/ml4science/DATASETS/iNaturalist/images" \
--mixed_precision="fp16" \
--resolution=512 \
--train_batch_size=64 \
--dataloader_num_workers=4 \
--learning_rate=1e-04 \
--weight_decay=0.01 \
--output_dir="/scratch/bio_diffusion/ip-adapter_runs/bioclip/taxabind_location_context4_concat" \
--save_steps=2000 \
--report_to="wandb" \
--clip_extra_context_tokens=4 \
--image_encoder='taxabind' \
--image_encoder_secondary='location' 








#################### location adapter only runs
accelerate launch --num_processes 4 --multi_gpu tutorial_train.py \
--pretrained_model_name_or_path="runwayml/stable-diffusion-v1-5" \
--image_encoder_path="/home/mridul/IP-Adapter-fork/models/image_encoder" \
--data_json_file="/projects/ml4science/DATASETS/iNaturalist/train_mini_birds_subset_arc.json" \
--data_root_path="/projects/ml4science/DATASETS/iNaturalist/images" \
--mixed_precision="fp16" \
--resolution=512 \
--train_batch_size=64 \
--dataloader_num_workers=8 \
--learning_rate=1e-04 \
--weight_decay=0.01 \
--output_dir="/scratch/bio_diffusion/ip-adapter_runs/bioclip/location_only" \
--save_steps=2000 \
--report_to="wandb" \
--clip_extra_context_tokens=4 \
--image_encoder='location'










########################
##########. EVALS ##############
python inference.py --ip_ckpt "/scratch/bio_diffusion/ip-adapter_runs/bioclip/sdxl_extra_context4/checkpoint-28000/ip_adapter.bin" \
--json_file "/projects/ml4science/DATASETS/iNaturalist/train_mini_birds_subset_arc.json" \
--out_dir /scratch/bio_diffusion/ip-adapter_runs/samples/sdxl_extra_context4_bioclip --num_samples 20 



python inference.py --ip_ckpt "/scratch/bio_diffusion/ip-adapter_runs/bioclip/extra_context4/checkpoint-28000/ip_adapter.bin" \
--json_file "/projects/ml4science/DATASETS/iNaturalist/train_mini_birds_subset_arc.json" \
--out_dir /scratch/bio_diffusion/ip-adapter_runs/samples/extra_context4_bioclip --num_samples 20 


########### Location + bioclip / taxabind evals

python inference_location.py --ip_ckpt "/scratch/bio_diffusion/ip-adapter_runs/bioclip/taxabind_location_context4_concat/checkpoint-16000/ip_adapter.bin" \
--json_file "/projects/ml4science/DATASETS/iNaturalist/train_mini_birds_subset_arc.json" \
--out_dir /scratch/bio_diffusion/ip-adapter_runs/samples/taxabind_location_context4_concat --num_samples 20 \
--model_type taxabind --model_type_secondary location

python inference_location.py --ip_ckpt "/scratch/bio_diffusion/ip-adapter_runs/bioclip/bioclip_location_context4/checkpoint-22000/ip_adapter.bin" \
--json_file "/projects/ml4science/DATASETS/iNaturalist/train_mini_birds_subset_arc.json" \
--out_dir /scratch/bio_diffusion/ip-adapter_runs/samples/bioclip_location_context4 --num_samples 20 \
--model_type bioclip --model_type_secondary location --image_proj_secondary


python inference_location.py --ip_ckpt "/scratch/bio_diffusion/ip-adapter_runs/bioclip/taxabind_location_context4_doubleproj/checkpoint-20000/ip_adapter.bin" \
--json_file "/projects/ml4science/DATASETS/iNaturalist/train_mini_birds_subset_arc.json" \
--out_dir /scratch/bio_diffusion/ip-adapter_runs/samples/taxabind_location_context4_doubleproj --num_samples 20 \
--model_type taxabind --model_type_secondary location --image_proj_secondary







################## FID try
# python -m torch_fidelity --input1 /projects/ml4science/DATASETS/iNaturalist/images/train_mini --input2 /scratch/bio_diffusion/ip-adapter_runs/samples/extra_context4_bioclip --fid --cuda

# print("FID:", fid.compute_fid("/projects/ml4science/DATASETS/iNaturalist/images/train_mini", "/scratch/bio_diffusion/ip-adapter_runs/samples/extra_context4_bioclip"))

# python classwise_fid.py \
#   --real_root /projects/ml4science/DATASETS/iNaturalist/images/train_mini \
#   --gen_root  /scratch/bio_diffusion/ip-adapter_runs/samples/extra_context4_bioclip \
#   --out_csv   /scratch/bio_diffusion/ip-adapter_runs/fid/classwise_fid.csv \
#   --batch_size 128 --num_workers 8


# python classwise_fid.py \
#   --real_root /projects/ml4science/DATASETS/iNaturalist/images/train_mini \
#   --gen_root  /scratch/bio_diffusion/ip-adapter_runs/samples/extra_context4_bioclip \
#   --out_csv   /scratch/bio_diffusion/ip-adapter_runs/fid/classwise_fid.csv \
#   --macro_txt /scratch/bio_diffusion/ip-adapter_runs/fid/latest_macro_fid.txt \
#   --device cuda --batch_size 256 --num_workers 12


