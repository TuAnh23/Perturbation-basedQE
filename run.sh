#!/bin/bash
# Setting environment
source /home/tdinh/.bashrc
conda activate KIT_start
which python

export CUDA_VISIBLE_DEVICES=1
export CUDA_DEVICE_ORDER=PCI_BUS_ID  # make sure the GPU order is correct
export TORCH_HOME=/project/OML/tdinh/.cache/torch

nvidia-smi

dataname="masked_regional_covost2_for_en2de"
data_root_dir="data"
batch_size=10
seed=0
replacement_strategy="masking_language_model"
number_of_replacement=5

declare -a perturbation_types=("noun" "verb" "adjective" "adverb" "pronoun" )

#for beam in {5..1}; do
#  for perturbation_type in ${perturbation_types[@]}; do
#    timestamp=$(date +"%d-%m-%y-%T")
#    output_dir=output/${dataname}/${replacement_strategy}/beam${beam}_perturb${perturbation_type}/seed${seed}
#    mkdir -p ${output_dir}
#    python -u translate.py \
#      --data_root_dir ${data_root_dir} \
#      --dataname ${dataname} \
#      --perturbation_type $perturbation_type \
#      --output_dir ${output_dir} \
#      --beam ${beam} \
#      --seed ${seed} \
#      --batch_size ${batch_size} \
#      --replacement_strategy ${replacement_strategy} \
#      --number_of_replacement ${number_of_replacement} \
#      |& tee -a ${output_dir}/output_job_${timestamp}.txt
#  done
#done

beam=5
perturbation_type="None"
timestamp=$(date +"%d-%m-%y-%T")
output_dir=output/${dataname}/${replacement_strategy}/beam${beam}_perturb${perturbation_type}/seed${seed}
mkdir -p ${output_dir}
python -u translate.py \
  --data_root_dir ${data_root_dir} \
  --dataname ${dataname} \
  --perturbation_type $perturbation_type \
  --output_dir ${output_dir} \
  --beam ${beam} \
  --seed ${seed} \
  --batch_size ${batch_size} \
  --replacement_strategy ${replacement_strategy} \
  --number_of_replacement ${number_of_replacement} \
  |& tee -a ${output_dir}/output_job_${timestamp}.txt