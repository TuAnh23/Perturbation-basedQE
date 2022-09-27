#!/bin/bash
# Setting environment
source /home/tdinh/.bashrc
conda activate KIT_start
which python

export CUDA_VISIBLE_DEVICES=5
export CUDA_DEVICE_ORDER=PCI_BUS_ID  # make sure the GPU order is correct
export TORCH_HOME=/project/OML/tdinh/.cache/torch

nvidia-smi

dataname="Europarl-en2de"
data_root_dir="data"
batch_size=5
seed=0

declare -a StringArray=("None" "noun" "verb" "adjective" "adverb" "pronoun" )
for perturbation_type in ${StringArray[@]}; do
  for beam in {1..5}; do
    timestamp=$(date +"%d-%m-%y-%T")
    output_dir=output/${dataname}/beam${beam}_perturb${perturbation_type}/seed${seed}
    mkdir -p ${output_dir}
    python -u translate.py \
      --data_root_dir ${data_root_dir} \
      --dataname ${dataname} \
      --perturbation_type $perturbation_type \
      --output_dir ${output_dir} \
      --beam ${beam} \
      --seed ${seed} \
      --batch_size ${batch_size} \
      |& tee -a ${output_dir}/output_job_${timestamp}.txt
  done
done
