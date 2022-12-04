#!/bin/bash
# Setting environment
source /home/tdinh/.bashrc
conda activate KIT_start
which python

export CUDA_VISIBLE_DEVICES=3
export CUDA_DEVICE_ORDER=PCI_BUS_ID  # make sure the GPU order is correct
export TORCH_HOME=/project/OML/tdinh/.cache/torch

nvidia-smi

dataname="masked_content_mustSHE"
trans_direction="en2de"
premasked_groupped_by_word="False"
data_root_dir="data"
batch_size=60
seed=0
replacement_strategy="masking_language_model"
number_of_replacement=30

declare -a perturbation_types=("noun" "verb" "adjective" "adverb" "pronoun" )

#for beam in {5..1}; do
#  for perturbation_type in ${perturbation_types[@]}; do
#    timestamp=$(date +"%d-%m-%y-%T")
#    output_dir=output/${dataname}_${trans_direction}/${replacement_strategy}/beam${beam}_perturb${perturbation_type}/${number_of_replacement}replacements/seed${seed}
#    mkdir -p ${output_dir}
#    python -u generate_input_SRC.py ...
#    python -u translate.py ...
#  done
#done

beam=5
perturbation_type="content"
output_dir=output/${dataname}_${trans_direction}/${replacement_strategy}/beam${beam}_perturb${perturbation_type}/${number_of_replacement}replacements/seed${seed}
mkdir -p ${output_dir}
python -u generate_input_SRC.py \
  --data_root_dir ${data_root_dir} \
  --dataname ${dataname} \
  --perturbation_type ${perturbation_type} \
  --output_dir ${output_dir} \
  --seed ${seed} \
  --replacement_strategy ${replacement_strategy} \
  --number_of_replacement ${number_of_replacement} \
  --premasked_groupped_by_word ${premasked_groupped_by_word} \
  |& tee -a ${output_dir}/generate_input_SRC.log

python -u translate.py \
  --output_dir ${output_dir} \
  --beam ${beam} \
  --seed ${seed} \
  --batch_size ${batch_size} \
  --trans_direction ${trans_direction} \
  --SRC_perturbed_type ${perturbation_type} \
  |& tee -a ${output_dir}/translate.log
