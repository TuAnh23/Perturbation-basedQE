#!/bin/bash
# Setting environment
source /home/tdinh/.bashrc
conda activate KIT_start
which python

export CUDA_VISIBLE_DEVICES=1
export CUDA_DEVICE_ORDER=PCI_BUS_ID  # make sure the GPU order is correct
export TORCH_HOME=/project/OML/tdinh/.cache/torch

nvidia-smi

lang_pair="en2de"

dataname="WMT21_DA_test"
SRC_LANG=${lang_pair:0:2}
TGT_LANG=${lang_pair:3:2}

QE_method="openkiwi_2.1.0"
analyse_output_path="analyse_output/${dataname}_${SRC_LANG}2${TGT_LANG}_${QE_method}"
data_root_dir="data"
OUTPUT_dir=output/${dataname}_${lang_pair}
output_dir_original_SRC=${OUTPUT_dir}/original

mkdir -p ${analyse_output_path}

if [[ ! -f ${analyse_output_path}/pred_labels_${QE_method}.pkl ]]; then
  python -u tokenize_original.py \
    --original_translation_output_dir ${output_dir_original_SRC} \
    --dataset ${dataname} \
    --data_root_path ${data_root_dir} \
    --src_lang ${SRC_LANG} \
    --tgt_lang ${TGT_LANG}
  conda activate openkiwi
  python -u openkiwi_qe.py \
    --original_translation_output_dir ${output_dir_original_SRC} \
    --model_path "models/xlmr-en-de.ckpt" \
    --label_output_path ${analyse_output_path}/pred_labels_${QE_method}.pkl
  conda activate KIT_start
fi

python -u eval_on_WMT21.py \
  --dataset ${dataname} \
  --data_root_path ${data_root_dir} \
  --src_lang ${SRC_LANG} \
  --tgt_lang ${TGT_LANG} \
  --qe_pred_labels_path ${analyse_output_path}/pred_labels_${QE_method}.pkl \
  --output_path ${analyse_output_path}/scores.txt