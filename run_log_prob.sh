#!/bin/bash
# Setting environment
source /home/tdinh/.bashrc
conda activate KIT_start
which python

export CUDA_VISIBLE_DEVICES=5
export CUDA_DEVICE_ORDER=PCI_BUS_ID  # make sure the GPU order is correct
export TORCH_HOME=/project/OML/tdinh/.cache/torch


# Hyperparam tuning for the pairs that has dev data
declare -a lang_pairs=("en2de" "en2zh" )

data_root_dir="data"
analyse_output_path="analyse_output"
seed=0
sentence_level_eval_da="False"
trans_word_level_eval="True"
src_word_level_eval="False"
OUTPUT_dir=output/${dataname}_${trans_direction}
output_dir_original_SRC=${OUTPUT_dir}/original

dataname="WMT21_DA_dev"
test_dataname="WMT21_DA_test"

for lang_pair in ${lang_pairs[@]}; do
  SRC_LANG=${lang_pair:0:2}
  TGT_LANG=${lang_pair:3:2}

  trans_direction="${SRC_LANG}2${TGT_LANG}"
  OUTPUT_dir=output/${dataname}_${trans_direction}
  output_dir_original_SRC=${OUTPUT_dir}/original

  python -u quality_estimation.py \
  --original_translation_output_dir ${output_dir_original_SRC} \
  --dataset ${dataname} \
  --data_root_path ${data_root_dir} \
  --src_lang ${SRC_LANG} \
  --tgt_lang ${TGT_LANG} \
  --seed ${seed} \
  --sentence_level_eval_da ${sentence_level_eval_da} \
  --trans_word_level_eval ${trans_word_level_eval} \
  --trans_word_level_eval_methods 'nmt_log_prob' \
  --nmt_log_prob_thresholds 0.4 0.45 0.5 0.55 0.6 0.65 0.7 0.75 \
  --src_word_level_eval ${src_word_level_eval} \
  --src_word_level_eval_methods 'nmt_log_prob' 'nr_effecting_src_words' |& tee ${analyse_output_path}/${dataname}_${SRC_LANG}2${TGT_LANG}/logprob.log
  logprob_htune_output=$(tail -1 "${analyse_output_path}/${dataname}_${SRC_LANG}2${TGT_LANG}/logprob.log")
  best_prob_threshold=$(echo $logprob_htune_output | awk -F'best params ' '{print $2}')

  if [ "$lang_pair" = "en2zh" ]; then
    # Apply the best threshold also on en2ja, bc they similar
    declare -a test_lang_pairs=("en2zh" "en2ja" )
  elif [ "$lang_pair" = "en2de" ]; then
    # Apply the best threshold also on en2cs, bc they similar
    declare -a test_lang_pairs=("en2de" "en2cs" )
  fi

  for test_lang_pair in ${test_lang_pairs[@]}; do
    SRC_LANG=${test_lang_pair:0:2}
    TGT_LANG=${test_lang_pair:3:2}

    trans_direction="${SRC_LANG}2${TGT_LANG}"
    OUTPUT_dir=output/${test_dataname}_${trans_direction}
    output_dir_original_SRC=${OUTPUT_dir}/original
    echo ${output_dir_original_SRC}

    python -u quality_estimation.py \
    --original_translation_output_dir ${output_dir_original_SRC} \
    --dataset ${test_dataname} \
    --data_root_path ${data_root_dir} \
    --src_lang ${SRC_LANG} \
    --tgt_lang ${TGT_LANG} \
    --seed ${seed} \
    --sentence_level_eval_da ${sentence_level_eval_da} \
    --trans_word_level_eval ${trans_word_level_eval} \
    --trans_word_level_eval_methods 'nmt_log_prob' \
    --nmt_log_prob_thresholds ${best_prob_threshold} \
    --src_word_level_eval ${src_word_level_eval} \
    --src_word_level_eval_methods 'nmt_log_prob' 'nr_effecting_src_words' |& tee ${analyse_output_path}/${test_dataname}_${SRC_LANG}2${TGT_LANG}/logprob.log
  done
done