#!/bin/bash
# Setting environment
source /home/tdinh/.bashrc
conda activate KIT_start
which python

export CUDA_VISIBLE_DEVICES=0
export CUDA_DEVICE_ORDER=PCI_BUS_ID  # make sure the GPU order is correct
export TORCH_HOME=/project/OML/tdinh/.cache/torch

nvidia-smi

declare -a lang_pairs=("en2ja" "en2cs" )

for lang_pair in ${lang_pairs[@]}; do
  SRC_LANG=${lang_pair:0:2}
  TGT_LANG=${lang_pair:3:2}
  testdataname="WMT21_DA_test"
  if [ "$lang_pair" = "en2ja" ]; then
    effecting_words_threshold=4
    consistence_trans_portion_threshold=0.95
    uniques_portion_for_noiseORperturbed_threshold=0.8
    mask_type="MultiplePerSentence_allTokens"
    unmasking_model="roberta-base"
    alignment_tool="Tercom"
  elif [ "$lang_pair" = "en2cs" ]; then
    effecting_words_threshold=2
    consistence_trans_portion_threshold=0.95
    uniques_portion_for_noiseORperturbed_threshold=0.9
    mask_type="MultiplePerSentence_content"
    unmasking_model="roberta-base"
    alignment_tool="Tercom"
  fi
  test_analyse_output_path="analyse_output/${testdataname}_${SRC_LANG}2${TGT_LANG}"
  test_analyse_output_path_per_setting=${test_analyse_output_path}/${mask_type}_${unmasking_model}
  bash run_perturbation_and_translation.sh ${testdataname} ${SRC_LANG} ${TGT_LANG} ${mask_type} ${unmasking_model}
  bash run_quality_estimation.sh ${testdataname} ${SRC_LANG} ${TGT_LANG} ${mask_type} ${unmasking_model} ${test_analyse_output_path_per_setting} ${alignment_tool} ${effecting_words_threshold} ${consistence_trans_portion_threshold} ${uniques_portion_for_noiseORperturbed_threshold}
done