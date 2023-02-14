#!/bin/bash
# Setting environment
source /home/tdinh/.bashrc
conda activate KIT_start
which python

export CUDA_VISIBLE_DEVICES=0
export CUDA_DEVICE_ORDER=PCI_BUS_ID  # make sure the GPU order is correct
export TORCH_HOME=/project/OML/tdinh/.cache/torch

nvidia-smi

dataname="WMT21_DA_dev"
SRC_LANG="en"
TGT_LANG="de"

declare -a unmasking_models=("bert-large-cased-whole-word-masking" "bert-large-cased" "distilbert-base-cased" "roberta-base" "bert-base-cased" )

declare -a mask_types=("MultiplePerSentence_content" ) # "MultiplePerSentence_content" "MultiplePerSentence_allWords" "MultiplePerSentence_allTokens" )

analyse_output_path="analyse_output/${dataname}_${SRC_LANG}2${TGT_LANG}"
mkdir -p ${analyse_output_path}

# Hyperparams tuning
best_score=0
best_qe_params=""
best_unmasking_model=""
best_mask_type=""
for unmasking_model in ${unmasking_models[@]}; do
  for mask_type in ${mask_types[@]}; do
    analyse_output_path_per_setting=${analyse_output_path}/${mask_type}_${unmasking_model}
    echo "Runing perturbation and translation"
    bash run_perturbation_and_translation.sh ${dataname} ${SRC_LANG} ${TGT_LANG} ${mask_type} ${unmasking_model}
    echo "Runing QE"
    bash run_quality_estimation.sh ${dataname} ${SRC_LANG} ${TGT_LANG} ${mask_type} ${unmasking_model} ${analyse_output_path_per_setting}
    qe_htune_output=$(tail -1 "${analyse_output_path_per_setting}/quality_estimation.log")
    score=$(echo $qe_htune_output | awk -F'best score ' '{print $2}' | awk -F',' '{print $1}')
    qe_params=$(echo $qe_htune_output | awk -F'best params ' '{print $2}')
    echo "mask_type: ${mask_type}, unmasking_models: ${unmasking_model}, QE_params: ${params}, score: ${score}" | tee -a ${analyse_output_path}/qe_hyperparam_tuning.txt
    if (( $(echo "$score > $best_score" | bc -l) )); then
      best_score=${score}
      best_qe_params=${qe_params}
      best_unmasking_model=${unmasking_model}
      best_mask_type=${mask_type}
    fi
  done
done

echo "-------------- BEST SETTING --------------" | tee -a ${analyse_output_path}/qe_hyperparam_tuning.txt
echo "mask_type: ${best_mask_type}, unmasking_models: ${best_unmasking_model}, QE_params: ${best_qe_params}, score: ${best_score}" | tee -a ${analyse_output_path}/qe_hyperparam_tuning.txt


# Run the best hyperparams on the test set
dataname="WMT21_DA_test"
SRC_LANG="en"
TGT_LANG="de"

best_setting_str=$(tail -1 "${analyse_output_path}/qe_hyperparam_tuning.txt")
best_mask_type=$(echo "${best_setting_str}" | awk -F'mask_type: ' '{print $2}' | awk -F',' '{print $1}')
best_unmasking_model=$(echo "${best_setting_str}" | awk -F'unmasking_models: ' '{print $2}' | awk -F',' '{print $1}')
best_qe_params=$(echo "${best_setting_str}" | awk -F'[()]' '{print $2}')


effecting_words_threshold=$(echo "${best_qe_params}" | awk -F',' '{print $1}')
consistence_trans_portion_threshold=$(echo "${best_qe_params}" | awk -F',' '{print $2}')
uniques_portion_for_noiseORperturbed_threshold=$(echo "${best_qe_params}" | awk -F',' '{print $3}')


analyse_output_path="analyse_output/${dataname}_${SRC_LANG}2${TGT_LANG}"
analyse_output_path_per_setting=${analyse_output_path}/${best_mask_type}_${best_unmasking_model}
bash run_perturbation_and_translation.sh ${dataname} ${SRC_LANG} ${TGT_LANG} ${best_mask_type} ${best_unmasking_model}
bash run_quality_estimation.sh ${dataname} ${SRC_LANG} ${TGT_LANG} ${best_mask_type} ${best_unmasking_model} ${analyse_output_path_per_setting} ${effecting_words_threshold} ${consistence_trans_portion_threshold} ${uniques_portion_for_noiseORperturbed_threshold}
