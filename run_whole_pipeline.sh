#!/bin/bash
# Setting environment
source /home/tdinh/.bashrc
conda activate KIT_start
which python

export CUDA_VISIBLE_DEVICES=3
export CUDA_DEVICE_ORDER=PCI_BUS_ID  # make sure the GPU order is correct
export TORCH_HOME=/project/OML/tdinh/.cache/torch
export HF_HOME=/project/OML/tdinh/.cache/huggingface

nvidia-smi

if [ -z "$1" ]; then
  dataname_prefix="WMT21_DA"  # WMT20_HJQ[E
else
  dataname_prefix=$1
fi

declare -a lang_pairs=( "en2de" "en2zh" )

for lang_pair in ${lang_pairs[@]}; do
  SRC_LANG=${lang_pair:0:2}
  TGT_LANG=${lang_pair:3:2}

  dataname="${dataname_prefix}_dev"

  declare -a unmasking_models=( "roberta-base" )  # ( "bert-large-cased-whole-word-masking" "bert-large-cased" "distilbert-base-cased" "roberta-base" "bert-base-cased" )

  declare -a mask_types=( "MultiplePerSentence_content" )  # ( "MultiplePerSentence_content" "MultiplePerSentence_allWords" "MultiplePerSentence_allTokens" )

  declare -a alignment_tools=( "Tercom" )  # ( "Tercom" "Levenshtein" )

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

      for alignment_tool in ${alignment_tools[@]}; do
        bash run_quality_estimation.sh ${dataname} ${SRC_LANG} ${TGT_LANG} ${mask_type} ${unmasking_model} ${analyse_output_path_per_setting} ${alignment_tool}
        qe_htune_output=$(tail -1 "${analyse_output_path_per_setting}/quality_estimation_${alignment_tool}.log")
        score=$(echo $qe_htune_output | awk -F'best score ' '{print $2}' | awk -F',' '{print $1}')
        qe_params=$(echo $qe_htune_output | awk -F'[()]' '{print $2}')
        echo "mask_type: ${mask_type}, unmasking_models: ${unmasking_model}, QE_params: ${qe_params}, score: ${score}" | tee -a ${analyse_output_path}/qe_hyperparam_tuning_${alignment_tool}.txt

        echo "Eval ${mask_type} ${unmasking_model} also on the test set"
        testdataname="${dataname_prefix}_test"
        effecting_words_threshold=$(echo "${qe_params}" | awk -F',' '{print $1}')
        consistence_trans_portion_threshold=$(echo "${qe_params}" | awk -F',' '{print $2}')
        uniques_portion_for_noiseORperturbed_threshold=$(echo "${qe_params}" | awk -F',' '{print $3}')
        test_analyse_output_path="analyse_output/${testdataname}_${SRC_LANG}2${TGT_LANG}"
        test_analyse_output_path_per_setting=${test_analyse_output_path}/${mask_type}_${unmasking_model}
        bash run_perturbation_and_translation.sh ${testdataname} ${SRC_LANG} ${TGT_LANG} ${mask_type} ${unmasking_model}
        bash run_quality_estimation.sh ${testdataname} ${SRC_LANG} ${TGT_LANG} ${mask_type} ${unmasking_model} ${test_analyse_output_path_per_setting} ${alignment_tool} ${effecting_words_threshold} ${consistence_trans_portion_threshold} ${uniques_portion_for_noiseORperturbed_threshold}

        # Log scores and hyperparams
        if (( $(echo "$score > $best_score" | bc -l) )); then
          best_score=${score}
          best_qe_params=${qe_params}
          best_unmasking_model=${unmasking_model}
          best_mask_type=${mask_type}
        fi
      done
    done
  done

  for alignment_tool in ${alignment_tools[@]}; do
    echo "-------------- BEST SETTING --------------" | tee -a ${analyse_output_path}/qe_hyperparam_tuning_${alignment_tool}.txt
    echo "mask_type: ${best_mask_type}, unmasking_models: ${best_unmasking_model}, QE_params: ${best_qe_params}, score: ${best_score}" | tee -a ${analyse_output_path}/qe_hyperparam_tuning_${alignment_tool}.txt
  done
done
