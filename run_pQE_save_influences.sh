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

# Given an English dataset (monolingual), this script use an MT model to translate it to the target language, then run
# perturbation based QE, then save the bad word indices and the src-tgt influences

if [ -z "$1" ]; then
  MTmodel="LLM"  ## "LLM" or "qe_wmt21"
else
  MTmodel=$1
fi

if [ -z "$2" ]; then
  dataname="winoMT"
else
  dataname=$2
fi

if [ -z "$3" ]; then
  lang_pair="en2de"
else
  lang_pair=$3
fi

SRC_LANG=${lang_pair:0:2}
TGT_LANG=${lang_pair:3:2}

analyse_output_path="analyse_output/${dataname}_${SRC_LANG}2${TGT_LANG}_${MTmodel}"
analyse_output_path=$(realpath $analyse_output_path)
mkdir -p ${analyse_output_path}


# Get the translations (original and perturbed)
if [[ (${lang_pair} == "en2de") || (${lang_pair} == "en2cs") || (${lang_pair} == "en2vi") ]]; then
  effecting_words_threshold=2
  consistence_trans_portion_threshold=0.95
  uniques_portion_for_noiseORperturbed_threshold=0.9
  mask_type="MultiplePerSentence_content"
  unmasking_model="roberta-base"
  alignment_tool="Tercom"
  nmt_log_prob_threshold=0.45
elif [[ (${lang_pair} == "en2zh") || (${lang_pair} == "en2ja") ]]; then
  effecting_words_threshold=4
  consistence_trans_portion_threshold=0.95
  uniques_portion_for_noiseORperturbed_threshold=0.8
  mask_type="MultiplePerSentence_allTokens"
  unmasking_model="roberta-base"
  alignment_tool="Tercom"
  nmt_log_prob_threshold=0.6
fi
bash run_perturbation_and_translation.sh ${dataname} ${SRC_LANG} ${TGT_LANG} ${mask_type} ${unmasking_model} ${MTmodel}

if [[ ${dataname} == "WMT21_DA"* ]]; then
  # Always uses qe_wmt21 MT models for consistent with gold label, so no need to specify
  OUTPUT_dir=output/${dataname}_${lang_pair}
else
  OUTPUT_dir=output/${dataname}_${lang_pair}_${MTmodel}
fi
output_dir_original_SRC=${OUTPUT_dir}/original


# Save the tokenized original src and trans
if [[ ! -f ${analyse_output_path}/tok_src.${SRC_LANG} ]]; then
  cp ${output_dir_original_SRC}/input.${SRC_LANG} ${analyse_output_path}/src.${SRC_LANG}
  if [[ ${dataname} == "WMT21_DA_test" ]]; then
    cp "data/wmt-qe-2021-data/${SRC_LANG}-${TGT_LANG}-test21/test21.tok.src" ${analyse_output_path}/tok_src.${SRC_LANG}
  elif [[ ${dataname} == "WMT20_HJQE_test" ]]; then
    cp "data/HJQE/${SRC_LANG}-${TGT_LANG}/test/test.tok.src" ${analyse_output_path}/tok_src.${SRC_LANG}
  else
    python -u tokenization.py \
      --text_file_path ${output_dir_original_SRC}/input.${SRC_LANG} \
      --lang ${SRC_LANG} \
      --output_tok_path ${analyse_output_path}/tok_src.${SRC_LANG}
  fi
fi

if [[ ! -f ${analyse_output_path}/tok_trans.${TGT_LANG} ]]; then
  cp ${output_dir_original_SRC}/trans_sentences.txt ${analyse_output_path}/trans.${TGT_LANG}
  if [[ ${dataname} == "WMT21_DA_test" ]]; then
    cp "data/wmt-qe-2021-data/${SRC_LANG}-${TGT_LANG}-test21/test21.tok.mt" ${analyse_output_path}/tok_trans.${TGT_LANG}
  elif [[ ${dataname} == "WMT20_HJQE_test" ]]; then
    cp "data/HJQE/${SRC_LANG}-${TGT_LANG}/test/test.tok.mt" ${analyse_output_path}/tok_trans.${TGT_LANG}
  else
    python -u tokenization.py \
      --text_file_path ${output_dir_original_SRC}/trans_sentences.txt \
      --lang ${TGT_LANG} \
      --output_tok_path ${analyse_output_path}/tok_trans.${TGT_LANG}
  fi
fi


# Run quality estimation
sentence_level_eval_da="False"
trans_word_level_eval="True"
src_word_level_eval="False"
use_src_tgt_alignment="False"
data_root_dir="data"
output_root_path="output"
seed=0
replacement_strategy="masking_language_model_${unmasking_model}"
number_of_replacement=30
beam=5
if [[ ! -f ${analyse_output_path}/analyse_${dataname}_${SRC_LANG}2${TGT_LANG}_${mask_type}.pkl ]]; then
  # First process the dataframe
  python -u read_and_analyse_df.py \
    --df_root_path ${output_root_path} \
    --data_root_path ${data_root_dir} \
    --src_lang ${SRC_LANG} \
    --tgt_lang ${TGT_LANG} \
    --replacement_strategy ${replacement_strategy} \
    --number_of_replacement ${number_of_replacement} \
    --dataname ${dataname} \
    --MTmodel ${MTmodel} \
    --seed ${seed} \
    --beam ${beam} \
    --mask_type ${mask_type} \
    --output_dir ${analyse_output_path} \
    --tokenize_sentences "True" \
    --use_src_tgt_alignment ${use_src_tgt_alignment} \
    --analyse_feature "edit_distance" "change_spread"
fi

# Run quality estimation and save the predicted labels
declare -a QE_methods=( 'nr_effecting_src_words' 'openkiwi_2.1.0' )
for QE_method in ${QE_methods[@]}; do
  echo "-------------------------------------------------"
  if [[ ( ! -f ${analyse_output_path}/pred_labels_${QE_method}.pkl ) ||
      (! -f ${analyse_output_path}/src_tgt_influence.pkl) ]]; then
    if [[ (${QE_method} == "openkiwi_2.1.0") || (${QE_method} == "openkiwi_wmt21") ]]; then
      python -u tokenize_original.py \
        --original_translation_output_dir ${output_dir_original_SRC} \
        --dataset ${dataname} \
        --data_root_path ${data_root_dir} \
        --src_lang ${SRC_LANG} \
        --tgt_lang ${TGT_LANG}
      conda activate openkiwi
      if [[ ${QE_method} == "openkiwi_2.1.0" ]]; then
        model_path="models/xlmr-en-de.ckpt"
      elif [[ ${QE_method} == "openkiwi_wmt21" ]]; then
        model_path="models/updated_models/Task2/checkpoints/model_epoch=01-val_WMT19_MCC+PEARSON=1.30.ckpt"
      fi
      python -u openkiwi_qe.py \
        --original_translation_output_dir ${output_dir_original_SRC} \
        --model_path ${model_path} \
        --label_output_path ${analyse_output_path}/pred_labels_${QE_method}.pkl
      conda activate KIT_start
    else
      python -u quality_estimation.py \
        --perturbed_trans_df_path ${analyse_output_path}/analyse_${dataname}_${SRC_LANG}2${TGT_LANG}_${mask_type}.pkl \
        --original_translation_output_dir ${output_dir_original_SRC} \
        --dataset ${dataname} \
        --data_root_path ${data_root_dir} \
        --src_lang ${SRC_LANG} \
        --tgt_lang ${TGT_LANG} \
        --seed ${seed} \
        --method ${QE_method} \
        --nmt_log_prob_threshold ${nmt_log_prob_threshold} \
        --effecting_words_threshold ${effecting_words_threshold} \
        --consistence_trans_portion_threshold ${consistence_trans_portion_threshold} \
        --uniques_portion_for_noiseORperturbed_threshold ${uniques_portion_for_noiseORperturbed_threshold} \
        --alignment_tool ${alignment_tool} \
        --label_output_path ${analyse_output_path}/pred_labels_${QE_method}.pkl \
        --src_tgt_influence_output_path ${analyse_output_path}/src_tgt_influence.pkl \
        --include_direct_influence "True"
    fi
  fi
done