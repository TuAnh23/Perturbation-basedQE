#!/bin/bash
# Setting environment
source /home/tdinh/.bashrc
conda activate KIT_start
which python

export CUDA_VISIBLE_DEVICES=5
export CUDA_DEVICE_ORDER=PCI_BUS_ID  # make sure the GPU order is correct
export TORCH_HOME=/project/OML/tdinh/.cache/torch
export HF_HOME=/project/OML/tdinh/.cache/huggingface

nvidia-smi

if [ -z "$1" ]; then
  dataname="WMT21_DA_dev"
else
  dataname=$1
fi

if [ -z "$2" ]; then
  SRC_LANG="en"
else
  SRC_LANG=$2
fi

if [ -z "$3" ]; then
  TGT_LANG="de"
else
  TGT_LANG=$3
fi

if [ -z "$4" ]; then
  mask_type="MultiplePerSentence_content"  # "MultiplePerSentence_content" "MultiplePerSentence_allWords" "MultiplePerSentence_allTokens" )
else
  mask_type=$4
fi

if [ -z "$5" ]; then
  unmasking_model='bert-base-cased'
else
  unmasking_model=$5
fi

if [ -z "$6" ]; then
  analyse_output_path="analyse_output/${dataname}_${SRC_LANG}2${TGT_LANG}/${mask_type}_${unmasking_model}"
else
  analyse_output_path=$6
fi

if [ -z "$7" ]; then
  alignment_tool="Tercom"
else
  alignment_tool=$7
fi

if [ -z "$8" ]; then
  declare -a effecting_words_thresholds=(1 2 3 4 )
else
  effecting_words_thresholds=$8
fi

if [ -z "$9" ]; then
  declare -a consistence_trans_portion_thresholds=(0.9 0.95 )
else
  consistence_trans_portion_thresholds=$9
fi

if [ -z "${10}" ]; then
  declare -a uniques_portion_for_noiseORperturbed_thresholds=(0.4 0.5 0.6 0.7 0.8 0.9 )
else
  uniques_portion_for_noiseORperturbed_thresholds=${10}
fi

sentence_level_eval_da="False"
trans_word_level_eval="True"
src_word_level_eval="False"

use_src_tgt_alignment="False"

dev=False
grouped_mask=False
trans_direction="${SRC_LANG}2${TGT_LANG}"
data_root_dir="data"
batch_size=100
seed=0
replacement_strategy="masking_language_model_${unmasking_model}"
number_of_replacement=30
beam=5

output_root_path="output"
OUTPUT_dir=${output_root_path}/${dataname}_${trans_direction}
output_dir_original_SRC=${OUTPUT_dir}/original


if [ -f "${analyse_output_path}/quality_estimation_${alignment_tool}.log" ]; then
  echo "Output files exists. Skip QE."
  exit 0
fi

mkdir -p ${analyse_output_path}

# Read and process the output perturbation translations df
if [ "$use_src_tgt_alignment" = "True" ]; then
  # First generate the reformatted src-trans files to be used by awesome-align
  python -u src_tgt_alignment.py \
    --output_root_path ${output_root_path} \
    --data_root_path ${data_root_dir} \
    --src_lang ${SRC_LANG} \
    --tgt_lang ${TGT_LANG} \
    --replacement_strategy ${replacement_strategy} \
    --number_of_replacement ${number_of_replacement} \
    --dataname ${dataname} \
    --seed ${seed} \
    --beam ${beam} \
    --mask_type ${mask_type}
  # Run awesome-align
  abs_output_root_path=$(realpath $output_root_path)
  cd ../awesome-align
  original_path_prefix=${abs_output_root_path}/${dataname}_${SRC_LANG}2${TGT_LANG}/original/translations
  perturbed_path_prefix=${abs_output_root_path}/${dataname}_${SRC_LANG}2${TGT_LANG}/${replacement_strategy}/beam${beam}_perturb${mask_type}/${number_of_replacement}replacements/seed${seed}/translations
  declare -a path_prefixes=(${original_path_prefix} ${perturbed_path_prefix} )
  for PATH_PREFIX in ${path_prefixes[@]}; do
    echo $PATH_PREFIX
    DATA_FILE=${PATH_PREFIX}_reformatted.txt
    MODEL_NAME_OR_PATH=bert-base-multilingual-cased
    OUTPUT_WORDPAIR=${PATH_PREFIX}_word_alignment.txt
    OUT_PROB=${PATH_PREFIX}_prob_alignment.txt
    OUT_IDX=${PATH_PREFIX}_index_alignment.txt

    awesome-align \
      --output_file=$OUT_IDX \
      --output_word_file=$OUTPUT_WORDPAIR \
      --output_prob_file=$OUT_PROB\
      --model_name_or_path=$MODEL_NAME_OR_PATH \
      --data_file=$DATA_FILE \
      --extraction 'softmax' \
      --num_workers 0 \
      --batch_size 32
  done
  cd ../KIT_start
fi

python -u read_and_analyse_df.py \
  --output_root_path ${output_root_path} \
  --data_root_path ${data_root_dir} \
  --src_lang ${SRC_LANG} \
  --tgt_lang ${TGT_LANG} \
  --replacement_strategy ${replacement_strategy} \
  --number_of_replacement ${number_of_replacement} \
  --dataname ${dataname} \
  --seed ${seed} \
  --beam ${beam} \
  --mask_type ${mask_type} \
  --output_dir ${analyse_output_path} \
  --tokenize_sentences "True" \
  --use_src_tgt_alignment ${use_src_tgt_alignment} \
  --analyse_feature "edit_distance" "change_spread"

python -u tune_quality_estimation.py \
  --perturbed_trans_df_path ${analyse_output_path}/analyse_${dataname}_${SRC_LANG}2${TGT_LANG}_${mask_type}.pkl \
  --original_translation_output_dir ${output_dir_original_SRC} \
  --dataset ${dataname} \
  --data_root_path ${data_root_dir} \
  --src_lang ${SRC_LANG} \
  --tgt_lang ${TGT_LANG} \
  --seed ${seed} \
  --sentence_level_eval_da ${sentence_level_eval_da} \
  --trans_word_level_eval ${trans_word_level_eval} \
  --trans_word_level_eval_methods 'nr_effecting_src_words' \
  --nmt_log_prob_thresholds 0.45 \
  --src_word_level_eval ${src_word_level_eval} \
  --src_word_level_eval_methods 'nmt_log_prob' 'nr_effecting_src_words' \
  --effecting_words_thresholds ${effecting_words_thresholds[@]} \
  --consistence_trans_portion_thresholds ${consistence_trans_portion_thresholds[@]} \
  --uniques_portion_for_noiseORperturbed_thresholds ${uniques_portion_for_noiseORperturbed_thresholds[@]} \
  --alignment_tool ${alignment_tool} |& tee ${analyse_output_path}/quality_estimation_${alignment_tool}.log

