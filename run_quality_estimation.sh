#!/bin/bash
# Setting environment
source /home/tdinh/.bashrc
conda activate KIT_start
which python

export CUDA_VISIBLE_DEVICES=1
export CUDA_DEVICE_ORDER=PCI_BUS_ID  # make sure the GPU order is correct
export TORCH_HOME=/project/OML/tdinh/.cache/torch

nvidia-smi

dataname="WMT21_DA_test"
SRC_LANG="en"
TGT_LANG="de"
sentence_level_eval_da="False"

trans_word_level_eval="True"
src_word_level_eval="True"
declare -a mask_types=("MultiplePerSentence_content" ) # "MultiplePerSentence_allWords" "MultiplePerSentence_allTokens" )
use_src_tgt_alignment="True"

dev=False
grouped_mask=False
trans_direction="${SRC_LANG}2${TGT_LANG}"
data_root_dir="data"
batch_size=100
seed=0
replacement_strategy="masking_language_model"
number_of_replacement=30
beam=5

for mask_type in ${mask_types[@]}; do
  df_root_path="output"
  analyse_output_path="analyse_output/${dataname}_${SRC_LANG}2${TGT_LANG}_${mask_type}"

  mkdir -p ${analyse_output_path}

  # Read and process the output perturbation translations df
  if [ "$use_src_tgt_alignment" = "True" ]; then
    # First generate the reformatted src-trans files to be used by awesome-align
    python -u src_tgt_alignment.py \
      --df_root_path ${df_root_path} \
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
    abs_df_root_path=$(realpath $df_root_path)
    cd ../awesome-align
    original_path_prefix=${abs_df_root_path}/${dataname}_${SRC_LANG}2${TGT_LANG}/original/translations
    perturbed_path_prefix=${abs_df_root_path}/${dataname}_${SRC_LANG}2${TGT_LANG}/${replacement_strategy}/beam${beam}_perturb${mask_type}/${number_of_replacement}replacements/seed${seed}/translations
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
    --df_root_path ${df_root_path} \
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
    --analyse_feature "highlight_changes" "edit_distance" "change_spread"

  python -u quality_estimation.py \
    --perturbed_trans_df_path ${analyse_output_path}/analyse_${dataname}_${SRC_LANG}2${TGT_LANG}_${mask_type}.pkl \
    --dataset ${dataname} \
    --data_root_path ${data_root_dir} \
    --src_lang ${SRC_LANG} \
    --tgt_lang ${TGT_LANG} \
    --seed ${seed} \
    --sentence_level_eval_da ${sentence_level_eval_da} \
    --trans_word_level_eval ${trans_word_level_eval} \
    --trans_word_level_eval_methods 'nmt_log_prob' 'nr_effecting_src_words' \
    --nmt_log_prob_thresholds 0.45 \
    --src_word_level_eval ${src_word_level_eval} \
    --src_word_level_eval_methods 'nmt_log_prob' 'nr_effecting_src_words' \
    --effecting_words_thresholds 1 \
    --consistence_trans_portion_thresholds 0.9 \
    --uniques_portion_for_noiseORperturbed_thresholds 0.4 \
    |& tee ${analyse_output_path}/quality_estimation.log
done
