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
  MTmodel="LLM"  ## "LLM" or "qe_wmt21"
else
  MTmodel=$1
fi

#declare -a lang_pairs=("en2de" "ro2en" "et2en" "en2zh" )
lang_pair="en2de"

dataname="mucow"
SRC_LANG=${lang_pair:0:2}
TGT_LANG=${lang_pair:3:2}

analyse_output_path="analyse_output/${dataname}_${SRC_LANG}2${TGT_LANG}_${MTmodel}"
analyse_output_path=$(realpath $analyse_output_path)
mkdir -p ${analyse_output_path}


# Get the translations (original and perturbed)
if [[ (${lang_pair} == "en2de") || (${lang_pair} == "en2cs") ]]; then
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
OUTPUT_dir=output/${dataname}_${lang_pair}_${MTmodel}
output_dir_original_SRC=${OUTPUT_dir}/original
output_dir_original_SRC=$(realpath $output_dir_original_SRC)



# Run MuCoW WSD labels
cd "../MuCoW/WMT2019/translation test suite" || exit
python evaluate.py \
  --lgpair ${SRC_LANG}-${TGT_LANG} \
  --rawtranslations ${output_dir_original_SRC}/trans_sentences.txt \
  --output_path ${analyse_output_path}/wsd_labels.csv | tee ${analyse_output_path}/MuCoW_eval.log
cd ../../../KIT_start || exit



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
declare -a QE_methods=('nmt_log_prob' 'nr_effecting_src_words' 'openkiwi_2.1.0' 'openkiwi_wmt21' )
for QE_method in ${QE_methods[@]}; do
  if [[ ! -f ${analyse_output_path}/pred_labels_${QE_method}.pkl ]]; then
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
        --label_output_path ${analyse_output_path}/pred_labels_${QE_method}.pkl
    fi
  fi

  echo "Eval WSD error by QE"
  python -u find_wsd_utils.py \
    --wsd_label_path ${analyse_output_path}/wsd_labels.csv \
    --qe_pred_labels_path ${analyse_output_path}/pred_labels_${QE_method}.pkl \
    --output_path_eval_wsd_error ${analyse_output_path}/wrong_wsd_recall_${QE_method}.txt

  echo "QE method ${QE_method}"
  head ${analyse_output_path}/wrong_wsd_recall_${QE_method}.txt
  echo "-------------------------------------------------"
done
