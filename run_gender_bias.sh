#!/bin/bash
# Setting environment
source /home/tdinh/.bashrc
conda activate KIT_start
which python

export CUDA_VISIBLE_DEVICES=1
export CUDA_DEVICE_ORDER=PCI_BUS_ID  # make sure the GPU order is correct
export TORCH_HOME=/project/OML/tdinh/.cache/torch

nvidia-smi

#declare -a lang_pairs=("en2de" "ro2en" "et2en" "en2zh" )
lang_pair="en2de"
MTmodel="qe_wmt21"

dataname="winoMT"
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





# Run winoMT on the original translations
if [[ ! -f "../mt_gender/translations/${MTmodel}/${SRC_LANG}-${TGT_LANG}.txt" ]]; then
  echo "Reformat translation for winoMT"
  python -u find_gender_bias_utils.py \
    --function "reformat_trans" \
    --translation_df_path ${output_dir_original_SRC}/translations.csv \
    --output_reformat_dir ${output_dir_original_SRC} \
    --src_lang ${SRC_LANG} \
    --tgt_lang ${TGT_LANG}
  # Move the output to the winoMT directory
  mkdir -p ../mt_gender/translations/${MTmodel}
  mv ${output_dir_original_SRC}/reformatted_forWinoMT.txt ../mt_gender/translations/${MTmodel}/${SRC_LANG}-${TGT_LANG}.txt
fi

corpus_fn=/project/OML/tdinh/mt_gender/data/aggregates/en.txt
if [[ ! -f "${analyse_output_path}/gender_pred.csv" ]]; then
  echo "Eval gender bias on WinoMT"
  echo "Evaluating ${TGT_LANG} into ${analyse_output_path}/winoMT_eval.log"
  cd ../mt_gender/src
  export FAST_ALIGN_BASE=/project/OML/tdinh/fast_align
  ../scripts/evaluate_language.sh $corpus_fn ${TGT_LANG} ${MTmodel} | tee ${analyse_output_path}/winoMT_eval.log
  cd ../../KIT_start
  cp ../mt_gender/data/human/qe_wmt21/de/de.pred.csv ${analyse_output_path}/gender_pred.csv
fi




# Run quality estimation
sentence_level_eval_da="False"
trans_word_level_eval="True"
src_word_level_eval="False"
use_src_tgt_alignment="False"
data_root_dir="data"
output_root_path="output"
OUTPUT_dir=${output_root_path}/${dataname}_${lang_pair}
output_dir_original_SRC=${OUTPUT_dir}/original
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
    --seed ${seed} \
    --beam ${beam} \
    --mask_type ${mask_type} \
    --output_dir ${analyse_output_path} \
    --tokenize_sentences "True" \
    --use_src_tgt_alignment ${use_src_tgt_alignment} \
    --analyse_feature "edit_distance" "change_spread"
fi

# Run quality estimation and save the predicted labels
declare -a QE_methods=('nmt_log_prob' 'nr_effecting_src_words' 'wmt18_openkiwi' )
for QE_method in ${QE_methods[@]}; do
  if [[ ! -f ${analyse_output_path}/pred_labels_${QE_method}.pkl ]]; then
    if [[ ${QE_method} == "wmt18_openkiwi" ]]; then
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

  echo "Eval recall on gender bias by QE"
  python -u find_gender_bias_utils.py \
    --function "eval_gender_bias" \
    --gender_pred_path ${analyse_output_path}/gender_pred.csv \
    --winoMT_data_path ${corpus_fn} \
    --qe_pred_labels_path ${analyse_output_path}/pred_labels_${QE_method}.pkl \
    --output_path_eval_gender_bias ${analyse_output_path}/wrong_gender_recall_${QE_method}.txt

  echo "QE method ${QE_method}"
  head ${analyse_output_path}/wrong_gender_recall_${QE_method}.txt
  echo "-------------------------------------------------"
done


