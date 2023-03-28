#!/bin/bash
# Setting environment
source /home/tdinh/.bashrc
conda activate KIT_start
which python

export CUDA_VISIBLE_DEVICES=1
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
  MTmodel="qe_wmt21"  # LLM, qe_wmt21
else
  MTmodel=$6
fi

dev=False
grouped_mask=False
trans_direction="${SRC_LANG}2${TGT_LANG}"
data_root_dir="data"
batch_size=100
seed=0
replacement_strategy="masking_language_model_${unmasking_model}"
number_of_replacement=30
beam=5


if [[ $mask_type == *"occupation"* ]]; then
  # Data from https://www.enchantedlearning.com/wordlist/jobs.shtml
  masked_vocab_path=${data_root_dir}/JobTitles.txt
elif [[ $mask_type == *"country"* ]]; then
  # Data from https://www.kaggle.com/datasets/alexeyblinnikov/country-adjective-pairs
  python -u process_vocab_data.py \
    --data_root_dir ${data_root_dir} \
    --dataname "Country_Adjective"
  masked_vocab_path=${data_root_dir}/Country_Adjective_vocab.txt
else
  masked_vocab_path="None"
fi

if [[ ${dataname} == "WMT21_DA"* ]]; then
  # Always uses qe_wmt21 MT models for consistent with gold label, so no need to specify
  OUTPUT_dir=output/${dataname}_${trans_direction}
else
  OUTPUT_dir=output/${dataname}_${trans_direction}_${MTmodel}
fi

TMP_dir=/export/data1/tdinh/${dataname}_${trans_direction}
output_dir_original_SRC=${OUTPUT_dir}/original
output_dir_perturbed_SRC=${OUTPUT_dir}/${replacement_strategy}/beam${beam}_perturb${mask_type}/${number_of_replacement}replacements/seed${seed}
TMP_dir_original_SRC=${TMP_dir}/original
TMP_dir_perturbed_SRC=${TMP_dir}/${replacement_strategy}/beam${beam}_perturb${mask_type}/${number_of_replacement}replacements/seed${seed}

#if [ -d "$output_dir_perturbed_SRC" ]; then
#  echo "Output files exists. Skip perturbation and translation."
#  exit 0
#fi

mkdir -p ${output_dir_original_SRC}
mkdir -p ${output_dir_perturbed_SRC}
mkdir -p ${TMP_dir_original_SRC}
mkdir -p ${TMP_dir_perturbed_SRC}

# Process the data
if [ ! -f "${output_dir_original_SRC}/src_df.csv" ]; then
  python -u process_src_data.py \
    --data_root_dir ${data_root_dir} \
    --src_lang ${SRC_LANG} \
    --tgt_lang ${TGT_LANG} \
    --dataname ${dataname} \
    --seed ${seed} \
    --output_dir ${output_dir_original_SRC} \
    --dev ${dev}
fi

if [ ! -f "${output_dir_perturbed_SRC}/masked_df.csv" ]; then
  # Mask the data
  python -u mask_src_data.py \
    --original_src_path ${output_dir_original_SRC}/src_df.csv \
    --src_lang ${SRC_LANG} \
    --mask_type ${mask_type} \
    --seed ${seed} \
    --output_dir ${output_dir_perturbed_SRC} \
    --masked_vocab_path ${masked_vocab_path}
fi

if [ ! -f "${output_dir_perturbed_SRC}/unmasked_df.csv" ]; then
  # Unmask the data to generate perturbed sentences
  python -u unmask.py \
    --masked_src_path ${output_dir_perturbed_SRC}/masked_df.csv \
    --src_lang ${SRC_LANG} \
    --seed ${seed} \
    --output_dir ${output_dir_perturbed_SRC} \
    --replacement_strategy ${replacement_strategy} \
    --number_of_replacement ${number_of_replacement} \
    --grouped_mask ${grouped_mask} \
    --unmasking_model ${unmasking_model}
fi

# Translate original and perturbed sentences
declare -a input_SRC_columns=("SRC" "SRC_perturbed" )

for input_SRC_column in ${input_SRC_columns[@]}; do
  if [ "$input_SRC_column" = "SRC" ]; then
    output_dir=${output_dir_original_SRC}
    TMP=${TMP_dir_original_SRC}
    input_src_path=${output_dir_original_SRC}/src_df.csv
  else
    output_dir=${output_dir_perturbed_SRC}
    TMP=${TMP_dir_perturbed_SRC}
    input_src_path=${output_dir_perturbed_SRC}/unmasked_df.csv
  fi

  if [ -f "${output_dir}/translations.csv" ]; then
      continue
  fi

  if [ "$MTmodel" = "qe_wmt21" ]; then
    # Preprocess input data
    python -u QE_WMT21_format_utils.py \
      --func "format_input" \
      --input_src_path ${input_src_path} \
      --output_dir ${output_dir} \
      --input_SRC_column ${input_SRC_column} \
      --src_lang ${SRC_LANG} \
      --tgt_lang ${TGT_LANG} \
      --tmp_dir ${TMP}

    # Use the same NMT model as the ones used to create the QE dataset to translate
    if [[ ($SRC_LANG == "en" && $TGT_LANG == "de") || ($SRC_LANG == "en" && $TGT_LANG == "zh") || ($SRC_LANG == "ro" && $TGT_LANG == "en") || ($SRC_LANG == "et" && $TGT_LANG == "en") ]]; then
      # Instructions: https://github.com/facebookresearch/mlqe/blob/main/nmt_models/README-translate.md
      # Define vars
      INPUT=${output_dir}/input
      OUTPUT=${output_dir}/trans_sentences.txt
      BPE_ROOT=/home/tdinh/miniconda3/envs/KIT_start/lib/python3.9/site-packages/subword_nmt
      BPE=models/${SRC_LANG}-${TGT_LANG}/bpecodes
      MODEL_DIR=models/${SRC_LANG}-${TGT_LANG}
      # Tokenize data
      for LANG in $SRC_LANG $TGT_LANG; do
#        sacremoses -l $LANG tokenize -a < $INPUT.$LANG > $TMP/preprocessed.tok.$LANG
        perl ../mosesdecoder/scripts/tokenizer/tokenizer.perl -threads 80 -a -l $LANG < $INPUT.$LANG > $TMP/preprocessed.tok.$LANG
        python $BPE_ROOT/apply_bpe.py -c ${BPE} < $TMP/preprocessed.tok.$LANG > $TMP/preprocessed.tok.bpe.$LANG
      done
      # Apply bpe
      for LANG in $SRC_LANG $TGT_LANG; do
        python $BPE_ROOT/apply_bpe.py -c ${BPE} < $TMP/preprocessed.tok.$LANG > $TMP/preprocessed.tok.bpe.$LANG
      done
      # Binarize the data for faster translation
      fairseq-preprocess --srcdict $MODEL_DIR/dict.$SRC_LANG.txt --tgtdict $MODEL_DIR/dict.$TGT_LANG.txt --source-lang ${SRC_LANG} --target-lang ${TGT_LANG} --testpref $TMP/preprocessed.tok.bpe --destdir $TMP/bin --workers 4
      # Translate
      fairseq-generate $TMP/bin --path ${MODEL_DIR}/${SRC_LANG}-${TGT_LANG}.pt --beam 5 --source-lang $SRC_LANG --target-lang $TGT_LANG --no-progress-bar --unkpen 5 --batch-size ${batch_size} > ${output_dir}/fairseq.out
      grep ^H ${output_dir}/fairseq.out | cut -d- -f2- | sort -n | cut -f3- > ${output_dir}/mt.out
      grep ^P ${output_dir}/fairseq.out | cut -d- -f2- | sort -n | cut -f2- > ${output_dir}/log_prob.out
      # Post-process
#      sed -r 's/(@@ )| (@@ ?$)//g' < ${output_dir}/mt.out | sacremoses -l $TGT_LANG detokenize > $OUTPUT
      sed -r 's/(@@ )| (@@ ?$)//g' < ${output_dir}/mt.out | perl ../mosesdecoder/scripts/tokenizer/detokenizer.perl -l $TGT_LANG > $OUTPUT
    elif [[ ($SRC_LANG == "en" && $TGT_LANG == "ja") || ($SRC_LANG == "en" && $TGT_LANG == "cs") ]]; then
      model_path=models/mbart50.ft.1n
      data_root=${TMP}
      raw_folder=raw
      if [ "$SRC_LANG" = "en" ]; then
        SRC_LANG_formatted="en_XX"
      fi
      if [ "$TGT_LANG" = "ja" ]; then
        TGT_LANG_formatted="ja_XX"
      elif [ "$TGT_LANG" = "cs" ]; then
        TGT_LANG_formatted="cs_CZ"
      fi

      mkdir ${data_root}/${raw_folder}
      cp ${output_dir}/input.${SRC_LANG} ${data_root}/${raw_folder}/test.${SRC_LANG_formatted}-${TGT_LANG_formatted}.${SRC_LANG_formatted}
      cp ${output_dir}/input.${TGT_LANG} ${data_root}/${raw_folder}/test.${SRC_LANG_formatted}-${TGT_LANG_formatted}.${TGT_LANG_formatted}
      python -u binarize.py \
        --data_root ${data_root} \
        --raw-folder ${raw_folder} \
        --spm_model ${model_path}/sentence.bpe.model \
        --spm_vocab ${model_path}/dict.${SRC_LANG_formatted}.txt ${model_path}/dict.${TGT_LANG_formatted}.txt

      path_2_data=${data_root}/databin
      model=${model_path}/model.pt
      lang_list=${model_path}/ML50_langs.txt

      fairseq-generate $path_2_data \
        --path $model \
        --task translation_multi_simple_epoch \
        --gen-subset test \
        --source-lang ${SRC_LANG_formatted} \
        --target-lang ${TGT_LANG_formatted} \
        --sacrebleu \
        --batch-size 32 \
        --encoder-langtok "src" \
        --decoder-langtok \
        --lang-dict "$lang_list" \
        --lang-pairs "${SRC_LANG_formatted}-${TGT_LANG_formatted}" > ${output_dir}/fairseq.out
      grep ^H ${output_dir}/fairseq.out | cut -d- -f2- | sort -n | cut -f3- > ${output_dir}/mt.out
      grep ^P ${output_dir}/fairseq.out | cut -d- -f2- | sort -n | cut -f2- > ${output_dir}/log_prob.out
      # Post-process
      sed -r 's/ //g; s/▁/ /g; s/^[[:space:]]*//' < ${output_dir}/mt.out > ${output_dir}/trans_sentences.txt
    fi

    # Put the translation to the dataframe
    python -u QE_WMT21_format_utils.py \
      --func "format_translation_file" \
      --input_src_path ${input_src_path} \
      --output_dir ${output_dir} \
      --input_SRC_column ${input_SRC_column} \
      --src_lang ${SRC_LANG} \
      --tgt_lang ${TGT_LANG} \
      --tmp_dir ${TMP}
  elif [[ ${MTmodel} == "LLM" ]]; then
    shot=0
    LLM_model=google/flan-ul2  # google/flan-t5-small, google/flan-ul2
    # Preprocess input data
    python -u QE_WMT21_format_utils.py \
      --func "format_input" \
      --input_src_path ${input_src_path} \
      --output_dir ${output_dir} \
      --input_SRC_column ${input_SRC_column} \
      --src_lang ${SRC_LANG} \
      --tgt_lang ${TGT_LANG} \
      --tmp_dir ${TMP}

    source /home/skoneru/miniconda3/bin/activate llm
    python llm_eval.py \
      --input_file ${output_dir}/input.${SRC_LANG} \
      --hyp_output_file ${output_dir}/trans_sentences.txt \
      --tokenized_hyp_output_file ${output_dir}/mt.out \
      --logprobs_hyp_output_file ${output_dir}/log_prob.out \
      --target_file ${output_dir}/input.${TGT_LANG} \
      --model $LLM_model \
      --shot $shot
    source /home/tdinh/miniconda3/bin/activate KIT_start
    # Put the translation to the dataframe
    python -u QE_WMT21_format_utils.py \
      --func "format_translation_file" \
      --input_src_path ${input_src_path} \
      --output_dir ${output_dir} \
      --input_SRC_column ${input_SRC_column} \
      --src_lang ${SRC_LANG} \
      --tgt_lang ${TGT_LANG} \
      --tmp_dir ${TMP}
  else
    echo "MT MODEL ${MTmodel} NOT AVAILABLE!!"
#    python -u translate.py \
#      --input_path ${input_src_path} \
#      --output_dir ${output_dir} \
#      --beam ${beam} \
#      --seed ${seed} \
#      --batch_size ${batch_size} \
#      --trans_direction ${trans_direction} \
#      --input_SRC_column ${input_SRC_column} \
#      |& tee -a ${output_dir}/translate.log
  fi
done
