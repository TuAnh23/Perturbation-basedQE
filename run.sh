#!/bin/bash
# Setting environment
source /home/tdinh/.bashrc
conda activate KIT_start
which python

export CUDA_VISIBLE_DEVICES=0
export CUDA_DEVICE_ORDER=PCI_BUS_ID  # make sure the GPU order is correct
export TORCH_HOME=/project/OML/tdinh/.cache/torch

nvidia-smi

dataname="covost2_all"
qe_wmt21="False"
SRC_LANG="en"
TGT_LANG="vi"
mask_type="MultiplePerSentence_country"
dev=False
grouped_mask=False
trans_direction="${SRC_LANG}2${TGT_LANG}"
data_root_dir="data"
batch_size=100
seed=0
replacement_strategy="masking_language_model"
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

OUTPUT_dir=output/${dataname}_${trans_direction}
TMP_dir=/export/data1/tdinh/${dataname}_${trans_direction}
output_dir_original_SRC=${OUTPUT_dir}/original
output_dir_perturbed_SRC=${OUTPUT_dir}/${replacement_strategy}/beam${beam}_perturb${mask_type}/seed${seed}
TMP_dir_original_SRC=${TMP_dir}/original
TMP_dir_perturbed_SRC=${TMP_dir}/${replacement_strategy}/beam${beam}_perturb${mask_type}/seed${seed}
mkdir -p ${output_dir_original_SRC}
mkdir -p ${output_dir_perturbed_SRC}
mkdir -p ${TMP_dir_original_SRC}
mkdir -p ${TMP_dir_perturbed_SRC}

# Process the data
python -u process_src_data.py \
  --data_root_dir ${data_root_dir} \
  --src_lang ${SRC_LANG} \
  --tgt_lang ${TGT_LANG} \
  --dataname ${dataname} \
  --seed ${seed} \
  --output_dir ${output_dir_original_SRC} \
  --dev ${dev}

# Mask the data
python -u mask_src_data.py \
  --original_src_path ${output_dir_original_SRC}/src_df.csv \
  --src_lang ${SRC_LANG} \
  --mask_type ${mask_type} \
  --seed ${seed} \
  --output_dir ${output_dir_perturbed_SRC} \
  --masked_vocab_path ${masked_vocab_path}

# Unmask the data to generate perturbed sentences
python -u unmask.py \
  --masked_src_path ${output_dir_perturbed_SRC}/masked_df.csv \
  --src_lang ${SRC_LANG} \
  --seed ${seed} \
  --output_dir ${output_dir_perturbed_SRC} \
  --replacement_strategy ${replacement_strategy} \
  --number_of_replacement ${number_of_replacement} \
  --grouped_mask ${grouped_mask}

# Translate original and perturbed sentences
declare -a columns_to_be_translated=("SRC" "SRC_perturbed" )

for column_to_be_translated in ${columns_to_be_translated[@]}; do
  if [ "$column_to_be_translated" = "SRC" ]; then
    output_dir=${output_dir_original_SRC}
    TMP=${TMP_dir_original_SRC}
    input_src_path=${output_dir_original_SRC}/src_df.csv
  else
    output_dir=${output_dir_perturbed_SRC}
    TMP=${TMP_dir_perturbed_SRC}
    input_src_path=${output_dir_perturbed_SRC}/unmasked_df.csv
  fi

  if [ "$qe_wmt21" = "True" ]; then
    # Use the same NMT model as the ones used to create the QE dataset to translate
    # Instructions: https://github.com/facebookresearch/mlqe/blob/main/nmt_models/README-translate.md
    # Define vars
    INPUT=${output_dir}/input
    OUTPUT=${output_dir}/trans_sentences.txt
    BPE_ROOT=/home/tdinh/miniconda3/envs/KIT_start/lib/python3.9/site-packages/subword_nmt
    BPE=models/${SRC_LANG}-${TGT_LANG}/bpecodes
    MODEL_DIR=models/${SRC_LANG}-${TGT_LANG}
    # Preprocess input data
    python -u QE_WMT21_format_utils.py \
      --func "format_input" \
      --input_src_path ${input_src_path} \
      --output_dir ${output_dir} \
      --column_to_be_formatted ${column_to_be_translated} \
      --src_lang ${SRC_LANG} \
      --tgt_lang ${TGT_LANG} \
      --tmp_dir ${TMP}
    # Tokenize data
    for LANG in $SRC_LANG $TGT_LANG; do
      sacremoses -l $LANG tokenize -a < $INPUT.$LANG > $TMP/preprocessed.tok.$LANG
      python $BPE_ROOT/apply_bpe.py -c ${BPE} < $TMP/preprocessed.tok.$LANG > $TMP/preprocessed.tok.bpe.$LANG
    done
    # Apply bpe
    for LANG in $SRC_LANG $TGT_LANG; do
      python $BPE_ROOT/apply_bpe.py -c ${BPE} < $TMP/preprocessed.tok.$LANG > $TMP/preprocessed.tok.bpe.$LANG
    done
    # Binarize the data for faster translation
    fairseq-preprocess --srcdict $MODEL_DIR/dict.$SRC_LANG.txt --tgtdict $MODEL_DIR/dict.$TGT_LANG.txt --source-lang ${SRC_LANG} --target-lang ${TGT_LANG} --testpref $TMP/preprocessed.tok.bpe --destdir $TMP/bin --workers 4
    # Translate
    fairseq-generate $TMP/bin --path ${MODEL_DIR}/${SRC_LANG}-${TGT_LANG}.pt --beam 5 --source-lang $SRC_LANG --target-lang $TGT_LANG --no-progress-bar --unkpen 5 --batch-size ${batch_size} > $TMP/fairseq.out
    grep ^H $TMP/fairseq.out | cut -d- -f2- | sort -n | cut -f3- > $TMP/mt.out
    # Post-process
    sed -r 's/(@@ )| (@@ ?$)//g' < $TMP/mt.out | sacremoses -l $TGT_LANG detokenize > $OUTPUT
    # Put the translation to the dataframe
    python -u QE_WMT21_format_utils.py \
      --func "format_translation_file" \
      --input_src_path ${input_src_path} \
      --output_dir ${output_dir} \
      --column_to_be_formatted ${column_to_be_translated} \
      --src_lang ${SRC_LANG} \
      --tgt_lang ${TGT_LANG} \
      --tmp_dir ${TMP}
  else
    python -u translate.py \
      --input_path ${input_src_path} \
      --output_dir ${output_dir} \
      --beam ${beam} \
      --seed ${seed} \
      --batch_size ${batch_size} \
      --trans_direction ${trans_direction} \
      --column_to_be_translated ${column_to_be_translated} \
      |& tee -a ${output_dir}/translate.log
  fi
done
