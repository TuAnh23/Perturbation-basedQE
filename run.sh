#!/bin/bash
# Setting environment
source /home/tdinh/.bashrc
conda activate KIT_start
which python

export CUDA_VISIBLE_DEVICES=3
export CUDA_DEVICE_ORDER=PCI_BUS_ID  # make sure the GPU order is correct
export TORCH_HOME=/project/OML/tdinh/.cache/torch

nvidia-smi

dataname="masked_content_mustSHE"
qe_wmt22="True"
SRC_LANG="en"
TGT_LANG="de"
trans_direction="${SRC_LANG}2${TGT_LANG}"
premasked_groupped_by_word="False"
data_root_dir="data"
batch_size=60
seed=0
replacement_strategy="masking_language_model"
number_of_replacement=30

declare -a perturbation_types=("noun" "verb" "adjective" "adverb" "pronoun" )

#for beam in {5..1}; do
#  for perturbation_type in ${perturbation_types[@]}; do
#    timestamp=$(date +"%d-%m-%y-%T")
#    output_dir=output/${dataname}_${trans_direction}/${replacement_strategy}/beam${beam}_perturb${perturbation_type}/${number_of_replacement}replacements/seed${seed}
#    mkdir -p ${output_dir}
#    python -u generate_input_SRC.py ...
#    python -u translate.py ...
#  done
#done

beam=5
perturbation_type="content"
output_dir=output/${dataname}_${trans_direction}/${replacement_strategy}/beam${beam}_perturb${perturbation_type}/${number_of_replacement}replacements/seed${seed}
mkdir -p ${output_dir}
python -u generate_input_SRC.py \
  --data_root_dir ${data_root_dir} \
  --dataname ${dataname} \
  --perturbation_type ${perturbation_type} \
  --output_dir ${output_dir} \
  --seed ${seed} \
  --replacement_strategy ${replacement_strategy} \
  --number_of_replacement ${number_of_replacement} \
  --premasked_groupped_by_word ${premasked_groupped_by_word} \
  |& tee -a ${output_dir}/generate_input_SRC.log

if [ "$qe_wmt22" = "True" ]; then
  # Use the same NMT model as the ones used to create the QE dataset to translate
  # Instructions: https://github.com/facebookresearch/mlqe/blob/main/nmt_models/README-translate.md
  # Define vars
  INPUT=${output_dir}/input
  BPE_ROOT=/home/tdinh/miniconda3/envs/KIT_start/lib/python3.9/site-packages/subword_nmt
  BPE=models/${SRC_LANG}-${TGT_LANG}/bpecodes
  MODEL_DIR=models/${SRC_LANG}-${TGT_LANG}
  TMP=/export/data1/tdinh
  # Preprocess input data
  python -u QE_WMT22_format_utils.py \
    --func "format_input" \
    --output_dir ${output_dir} \
    --SRC_perturbed_type ${perturbation_type} \
    --src_lang ${SRC_LANG} \
    --tgt_lang ${TGT_LANG} \
    --tmp_dir ${TMP}
  # Tokenize data
  python -u QE_WMT22_format_utils.py \
    --func "tokenize" \
    --output_dir ${output_dir} \
    --src_lang ${SRC_LANG} \
    --tgt_lang ${TGT_LANG} \
    --tmp_dir ${TMP}
  # Apply bpe
  for LANG in $SRC_LANG $TGT_LANG; do
    python $BPE_ROOT/apply_bpe.py -c ${BPE} < $TMP/preprocessed.tok.$LANG > $TMP/preprocessed.tok.bpe.$LANG
  done
  # Binarize the data for faster translation
  fairseq-preprocess --srcdict $MODEL_DIR/dict.$SRC_LANG.txt --tgtdict $MODEL_DIR/dict.$TGT_LANG.txt --source-lang ${SRC_LANG} --target-lang ${TGT_LANG} --testpref $TMP/preprocessed.tok.bpe --destdir $TMP/bin --workers 4
  # Translate
  fairseq-generate $TMP/bin --path ${MODEL_DIR}/${SRC_LANG}-${TGT_LANG}.pt --beam 5 --source-lang $SRC_LANG --target-lang $TGT_LANG --no-progress-bar --unkpen 5 > $TMP/fairseq.out
  grep ^H $TMP/fairseq.out | cut -d- -f2- | sort -n | cut -f3- > $TMP/mt.out
  # Post-process
  sed -r 's/(@@ )| (@@ ?$)//g' < $TMP/mt.out > $TMP/mt_filtered.out
  python -u QE_WMT22_format_utils.py \
    --func "detokenize" \
    --output_dir ${output_dir} \
    --src_lang ${SRC_LANG} \
    --tgt_lang ${TGT_LANG} \
    --tmp_dir ${TMP}
  # Put the translation to the dataframe
  python -u QE_WMT22_format_utils.py \
    --func "format_translation_file" \
    --output_dir ${output_dir} \
    --SRC_perturbed_type ${perturbation_type} \
    --src_lang ${SRC_LANG} \
    --tgt_lang ${TGT_LANG} \
    --tmp_dir ${TMP}
else
  python -u translate.py \
    --output_dir ${output_dir} \
    --beam ${beam} \
    --seed ${seed} \
    --batch_size ${batch_size} \
    --trans_direction ${trans_direction} \
    --SRC_perturbed_type ${perturbation_type} \
    |& tee -a ${output_dir}/translate.log
fi
