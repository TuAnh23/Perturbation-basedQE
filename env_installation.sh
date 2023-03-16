conda create -n KIT_start python=3.9.13
conda install pytorch cudatoolkit=11.6 -c pytorch -c conda-forge # version 1.12.1
pip install fairseq
pip install sacremoses
pip install fastBPE
pip install subword_nmt
conda install pandas
conda install -c conda-forge gensim
conda install -c anaconda nltk
conda install -c anaconda requests
conda install -c conda-forge sacrebleu
conda install -c conda-forge huggingface_hub
conda install -c huggingface transformers==4.14.1 tokenizers==0.10.3
pip install sentencepiece
#pip install tensor2tensor
#conda install libgcc
#conda install -c conda-forge gcc=12.1.0
#conda install -c anaconda cudnn
pip install "spacy<3.0.0"
python3 -m spacy download en_core_web_sm
python3 -m spacy download de_core_news_sm
pip install edist
pip install jieba==0.42.1
pip install fugashi==1.1.1
pip install indic-nlp-library==0.81
pip install unidic-lite

cd ../

git clone https://github.com/TuAnh23/mt_gender.git
cd mt_gender
./install.sh
cd ../

git clone https://github.com/clab/fast_align.git
cd fast_align
mkdir build
cd build
cmake ..
make
cd ../

git clone https://github.com/moses-smt/mosesdecoder.git

cd KIT_start

conda create -n openkiwi python=3.8
conda activate openkiwi
pip install openkiwi
pip install protobuf==3.20.1
conda install pandas=1.3