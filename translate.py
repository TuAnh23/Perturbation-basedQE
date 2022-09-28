import argparse
import torch
import pandas as pd
import nltk
from nltk.corpus import stopwords
import gensim.downloader as api
import random
import logging
from transformers import T5ForConditionalGeneration, T5Tokenizer

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)

LOGGER = logging.getLogger(__name__)


# nltk.download()


def main():
    def none_or_str(value: str):
        if value.lower() == 'none':
            return None
        return value

    def str_to_bool(value: str):
        if value.lower() == 'yes' or value.lower() == 'true':
            return True
        elif value.lower() == 'no' or value.lower() == 'false':
            return False
        else:
            raise ValueError

    parser = argparse.ArgumentParser(description='Translate with perturbation to source sentences.')
    parser.add_argument('--data_root_dir', type=str, default="data")
    parser.add_argument('--dataname', type=str, default="MuST-SHE-en2fr")
    parser.add_argument('--perturbation_type', type=none_or_str, default=None)
    parser.add_argument('--beam', type=int, default=5)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--batch_size', type=int, default=5)
    parser.add_argument('--output_dir', type=str)
    parser.add_argument("--dev", type=str_to_bool, default=False,
                        help="Run on a tiny amount of data for developing")
    args = parser.parse_args()
    print(args)

    random.seed(args.seed)

    if args.dataname == "MuST-SHE-en2fr":
        LOGGER.info("Loading pretrained translation model")
        src2tgt_model = torch.hub.load('pytorch/fairseq', 'transformer.wmt14.en-fr')
        src2tgt_model.eval()  # disable dropout
        src2tgt_model.cuda()  # move model to GPU

        # Load test translation data
        src_tgt_df = pd.read_csv(f"{args.data_root_dir}/MuST-SHE_v1.2/MuST-SHE-v1.2-data/tsv/MONOLINGUAL.fr_v1.2.tsv",
                                 sep='\t', index_col=0)[['SRC', "REF"]]

    elif args.dataname == "Europarl-en2de":
        # Use test data from Europarl at https://www.statmt.org/europarl/archives.html
        LOGGER.info("Loading pretrained translation model")
        src2tgt_model = torch.hub.load('pytorch/fairseq', 'transformer.wmt19.en-de', tokenizer='moses', bpe='fastbpe',
                                       checkpoint_file='model1.pt:model2.pt:model3.pt:model4.pt')
        src2tgt_model.eval()  # disable dropout
        src2tgt_model.cuda()  # move model to GPU

        # Load test translation data
        with open(f"{args.data_root_dir}/common-test/ep-test.en", encoding="ISO-8859-1") as f:
            en_sentences = f.readlines()
            en_sentences = [line.rstrip() for line in en_sentences]
        with open(f"{args.data_root_dir}/common-test/ep-test.de", encoding="ISO-8859-1") as f:
            de_sentences = f.readlines()
            de_sentences = [line.rstrip() for line in de_sentences]
        src_tgt_df = pd.DataFrame(data={'SRC': en_sentences, 'REF': de_sentences})

    elif args.dataname == "IWSLT'15-en2vi":
        with open(f"{args.data_root_dir}/IWSLT'15-en2vi/tst2013.en") as f:
            en_sentences = f.readlines()
            en_sentences = [line.rstrip() for line in en_sentences]
        with open(f"{args.data_root_dir}/IWSLT'15-en2vi/tst2013.vi") as f:
            vi_sentences = f.readlines()
            vi_sentences = [line.rstrip() for line in vi_sentences]
        src_tgt_df = pd.DataFrame(data={'SRC': en_sentences, 'REF': vi_sentences})

        # Use en-vi model at https://huggingface.co/NlpHUST/t5-en-vi-small
        if torch.cuda.is_available():
            device = torch.device("cuda")
            print('There are %d GPU(s) available.' % torch.cuda.device_count())
            print('We will use the GPU:', torch.cuda.get_device_name(0))
        else:
            print('No GPU available, using  CPU instead.')
            device = torch.device("cpu")

        src2tgt_model = T5ForConditionalGeneration.from_pretrained("NlpHUST/t5-en-vi-small")
        tokenizer = T5Tokenizer.from_pretrained("NlpHUST/t5-en-vi-small")
        src2tgt_model.to(device)
        src2tgt_model.eval()

    else:
        raise RuntimeError(f"Dataset {args.dataname} not available.")

    if args.dev:
        src_tgt_perturbed = src_tgt_df[:10]  # Select a small number of rows to dev
    else:
        src_tgt_perturbed = src_tgt_df

    if args.perturbation_type is None:
        # Translate the original source sentences
        LOGGER.info("Translating original SRC sentences.")
        if args.dataname == "IWSLT'15-en2vi":
            src_tgt_perturbed["OriginalSRC-Trans"] = src_tgt_perturbed['SRC'].apply(
                lambda x: translate(src2tgt_model, tokenizer, x, args.beam, device))
        else:
            if args.batch_size > 1:
                src_tgt_perturbed["OriginalSRC-Trans"] = batch_translation(model=src2tgt_model,
                                                                           src_sentences=src_tgt_perturbed['SRC'].tolist(),
                                                                           beam=args.beam, batch_size=args.batch_size)
            else:
                src_tgt_perturbed["OriginalSRC-Trans"] = src_tgt_perturbed['SRC'].apply(
                    lambda x: src2tgt_model.translate(x, beam=args.beam))
    else:
        # Generate the perturbed sentences
        LOGGER.info("Perturbing sentences")
        LOGGER.debug("Loading word embedding model")
        word2vec_model = api.load('glove-wiki-gigaword-100')
        perturbed_df = src_tgt_perturbed[['SRC']]. \
            apply(lambda x: perturb_sentence(args.perturbation_type, x.values[0], word2vec_model), axis='columns',
                  result_type='expand'). \
            rename(columns={0: f"original_{args.perturbation_type}",
                            1: f"perturbed_{args.perturbation_type}",
                            2: f"SRC-{args.perturbation_type}_perturbed"})
        # Concatenate to the original dataframe
        src_tgt_perturbed = pd.concat([src_tgt_perturbed, perturbed_df], axis='columns')

        # Translate the perturbed sentences
        LOGGER.info("Translating perturbed SRC sentences.")
        if args.dataname == "IWSLT'15-en2vi":
            src_tgt_perturbed[f"SRC-{args.perturbation_type}_perturbed-Trans"] = \
                src_tgt_perturbed[f"SRC-{args.perturbation_type}_perturbed"].apply(
                    lambda x: translate(src2tgt_model, tokenizer, x, args.beam, device))
        else:
            if args.batch_size > 1:
                src_tgt_perturbed[f"SRC-{args.perturbation_type}_perturbed-Trans"] = \
                    batch_translation(model=src2tgt_model,
                                      src_sentences=src_tgt_perturbed[f"SRC-{args.perturbation_type}_perturbed"].tolist(),
                                      beam=args.beam, batch_size=args.batch_size)
            else:
                src_tgt_perturbed[f"SRC-{args.perturbation_type}_perturbed-Trans"] = \
                    src_tgt_perturbed[f"SRC-{args.perturbation_type}_perturbed"].apply(
                        lambda x: src2tgt_model.translate(x, beam=args.beam))

    LOGGER.info("Saving output")
    src_tgt_perturbed.to_csv(f"{args.output_dir}/translations.csv")


def translate(model, tokenizer, src: str, beam: int, device) -> str:
    # Only compatible for huggingface transformer models
    tokenized_text = tokenizer.encode(src, return_tensors="pt").to(device)
    summary_ids = model.generate(
        tokenized_text,
        max_length=128,
        num_beams=beam,
        repetition_penalty=2.5,
        length_penalty=1.0,
        early_stopping=True
    )
    output = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return output


def batch_translation(model, src_sentences: list, batch_size: int, beam: int) -> list:
    # Only compatible for fairseq models
    translations = []
    for i in range(0, len(src_sentences), batch_size):
        if i + batch_size > len(src_sentences):
            translations = translations + model.translate(src_sentences[i:len(src_sentences)], beam=beam)
        else:
            translations = translations + model.translate(src_sentences[i:i + batch_size], beam=beam)
    return translations


def perturb_sentence(perturb_type, sentence, word2vec_model):
    """
    Perturb a sentence by replacing a word with its closest neighbor determined by word2vec
    `perturb_type`: can be 'noun', 'verb', 'adjective', 'adverb' or 'pronoun'
    """

    stop_words = set(stopwords.words('english'))

    # Word tokenizers is used to find the words
    # and punctuation in a string
    words = nltk.word_tokenize(sentence)

    # removing stop words from wordList
    words = [w for w in words if w not in stop_words]

    # Perform part of speech tagging on the sentence
    LOGGER.debug("POS tagging on the SRC sentences")
    tagged = nltk.pos_tag(words)

    if perturb_type == 'noun':
        tag_prefix = 'NN'
    elif perturb_type == 'verb':
        tag_prefix = 'V'
    elif perturb_type == 'adjective':
        tag_prefix = 'JJ'
    elif perturb_type == 'adverb':
        tag_prefix = 'RB'
    elif perturb_type == 'pronoun':
        tag_prefix = 'PRP'
    else:
        raise NotImplementedError

    # Randomly select one word of the given type to perturb
    words_with_perturb_type = [x for x in tagged if x[1].startswith(tag_prefix)]
    if len(words_with_perturb_type) == 0:
        # This sentence does not have the word of that type
        return None, None, sentence

    any_word_selected = False

    LOGGER.debug("Choosing the src word that exist in the embedding vocal to perturb later on")
    for selected_word_with_tag in words_with_perturb_type:
        # Select a word to perturb, that is in the word2vec model for convenience
        selected_word = selected_word_with_tag[0]
        selected_word_tag = selected_word_with_tag[1]

        if selected_word in word2vec_model.index_to_key:
            any_word_selected = True
            break

    if not any_word_selected:
        return None, None, sentence

    # Find the word's closet neighbor using word2vec, that has the exact same tagging to avoid gramartical error
    LOGGER.debug("Finding top similar words")
    similar_words = word2vec_model.most_similar(positive=[selected_word], topn=20)
    similar_words = [x[0] for x in similar_words]  # Only keep the word and not the similarity score
    selected_replacement_word = None
    LOGGER.debug("Choosing the replacement word within similar words")
    for similar_word in similar_words:
        if nltk.pos_tag([similar_word])[0][1] == selected_word_tag:
            selected_replacement_word = similar_word
            break
    perturbed_sentence = sentence
    if selected_replacement_word is not None:
        LOGGER.debug("Replacing old word with new word")
        # Replace the selected word with the new word
        perturbed_sentence = sentence.replace(selected_word, selected_replacement_word)

    return selected_word, selected_replacement_word, perturbed_sentence


if __name__ == "__main__":
    main()
