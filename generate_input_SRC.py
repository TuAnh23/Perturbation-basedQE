import argparse
import torch
import pandas as pd
import nltk
from nltk.corpus import stopwords
import gensim.downloader as api
import random
import logging
from transformers import pipeline
from html import unescape
import numpy as np
from nltk.stem.snowball import SnowballStemmer
from torch.utils.data import Dataset
from sacremoses import MosesTokenizer

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)

LOGGER = logging.getLogger(__name__)


# nltk.download()

class MaskedSentencesDataset(Dataset):
    def __init__(self, samples):
        """
        :param samples: list of masked sentences
        """
        self.samples = samples

    def __getitem__(self, index):
        return self.samples[index]

    def __len__(self):
        return len(self.samples)


def set_seed(seed=0):
    """Set the random seed for torch.
    Args:
        seed (int, optional): random seed. Default 0
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    # If CUDA is not available, this is silently ignored.
    torch.cuda.manual_seed_all(seed)


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

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root_dir', type=str, default="data")
    parser.add_argument('--src_lang', type=str, default="en")
    parser.add_argument('--dataname', type=str, default="MuST-SHE-en2fr",
                        help="[MuST-SHE-en2fr|Europarl-en2de|IWSLT15-en2vi|wmt19-newstest2019-en2de|"
                             "masked_covost2_for_en2de]")
    parser.add_argument('--perturbation_type', type=none_or_str, default=None)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--output_dir', type=str)
    parser.add_argument('--dev', type=str_to_bool, default=False,
                        help="Run on a tiny amount of data for developing")
    parser.add_argument('--ignore_case', type=str_to_bool, default=False)
    parser.add_argument('--replacement_strategy', type=str, default='word2vec_similarity',
                        help='[word2vec_similarity|masking_language_model]. The later option is context-based.')
    parser.add_argument('--premasked_groupped_by_word', type=str_to_bool, default=False,
                        help='Whether the data is already masked, and the masking strategy is same words across'
                             'multiple sentences.')
    parser.add_argument('--number_of_replacement', type=int, default=1,
                        help='The number of replacement for 1 SRC word')
    args = parser.parse_args()
    print(args)

    set_seed(args.seed)

    if args.dataname == "MuST-SHE-en2fr":
        # Load test translation data
        src_df = pd.read_csv(f"{args.data_root_dir}/MuST-SHE_v1.2/MuST-SHE-v1.2-data/tsv/MONOLINGUAL.fr_v1.2.tsv",
                             sep='\t', index_col=0)[['SRC', "REF"]]

    elif args.dataname == "Europarl-en2de":
        # Use test data from Europarl at https://www.statmt.org/europarl/archives.html
        # Load test translation data
        with open(f"{args.data_root_dir}/common-test/ep-test.en", encoding="ISO-8859-1") as f:
            en_sentences = f.readlines()
            en_sentences = [line.rstrip() for line in en_sentences]
        with open(f"{args.data_root_dir}/common-test/ep-test.de", encoding="ISO-8859-1") as f:
            de_sentences = f.readlines()
            de_sentences = [line.rstrip() for line in de_sentences]
        src_df = pd.DataFrame(data={'SRC': en_sentences, 'REF': de_sentences})

    elif args.dataname == "IWSLT15-en2vi":
        with open(f"{args.data_root_dir}/IWSLT15-en2vi/tst2013.en") as f:
            en_sentences = f.readlines()
            en_sentences = [line.rstrip() for line in en_sentences]
            en_sentences = [unescape(line) for line in en_sentences]
        with open(f"{args.data_root_dir}/IWSLT15-en2vi/tst2013.vi") as f:
            vi_sentences = f.readlines()
            vi_sentences = [line.rstrip() for line in vi_sentences]
            vi_sentences = [unescape(line) for line in vi_sentences]
        src_df = pd.DataFrame(data={'SRC': en_sentences, 'REF': vi_sentences})

    elif args.dataname in ["manual_en2vi_bias", "IWSLT15_en2vi_bias", "IWSLT15_en2vi_bias_train"]:
        src_df = pd.read_csv(f"{args.data_root_dir}/en2vi_bias/{args.dataname}.csv", index_col=0)

    elif args.dataname in ['covost2_countries_replacement_he', 'covost2_countries_replacement_she']:
        src_df = pd.read_csv(f"{args.data_root_dir}/covost2/{args.dataname}.csv", index_col=0)

    elif args.dataname == 'covost2_countries_replacement_you':
        # Test data at https://www.statmt.org/wmt19/metrics-task.html
        # Load translation data
        src_df = pd.read_csv(f"{args.data_root_dir}/covost2/{args.dataname}.csv", index_col=0)

    elif args.dataname == 'winoMT_src':
        # Load translation data
        src_df = pd.read_csv(f"{args.data_root_dir}/winoMT_src.csv", index_col=0)

    elif args.dataname.startswith('masked'):
        # Load translation data
        src_df = pd.read_csv(f"{args.data_root_dir}/{args.dataname}.csv", index_col=0)

    elif args.dataname == 'wmt19-newstest2019-en2de':
        # Test data at https://www.statmt.org/wmt19/metrics-task.html
        # Load test translation data
        with open(f"{args.data_root_dir}/wmt19-submitted-data-v3/txt/sources/newstest2019-ende-src.en") as f:
            en_sentences = f.readlines()
            en_sentences = [line.rstrip() for line in en_sentences]
        with open(f"{args.data_root_dir}/wmt19-submitted-data-v3/txt/references/newstest2019-ende-ref.de") as f:
            de_sentences = f.readlines()
            de_sentences = [line.rstrip() for line in de_sentences]
        src_df = pd.DataFrame(data={'SRC': en_sentences, 'REF': de_sentences})

    else:
        raise RuntimeError(f"Dataset {args.dataname} not available.")

    if "SRC_index" not in src_df.columns:
        src_df.insert(0, column="SRC_index", value=src_df.index.values)

    if args.ignore_case:
        src_df = src_df.applymap(lambda x: x.lower())

    if args.dev:
        src_perturbed = src_df[:10]  # Select a small number of rows to dev
    else:
        src_perturbed = src_df

    if args.perturbation_type is None:
        pass
    elif args.dataname.startswith('masked'):
        # Generate the unmasked replacement sentences
        LOGGER.info("Unmasking sentences")
        if args.replacement_strategy == 'masking_language_model':
            if torch.cuda.is_available():
                gpu = 0
            else:
                print('No GPU available, using  CPU instead.')
                gpu = -1

            if args.ignore_case:
                word_replacement_model = pipeline('fill-mask', model='bert-base-uncased', device=gpu)
            else:
                word_replacement_model = pipeline('fill-mask', model='bert-base-cased', device=gpu)
        else:
            raise RuntimeError(f"Replacement strategy {args.replacement_strategy} not available.")

        if args.premasked_groupped_by_word:
            src_perturbed = replace_mask_groupped_by_perturbed_word(
                masked_src_df=src_perturbed,
                replacement_strategy=args.replacement_strategy,
                number_of_replacement=args.number_of_replacement,
                replacement_model=word_replacement_model
            )
        else:
            src_perturbed = replace_mask(masked_src_df=src_perturbed,
                                         replacement_strategy=args.replacement_strategy,
                                         number_of_replacement=args.number_of_replacement,
                                         replacement_model=word_replacement_model)

    else:
        # Generate the perturbed sentences
        LOGGER.info("Perturbing sentences")
        LOGGER.debug("Loading word embedding / language masking model")
        if args.replacement_strategy == 'word2vec_similarity':
            word_replacement_model = api.load('glove-wiki-gigaword-100')
        elif args.replacement_strategy == 'masking_language_model':
            if torch.cuda.is_available():
                gpu = 0
            else:
                print('No GPU available, using  CPU instead.')
                gpu = -1

            if args.ignore_case:
                word_replacement_model = pipeline('fill-mask', model='bert-base-uncased', device=gpu)
            else:
                word_replacement_model = pipeline('fill-mask', model='bert-base-cased', device=gpu)
        else:
            raise RuntimeError(f"Replacement strategy {args.replacement_strategy} not available.")

        src_perturbed = perturb_sentences(src_perturbed, perturb_type=args.perturbation_type,
                                          model=word_replacement_model,
                                          replacement_strategy=args.replacement_strategy,
                                          number_of_replacement=args.number_of_replacement,
                                          src_lang=args.src_lang)

    LOGGER.info("Saving input SRC")
    # Saving the perturbed source sentences
    src_perturbed.to_csv(f"{args.output_dir}/input.csv")


def perturb_sentences(src_df, perturb_type, replacement_strategy, number_of_replacement, model, src_lang):
    """
    Perturb sentences (put in a dataframe) by replacing a word
    :param src_df: the dataframe containing the source sentences
    :param perturb_type: can be 'noun', 'verb', 'adjective', 'adverb' or 'pronoun'
    :param replacement_strategy: 'word2vec_similarity' or 'masking_language_model'
    :param number_of_replacement: number of words to replace the 1 SRC word
    :param model: word2vec model if replacement_strategy=='word2vec_similarity', a language model if
    replacement_strategy=='masking_language_model'
    :return: the perturbed sentence
    """

    stop_words = set(stopwords.words('english'))

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

    perturbed_df = pd.DataFrame(
        columns=list(src_df.columns) + [f"original_{perturb_type}", 'Replacement rank', f"perturbed_{perturb_type}",
                                        f"SRC-{perturb_type}_perturbed"])

    for index, row in src_df.iterrows():
        perturbed_row = row.copy()
        sentence = perturbed_row['SRC']

        tokenizer = MosesTokenizer(lang=src_lang)
        words = tokenizer.tokenize(sentence, escape=False, aggressive_dash_splits=False)

        # removing stop words from wordList
        words = [w for w in words if w not in stop_words]

        # Perform part of speech tagging on the sentence
        LOGGER.debug("POS tagging on the SRC sentences")
        tagged = nltk.pos_tag(words)

        words_with_perturb_type = [x for x in tagged if x[1].startswith(tag_prefix)]
        if len(words_with_perturb_type) == 0:
            # This sentence does not have the word of that type
            continue

        if replacement_strategy == 'word2vec_similarity':
            LOGGER.debug("Choosing the src word that exist in the embedding vocal to perturb later on")
            any_word_selected = False
            for selected_word_with_tag in words_with_perturb_type:
                # Select a word to perturb, that is in the word2vec model for convenience
                selected_word = selected_word_with_tag[0]
                selected_word_tag = selected_word_with_tag[1]

                if selected_word in model.index_to_key:
                    any_word_selected = True
                    break

            if not any_word_selected:
                continue

            # Find the word's closet neighbor using word2vec
            LOGGER.debug("Finding top similar words")
            similar_words = model.most_similar(positive=[selected_word], topn=20)
            similar_words = [x[0] for x in similar_words]  # Only keep the word and not the similarity score
            selected_replacement_words = []
            LOGGER.debug("Choosing the replacement word within similar words")
            for similar_word in similar_words:
                # New word with exact same tagging to avoid gramartical error
                # and not the same as the original word
                if nltk.pos_tag([similar_word])[0][1] == selected_word_tag \
                        and similar_word.lower() != selected_word.lower():
                    selected_replacement_words.append(similar_word)

            if len(selected_replacement_words) > 0:
                # Replace the selected word with the new words
                for replacement_i in range(0, min(number_of_replacement, len(selected_replacement_words))):
                    perturbed_sentence = sentence.replace(selected_word, selected_replacement_words[i])
                    perturbed_row[f"original_{perturb_type}"] = selected_word
                    perturbed_row['Replacement rank'] = i + 1
                    perturbed_row[f"perturbed_{perturb_type}"] = selected_replacement_words[i]
                    perturbed_row[f"SRC-{perturb_type}_perturbed"] = perturbed_sentence
                    perturbed_df = pd.concat([perturbed_df, perturbed_row.to_frame().T])
            else:
                continue

        elif replacement_strategy == 'masking_language_model':
            # Randomly select one word of the given type to perturb
            selected_word = random.choice(words_with_perturb_type)[0]
            # Mask the selected word in the original sentence
            masked_sentence = sentence.replace(selected_word, '[MASK]', 1)
            # Run model unmasking prediction
            pred = model(masked_sentence, top_k=40)
            if type(pred[0]) is list:
                # Unravel the list
                pred = pred[0]
            # Choose the most probable word, if it is not the same as the original word (ignoring case and word form)
            stemmer = SnowballStemmer("english")

            count_replacement = 0
            replacement_i = 0

            while count_replacement < number_of_replacement:
                candidate_replacement = pred[replacement_i]
                if stemmer.stem(candidate_replacement['token_str'].lower()) != stemmer.stem(selected_word.lower()):
                    count_replacement = count_replacement + 1
                    perturbed_row[f"original_{perturb_type}"] = selected_word
                    perturbed_row['Replacement rank'] = count_replacement
                    perturbed_row[f"perturbed_{perturb_type}"] = candidate_replacement['token_str']
                    perturbed_row[f"SRC-{perturb_type}_perturbed"] = candidate_replacement['sequence'].replace(
                        '[CLS] ', '').replace(' [SEP]', '')
                    perturbed_df = pd.concat([perturbed_df, perturbed_row.to_frame().T])
                replacement_i = replacement_i + 1
        else:
            raise RuntimeError(f"Replacement strategy {args.replacement_strategy} not available.")

    return perturbed_df


def replace_mask_single_group(masked_src_df, replacement_strategy, number_of_replacement, replacement_model):
    """
    Unmask a group of sentences that are masked with the same word. We chose the top-5 replacements across the whole
    group of sentences
    :param masked_src_df: df of sentences that are masked at the same word
    :param replacement_strategy: 'masking_language_model' ('word2vec_similarity' can be used in theory but not yet
    implemented)
    :param number_of_replacement: number of words to replace the 1 SRC word
    :param replacement_model: word2vec model if replacement_strategy=='word2vec_similarity', a language model if
    replacement_strategy=='masking_language_model'
    :return: unmasked sentences
    """

    if replacement_strategy != 'masking_language_model':
        raise RuntimeError(f"Replacement strategy {replacement_strategy} not available.")

    count_original_word = masked_src_df['original_word'].value_counts()
    assert count_original_word.shape[0] == 1  # Because this function is for a single group
    original_word = count_original_word.index[0]

    # For each sentence, get the unmasking candidates and the corresponding confidence score
    sentence_dict = {}
    for index, row in masked_src_df.iterrows():
        # Run model unmasking prediction
        pred = replacement_model(row['SRC_masked'], top_k=30)
        if type(pred[0]) is list:
            # Unravel the list
            pred = pred[0]

        # Put replacement with score and rank in a dict
        replacement_confidence_dict = {}
        for rank, replacement in enumerate(pred):
            replacement_confidence_dict[replacement['token_str']] = {'score': replacement['score'],
                                                                     'rank_within_sentence': rank}

        # Store the dict for the corresponding sentence
        sentence_dict[row['SRC_index']] = replacement_confidence_dict

    # Create a dataframe that stores all the replacements and their confidence scores for each sentence
    # If a replacement is not output for a sentence, then it's confidence score is 0

    # Collect all possible replacement across sentences
    all_replacements = set(
        sum(
            [list(replacement_confidence_dict.keys())
             for replacement_confidence_dict in sentence_dict.values()],
            []
        )
    )

    # Filter out the replacements that are the same as the original word (ignoring case and word form)
    # This will be used as the index for the replacement dataframe
    index = []
    stemmer = SnowballStemmer("english")
    for replacement in all_replacements:
        if stemmer.stem(str(replacement).lower()) != stemmer.stem(str(original_word).lower()):
            index.append(replacement)

    # Put to the dataframe
    replacement_df_score = pd.DataFrame(data=0, index=index, columns=sentence_dict.keys())
    replacement_df_rank = pd.DataFrame(data='>30', index=index, columns=sentence_dict.keys())
    for sentence, replacement_confidence_dict in sentence_dict.items():
        for replacement, score_rank in replacement_confidence_dict.items():
            if replacement in index:
                replacement_df_score.loc[replacement, sentence] = score_rank['score']
                replacement_df_rank.loc[replacement, sentence] = score_rank['rank_within_sentence']

    # Find the top_k best replacements across all sentences
    top_k = replacement_df_score.mean(axis=1).sort_values(ascending=False).index[:number_of_replacement].to_list()

    # Create a dataframe of unmasked sentences
    output_df = pd.DataFrame(
        columns=list(masked_src_df.columns) + ['Replacement_confidence', "perturbed_word", "SRC_perturbed"]
    )

    for sentence_index, sentence_row in masked_src_df.iterrows():
        unmasked_row = sentence_row.copy()
        for replacement in top_k:
            unmasked_row['Replacement_confidence'] = replacement_df_score.loc[replacement, sentence_row['SRC_index']]
            unmasked_row['Replacement_rank_within_sentence'] = replacement_df_rank.loc[
                replacement, sentence_row['SRC_index']]
            unmasked_row['perturbed_word'] = replacement
            unmasked_row['SRC_perturbed'] = sentence_row['SRC_masked'].replace('[MASK]', replacement)
            output_df = pd.concat([output_df, unmasked_row.to_frame().T])

    return output_df


def replace_mask_groupped_by_perturbed_word(masked_src_df, replacement_strategy, number_of_replacement,
                                            replacement_model):
    """
        Perturb sentences (put in a dataframe) by replacing a word. Here the provided sentences are already masked,
        we only need to provide the replacement. We chose the top-5 replacements across the whole group of sentences

        :param masked_src_df: the dataframe containing the source sentences whose a single word is masked
                                (i.e., containing one [MASK] token)
        :param replacement_strategy: 'masking_language_model' ('word2vec_similarity' can be used in theory but not yet
        implemented)
        :param number_of_replacement: number of words to replace the 1 SRC word
        :param replacement_model: word2vec model if replacement_strategy=='word2vec_similarity', a language model if
        replacement_strategy=='masking_language_model'
        :return: unmasked sentences
    """
    output_df = pd.DataFrame()

    groupped = masked_src_df.groupby('original_word')
    for original_word, group in groupped:
        tmp_df = replace_mask_single_group(group, replacement_strategy, number_of_replacement, replacement_model)
        output_df = pd.concat([output_df, tmp_df])

    return output_df


def replace_mask(masked_src_df, replacement_strategy, number_of_replacement, replacement_model):
    """
        Perturb sentences (put in a dataframe) by replacing a word. Here the provided sentences are already masked,
        we only need to provide the replacement

        :param masked_src_df: the dataframe containing the source sentences whose a single word is masked
                                (i.e., containing one [MASK] token)
        :param replacement_strategy: 'masking_language_model' ('word2vec_similarity' can be used in theory but not yet
        implemented)
        :param number_of_replacement: number of words to replace the 1 SRC word
        :param replacement_model: word2vec model if replacement_strategy=='word2vec_similarity', a language model if
        replacement_strategy=='masking_language_model'
        :return: (an) unmasked sentence(s)
    """
    # Unmask all sentences, save raw unmasking value from the bert model
    # Do not do it within the loop for better GPU efficiency
    masked_dataset = MaskedSentencesDataset(masked_src_df['SRC_masked'].values)
    masked_src_df['raw_unmasks_bert'] = replacement_model(masked_dataset, top_k=40)

    output_df = pd.DataFrame(
        columns=list(masked_src_df.columns) + ['Replacement rank', f"perturbed_word", f"SRC_perturbed"]
    )

    for index, row in masked_src_df.iterrows():
        unmasked_row = row.copy()
        masked_word = unmasked_row['original_word']

        if replacement_strategy == 'masking_language_model':
            pred = row['raw_unmasks_bert']

            # Choose the most probable word, if it is not the same as the original word (ignoring case and word form)
            stemmer = SnowballStemmer("english")

            count_replacement = 0
            replacement_i = 0

            while count_replacement < number_of_replacement:
                candidate_replacement = pred[replacement_i]
                if stemmer.stem(str(candidate_replacement['token_str']).lower()) != stemmer.stem(
                        str(masked_word).lower()):
                    count_replacement = count_replacement + 1
                    unmasked_row['Replacement rank'] = count_replacement
                    unmasked_row['perturbed_word'] = candidate_replacement['token_str']
                    unmasked_row['SRC_perturbed'] = candidate_replacement['sequence'].replace(
                        '[CLS] ', '').replace(' [SEP]', '')
                    output_df = pd.concat([output_df, unmasked_row.to_frame().T])
                replacement_i = replacement_i + 1

        else:
            raise RuntimeError(f"Replacement strategy {args.replacement_strategy} not available.")

    output_df = output_df.drop('raw_unmasks_bert', axis=1)

    return output_df


if __name__ == "__main__":
    main()
