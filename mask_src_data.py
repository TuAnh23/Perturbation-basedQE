import argparse
import string
import pandas as pd
import nltk
import random
import logging
import numpy as np
from sacremoses import MosesTokenizer, MosesDetokenizer
from utils import set_seed, none_or_str
import copy
from sklearn.feature_extraction.text import CountVectorizer


logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)

LOGGER = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--original_src_path', type=str)
    parser.add_argument('--src_lang', type=str, default="en")
    parser.add_argument('--mask_type', type=str)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--output_dir', type=str)
    parser.add_argument('--masked_vocab_path', type=none_or_str, default=None, help="Will be converted to lowercase")
    parser.add_argument('--n_sentence_when_grouped', type=int, default=50)

    args = parser.parse_args()
    print(args)

    set_seed(args.seed)

    src_df = pd.read_csv(args.original_src_path, index_col=0)

    masked_vocab = None
    if args.masked_vocab_path is not None:
        with open(args.masked_vocab_path) as file:
            masked_vocab = [line.rstrip().lower() for line in file]

    mask_sentence_type, mask_word_type = args.mask_type.split('_')

    if mask_sentence_type == 'SinglePerSentence':
        masked_df = random_single_mask(src_df, mask_word_type, args.src_lang)
    elif mask_sentence_type == 'MultiplePerSentence':
        masked_df = multiple_per_sentence_mask(src_df, mask_word_type, args.src_lang, masked_vocab=masked_vocab)
    elif mask_sentence_type == 'GroupedByWord':
        masked_df = mask_groupped_by_word(src_df, mask_word_type, args.src_lang,
                                          n_sentences=args.n_sentence_when_grouped, masked_vocab=masked_vocab)
    else:
        raise RuntimeError(f'Unknown mask_type: {args.mask_type}')

    masked_df.to_csv(f"{args.output_dir}/masked_df.csv")


def random_single_mask(src_df, masked_word_type, src_lang):
    """
    Randomly select one word of the type `masked_word_type` in each SRC sentence to mask
    :param src_df: the dataframe containing the source sentences
    :param masked_word_type: can be 'noun', 'verb', 'adjective', 'adverb' or 'pronoun'
    :param src_lang: language of the src sentences
    :return: df containing the masked sentences (not that some sentence might be dropped out, since a word to masked
    cannot be found)
    """

    if masked_word_type == 'noun':
        tag_prefix = 'NN'
    elif masked_word_type == 'verb':
        tag_prefix = 'V'
    elif masked_word_type == 'adjective':
        tag_prefix = 'JJ'
    elif masked_word_type == 'adverb':
        tag_prefix = 'RB'
    elif masked_word_type == 'pronoun':
        tag_prefix = 'PRP'
    else:
        raise NotImplementedError

    tokenizer = MosesTokenizer(lang=src_lang)
    detokenizer = MosesDetokenizer(lang=src_lang)

    masked_data = pd.DataFrame(columns=['SRC_original_idx', 'SRC', 'SRC_masked', 'original_word', 'original_word_tag'])

    for src_idx, src_row in src_df.iterrows():
        tokenized_src = tokenizer.tokenize(src_row['SRC'], escape=False, aggressive_dash_splits=False)
        pos_tags = nltk.pos_tag(tokenized_src)

        word_indices_with_masked_word_type = [
            i for i in range(len(tokenized_src)) if pos_tags[i][1].startswith(tag_prefix)
        ]
        if len(word_indices_with_masked_word_type) == 0:
            # This sentence does not have the word of that type
            continue

        # Randomly select one word of the given type to perturb
        selected_word_index = random.choice(word_indices_with_masked_word_type)

        # Mask the selected word in the original sentence
        masked_tokenized = copy.deepcopy(tokenized_src)
        masked_tokenized[selected_word_index] = '[MASK]'
        masked_sentence = detokenizer.detokenize(masked_tokenized)

        # Collect the info of this sentence and concatenate to the final df
        single_sentence_series = pd.Series()
        single_sentence_series['SRC_masked'] = masked_sentence
        single_sentence_series['SRC'] = src_row['SRC']
        single_sentence_series['SRC_original_idx'] = src_idx
        single_sentence_series['original_word'] = tokenized_src[selected_word_index]
        single_sentence_series['original_word_tag'] = pos_tags[selected_word_index][1]

        masked_data = pd.concat([masked_data, single_sentence_series.to_frame().T], ignore_index=True)

    masked_data['SRC_masked_index'] = masked_data.index

    return masked_data


def is_valid_word(word):
    # Filter out the words that has all punctuations in it

    all_puncts = string.punctuation + '—'
    contains_all_puncts = True
    for char in word:
        if char not in all_puncts:
            contains_all_puncts = False
            break
    if contains_all_puncts:
        return False

    return True


def is_content_tag(nltk_pos):
    content_tags_prefix = ['NN', 'V', 'JJ', 'RB', 'PRP']  # Noun, verb, adj, adv, pronoun
    for prefix in content_tags_prefix:
        if nltk_pos.startswith(prefix):
            return True
    return False


def is_valid_mask(word, pos_tag, masked_word_type, masked_vocab):
    """
    Check if a word is valid given the masked_word_type
    Args:
        word: the word to check
        pos_tag: POS tag of the word
        masked_word_type: can be `content`, `allTokens`, `allWords`, 'occupation', 'country'
        masked_vocab: the list of words to be masked
    Returns:
        bool
    """
    if masked_word_type == 'content':
        is_valid_token = is_valid_word(word) and is_content_tag(pos_tag)
    elif masked_word_type == 'allWords':
        is_valid_token = is_valid_word(word)
    elif masked_word_type == 'allTokens':
        is_valid_token = True
    elif masked_word_type in ['occupation', 'country']:
        is_valid_token = word.lower() in masked_vocab
    else:
        raise RuntimeError(f'masked_word_type {masked_word_type} not available.')

    return is_valid_token


def multiple_per_sentence_mask(src_df, masked_word_type, src_lang, masked_vocab=None):
    """
    Mask all the `masked_word_type` words in the sentences one by one
    :param src_df: the dataframe containing the source sentences
    :param masked_word_type: can be `content`, `allTokens`, `allWords`, 'occupation', 'country'
    :param src_lang: language of the src sentences
    :param masked_vocab: the list of words to be masked
    :return: df containing the masked sentences
    """

    if masked_word_type in ['occupation', 'country']:
        # Have to give the list of occupations or country words
        assert masked_vocab is not None

    tokenizer = MosesTokenizer(lang=src_lang)
    detokenizer = MosesDetokenizer(lang=src_lang)

    masked_data = pd.DataFrame(columns=['SRC_original_idx', 'SRC', 'SRC_masked', 'original_word', 'original_word_tag'])

    for src_idx, src_row in src_df.iterrows():
        tokenized_src = tokenizer.tokenize(src_row['SRC'], escape=False, aggressive_dash_splits=False)
        pos_tags = nltk.pos_tag(tokenized_src)
        original_words = []
        original_words_tags = []
        masked_sentences = []
        for i, word_tag in enumerate(pos_tags):
            word, pos_tag = word_tag

            is_valid_token = is_valid_mask(word, pos_tag, masked_word_type, masked_vocab)

            if is_valid_token:
                masked_tokenized = copy.deepcopy(tokenized_src)
                masked_tokenized[i] = '[MASK]'
                masked_sentences.append(
                    detokenizer.detokenize(masked_tokenized)
                )
                original_words.append(word)
                original_words_tags.append(pos_tag)
        single_sentence_df = pd.DataFrame()
        single_sentence_df['SRC_masked'] = masked_sentences
        single_sentence_df['SRC'] = src_row['SRC']
        single_sentence_df['SRC_original_idx'] = src_idx
        single_sentence_df['original_word'] = original_words
        single_sentence_df['original_word_tag'] = original_words_tags

        masked_data = pd.concat([masked_data, single_sentence_df], axis=0, ignore_index=True)

    masked_data['SRC_masked_index'] = masked_data.index

    return masked_data


def nltk_pos_tag_single_word(word):
    return nltk.pos_tag([word])[0][1]


def mask_sentence_given_word(sentence, tokenizer, masked_word):
    """
    Mask a word in a sentence (where the word index is not provided)
        sentence: the original sentence without preprocessing
        masked_word: the word to be masked (in lowercase)
    """
    # Find the location of the word in the sentence
    lowercased_sentence = sentence.lower()
    tokenized_sentence = tokenizer.tokenize(
        lowercased_sentence, escape=False, aggressive_dash_splits=False
    )
    masked_word_index = None
    prev_index = 0
    for word in tokenized_sentence:
        word_location = lowercased_sentence.find(word, prev_index)
        if word == masked_word:
            masked_word_index = word_location
        else:
            prev_index = word_location + len(word)

    assert masked_word_index is not None

    return sentence[:masked_word_index] + '[MASK]' + sentence[masked_word_index + len(masked_word):]


def mask_groupped_by_word(src_df, masked_word_type, src_lang, n_sentences, masked_vocab=None):
    """
    Select a list of words to mask. For each word, select `n_sentences` sentences containing that word to mask it
    Args:
        :param src_df: the dataframe containing the source sentences
        :param masked_word_type: can be `content`, `allTokens`, `allWords`, 'occupation', 'country'
        :param src_lang: language of the src sentences
        :param masked_vocab: the list of words to be masked
        :param n_sentences: number of sentences to mask for each word
    Returns:
        df containing the masked sentences
    """

    if masked_word_type in ['occupation', 'country']:
        # Have to give the list of occupations or country words
        assert masked_vocab is not None

    # Count words in the dataset
    corpus = src_df['SRC'].values
    tokenizer = MosesTokenizer(lang=src_lang)
    detokenizer = MosesDetokenizer(lang=src_lang)
    vectorizer = CountVectorizer(
        tokenizer=lambda x: tokenizer.tokenize(
            x,
            escape=False,
            aggressive_dash_splits=False
        ),
        lowercase=True,
    )
    count_fit = vectorizer.fit_transform(corpus)

    # Only consider the single occurance of a word in a sentence
    count_fit[count_fit > 1] = 0

    word_df = pd.DataFrame()
    word_df['word'] = vectorizer.get_feature_names_out()
    word_df['freq'] = np.asarray(count_fit.sum(axis=0)).flatten()
    word_df['pos_tag'] = word_df['word'].apply(nltk_pos_tag_single_word)

    # Filter out the words that are `masked_word_type`, and are frequent enough
    valid_word_bool = (word_df['freq'] > n_sentences) & \
                      word_df.apply(
                          lambda x: is_valid_mask(x['word'], x['pos_tag'], masked_word_type, masked_vocab),
                          axis=1
                      )



    masked_data = pd.DataFrame(columns=['SRC_original_idx', 'SRC', 'SRC_masked', 'original_word', 'original_word_tag'])

    filtered_word_df = word_df[valid_word_bool]

    for word_index, filtered_word_row in filtered_word_df.iterrows():
        # Indices of the sentences that contains the word
        sentence_indices = src_df.index[count_fit.transpose()[word_index].nonzero()[1]]

        # Randomly select a fixed number of sentences
        sentence_indices = np.random.choice(a=sentence_indices,
                                            size=n_sentences,
                                            replace=False)

        # Create a temporary df to store the sentences for this word
        tmp_df = pd.DataFrame()
        tmp_df['SRC_original_idx'] = sentence_indices
        tmp_df['SRC'] = src_df.loc[sentence_indices, 'SRC'].values
        tmp_df['original_word'] = filtered_word_row['word']
        tmp_df['original_word_tag'] = filtered_word_row['pos_tag']

        # Mask the word in those sentences
        tmp_df['SRC_masked'] = \
            src_df.loc[sentence_indices, 'SRC'].apply(
                lambda x: mask_sentence_given_word(
                    sentence=x, masked_word=filtered_word_row['word'], tokenizer=tokenizer,
                )
            ).values

        # Concat to the whole df
        masked_data = pd.concat([masked_data, tmp_df], axis=0, ignore_index=True)

    return masked_data


if __name__ == "__main__":
    main()
