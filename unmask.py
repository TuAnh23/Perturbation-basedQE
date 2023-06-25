import argparse
import pickle

import spacy
import torch
import pandas as pd
import gensim.downloader as api
import logging
from transformers import RobertaTokenizer, RobertaForMaskedLM, BertTokenizer, BertForMaskedLM, DistilBertTokenizer, \
    DistilBertForMaskedLM
from nltk.stem.snowball import SnowballStemmer
from torch.utils.data import Dataset, DataLoader
from utils import str_to_bool, set_seed
from tqdm import tqdm

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


def load_unmasker(pretrained_model_name, device):
    if pretrained_model_name.startswith("roberta"):
        model = RobertaForMaskedLM.from_pretrained(pretrained_model_name)
        tokenizer = RobertaTokenizer.from_pretrained(pretrained_model_name)
    elif pretrained_model_name.startswith("bert"):
        model = BertForMaskedLM.from_pretrained(pretrained_model_name)
        tokenizer = BertTokenizer.from_pretrained(pretrained_model_name)
    elif pretrained_model_name.startswith("distilbert"):
        model = BertForMaskedLM.from_pretrained(pretrained_model_name)
        tokenizer = BertTokenizer.from_pretrained(pretrained_model_name)
    else:
        raise RuntimeError(f"Unknown model {pretrained_model_name}")
    model = model.to(device)
    return tokenizer, model


def run_unmasker(unmasker_model, unmasker_tokenizer, device, top_k, sentence_list, batch_size):
    ret = []

    masked_dataset = MaskedSentencesDataset(sentence_list)
    dataloader = DataLoader(
        masked_dataset,
        batch_size=batch_size,
        shuffle=False
    )

    progress_bar = tqdm(total=len(dataloader))
    for batch in dataloader:
        inputs = unmasker_tokenizer.batch_encode_plus(
            batch,
            padding=True,
            truncation=True,
            return_tensors="pt"
        )
        inputs = inputs.to(device)

        with torch.no_grad():
            outputs = unmasker_model(**inputs)

        logits = outputs.logits  # nr_sentences_per_batch x nr_words_longest_sent x vocab_size
        mask_token_indices = (inputs.input_ids == unmasker_tokenizer.mask_token_id).nonzero(as_tuple=True)[1]  # 1 x nr_sentences_per_batch
        probs = logits.gather(
            dim=1, index=mask_token_indices.unsqueeze(1).unsqueeze(2).expand(logits.shape[0], 1, logits.shape[2])  # Select the logits of the masked word only
        ).softmax(dim=-1)  # nr_sentences_per_batch x 1 x vocab_size
        probs = probs.squeeze(dim=1)  # nr_sentences_per_batch x vocab_size
        values, predictions = probs.topk(top_k, dim=-1)  # nr_sentences_per_batch x top_k

        # decoded_strs = [unmasker_tokenizer.convert_ids_to_tokens(prediction) for prediction in predictions]
        decoded_strs = unmasker_tokenizer.convert_ids_to_tokens(predictions.flatten())
        decoded_strs = [token.replace("Ġ", "") for token in decoded_strs]
        decoded_strs = [decoded_strs[i:i + top_k] for i in range(0, len(decoded_strs), top_k)]

        for sent, v_per_sent, p_per_sent, d_per_sent in zip(batch, values, predictions, decoded_strs):
            ret_per_sent = []
            for v, p, d in zip(v_per_sent, p_per_sent, d_per_sent):
                result = {}
                result["score"] = float(v)
                result["token"] = int(p)
                result["token_str"] = d
                result["sequence"] = sent.replace("<mask>", d)
                ret_per_sent.append(result)
            ret.append(ret_per_sent)
        # Update the progress bar
        progress_bar.update(1)
    progress_bar.close()
    return ret


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--masked_src_path', type=str)
    parser.add_argument('--src_lang', type=str, default="en")
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--output_dir', type=str)
    parser.add_argument('--replacement_strategy', type=str, default='word2vec_similarity',
                        help='[word2vec_similarity|masking_language_model]. The later option is context-based.')
    parser.add_argument('--number_of_replacement', type=int, default=5,
                        help='The number of replacement for 1 SRC word')
    parser.add_argument('--grouped_mask', type=str_to_bool, default=False,
                        help='Whether the data is masked in the groupped setting, i.e., single word over multiple '
                             'sentences, and the replacement rank in across sentences')
    parser.add_argument('--unmasking_model', type=str, default='bert-base-cased')
    parser.add_argument('--batch_size', type=int, default=300)
    args = parser.parse_args()
    print(args)

    set_seed(args.seed)

    masked_src_df = pd.read_csv(args.masked_src_path, index_col=0)

    LOGGER.debug("Loading word embedding / language masking model")
    unmasker_tokenizer = None
    device = None
    if args.replacement_strategy == 'word2vec_similarity':
        word_replacement_model = api.load('glove-wiki-gigaword-100')
    elif args.replacement_strategy.startswith('masking_language_model'):
        if torch.cuda.is_available():
            device = torch.device("cuda")
            print('There are %d GPU(s) available.' % torch.cuda.device_count())
            print('We will use the GPU:', torch.cuda.get_device_name(0))
        else:
            print('No GPU available, using the CPU instead.')
            device = torch.device("cpu")
        unmasker_tokenizer, word_replacement_model = load_unmasker(args.unmasking_model, device)

    else:
        raise RuntimeError(f"Replacement strategy {args.replacement_strategy} not available.")

    if args.grouped_mask:
        unmasked_df = replace_grouped_mask(masked_src_df,
                                           args.replacement_strategy,
                                           args.number_of_replacement,
                                           word_replacement_model)
    else:
        unmasked_df = replace_mask(masked_src_df,
                                   args.replacement_strategy,
                                   args.number_of_replacement,
                                   word_replacement_model,
                                   unmasker_tokenizer,
                                   device,
                                   args.batch_size)

    unmasked_df.to_csv(f"{args.output_dir}/unmasked_df.csv")


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
    MASK_TOKEN = '[MASK]'
    if replacement_strategy.startswith('masking_language_model'):
        # Different unmasking model could have different mask token ('[MASK]' or '<mask>')
        # Have to adapt the data
        MASK_TOKEN = replacement_model.tokenizer.mask_token
        masked_src_df['SRC_masked'] = masked_src_df['SRC_masked'].apply(lambda x: x.replace('[MASK]', MASK_TOKEN))

    if not replacement_strategy.startswith('masking_language_model'):
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
        sentence_dict[row['SRC_masked_index']] = replacement_confidence_dict

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
            unmasked_row['Replacement_confidence'] = replacement_df_score.loc[
                replacement, sentence_row['SRC_masked_index']]
            unmasked_row['Replacement_rank_within_sentence'] = replacement_df_rank.loc[
                replacement, sentence_row['SRC_masked_index']]
            unmasked_row['perturbed_word'] = replacement
            unmasked_row['SRC_perturbed'] = sentence_row['SRC_masked'].replace(MASK_TOKEN, replacement)
            output_df = pd.concat([output_df, unmasked_row.to_frame().T])

    return output_df


def replace_grouped_mask(masked_src_df, replacement_strategy, number_of_replacement,
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


def lemmatize(spacy_model, token):
    doc = spacy_model(token)
    return [token.lemma_ for token in doc][0]


def replace_mask(masked_src_df, replacement_strategy, number_of_replacement, unmasker_model, unmasker_tokenizer=None,
                 device=None, batch_size=300):
    """
        Perturb sentences (put in a dataframe) by replacing a word. Here the provided sentences are already masked,
        we only need to provide the replacement

        :param masked_src_df: the dataframe containing the source sentences whose a single word is masked
                                (i.e., containing one [MASK] token)
        :param replacement_strategy: 'masking_language_model' ('word2vec_similarity' can be used in theory but not yet
        implemented)
        :param number_of_replacement: number of words to replace the 1 SRC word
        :param unmasker_model: word2vec model if replacement_strategy=='word2vec_similarity', a language model if
        replacement_strategy=='masking_language_model'
        :param unmasker_tokenizer: only required if replacement_model is a language model such as BERT
        :param device: device to perform unmasking. Only relevant if replacement_model is a language model such as BERT
        :param batch_size: Only relevant if replacement_model is a language model such as BERT
        :return: (an) unmasked sentence(s)
    """
    MASK_TOKEN = '[MASK]'
    stemmer = SnowballStemmer("english")
    # spacy_model = spacy.load("en_core_web_sm")

    if replacement_strategy.startswith('masking_language_model'):
        assert unmasker_tokenizer is not None
        # Different unmasking model could have different mask token ('[MASK]' or '<mask>')
        # Have to adapt the data
        MASK_TOKEN = unmasker_tokenizer.mask_token
        masked_src_df['SRC_masked'] = masked_src_df['SRC_masked'].apply(lambda x: x.replace('[MASK]', MASK_TOKEN))
    else:
        raise RuntimeError(f"Replacement strategy {replacement_strategy} not available")

    # Unmask all sentences, save raw unmasking value from the bert model
    print("Run LM unmasking")
    masked_src_df['raw_unmasks_bert'] = run_unmasker(unmasker_model, unmasker_tokenizer, device,
                                                     top_k=number_of_replacement + 10,
                                                     sentence_list=masked_src_df['SRC_masked'].tolist(),
                                                     batch_size=batch_size
                                                     )

    # Store values for new columns in the unmasking df
    replacement_ranks = []
    perturbed_words = []
    SRC_perturbeds = []

    print("Selecting the replacements")
    progress_bar = tqdm(total=masked_src_df.shape[0])
    for index, row in masked_src_df.iterrows():
        unmasked_row = row.copy()
        masked_word = unmasked_row['original_word']

        if replacement_strategy.startswith('masking_language_model'):
            pred = row['raw_unmasks_bert']

            # Select the replacements that is not the same as the original word
            # (ignoring case and word form)
            replacements_indices = [
                i for i in range(len(pred))
                if stemmer.stem(str(pred[i]['token_str']).lower()) != stemmer.stem(str(masked_word).lower())
            ][:number_of_replacement]

            if len(replacements_indices) < number_of_replacement:
                # If not enough distinct replacements, just choose the top replacements
                replacements_indices = list(range(number_of_replacement))

            for i, replacements_index in enumerate(replacements_indices):
                candidate_replacement = pred[replacements_index]
                replacement_ranks.append(i)
                perturbed_words.append(candidate_replacement['token_str'])
                SRC_perturbeds.append(
                    candidate_replacement['sequence'].replace('[CLS] ', '').replace(' [SEP]', '')
                )

        else:
            raise RuntimeError(f"Replacement strategy {replacement_strategy} not available.")

        # Update the progress bar
        progress_bar.update(1)
    progress_bar.close()

    masked_src_df = masked_src_df.drop('raw_unmasks_bert', axis=1)
    output_df = masked_src_df.loc[masked_src_df.index.repeat(number_of_replacement)].reset_index(drop=True)
    output_df['Replacement rank'] = replacement_ranks
    output_df['perturbed_word'] = perturbed_words
    output_df['SRC_perturbed'] = SRC_perturbeds

    return output_df


if __name__ == "__main__":
    main()
