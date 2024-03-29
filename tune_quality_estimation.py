"""
Used to perform hyperparams tuning for the proposed perturbation-based QE and log prob based QE
"""
import copy
import itertools

import pandas as pd
import numpy as np
from sklearn.metrics import matthews_corrcoef
import argparse
from utils import str_to_bool, set_seed
from align_and_analyse_ambiguous_trans import analyse_single_sentence, tercom_alignment, edist_alignment
from scipy.stats import zscore
from scipy.stats import pearsonr, spearmanr
from collections import defaultdict
from sacremoses import MosesTokenizer
from multiprocessing import Pool, cpu_count
from itertools import repeat
from utils import load_text_file
import re
from tqdm import tqdm


def load_gold_labels(dataset, data_root_path, src_lang, tgt_lang, task):
    """
    Args:
        dataset: 'WMT21_DA_test' or 'WMT21_DA_dev'
        task: 'sentence_level_eval_da', 'sentence_level_eval_hter', 'trans_word_level_eval' or 'src_word_level_eval'
    Returns:
        gold_labels: list for 'sentence_level_eval_da' task or
                    list of list for '*_word_level_eval' tasks
    """
    if dataset == 'WMT21_DA_test':
        if task == 'sentence_level_eval_da':
            with open(f"{data_root_path}/wmt-qe-2021-data/{src_lang}-{tgt_lang}-test21/goldlabels/test21.da", 'r') as f:
                da_scores = f.readlines()
                da_scores = [float(da_score.replace('\n', '')) for da_score in da_scores]
            return da_scores
        elif task == 'sentence_level_eval_hter':
            with open(f"{data_root_path}/wmt-qe-2021-data/{src_lang}-{tgt_lang}-test21/goldlabels/test21.da", 'r') as f:
                hter_scores = f.readlines()
                hter_scores = [float(da_score.replace('\n', '')) for da_score in hter_scores]
            return hter_scores
        elif task == 'trans_word_level_eval':
            gold_labels_path = f'{data_root_path}/wmt-qe-2021-data/{src_lang}-{tgt_lang}-test21/goldlabels/task2_wordlevel_mt.tags'
            gold_labels = pd.read_csv(
                gold_labels_path,
                header=None, sep='\t', quoting=3
            )
            return gold_labels.groupby(3)[6].apply(list).tolist()
        elif task == 'src_word_level_eval':
            gold_labels_path = f'{data_root_path}/wmt-qe-2021-data/{src_lang}-' \
                               f'{tgt_lang}-test21/goldlabels/task2_wordlevel_src.tags'
            gold_labels = pd.read_csv(
                gold_labels_path,
                header=None, sep='\t', quoting=3
            )
            return gold_labels.groupby(3)[6].apply(list).tolist()
    elif dataset == 'WMT21_DA_dev':
        if task == 'sentence_level_eval_da':
            gold_labels_path = f"{data_root_path}/wmt-qe-2021-data/{src_lang}-{tgt_lang}-dev/direct-assessments/{src_lang}-{tgt_lang}-dev/dev.{src_lang}{tgt_lang}.df.short.tsv"
            return pd.read_csv(gold_labels_path, sep='\t')['z_mean'].to_list()
        elif task == 'sentence_level_eval_hter':
            gold_labels_path = f"{data_root_path}/wmt-qe-2021-data/{src_lang}-{tgt_lang}-dev/post-editing/{src_lang}-{tgt_lang}-dev/dev.hter"
            with open(gold_labels_path, 'r') as f:
                scores = f.readlines()
                scores = [float(score.strip()) for score in scores]
            return scores
        elif task == 'trans_word_level_eval':
            gold_labels_path = f"{data_root_path}/wmt-qe-2021-data/{src_lang}-{tgt_lang}-dev/post-editing/{src_lang}-{tgt_lang}-dev/dev.tags"
            with open(gold_labels_path, 'r') as f:
                tags = f.readlines()
                tags = [line.replace('\n', '').split() for line in tags]
                # Since we are not considering the gap, only the tokens:
                tags = [line[1:-1] for line in tags]  # Remove the begin and end tag
                tags = [line[::2] for line in tags]  # Remove the gap tags
            return tags
        elif task == 'src_word_level_eval':
            gold_labels_path = f"{data_root_path}/wmt-qe-2021-data/{src_lang}-{tgt_lang}-dev/post-editing/{src_lang}-{tgt_lang}-dev/dev.source_tags"
            with open(gold_labels_path, 'r') as f:
                tags = f.readlines()
                tags = [line.replace('\n', '').split() for line in tags]
            return tags
    elif dataset == 'WMT22_MQM':
        with open(f"{data_root_path}/wmt-qe-2022-data/test_data-gold_labels/task1_mqm/{src_lang}-{tgt_lang}/test.2022.{src_lang}-{tgt_lang}.mqm_z_score",
                  'r') as f:
            mqm_scores = f.readlines()
            mqm_scores = [float(mqm_score.replace('\n', '')) for mqm_score in mqm_scores]
        return mqm_scores
    elif dataset.startswith("WMT20_HJQE"):
        split = dataset.split('_')[-1]
        if task == 'trans_word_level_eval':
            gold_labels_path = f"{data_root_path}/HJQE/{src_lang}-{tgt_lang}/{split}/{split}.tags"
        elif task == 'src_word_level_eval':
            gold_labels_path = f"{data_root_path}/HJQE/{src_lang}-{tgt_lang}/{split}/{split}.source_tags"
        else:
            raise RuntimeError(f"task {task} not available for dataset {dataset}")

        gold_labels = load_text_file(gold_labels_path)
        gold_labels = [x.split() for x in gold_labels]
        gold_labels = [line[1:-1] for line in gold_labels]  # Remove the begin and end tag
        gold_labels = [line[::2] for line in gold_labels]  # Remove the gap tags

        return gold_labels


def flatten_list(list_of_lists):
    """
    Args:
        list_of_lists
    Returns:
        1D list
    """
    return [item for sublist in list_of_lists for item in sublist]


def nmt_log_prob_eval(dataset, data_root_path, src_lang, tgt_lang, nmt_log_prob_threshold, perturbed_trans_df_path,
                      task, original_translation_output_dir, keep_unknown=False):
    word_log_probs = get_nmt_word_log_probs(dataset, data_root_path, src_lang, tgt_lang, original_translation_output_dir)
    threshold = np.log2(nmt_log_prob_threshold)
    if task == 'trans_word_level_eval':
        pred_labels = log_prob_to_label(threshold, word_log_probs)
    elif task == 'src_word_level_eval':
        # Have to first align src-trans
        perturbed_trans_df = pd.read_pickle(perturbed_trans_df_path)
        SRC_original_indices = perturbed_trans_df['SRC_original_idx'].unique()

        src_word_log_probs = []
        for i, SRC_original_idx in enumerate(SRC_original_indices):
            word_log_probs_sentence = word_log_probs[i]

            sentence_df = perturbed_trans_df[perturbed_trans_df['SRC_original_idx'] == SRC_original_idx]
            original_trans_alignment = sentence_df['original_trans_alignment_index'].values[0]
            original_src_tokenized = sentence_df['tokenized_SRC'].values[0]
            src_word_log_probs_sentence = [
                word_log_probs_sentence[original_trans_alignment[i]] if i in original_trans_alignment.keys() else np.nan \
                for i in range(0, len(original_src_tokenized))
            ]
            src_word_log_probs.append(src_word_log_probs_sentence)

        pred_labels = log_prob_to_label(threshold, src_word_log_probs)
    else:
        raise RuntimeError('Unknown task')

    if not keep_unknown:
        pred_labels = replace_unknown(pred_labels)
    return pred_labels


def replace_unknown(pred_labels):
    """
    If unknown --> failed word alignemnt --> the word is probably badly translated
    --> replace unknown with BAD
    Returns:
        new_pred_labels
    """
    # # Replace unknown predictions with either 'OK' or 'BAD' randomly.
    # new_pred_labels = copy.deepcopy(pred_labels)
    # for x in range(0, len(new_pred_labels)):
    #     for y in range(0, len(new_pred_labels[x])):
    #         if new_pred_labels[x][y] == 'unknown':
    #             new_pred_labels[x][y] = np.random.choice(['OK', 'BAD'])
    new_pred_labels = copy.deepcopy(pred_labels)
    for x in range(0, len(new_pred_labels)):
        for y in range(0, len(new_pred_labels[x])):
            if new_pred_labels[x][y] == 'unknown':
                new_pred_labels[x][y] = 'BAD'
    return new_pred_labels


def log_prob_to_label(threshold, word_log_probs):
    return [['unknown' if np.isnan(x) else 'BAD' if x < threshold else 'OK' for x in y]
            for y in word_log_probs]


def nr_effecting_src_words_eval(perturbed_trans_df_path, effecting_words_threshold, task,
                                consistence_trans_portion_threshold=0.8,
                                uniques_portion_for_noiseORperturbed_threshold=0.4,
                                no_effecting_words_portion_threshold=0.8,
                                keep_unknown=False, return_details=False,
                                alignment_tool='Levenshtein',
                                clean_up_return_details=False,
                                include_direct_influence=False):
    """
    *_word_level_eval by using nr_effecting_src_words
    return_details: if False, return only the word predicted tag. if True, also returns the list of SRC words that
                    effect the translated words
    clean_up_return_details: if not True, then return the raw details that can be used directly to detect BAD word
        indices {tgt_word_inx: {'no_effecting_words': [src_word_1, src_word_2], effecting_words[src_word_3, src_word_4]},
        where src_word still have the uniquify tags
        (when there are 2 same src words in a sentence).
        If True then clean up: return a dataframe with columns
        [Sentence_idx, Target_word, Target_word_idx, OK/BAD, Effecting_src_words]
        remove uniquify tags, only report the effecting words
    Returns:
        The flattened predictions
    """

    if task == 'trans_word_level_eval':
        align_type = "trans-only"
    elif task == 'src_word_level_eval':
        align_type = "src-trans"
    else:
        raise RuntimeError('Unknown task')

    word_tag = []
    details = []  # the effecting and no-effecting SRC words to every translated words
    clean_details_cols = \
        [
            'Sentence_idx', 'Target_word_idx', 'Target_word',
            'OK/BAD', 'Effecting_src_words', 'Effecting_src_words_idx', 'Effecting_src_words_influence',
            'inconsistent_versions',
            'Direct_src_word', 'Direct_src_word_idx'
        ]
    clean_details = pd.DataFrame(columns=clean_details_cols)  # the effecting SRC words to every translated words
    tmp_clean_details_dfs = [clean_details]
    perturbed_trans_df = pd.read_pickle(perturbed_trans_df_path)

    # Perform alignment here at once for efficiency
    original_trans_tokenized = perturbed_trans_df['tokenized_SRC-Trans'].tolist()
    perturbed_trans_tokenized = perturbed_trans_df['tokenized_SRC_perturbed-Trans'].tolist()
    if alignment_tool == 'Levenshtein':
        aligments = [edist_alignment(s1, s2) for s1, s2 in zip(original_trans_tokenized, perturbed_trans_tokenized)]
    elif alignment_tool == 'Tercom':
        aligments = tercom_alignment(original_trans_tokenized, perturbed_trans_tokenized)
    else:
        raise RuntimeError(f"Unknown alignment tool {alignment_tool}")
    perturbed_trans_df['trans-only-alignment'] = aligments

    SRC_original_indices = perturbed_trans_df['SRC_original_idx'].unique()

    progress_bar = tqdm(total=len(SRC_original_indices))
    for SRC_original_idx in SRC_original_indices:
        sentence_df = perturbed_trans_df[perturbed_trans_df['SRC_original_idx'] == SRC_original_idx]
        original_trans_length = len(sentence_df['tokenized_SRC-Trans'].values[0])
        original_src_length = len(sentence_df['tokenized_SRC'].values[0])
        tgt_src_effects = analyse_single_sentence(
            sentence_df,
            align_type=align_type, return_word_index=True,
            consistence_trans_portion_threshold=consistence_trans_portion_threshold,
            uniques_portion_for_noiseORperturbed_threshold=uniques_portion_for_noiseORperturbed_threshold,
            alignment_tool=alignment_tool,
            include_direct_influence=include_direct_influence
        )
        bad_words = find_bad_word(tgt_src_effects, effecting_words_threshold)
        # ok_words = find_ok_word(tgt_src_effects,
        #                         no_effecting_words_threshold=original_src_length*no_effecting_words_portion_threshold)
        sentence_word_tags = ['BAD' if x in bad_words else 'OK'
                              for x in range(0,
                                             original_trans_length if task == 'trans_word_level_eval' else original_src_length)]
        # sentence_word_tags = ['OK' if x in ok_words else 'BAD' for x in range(0,
        #                       original_trans_length if task == 'trans_word_level_eval' else original_src_length)]
        word_tag.append(sentence_word_tags)
        if return_details:
            if clean_up_return_details:
                tmp_clean_details = pd.DataFrame(columns=clean_details_cols)
                tmp_clean_details['Target_word_idx'] = tgt_src_effects.keys()
                tmp_clean_details['Target_word'] = tmp_clean_details['Target_word_idx'].apply(
                    lambda x: sentence_df['tokenized_SRC-Trans'].values[0][x]
                )
                tmp_clean_details['Effecting_src_words'] = tgt_src_effects.values()
                tmp_clean_details['Effecting_src_words_idx'] = \
                    tmp_clean_details['Effecting_src_words'].apply(
                        lambda x: x['effecting_words_idx']
                    )
                tmp_clean_details['Effecting_src_words_influence'] = \
                    tmp_clean_details['Effecting_src_words'].apply(
                        lambda x: x['effecting_words_influence']
                    )
                tmp_clean_details['Direct_src_word'] = tgt_src_effects.values()
                tmp_clean_details['Direct_src_word_idx'] = \
                    tmp_clean_details['Direct_src_word'].apply(
                        lambda x: x['direct_perturbation_words_idx']
                    )
                tmp_clean_details['inconsistent_versions'] = tgt_src_effects.values()
                tmp_clean_details['inconsistent_versions'] = \
                    tmp_clean_details['inconsistent_versions'].apply(
                        lambda x: x['inconsistent_versions']
                    )
                # Remove the uniquify tags from src_words
                tmp_clean_details['Effecting_src_words'] = \
                    tmp_clean_details['Effecting_src_words'].apply(
                        lambda x: [re.sub(r"_\d+$", "", i) for i in x['effecting_words']]
                    )
                tmp_clean_details['Direct_src_word'] = \
                    tmp_clean_details['Direct_src_word'].apply(
                        lambda x: [re.sub(r"_\d+$", "", i) for i in x['direct_perturbation_words']]
                    )
                tmp_clean_details['OK/BAD'] = sentence_word_tags
                tmp_clean_details['Sentence_idx'] = SRC_original_idx

                # Append to the final dataframe
                tmp_clean_details_dfs.append(tmp_clean_details)
            else:
                details.append(tgt_src_effects)

        # Update the progress bar
        progress_bar.update(1)

    progress_bar.close()

    if not keep_unknown:
        word_tag = replace_unknown(word_tag)

    if return_details:
        if clean_up_return_details:
            clean_details = pd.concat(tmp_clean_details_dfs, ignore_index=True)
            return word_tag, clean_details
        else:
            return word_tag, details
    else:
        return word_tag


def get_nmt_word_log_probs(dataset, data_root_path, src_lang, tgt_lang, original_translation_output_dir):
    subwords_path = f"{original_translation_output_dir}/mt.out"
    subword_log_probs_path = f"{original_translation_output_dir}/log_prob.out"

    if dataset == 'WMT21_DA_test':
        tokenized_path = f"{data_root_path}/wmt-qe-2021-data/{src_lang}-{tgt_lang}-test21/test21.tok.mt"
        with open(tokenized_path, 'r') as f:
            tokenized_translations = f.readlines()
            tokenized_translations = [tokenized_trans.strip().split() for tokenized_trans in tokenized_translations]
    elif dataset == 'WMT21_DA_dev':
        tokenized_path = f"{data_root_path}/wmt-qe-2021-data/{src_lang}-{tgt_lang}-dev/post-editing/{src_lang}-{tgt_lang}-dev/dev.mt"
        with open(tokenized_path, 'r') as f:
            tokenized_translations = f.readlines()
            tokenized_translations = [tokenized_trans.strip().split() for tokenized_trans in tokenized_translations]
    elif dataset.startswith('WMT20_HJQE'):
        split = dataset.split('_')
        tokenized_path = f"{data_root_path}/HJQE/{src_lang}-{tgt_lang}/{split}/{split}.mt"
        with open(tokenized_path, 'r') as f:
            tokenized_translations = f.readlines()
            tokenized_translations = [tokenized_trans.strip().split() for tokenized_trans in tokenized_translations]
    else:
        with open(f"{original_translation_output_dir}/trans_sentences.txt", 'r') as f:
            translations = f.readlines()
            translations = [line.rstrip() for line in translations]

        tgt_tokenizer = MosesTokenizer(lang=tgt_lang)
        tokenized_translations = [tgt_tokenizer.tokenize(x, escape=False, aggressive_dash_splits=False)
                                  for x in translations]

    subword_type = get_subword_type(subwords_path)
    subwords = load_subwords_from_file(subwords_path, subword_type)
    subword_log_probs = load_subword_log_probs_from_file(subword_log_probs_path)

    if subword_type == "double_at_separator":
        # We do not care about the end-of-sentence marker which is omitted from the text.
        subword_log_probs = [x[:-1] for x in subword_log_probs]
    elif subword_type == "underscore_separator":
        subword_lines = load_text_file(subwords_path)
        for sentence_idx in range(len(subword_lines)):
            if subword_lines[sentence_idx].startswith('▁ '):
                if len(subwords[sentence_idx]) == len(subword_log_probs[sentence_idx]) - 3:
                    subword_log_probs[sentence_idx] = subword_log_probs[sentence_idx][2:-1]
                else:
                    subword_log_probs[sentence_idx] = subword_log_probs[sentence_idx][1:]
            elif subword_lines[sentence_idx].startswith('▁'):
                if len(subwords[sentence_idx]) == len(subword_log_probs[sentence_idx]) - 2:
                    subword_log_probs[sentence_idx] = subword_log_probs[sentence_idx][1:-1]
            else:
                if len(subwords[sentence_idx]) == len(subword_log_probs[sentence_idx]) - 1:
                    subword_log_probs[sentence_idx] = subword_log_probs[sentence_idx][:-1]

    for sentence_idx in range(len(subwords)):
        if len(subwords[sentence_idx]) != len(subword_log_probs[sentence_idx]):
            print(sentence_idx)
            print(subwords[sentence_idx])
            print(subword_log_probs[sentence_idx])
            print(len(subwords[sentence_idx]), len(subword_log_probs[sentence_idx]))
        assert len(subwords[sentence_idx]) == len(subword_log_probs[sentence_idx])

    word_log_probs = merge_subwords_log_prob_to_words_log_prob(subwords, subword_log_probs, tokenized_translations)
    return word_log_probs


def load_subword_log_probs_from_file(subword_log_probs_path):
    with open(subword_log_probs_path, 'r') as f:
        subword_log_probs = f.readlines()
        subword_log_probs = [line.replace('\n', '').split() for line in subword_log_probs]
        # Cast all values to float
        subword_log_probs = [[float(x) for x in y] for y in subword_log_probs]
    return subword_log_probs


def load_subwords_from_file(subwords_path, subword_type):
    if subword_type == "double_at_separator":
        # It used the default bpe, where word continuation markers is @@
        with open(subwords_path, 'r') as f:
            subwords = f.readlines()
            subwords = [line.replace('\n', '').split() for line in subwords]
    elif subword_type == "underscore_separator":
        # It used sentence bpe, where word continuation markers is space, word split is ▁
        # Convert to the default bpe
        with open(subwords_path, 'r') as f:
            subwords = f.readlines()
            subwords = [line[1:].strip().replace('\n', '').replace(' ▁', '▁').replace(' ', '@@▁').split('▁') for line in subwords]
    else:
        raise RuntimeError(f"Unknown subword_type {subword_type}")
    return subwords


def get_nmt_word_log_probs_avg_perturbed(dataset, data_root_path, src_lang, tgt_lang, original_translation_output_dir,
                                         perturbed_translation_output_dir, perturbed_trans_df_path,
                                         alignment_tool='Levenshtein'):
    """
    Align the perturbed translations with the original translation, and take the average of the log probabilities of all
    aligned words in the perturbed translations with each original word.

    """
    original_word_log_probs = get_nmt_word_log_probs(
        dataset, data_root_path, src_lang, tgt_lang, original_translation_output_dir
    )

    all_perturbed_word_log_probs = get_nmt_word_log_probs_perturbed(perturbed_translation_output_dir,
                                                                    perturbed_trans_df_path)

    perturbed_trans_df = pd.read_pickle(perturbed_trans_df_path)
    # Perform alignment
    original_trans_tokenized = perturbed_trans_df['tokenized_SRC-Trans'].tolist()
    perturbed_trans_tokenized = perturbed_trans_df['tokenized_SRC_perturbed-Trans'].tolist()
    if alignment_tool == 'Levenshtein':
        aligments = [edist_alignment(s1, s2) for s1, s2 in zip(original_trans_tokenized, perturbed_trans_tokenized)]
    elif alignment_tool == 'Tercom':
        aligments = tercom_alignment(original_trans_tokenized, perturbed_trans_tokenized)
    else:
        raise RuntimeError(f"Unknown alignment tool {alignment_tool}")

    # Collect the probs
    perturbed_count_per_sentence = perturbed_trans_df.groupby('SRC_original_idx').size()
    pertubed_i = 0
    avg_perturbed_log_probs = []
    for sentence_i in perturbed_count_per_sentence.index:
        avg_perturbed_log_probs_per_sentence = []
        aligments_per_sentence = aligments[
                                 pertubed_i:pertubed_i+perturbed_count_per_sentence[sentence_i]
                                 ]
        perturbed_word_log_probs_per_sentence = all_perturbed_word_log_probs[
                                                pertubed_i:pertubed_i+perturbed_count_per_sentence[sentence_i]
                                                ]
        pertubed_i = pertubed_i + perturbed_count_per_sentence[sentence_i]

        for word_i in range(len(original_word_log_probs[sentence_i])):
            log_probs = [original_word_log_probs[sentence_i][word_i]]
            for pertubed_i_per_sentence in range(len(aligments_per_sentence)):
                if word_i not in dict(aligments_per_sentence[pertubed_i_per_sentence]).keys()\
                        or pd.isna(dict(aligments_per_sentence[pertubed_i_per_sentence])[word_i]):
                    continue
                log_probs.append(
                    perturbed_word_log_probs_per_sentence[pertubed_i_per_sentence][
                        dict(aligments_per_sentence[pertubed_i_per_sentence])[word_i]
                    ]
                )
            avg_perturbed_log_probs_per_sentence.append(np.mean(np.array(log_probs)))
        avg_perturbed_log_probs.append(avg_perturbed_log_probs_per_sentence)

    return avg_perturbed_log_probs


def get_nmt_word_log_probs_perturbed(perturbed_translation_output_dir, perturbed_trans_df_path):
    subwords_path = f"{perturbed_translation_output_dir}/mt.out"
    subword_log_probs_path = f"{perturbed_translation_output_dir}/log_prob.out"

    perturbed_trans_df = pd.read_pickle(perturbed_trans_df_path)
    tokenized_translations = perturbed_trans_df['tokenized_SRC_perturbed-Trans'].tolist()

    subword_type = get_subword_type(subwords_path)
    subwords = load_subwords_from_file(subwords_path, subword_type)
    subword_log_probs = load_subword_log_probs_from_file(subword_log_probs_path)
    word_log_probs = merge_subwords_log_prob_to_words_log_prob(subwords, subword_log_probs, tokenized_translations)

    return word_log_probs


def get_subword_type(subwords_path):
    # First check subword type
    with open(subwords_path, 'r') as f:
        text = f.read()
    count_underscore = text.count('▁')
    count_double_at = text.count('@@')
    if count_double_at > count_underscore:
        subword_type = "double_at_separator"
    else:
        subword_type = "underscore_separator"
    if (count_underscore == 0) and (count_double_at == 0):
        raise RuntimeError("Unknown subword_type")
    return subword_type


def merge_subwords_log_prob_to_words_log_prob(subwords, subword_log_probs, tokenized_translations):
    """
    Concat subwords wih '@@' and hyphen tokens @-@
    """
    subwords = [[x.replace('@@', '').replace('@-@', '-') for x in y] for y in subwords]
    word_log_probs = []
    for sentence_index in range(len(subwords)):
        word_log_probs_per_sentence = []
        subword_log_probs_per_sentence = subword_log_probs[sentence_index]
        subwords_per_sentence = subwords[sentence_index]
        start_pointer = 0  # Will move from left to right along the sentence, subword by subword
        for word in tokenized_translations[sentence_index]:
            # Finding the start where the word and the subword first align
            found = False
            tmp_start_pointer = start_pointer
            while tmp_start_pointer < len(subwords_per_sentence):
                if word.startswith(subwords_per_sentence[tmp_start_pointer]):
                    start_pointer = tmp_start_pointer
                    found = True
                    break
                else:
                    tmp_start_pointer = tmp_start_pointer + 1
            if not found:
                word_log_probs_per_sentence.append(np.nan)
                continue
            # Finding the end where the word and the subword first align
            tmp_end_pointer = start_pointer
            found = False
            while tmp_end_pointer < len(subwords_per_sentence):
                if ''.join(subwords_per_sentence[start_pointer:tmp_end_pointer+1]) == word:
                    word_log_probs_per_sentence.append(
                        np.sum(np.array(subword_log_probs_per_sentence[start_pointer:tmp_end_pointer+1]))
                    )
                    found = True
                    start_pointer = tmp_end_pointer + 1
                    break
                else:
                    tmp_end_pointer = tmp_end_pointer + 1
            if not found:
                word_log_probs_per_sentence.append(np.nan)
                continue
        word_log_probs.append(word_log_probs_per_sentence)
    return word_log_probs


def find_bad_word(tgt_src_effects, effecting_words_threshold):
    """

    Args:
        tgt_src_effects: output of the analysis
        effecting_words_threshold: if more than `effecting_words_threshold` src words changes effect the
            translation, then it's a bad translation

    Returns:

    """
    bad_tgt_words = []
    for tgt_word, src_effects in tgt_src_effects.items():
        if len(src_effects['effecting_words']) > effecting_words_threshold:
            bad_tgt_words.append(tgt_word)
    return bad_tgt_words


def find_ok_word(tgt_src_effects, no_effecting_words_threshold):
    """

    Args:
        tgt_src_effects: output of the analysis
        no_effecting_words_threshold: if more than `no_effecting_words_threshold` src words changes do not effect the
            translation, then it's a good translation

    Returns:

    """
    ok_tgt_words = []
    for tgt_word, src_effects in tgt_src_effects.items():
        if len(src_effects['no_effecting_words']) > no_effecting_words_threshold:
            ok_tgt_words.append(tgt_word)
    return ok_tgt_words


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--perturbed_trans_df_path', type=str, default=None)
    parser.add_argument('--original_translation_output_dir', type=str,
                        help='Folder containing the translation output, including the log probabilities')
    parser.add_argument('--dataset', type=str,
                        choices=['WMT21_DA_test', 'WMT21_DA_dev', 'WMT20_HJQE_dev', 'WMT20_HJQE_test']
    )
    parser.add_argument('--data_root_path', type=str)
    parser.add_argument('--src_lang', type=str)
    parser.add_argument('--tgt_lang', type=str)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--sentence_level_eval_da', type=str_to_bool, default=False)
    parser.add_argument('--trans_word_level_eval', type=str_to_bool, default=False)
    parser.add_argument('--trans_word_level_eval_methods', type=str, nargs="*",
                        help="Provide options for hyperparams tuning."
                             "Any non-empty sublist of ['nmt_log_prob', 'nr_effecting_src_words']",
                        default=['nmt_log_prob', 'nr_effecting_src_words'])
    parser.add_argument('--nmt_log_prob_thresholds', type=float, nargs="*",
                        help="If nmt log prob of a word < threshold, mark it as BAD."
                             "Provide a list of options for hyperparams tuning."
                             "E.g., [0.4, 0.5, 0.6]",
                        default=[0.4, 0.5, 0.6])
    parser.add_argument('--src_word_level_eval', type=str_to_bool, default=False)
    parser.add_argument('--src_word_level_eval_methods', type=str, nargs="*",
                        help="Provide options for hyperparams tuning."
                             "Any non-empty sublist of ['nmt_log_prob', 'nr_effecting_src_words']",
                        default=['nmt_log_prob', 'nr_effecting_src_words'])
    parser.add_argument('--sentence_level_eval_da_method', type=str,
                        choices=[
                            'trans_edit_distance',  # avg edit distance of the perturbed trans' vs original trans
                            'trans_edit_distance/sentence_length',  # same as above but normalize by length
                            'change_spread',  # Avg ongest distance between 2 changes of the perturbed trans' vs original trans
                            'change_spread/sentence_length',  # same as above but normalize by length
                            'word_level_agg',  # aggreate word_level eval (count)
                        ])
    parser.add_argument('--effecting_words_thresholds', type=int, nargs="*",
                        help="For a word, if the number of SRC words in the sentence effecting its translation"
                             "is > threshold, that word is marked as BAD"
                             "Provide a list of options for hyperparams tuning."
                             "E.g., [1, 2, 3, 4]",
                        default=[1])
    parser.add_argument('--consistence_trans_portion_thresholds', type=float, nargs="*",
                        help="1 sentence 1 perturbed word different replacement."
                             "For a translated word, if the frequency of the most common translation among different perturbation"
                             " > consistence_trans_portion_threshold"
                             "then it is a consistence translation."
                             "Provide a list of options for hyperparams tuning."
                             "E.g., [0.6, 0.7, 0.8, 0.9]",
                        default=[0.9])
    parser.add_argument('--uniques_portion_for_noiseORperturbed_thresholds', type=float, nargs="*",
                        help="1 sentence 1 perturbed word different replacement."
                             "For a translated word, if the portion of unique translation among different perturbation"
                             " > uniques_portion_for_noiseORperturbed_threshold"
                             "then it is the perturbed word or noise."
                             "Provide a list of options for hyperparams tuning."
                             "E.g., [0.4, 0.6, 0.8]",
                        default=[0.4])
    parser.add_argument('--no_effecting_words_portion_thresholds', type=float, nargs="*",
                        help="For a word, if the number of SRC words in the sentence NOT effecting its translation"
                             "is > threshold, that word is marked as OK"
                             "Provide a list of options for hyperparams tuning."
                             "E.g., [0.4, 0.5, 0.6, 0.7, 0.8, 0.9]",
                        default=[0.6])
    parser.add_argument('--alignment_tool', type=str, choices=['Levenshtein', 'Tercom'], default='Tercom')

    args = parser.parse_args()
    print(args)

    set_seed(args.seed)

    if args.trans_word_level_eval:
        task = 'trans_word_level_eval'
        hyperparams_tune_word_level_eval(methods=args.trans_word_level_eval_methods,
                                         nmt_log_prob_thresholds=args.nmt_log_prob_thresholds,
                                         dataset=args.dataset,
                                         data_root_path=args.data_root_path,
                                         src_lang=args.src_lang,
                                         tgt_lang=args.tgt_lang,
                                         perturbed_trans_df_path=args.perturbed_trans_df_path,
                                         original_translation_output_dir=args.original_translation_output_dir,
                                         effecting_words_thresholds=args.effecting_words_thresholds,
                                         consistence_trans_portion_thresholds=args.consistence_trans_portion_thresholds,
                                         uniques_portion_for_noiseORperturbed_thresholds=args.uniques_portion_for_noiseORperturbed_thresholds,
                                         no_effecting_words_portion_thresholds=args.no_effecting_words_portion_thresholds,
                                         task=task,
                                         alignment_tool=args.alignment_tool)

    if args.src_word_level_eval:
        task = 'src_word_level_eval'
        hyperparams_tune_word_level_eval(methods=args.trans_word_level_eval_methods,
                                         nmt_log_prob_thresholds=args.nmt_log_prob_thresholds,
                                         dataset=args.dataset,
                                         data_root_path=args.data_root_path,
                                         src_lang=args.src_lang,
                                         tgt_lang=args.tgt_lang,
                                         perturbed_trans_df_path=args.perturbed_trans_df_path,
                                         original_translation_output_dir=args.original_translation_output_dir,
                                         effecting_words_thresholds=args.effecting_words_thresholds,
                                         consistence_trans_portion_thresholds=args.consistence_trans_portion_thresholds,
                                         uniques_portion_for_noiseORperturbed_thresholds=args.uniques_portion_for_noiseORperturbed_thresholds,
                                         no_effecting_words_portion_thresholds=args.no_effecting_words_portion_thresholds,
                                         task=task,
                                         alignment_tool=args.alignment_tool)

    if args.sentence_level_eval_da:
        perturbed_trans_df = pd.read_pickle(args.perturbed_trans_df_path)
        approximations = perturbed_trans_df[
            ["SRC_original_idx",
             "Trans-edit_distance",
             "#TransChanges/SentenceLength",
             "ChangesSpread",
             "ChangesSpread/SentenceLength"
             ]
        ].groupby("SRC_original_idx").mean()

        word_level_tags = nr_effecting_src_words_eval(args.perturbed_trans_df_path, args.effecting_words_threshold,
                                                      task='trans_word_level_eval',
                                                      consistence_trans_portion_threshold=args.consistence_trans_portion_threshold,
                                                      uniques_portion_for_noiseORperturbed_threshold=args.uniques_portion_for_noiseORperturbed_threshold,
                                                      no_effecting_words_portion_threshold=args.no_effecting_words_portion_threshold,
                                                      alignment_tool=args.alignment_tool)
        approximations['word_level_agg'] = [x.count('BAD') for x in word_level_tags]

        for col in approximations.columns:
            # Normalize the approximations, invert the sign
            approximations[col] = -approximations[col]
            approximations[col] = zscore(approximations[col].values)

            print(f"-----------------{col}-----------------")
            print(
                pearsonr(
                    load_gold_labels(
                        args.dataset, args.data_root_path, args.src_lang, args.tgt_lang, task='sentence_level_eval_da'
                    ),
                    approximations[col].values
                )
            )


def hyperparams_tune_word_level_eval(methods, nmt_log_prob_thresholds, dataset, data_root_path, src_lang,
                                     tgt_lang, perturbed_trans_df_path, original_translation_output_dir,
                                     effecting_words_thresholds,
                                     consistence_trans_portion_thresholds,
                                     uniques_portion_for_noiseORperturbed_thresholds,
                                     no_effecting_words_portion_thresholds,
                                     task,
                                     alignment_tool='Levenshtein'):
    print("---------------------------------------")
    print(f"Hyperparams tuning for task {task}")
    for method in methods:
        print(f"\tMethod: {method}")
        max_score = 0
        best_hyperparams = None
        if method == 'nmt_log_prob':
            for nmt_log_prob_threshold in nmt_log_prob_thresholds:
                matthews_corrcoef_score = word_level_eval(task, dataset, data_root_path, src_lang,
                                                          tgt_lang,
                                                          perturbed_trans_df_path,
                                                          original_translation_output_dir=original_translation_output_dir,
                                                          word_level_eval_method=method,
                                                          nmt_log_prob_threshold=nmt_log_prob_threshold
                                                          )
                print(f"\t\tnmt_log_prob_thresholds: {nmt_log_prob_threshold}")
                print(f"\t\t\tmatthews_corrcoef_score: {matthews_corrcoef_score}")
                if matthews_corrcoef_score >= max_score:
                    max_score = matthews_corrcoef_score
                    best_hyperparams = nmt_log_prob_threshold

        elif method == 'nr_effecting_src_words':
            hyperparams_choices = [effecting_words_thresholds, consistence_trans_portion_thresholds,
                                   uniques_portion_for_noiseORperturbed_thresholds, no_effecting_words_portion_thresholds]
            choice_tuples = list(itertools.product(*hyperparams_choices))

            # Multiprocessing for speeding up
            # Flatten for each params
            all_effecting_words_threshold = [choice_tuple[0] for choice_tuple in choice_tuples]
            all_consistence_trans_portion_threshold = [choice_tuple[1] for choice_tuple in choice_tuples]
            all_uniques_portion_for_noiseORperturbed_threshold = [choice_tuple[2] for choice_tuple in choice_tuples]
            all_no_effecting_words_portion_threshold = [choice_tuple[3] for choice_tuple in choice_tuples]

            # num_processes = cpu_count() - 1 if cpu_count() > 1 else cpu_count()
            num_processes = 10
            with Pool(num_processes) as pool:
                matthews_corrcoef_scores = pool.starmap(word_level_eval,
                                                       zip(repeat(task),
                                                           repeat(dataset),
                                                           repeat(data_root_path),
                                                           repeat(src_lang),
                                                           repeat(tgt_lang),
                                                           repeat(perturbed_trans_df_path),
                                                           repeat(original_translation_output_dir),
                                                           repeat(method),
                                                           repeat(0.5),  # Log prob threshold for the other method, not used, just for syntax
                                                           all_effecting_words_threshold,
                                                           all_consistence_trans_portion_threshold,
                                                           all_uniques_portion_for_noiseORperturbed_threshold,
                                                           all_no_effecting_words_portion_threshold,
                                                           repeat(alignment_tool)
                                                           ))

            for idx, choice_tuple in enumerate(choice_tuples):
                matthews_corrcoef_score = matthews_corrcoef_scores[idx]
                print(f"\t\tchoice_tuple: {choice_tuple}")
                print(f"\t\t\tmatthews_corrcoef_score: {matthews_corrcoef_score}")
                if matthews_corrcoef_score >= max_score:
                    max_score = matthews_corrcoef_score
                    best_hyperparams = choice_tuple
        else:
            raise RuntimeError(f"Unknown method {method} for task {task}")

        print(f"*************** FINAL BEST FOR METHOD {method}: best score {max_score}, best params {best_hyperparams}")


def word_level_eval(task, dataset, data_root_path, src_lang, tgt_lang, perturbed_trans_df_path,
                    original_translation_output_dir, word_level_eval_method,
                    nmt_log_prob_threshold=0.5, effecting_words_threshold=2, consistence_trans_portion_threshold=0.8,
                    uniques_portion_for_noiseORperturbed_threshold=0.4,
                    no_effecting_words_portion_threshold=0.8,
                    alignment_tool='Levenshtein'):
    gold_labels = load_gold_labels(dataset, data_root_path, src_lang, tgt_lang,
                                   task=task)
    if word_level_eval_method == 'nmt_log_prob':
        pred_labels = nmt_log_prob_eval(dataset, data_root_path, src_lang,
                                        tgt_lang, nmt_log_prob_threshold,
                                        perturbed_trans_df_path, task=task,
                                        original_translation_output_dir=original_translation_output_dir)
    elif word_level_eval_method == 'nr_effecting_src_words':
        pred_labels = nr_effecting_src_words_eval(
            perturbed_trans_df_path, effecting_words_threshold,
            task=task,
            consistence_trans_portion_threshold=consistence_trans_portion_threshold,
            uniques_portion_for_noiseORperturbed_threshold=uniques_portion_for_noiseORperturbed_threshold,
            no_effecting_words_portion_threshold=no_effecting_words_portion_threshold,
            alignment_tool=alignment_tool
        )
    else:
        raise RuntimeError(f"Method {word_level_eval_method} not available for task {task}.")

    return matthews_corrcoef(y_true=flatten_list(gold_labels), y_pred=flatten_list(pred_labels))


if __name__ == "__main__":
    main()
