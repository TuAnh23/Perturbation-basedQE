import copy
import itertools

import pandas as pd
import numpy as np
from sklearn.metrics import matthews_corrcoef
import argparse
from utils import str_to_bool, set_seed
from align_and_analyse_ambiguous_trans import analyse_single_sentence
from scipy.stats import zscore
from scipy.stats import pearsonr, spearmanr
from collections import defaultdict
from sacremoses import MosesTokenizer


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
    if dataset == 'WMT21_DA_dev':
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
                                keep_unknown=False):
    """
    *_word_level_eval by using nr_effecting_src_words
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
    perturbed_trans_df = pd.read_pickle(perturbed_trans_df_path)
    SRC_original_indices = perturbed_trans_df['SRC_original_idx'].unique()
    for SRC_original_idx in SRC_original_indices:
        sentence_df = perturbed_trans_df[perturbed_trans_df['SRC_original_idx'] == SRC_original_idx]
        original_trans_length = len(sentence_df['tokenized_SRC-Trans'].values[0])
        original_src_length = len(sentence_df['tokenized_SRC'].values[0])
        tgt_src_effects = analyse_single_sentence(
            sentence_df,
            align_type=align_type, return_word_index=True,
            consistence_trans_portion_threshold=consistence_trans_portion_threshold,
            uniques_portion_for_noiseORperturbed_threshold=uniques_portion_for_noiseORperturbed_threshold
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

    if not keep_unknown:
        word_tag = replace_unknown(word_tag)
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
    else:
        with open(f"{original_translation_output_dir}/trans_sentences.txt", 'r') as f:
            translations = f.readlines()
            translations = [line.rstrip() for line in translations]

        tgt_tokenizer = MosesTokenizer(lang=tgt_lang)
        tokenized_translations = [tgt_tokenizer.tokenize(x, escape=False, aggressive_dash_splits=False)
                                  for x in translations]

    with open(subwords_path, 'r') as f:
        subwords = f.readlines()
        subwords = [line.replace('\n', '').split() for line in subwords]

    with open(subword_log_probs_path, 'r') as f:
        subword_log_probs = f.readlines()
        subword_log_probs = [line.replace('\n', '').split() for line in subword_log_probs]
        # Cast all values to float
        subword_log_probs = [[float(x) for x in y] for y in subword_log_probs]

    word_log_probs = merge_subwords_log_prob_to_words_log_prob(subwords, subword_log_probs, tokenized_translations)
    return word_log_probs


def merge_subwords_log_prob_to_words_log_prob(subwords, subword_log_probs, tokenized_translations):
    """
    Concat subwords wih '@@' and hyphen tokens @-@
    """
    # Merge the subword_log_probs to word_log_probs
    # Concat subwords wih '@@' and hyphen tokens @-@
    word_log_probs = []
    for sentence_index in range(len(subwords)):
        word_log_probs_per_sentence = defaultdict(lambda: np.nan)  # word:log_prob
        subword_log_probs_per_sentence = subword_log_probs[sentence_index]
        subwords_per_sentence = subwords[sentence_index]
        current_word_log_prob = 0
        current_word = ''
        for subword_index in range(len(subwords_per_sentence)):
            current_word_log_prob = current_word_log_prob + subword_log_probs_per_sentence[subword_index]
            if subwords_per_sentence[subword_index].endswith('@@'):
                current_word = current_word + subwords_per_sentence[subword_index][:-2]
            elif subwords_per_sentence[subword_index] == '@-@':
                current_word = current_word + '-'
            else:
                current_word = current_word + subwords_per_sentence[subword_index]
            if (not subwords_per_sentence[subword_index].endswith('@@')) and \
                    (not subwords_per_sentence[subword_index] == '@-@') and \
                    (not (((subword_index + 1) < len(subwords_per_sentence)) and (
                            subwords_per_sentence[subword_index + 1] == '@-@'))):
                word_log_probs_per_sentence[current_word] = current_word_log_prob
                current_word_log_prob = 0
                current_word = ''

        fixed_word_log_probs_per_sentence = [word_log_probs_per_sentence[word] for word in
                                             tokenized_translations[sentence_index]]
        word_log_probs.append(fixed_word_log_probs_per_sentence)
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
    parser.add_argument('--perturbed_trans_df_path', type=str)
    parser.add_argument('--original_translation_output_dir', type=str,
                        help='Folder containing the translation output, including the log probabilities')
    parser.add_argument('--dataset', type=str, choices=['WMT21_DA_test', 'WMT21_DA_dev'])
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
                                         task=task)

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
                                         task=task)

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
                                                      no_effecting_words_portion_threshold=args.no_effecting_words_portion_threshold)
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
                                     task):
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
            for choice_tuple in choice_tuples:
                effecting_words_threshold, consistence_trans_portion_threshold, \
                uniques_portion_for_noiseORperturbed_threshold, no_effecting_words_portion_threshold = choice_tuple
                matthews_corrcoef_score = word_level_eval(task, dataset, data_root_path, src_lang,
                                                          tgt_lang,
                                                          perturbed_trans_df_path,
                                                          original_translation_output_dir=original_translation_output_dir,
                                                          word_level_eval_method=method,
                                                          effecting_words_threshold=effecting_words_threshold,
                                                          consistence_trans_portion_threshold=consistence_trans_portion_threshold,
                                                          uniques_portion_for_noiseORperturbed_threshold=uniques_portion_for_noiseORperturbed_threshold,
                                                          no_effecting_words_portion_threshold=no_effecting_words_portion_threshold
                                                          )
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
                    no_effecting_words_portion_threshold=0.8):
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
            no_effecting_words_portion_threshold=no_effecting_words_portion_threshold
        )
    else:
        raise RuntimeError(f"Method {word_level_eval_method} not available for task {task}.")

    return matthews_corrcoef(y_true=flatten_list(gold_labels), y_pred=flatten_list(pred_labels))


if __name__ == "__main__":
    main()
