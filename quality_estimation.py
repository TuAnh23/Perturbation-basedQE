import pandas as pd
import numpy as np
from sklearn.metrics import matthews_corrcoef
import argparse
from utils import str_to_bool
from align_and_analyse_ambiguous_trans import analyse_single_sentence
from scipy.stats import zscore
from scipy.stats import pearsonr, spearmanr
from collections import defaultdict


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


def mnt_log_prob_eval(dataset, data_root_path, src_lang, tgt_lang, nmt_log_prob_threshold, perturbed_trans_df_path, task):
    word_log_probs = get_nmt_word_log_probs(dataset, data_root_path, src_lang, tgt_lang)
    threshold = np.log(nmt_log_prob_threshold)
    if task == 'trans_word_level_eval':
        pred_labels = [['unknown' if np.isnan(x) else 'BAD' if x < threshold else 'OK' for x in y]
                       for y in word_log_probs]
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

        pred_labels = [['unknown' if np.isnan(x) else 'BAD' if x < threshold else 'OK' for x in y]
                       for y in src_word_log_probs]
    else:
        raise RuntimeError('Unknown task')
    return pred_labels


def nr_effecting_src_words_eval(perturbed_trans_df_path, effecting_words_threshold, task):
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
        tgt_src_effects = analyse_single_sentence(sentence_df,
                                                  align_type=align_type, return_word_index=True)
        bad_words = find_bad_word(tgt_src_effects, effecting_words_threshold)
        sentence_word_tags = ['BAD' if x in bad_words else 'OK'
                              for x in range(0,
                                             original_trans_length if task == 'trans_word_level_eval' else original_src_length)]
        word_tag.append(sentence_word_tags)
    return word_tag


def get_nmt_word_log_probs(dataset, data_root_path, src_lang, tgt_lang):
    if dataset == 'WMT21_DA_test':
        subwords_path = f"{data_root_path}/wmt-qe-2021-data/{src_lang}-{tgt_lang}-test21/word-probas/mt.test21.{src_lang}{tgt_lang}"
        subword_log_probs_path = f"{data_root_path}/wmt-qe-2021-data/{src_lang}-{tgt_lang}-test21/word-probas/word_probas.test21.{src_lang}{tgt_lang}"
        tokenized_path = f"{data_root_path}/wmt-qe-2021-data/{src_lang}-{tgt_lang}-test21/test21.tok.mt"
    elif dataset == 'WMT21_DA_dev':
        subwords_path = f"{data_root_path}/wmt-qe-2021-data/{src_lang}-{tgt_lang}-dev/direct-assessments/{src_lang}-{tgt_lang}-dev/word-probas/mt.dev.{src_lang}{tgt_lang}"
        subword_log_probs_path = f"{data_root_path}/wmt-qe-2021-data/{src_lang}-{tgt_lang}-dev/direct-assessments/{src_lang}-{tgt_lang}-dev/word-probas/word_probas.dev.{src_lang}{tgt_lang}"
        tokenized_path = f"{data_root_path}/wmt-qe-2021-data/{src_lang}-{tgt_lang}-dev/post-editing/{src_lang}-{tgt_lang}-dev/dev.mt"
    else:
        raise RuntimeError(f"get_nmt_word_log_probs not available for dataset {dataset}")

    with open(subwords_path, 'r') as f:
        subwords = f.readlines()
        subwords = [line.replace('\n', '').split() for line in subwords]

    with open(subword_log_probs_path, 'r') as f:
        subword_log_probs = f.readlines()
        subword_log_probs = [line.replace('\n', '').split() for line in subword_log_probs]
        # Cast all values to float
        subword_log_probs = [[float(x) for x in y] for y in subword_log_probs]

    with open(tokenized_path, 'r') as f:
        tokenized_translations = f.readlines()
        tokenized_translations = [tokenized_trans.strip().split() for tokenized_trans in tokenized_translations]

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
    bad_tgt_words = []
    for tgt_word, src_effects in tgt_src_effects.items():
        if len(src_effects['effecting_words']) > effecting_words_threshold:
            bad_tgt_words.append(tgt_word)
    return bad_tgt_words


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--perturbed_trans_df_path', type=str)
    parser.add_argument('--dataset', type=str, choices=['WMT21_DA_test', 'WMT21_DA_dev'])
    parser.add_argument('--data_root_path', type=str)
    parser.add_argument('--src_lang', type=str)
    parser.add_argument('--tgt_lang', type=str)
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
                        default=[1, 2, 3, 4])

    args = parser.parse_args()
    print(args)

    if args.trans_word_level_eval:
        task = 'trans_word_level_eval'
        print("---------------------------------------")
        print(f"Hyperparams tuning for task {task}")
        max_score = 0
        for trans_word_level_eval_method in args.trans_word_level_eval_methods:
            print(f"\tMethod: {trans_word_level_eval_method}")
            if trans_word_level_eval_method == 'nmt_log_prob':
                for nmt_log_prob_threshold in args.nmt_log_prob_thresholds:
                    matthews_corrcoef_score = word_level_eval(task, args.dataset, args.data_root_path, args.src_lang, args.tgt_lang,
                                                              args.perturbed_trans_df_path,
                                                              word_level_eval_method=trans_word_level_eval_method,
                                                              nmt_log_prob_threshold=nmt_log_prob_threshold)
                    print(f"\t\tnmt_log_prob_thresholds: {nmt_log_prob_threshold}")
                    print(f"\t\t\tmatthews_corrcoef_score: {matthews_corrcoef_score}")
                    if matthews_corrcoef_score >= max_score:
                        print("\t\t\tCurrent best")
            elif trans_word_level_eval_method == 'nr_effecting_src_words':
                for effecting_words_threshold in args.effecting_words_thresholds:
                    matthews_corrcoef_score = word_level_eval(task, args.dataset, args.data_root_path, args.src_lang,
                                                              args.tgt_lang,
                                                              args.perturbed_trans_df_path,
                                                              word_level_eval_method=trans_word_level_eval_method,
                                                              effecting_words_threshold=effecting_words_threshold)
                    print(f"\t\teffecting_words_threshold: {effecting_words_threshold}")
                    print(f"\t\t\tmatthews_corrcoef_score: {matthews_corrcoef_score}")
                    if matthews_corrcoef_score >= max_score:
                        max_score = matthews_corrcoef_score
                        print("\t\t\tCurrent best")
            else:
                raise RuntimeError(f"Unknown method {trans_word_level_eval_method} for task {task}")

    if args.src_word_level_eval:
        task = 'src_word_level_eval'
        print("---------------------------------------")
        print(f"Hyperparams tuning for task {task}")
        max_score = 0
        for src_word_level_eval_method in args.src_word_level_eval_methods:
            print(f"\tMethod: {src_word_level_eval_method}")
            if src_word_level_eval_method == 'nmt_log_prob':
                for nmt_log_prob_threshold in args.nmt_log_prob_thresholds:
                    matthews_corrcoef_score = word_level_eval(task, args.dataset, args.data_root_path, args.src_lang,
                                                              args.tgt_lang,
                                                              args.perturbed_trans_df_path,
                                                              word_level_eval_method=src_word_level_eval_method,
                                                              nmt_log_prob_threshold=nmt_log_prob_threshold)
                    print(f"\t\tnmt_log_prob_thresholds: {nmt_log_prob_threshold}")
                    print(f"\t\t\tmatthews_corrcoef_score: {matthews_corrcoef_score}")
                    if matthews_corrcoef_score >= max_score:
                        print("\t\t\tCurrent best")
            elif src_word_level_eval_method == 'nr_effecting_src_words':
                for effecting_words_threshold in args.effecting_words_thresholds:
                    matthews_corrcoef_score = word_level_eval(task, args.dataset, args.data_root_path, args.src_lang,
                                                              args.tgt_lang,
                                                              args.perturbed_trans_df_path,
                                                              word_level_eval_method=src_word_level_eval_method,
                                                              effecting_words_threshold=effecting_words_threshold)
                    print(f"\t\teffecting_words_threshold: {effecting_words_threshold}")
                    print(f"\t\t\tmatthews_corrcoef_score: {matthews_corrcoef_score}")
                    if matthews_corrcoef_score >= max_score:
                        max_score = matthews_corrcoef_score
                        print("\t\t\tCurrent best")
            else:
                raise RuntimeError(f"Unknown method {src_word_level_eval_method} for task {task}")

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
                                                      task='trans_word_level_eval')
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


def word_level_eval(task, dataset, data_root_path, src_lang, tgt_lang, perturbed_trans_df_path, word_level_eval_method,
                    nmt_log_prob_threshold=0.5, effecting_words_threshold=2):
    gold_labels = load_gold_labels(dataset, data_root_path, src_lang, tgt_lang,
                                   task=task)
    if word_level_eval_method == 'nmt_log_prob':
        pred_labels = mnt_log_prob_eval(dataset, data_root_path, src_lang,
                                        tgt_lang, nmt_log_prob_threshold,
                                        perturbed_trans_df_path, task=task)
    elif word_level_eval_method == 'nr_effecting_src_words':
        pred_labels = nr_effecting_src_words_eval(perturbed_trans_df_path, effecting_words_threshold,
                                                  task=task)
    else:
        raise RuntimeError(f"Method {word_level_eval_method} not available for task {task}.")

    return matthews_corrcoef(y_true=flatten_list(gold_labels), y_pred=flatten_list(pred_labels))


if __name__ == "__main__":
    main()
