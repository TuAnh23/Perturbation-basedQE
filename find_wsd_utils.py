"""
Containing functions for running finding Word Sense Disambiguation (WSD) error using QE
"""
import argparse
import pickle

import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--wsd_label_path', type=str,
                        help="Path to the csv that contains information about the location of the WSD words in the "
                             "translation, and what is the correct/incorrect WSD."
                        )
    parser.add_argument('--qe_pred_labels_path', type=str,
                        help="Path to the word-level QE OK/BAD labels."
                        )
    parser.add_argument('--output_path_eval_wsd_error', type=str,
                        help="Path the file that store the score for "
                             "evaluating whether QE methods can detect WSD errors."
                        )

    args = parser.parse_args()
    print(args)

    wsd_info = pd.read_csv(args.wsd_label_path)

    # Calculate the percentage of wrong-gender token being labeled as BAD by QE
    with open(args.qe_pred_labels_path, 'rb') as f:
        qe_ok_bad_preds = pickle.load(f)
    nr_pred_bads = sum([x.count('BAD') for x in qe_ok_bad_preds])

    nr_true_positives = 0  # The number of wrongly outputted gender translation that is labeled as BAD
    nr_positives = 0  # The total number of wrongly outputted gender translation
    for sentence_index in range(wsd_info.shape[0]):
        if (wsd_info['Correct WSD output'].iloc[sentence_index] is not None) and \
                (not wsd_info['Correct WSD output'].iloc[sentence_index]):
            wrong_wsd_word_indices = str_to_list(wsd_info['Wrong WSD words indices'].iloc[sentence_index])
            nr_positives = nr_positives + len(wrong_wsd_word_indices)

            for gender_word_index in wrong_wsd_word_indices:
                if (gender_word_index < len(qe_ok_bad_preds[sentence_index]) and
                        qe_ok_bad_preds[sentence_index][gender_word_index] == "BAD"):
                    nr_true_positives = nr_true_positives + 1

    wrong_wsd_recall = nr_true_positives/nr_positives
    wrong_wsd_precision = nr_true_positives/nr_pred_bads

    with open(args.output_path_eval_wsd_error, 'w') as f:
        f.write(f"wrong_wsd_recall: {wrong_wsd_recall}\n")
        f.write(f"wrong_wsd_precision (only as an indication): {wrong_wsd_precision}\n")
        f.write(f"total number of predicted BAD labels: {nr_pred_bads}\n")
        f.write(f"total number of actual WSD-BAD labels: {nr_positives}\n")


def str_to_list(list_as_str):
    """
    E.g.,
    "[0, 1]" --> [0, 1]
    """
    if list_as_str == '[]':
        return []
    else:
        l = list_as_str[1:-1].split(', ')
        l = [int(x) for x in l]
        return l


if __name__ == "__main__":
    main()
