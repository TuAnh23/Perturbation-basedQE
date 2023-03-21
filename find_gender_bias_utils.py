"""
Containing functions for running finding gender bias using QE
"""
import argparse
import pickle

import pandas as pd
from read_and_analyse_df import perform_tokenization


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--function',
                        type=str,
                        choices=['reformat_trans', 'eval_gender_bias']
                        )
    parser.add_argument('--translation_df_path', type=str,
                        help="Required for 'reformat_trans'"
                        )
    parser.add_argument('--output_reformat_dir', type=str,
                        help="Required for 'reformat_trans'. Dir location to save the reformatted files."
                        )
    parser.add_argument('--src_lang', type=str,
                        help="Required for 'reformat_trans'."
                        )
    parser.add_argument('--tgt_lang', type=str,
                        help="Required for 'reformat_trans'."
                        )
    parser.add_argument('--gender_pred_path', type=str,
                        help="Path to the csv that contains information about the location of the gender words in the "
                             "translation, and what is the predicted gender."
                             "Required for 'eval_gender_bias'."
                        )
    parser.add_argument('--winoMT_data_path', type=str,
                        help="Path to the csv that contains the original SRC data along with the correct gender."
                             "Required for 'eval_gender_bias'."
                        )
    parser.add_argument('--qe_pred_labels_path', type=str,
                        help="Path to the word-level QE OK/BAD labels."
                             "Required for 'eval_gender_bias'."
                        )
    parser.add_argument('--output_path_eval_gender_bias', type=str,
                        help="Path the file that store the score for eval_gender_bias."
                             "Required for 'eval_gender_bias'."
                        )

    args = parser.parse_args()
    print(args)

    if args.function == 'reformat_trans':
        trans_df = pd.read_csv(args.translation_df_path)
        trans_df['tokenized_SRC'] = perform_tokenization(args.src_lang, trans_df['SRC'].tolist())
        trans_df['tokenized_SRC-Trans'] = perform_tokenization(args.tgt_lang, trans_df['SRC-Trans'].tolist())
        with open(f"{args.output_reformat_dir}/reformatted_forWinoMT.txt", 'w') as file:
            for _, row in trans_df.iterrows():
                file.write(f"{' '.join(row['tokenized_SRC'])} ||| {' '.join(row['tokenized_SRC-Trans'])}\n")
    elif args.function == 'eval_gender_bias':
        gender_pred = pd.read_csv(args.gender_pred_path)
        winoMT_data = pd.read_csv(
            args.winoMT_data_path, sep='\t', header=None, names=['Gender', 'x', 'SRC', 'noun']
        )
        # Merge the two df
        gender_info = pd.merge(winoMT_data, gender_pred, left_index=True, right_index=True)
        assert gender_info.shape[0] == winoMT_data.shape[0]
        assert gender_info.shape[0] == gender_pred.shape[0]

        # Find out the wrong gender output
        gender_info['Correct gender prediction'] = gender_info['Gender'] == gender_info['Predicted gender']

        # Calculate the percentage of wrong-gender token being labeled as BAD by QE
        with open(args.qe_pred_labels_path, 'rb') as f:
            qe_ok_bad_preds = pickle.load(f)
        nr_pred_bads = sum([x.count('BAD') for x in qe_ok_bad_preds])

        nr_true_positives = 0  # The number of wrongly outputted gender translation that is labeled as BAD
        nr_positives = 0  # The total number of wrongly outputted gender translation
        for sentence_index in range(winoMT_data.shape[0]):
            if not gender_info['Correct gender prediction'].iloc[sentence_index]:
                gender_word_indices = str_to_list(gender_info['Gender words indices'].iloc[sentence_index])
                nr_positives = nr_positives + len(gender_word_indices)

                for gender_word_index in gender_word_indices:
                    if qe_ok_bad_preds[sentence_index][gender_word_index] == "BAD":
                        nr_true_positives = nr_true_positives + 1

        nr_bad = sum([x.count('BAD') for x in qe_ok_bad_preds], 0)
        wrong_gender_recall = nr_true_positives/nr_positives
        wrong_gender_precision = nr_true_positives/nr_bad
        with open(args.output_path_eval_gender_bias, 'w') as f:
            f.write(f"wrong_gender_recall: {wrong_gender_recall}\n")
            f.write(f"wrong_gender_precision (only as an indication): {wrong_gender_precision}\n")
            f.write(f"total number of predicted BAD labels: {nr_pred_bads}\n")

    else:
        raise RuntimeError(f"function {args.function} not available.")


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
