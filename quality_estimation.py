"""
Run word-level quality estimation and output the predicted labels
"""
import argparse
import pandas as pd
from tune_quality_estimation import nr_effecting_src_words_eval, nmt_log_prob_eval
import pickle


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--perturbed_trans_df_path', type=str, default=None)
    parser.add_argument('--original_translation_output_dir', type=str,
                        help='Folder containing the translation output, including the log probabilities')
    parser.add_argument('--dataset', type=str)
    parser.add_argument('--data_root_path', type=str)
    parser.add_argument('--src_lang', type=str)
    parser.add_argument('--tgt_lang', type=str)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--task', type=str, default='trans_word_level_eval')
    parser.add_argument('--method', type=str,
                        choices=['nmt_log_prob', 'nr_effecting_src_words'])
    parser.add_argument('--nmt_log_prob_threshold', type=float,
                        help="If nmt log prob of a word < threshold, mark it as BAD.")
    parser.add_argument('--effecting_words_threshold', type=int,
                        help="For a word, if the number of SRC words in the sentence effecting its translation"
                             "is > threshold, that word is marked as BAD")
    parser.add_argument('--consistence_trans_portion_threshold', type=float,
                        help="1 sentence 1 perturbed word different replacement."
                             "For a translated word, if the frequency of the most common translation among different "
                             "perturbation"
                             " > consistence_trans_portion_threshold"
                             "then it is a consistence translation.",
                        default=0.9)
    parser.add_argument('--uniques_portion_for_noiseORperturbed_threshold', type=float,
                        help="1 sentence 1 perturbed word different replacement."
                             "For a translated word, if the portion of unique translation among different perturbation"
                             " > uniques_portion_for_noiseORperturbed_threshold"
                             "then it is the perturbed word or noise.",
                        default=0.4)
    parser.add_argument('--no_effecting_words_portion_threshold', type=float,
                        help="For a word, if the number of SRC words in the sentence NOT effecting its translation"
                             "is > threshold, that word is marked as OK",
                        default=0.6)
    parser.add_argument('--alignment_tool', type=str, choices=['Levenshtein', 'Tercom'], default='Tercom')
    parser.add_argument('--label_output_path', type=str)
    parser.add_argument('--src_tgt_influence_output_path', type=str,
                        help='Only available for method `nr_effecting_src_words`', default=None)

    args = parser.parse_args()
    print(args)

    if args.method == 'nmt_log_prob':
        pred_labels = nmt_log_prob_eval(args.dataset, args.data_root_path, args.src_lang,
                                        args.tgt_lang, args.nmt_log_prob_threshold,
                                        args.perturbed_trans_df_path, task=args.task,
                                        original_translation_output_dir=args.original_translation_output_dir)
    elif args.method == 'nr_effecting_src_words':
        pred_labels, src_tgt_influence = nr_effecting_src_words_eval(
            args.perturbed_trans_df_path, args.effecting_words_threshold,
            task=args.task,
            consistence_trans_portion_threshold=args.consistence_trans_portion_threshold,
            uniques_portion_for_noiseORperturbed_threshold=args.uniques_portion_for_noiseORperturbed_threshold,
            no_effecting_words_portion_threshold=args.no_effecting_words_portion_threshold,
            alignment_tool=args.alignment_tool,
            return_details=True
        )

        if args.src_tgt_influence_output_path is not None:
            with open(args.src_tgt_influence_output_path, 'wb') as f:
                pickle.dump(src_tgt_influence, f)
    else:
        raise RuntimeError(f"QE method {args.method} not available")

    with open(args.label_output_path, 'wb') as f:
        pickle.dump(pred_labels, f)


if __name__ == "__main__":
    main()
