"""
Word alignment using awesome-align: https://github.com/neulab/awesome-align
Prepare src-tgt to the correct format
"Inputs should be *tokenized* and each line is a source language sentence and its target language translation,
separated by (`|||`)"
"""

import argparse
from utils import str_to_bool
from read_and_analyse_df import read_output_df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--df_root_path', type=str)
    parser.add_argument('--data_root_path', type=str, default='data')
    parser.add_argument('--src_lang', type=str, default="en")
    parser.add_argument('--tgt_lang', type=str, default="de")
    parser.add_argument('--replacement_strategy', type=str, default='word2vec_similarity',
                        help='[word2vec_similarity|masking_language_model]. The later option is context-based.')
    parser.add_argument('--number_of_replacement', type=int, default=5,
                        help='The number of replacement for 1 SRC word')
    parser.add_argument('--dataname', type=str)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--beam', type=int, default=5)
    parser.add_argument('--mask_type', type=str)
    parser.add_argument('--winoMT', type=str_to_bool, default=False)

    args = parser.parse_args()
    print(args)

    if args.winoMT:
        args.mask_type = 'pronoun'
        args.number_of_replacement = 1

    # Output the reformatted src-trans file to be used for awesome align
    read_output_df(df_root_path=args.df_root_path, data_root_path=args.data_root_path,
                   dataset=f"{args.dataname}_{args.src_lang}2{args.tgt_lang}",
                   src_lang=args.src_lang, tgt_lang=args.tgt_lang, mask_type=args.mask_type,
                   beam=args.beam, replacement_strategy=args.replacement_strategy, ignore_case=False,
                   no_of_replacements=args.number_of_replacement, winoMT=args.winoMT,
                   tokenize_sentences=True, reformat_for_src_tgt_alignment=True)


if __name__ == "__main__":
    main()
