import pandas as pd
import argparse
from mosestokenizer import *


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

    parser = argparse.ArgumentParser(description='Translate with perturbation to source sentences.')
    parser.add_argument('--func', type=str, help="pass in the desired functionality")
    parser.add_argument('--output_dir', type=str, default=None)
    parser.add_argument('--SRC_perturbed_type', type=none_or_str, default=None)
    parser.add_argument('--src_lang', type=str, default=None)
    parser.add_argument('--tgt_lang', type=str, default=None)
    parser.add_argument('--tmp_dir', type=str, default=None)

    args = parser.parse_args()
    print(args)

    if args.func == 'format_input':
        format_input(args.output_dir, args.SRC_perturbed_type, args.src_lang, args.tgt_lang)
    elif args.func == 'tokenize':
        for lang in [args.src_lang, args.tgt_lang]:
            tokenize(f"{args.output_dir}/input.{lang}", lang, args.tmp_dir)
    elif args.func == 'detokenize':
        detokenize(f"{args.tmp_dir}/mt_filtered.out", args.tgt_lang, args.output_dir)
    elif args.func == 'format_translation_file':
        format_translation_file(args.output_dir, args.SRC_perturbed_type)


def format_input(output_dir, SRC_perturbed_type, src_lang, tgt_lang):
    input_df = pd.read_csv(f"{output_dir}/input.csv", index_col=0)
    if SRC_perturbed_type is None:
        input_sentences = input_df['SRC'].tolist()
    else:
        input_sentences = input_df['SRC_perturbed'].tolist()

    with open(f'{output_dir}/input.{src_lang}', 'w') as f:
        for line in input_sentences:
            f.write(f"{line}\n")

    # Create a dummy placeholder file for the target. We will not use this tho, just for fairseq compatibility
    with open(f'{output_dir}/input.{tgt_lang}', 'w') as f:
        for line in input_sentences:
            f.write(f"{line}\n")


def tokenize(input_file, lang, out_dir):
    # Read in the input line by line
    with open(input_file, 'r') as f:
        input_lines = f.readlines()
        input_lines = [line.rstrip() for line in input_lines]

    with open(f"{out_dir}/preprocessed.tok.{lang}", 'w') as f:
        with MosesTokenizer(lang) as tokenizer:
            for input_line in input_lines:
                f.write(f"{' '.join(tokenizer(input_line))}\n")


def detokenize(input_file, lang, out_dir):
    # Read in the input line by line
    with open(input_file, 'r') as f:
        input_lines = f.readlines()
        input_lines = [line.rstrip() for line in input_lines]
        input_token_lists = [line.split() for line in input_lines]

    with open(f"{out_dir}/trans_sentences.txt", 'w') as f:
        with MosesDetokenizer(lang) as detokenizer:
            for input_token_list in input_token_lists:
                f.write(f"{detokenizer(input_token_list)}\n")


def format_translation_file(output_dir, SRC_perturbed_type):
    with open(f"{output_dir}/trans_sentences.txt", 'r') as f:
        translations = f.readlines()
        translations = [line.rstrip() for line in translations]

    output_df = pd.read_csv(f"{output_dir}/input.csv", index_col=0)

    if SRC_perturbed_type is None:
        output_df["OriginalSRC-Trans"] = translations
    else:
        output_df["SRC_perturbed-Trans"] = translations

    output_df.to_csv(f"{output_dir}/translations.csv")


if __name__ == "__main__":
    main()
