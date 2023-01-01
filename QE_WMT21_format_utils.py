import pandas as pd
import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--func', type=str, help="pass in the desired functionality")
    parser.add_argument('--input_src_path', type=str, default=None)
    parser.add_argument('--output_dir', type=str, default=None)
    parser.add_argument('--input_SRC_column', type=str, help="['SRC'|'SRC_perturbed']")
    parser.add_argument('--src_lang', type=str, default=None)
    parser.add_argument('--tgt_lang', type=str, default=None)
    parser.add_argument('--tmp_dir', type=str, default=None)

    args = parser.parse_args()
    print(args)

    if args.func == 'format_input':
        format_input(args.input_src_path, args.output_dir, args.input_SRC_column, args.src_lang, args.tgt_lang)
    elif args.func == 'format_translation_file':
        format_translation_file(args.input_src_path, args.output_dir, args.input_SRC_column)


def format_input(input_src_path, output_dir, input_SRC_column, src_lang, tgt_lang):
    input_df = pd.read_csv(input_src_path, index_col=0)
    input_sentences = input_df[input_SRC_column].tolist()

    with open(f'{output_dir}/input.{src_lang}', 'w') as f:
        for line in input_sentences:
            f.write(f"{line}\n")

    # Create a dummy placeholder file for the target. We will not use this tho, just for fairseq compatibility
    with open(f'{output_dir}/input.{tgt_lang}', 'w') as f:
        for line in input_sentences:
            f.write(f"{line}\n")


def format_translation_file(input_src_path, output_dir, input_SRC_column):
    with open(f"{output_dir}/trans_sentences.txt", 'r') as f:
        translations = f.readlines()
        translations = [line.rstrip() for line in translations]

    output_df = pd.read_csv(input_src_path, index_col=0)

    output_df[f"{input_SRC_column}-Trans"] = translations

    output_df.to_csv(f"{output_dir}/translations.csv")


if __name__ == "__main__":
    main()
