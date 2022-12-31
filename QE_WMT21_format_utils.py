import pandas as pd
import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--func', type=str, help="pass in the desired functionality")
    parser.add_argument('--input_src_path', type=str, default=None)
    parser.add_argument('--output_dir', type=str, default=None)
    parser.add_argument('--column_to_be_formatted', type=str, help="['SRC'|'SRC_perturbed']")
    parser.add_argument('--src_lang', type=str, default=None)
    parser.add_argument('--tgt_lang', type=str, default=None)
    parser.add_argument('--tmp_dir', type=str, default=None)

    args = parser.parse_args()
    print(args)

    if args.func == 'format_input':
        format_input(args.input_src_path, args.output_dir, args.column_to_be_formatted, args.src_lang, args.tgt_lang)
    elif args.func == 'format_translation_file':
        format_translation_file(args.input_src_path, args.output_dir, args.column_to_be_formatted)


def format_input(input_src_path, output_dir, column_to_be_formatted, src_lang, tgt_lang):
    input_df = pd.read_csv(input_src_path, index_col=0)
    input_sentences = input_df[column_to_be_formatted].tolist()

    with open(f'{output_dir}/input.{src_lang}', 'w') as f:
        for line in input_sentences:
            f.write(f"{line}\n")

    # Create a dummy placeholder file for the target. We will not use this tho, just for fairseq compatibility
    with open(f'{output_dir}/input.{tgt_lang}', 'w') as f:
        for line in input_sentences:
            f.write(f"{line}\n")


def format_translation_file(input_src_path, output_dir, column_to_be_formatted):
    with open(f"{output_dir}/trans_sentences.txt", 'r') as f:
        translations = f.readlines()
        translations = [line.rstrip() for line in translations]

    output_df = pd.read_csv(input_src_path, index_col=0)

    output_df[f"{column_to_be_formatted}-Trans"] = translations

    output_df.to_csv(f"{output_dir}/translations.csv")


if __name__ == "__main__":
    main()
