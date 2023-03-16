from read_and_analyse_df import tokenization_per_dataset
import argparse
import pandas as pd
import pickle


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--original_translation_output_dir', type=str,
                        help='Folder containing the translation output')
    parser.add_argument('--data_root_path', type=str, default='data')
    parser.add_argument('--dataset', type=str)
    parser.add_argument('--src_lang', type=str)
    parser.add_argument('--tgt_lang', type=str)

    args = parser.parse_args()
    print(args)

    original_src_trans_df = pd.read_csv(f"{args.original_translation_output_dir}/translations.csv")
    tokenize_result = tokenization_per_dataset(args.dataset, args.data_root_path, args.src_lang, args.tgt_lang,
                                               original_src_trans_df['SRC'].tolist(),
                                               original_src_trans_df['SRC-Trans'].tolist()
                                               )
    original_src_trans_df['tokenized_SRC'] = tokenize_result['tokenized_srcs']
    original_src_trans_df['tokenized_SRC-Trans'] = tokenize_result['tokenized_translations']
    original_src_trans_df['tokenized_SRC'] = original_src_trans_df['tokenized_SRC'].apply(lambda x: ' '.join(x))
    original_src_trans_df['tokenized_SRC-Trans'] = original_src_trans_df['tokenized_SRC-Trans'].apply(lambda x: ' '.join(x))
    original_src_trans_df.to_csv(f"{args.original_translation_output_dir}/translations.csv")


if __name__ == "__main__":
    main()
