"""
Load and process a given dataset to a common format: a dataframe with SRC and REF columns (REF is optional), index
from 0 to n.
For WMT QE data, also the add the provided tokenized SRC (column tokenized_SRC)
"""

import argparse
import json

import pandas as pd
from html import unescape
import os
import tarfile
from utils import set_seed, str_to_bool


def remove_quotes(sentence):
    """
    Remove the begining and end quotes
    """
    if (sentence.startswith('\"') and sentence.endswith('\"')) or \
        (sentence.startswith('“') and sentence.endswith('”')):
        return sentence[1:-1]
    return sentence


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root_dir', type=str, default="data")
    parser.add_argument('--src_lang', type=str, default="en")
    parser.add_argument('--tgt_lang', type=str, default="de")
    parser.add_argument('--dataname', type=str, default="MuST-SHE-en2fr",
                        help="[MuST-SHE-en2fr|Europarl-en2de|IWSLT15-en2vi|wmt19-newstest2019-en2de|"
                             "masked_covost2_for_en2de|cherry-picked|mucow]")
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--output_dir', type=str)
    parser.add_argument('--dev', type=str_to_bool, help="Whether to create a tiny dataset for testing")
    args = parser.parse_args()
    print(args)

    set_seed(args.seed)

    if args.dataname == "MuST-SHE-en2fr":
        assert args.src_lang == 'en'
        # Load test translation data
        src_df = pd.read_csv(f"{args.data_root_dir}/MuST-SHE_v1.2/MuST-SHE-v1.2-data/tsv/MONOLINGUAL.fr_v1.2.tsv",
                             sep='\t')[['SRC', "REF"]]

    elif args.dataname == "Europarl-en2de":
        # Use test data from Europarl at https://www.statmt.org/europarl/archives.html
        assert args.src_lang == 'en'
        # Load test translation data
        with open(f"{args.data_root_dir}/common-test/ep-test.en", encoding="ISO-8859-1") as f:
            en_sentences = f.readlines()
            en_sentences = [line.rstrip() for line in en_sentences]
        with open(f"{args.data_root_dir}/common-test/ep-test.de", encoding="ISO-8859-1") as f:
            de_sentences = f.readlines()
            de_sentences = [line.rstrip() for line in de_sentences]
        src_df = pd.DataFrame(data={'SRC': en_sentences, 'REF': de_sentences})

    elif args.dataname == "IWSLT15-en2vi":
        assert args.src_lang == 'en'
        with open(f"{args.data_root_dir}/IWSLT15-en2vi/tst2013.en") as f:
            en_sentences = f.readlines()
            en_sentences = [line.rstrip() for line in en_sentences]
            en_sentences = [unescape(line) for line in en_sentences]
        with open(f"{args.data_root_dir}/IWSLT15-en2vi/tst2013.vi") as f:
            vi_sentences = f.readlines()
            vi_sentences = [line.rstrip() for line in vi_sentences]
            vi_sentences = [unescape(line) for line in vi_sentences]
        src_df = pd.DataFrame(data={'SRC': en_sentences, 'REF': vi_sentences})

    elif args.dataname == 'winoMT':
        assert args.src_lang == 'en'
        src_df = pd.read_csv(
            "../mt_gender/data/aggregates/en.txt", sep='\t', header=None, names=['gender', 'x', 'SRC', 'noun']
        )[['SRC']]

    elif args.dataname == 'mucow':
        assert args.src_lang == 'en'

        # Test data at https://github.com/Helsinki-NLP/MuCoW
        with open("../MuCoW/WMT2019/translation test suite/txt/en-de.text.txt") as f:
            en_sentences = f.readlines()
            en_sentences = [line.rstrip() for line in en_sentences]
        with open("../MuCoW/WMT2019/translation test suite/txt/en-de.ref.txt") as f:
            de_sentences = f.readlines()
            de_sentences = [line.rstrip() for line in de_sentences]
        src_df = pd.DataFrame(data={'SRC': en_sentences, 'REF': de_sentences})

    elif args.dataname == 'wmt19-newstest2019-en2de':
        assert args.src_lang == 'en'
        # Test data at https://www.statmt.org/wmt19/metrics-task.html
        # Load test translation data
        with open(f"{args.data_root_dir}/wmt19-submitted-data-v3/txt/sources/newstest2019-ende-src.en") as f:
            en_sentences = f.readlines()
            en_sentences = [line.rstrip() for line in en_sentences]
        with open(f"{args.data_root_dir}/wmt19-submitted-data-v3/txt/references/newstest2019-ende-ref.de") as f:
            de_sentences = f.readlines()
            de_sentences = [line.rstrip() for line in de_sentences]
        src_df = pd.DataFrame(data={'SRC': en_sentences, 'REF': de_sentences})

    elif args.dataname == 'occupations_test':
        assert args.src_lang == 'en'
        data_df = pd.read_pickle(f"{args.data_root_dir}/occupations_test.pkl")
        src_df = pd.DataFrame(data={'SRC': data_df['sentence'].tolist(),
                                    'tokenized_SRC': data_df['sentence_tok'].tolist()})

    elif args.dataname == "covost2_all":
        assert args.src_lang == 'en'
        # Collect the SRC sentences from the test split of all pairs
        src_df = pd.DataFrame(columns=['SRC'])
        data_dir = f"{args.data_root_dir}/covost2/EN-translations"

        for filename in os.listdir(data_dir):
            if filename.endswith(".tar.gz") and ('de' not in filename):
                unzipped_file_name = filename.replace(".tar.gz", "")

                # Extract the file if not yet done so
                if not os.path.exists(os.path.join(data_dir, unzipped_file_name)):
                    tar = tarfile.open(os.path.join(data_dir, filename))
                    tar.extractall(data_dir)
                    tar.close()

                tmp_df = pd.DataFrame()
                original_df = pd.read_csv(os.path.join(data_dir, unzipped_file_name), sep='\t')
                original_df = original_df[original_df['split'] == 'test']
                tmp_df['SRC'] = original_df['translation']

                src_df = pd.concat([src_df, tmp_df], axis=0, ignore_index=True)

        # Filter out the errornously long and short sentences
        sentence_lengths = src_df['SRC'].apply(lambda x: len(x))
        length_stats = sentence_lengths.describe(percentiles=[0.05, .95])
        src_df = src_df[(sentence_lengths < length_stats['95%']) & (sentence_lengths > length_stats['5%'])]

        # Remove empty sentences
        src_df = src_df[src_df['SRC'] != ""]

        # Remove the begining and end quotes for consistency
        src_df['SRC'] = src_df['SRC'].apply(lambda x: remove_quotes(x))

        # Reindex after filtering the data
        src_df.reset_index(drop=True, inplace=True)

    elif args.dataname == 'WMT21_DA_test':
        assert args.src_lang == 'en'
        with open(f"{args.data_root_dir}/wmt-qe-2021-data/{args.src_lang}-{args.tgt_lang}-test21/test21.src") as f:
            en_sentences = f.readlines()
            en_sentences = [line.rstrip() for line in en_sentences]

        with open(f"{args.data_root_dir}/wmt-qe-2021-data/{args.src_lang}-{args.tgt_lang}-test21/test21.tok.src") as f:
            en_sentences_tok = f.readlines()
            en_sentences_tok = [line.rstrip() for line in en_sentences_tok]

        src_df = pd.DataFrame(data={'SRC': en_sentences, 'tokenized_SRC': en_sentences_tok})

    elif args.dataname == 'WMT21_DA_dev':
        assert args.src_lang == 'en'
        with open(f"{args.data_root_dir}/wmt-qe-2021-data/{args.src_lang}-{args.tgt_lang}-dev/post-editing/{args.src_lang}-{args.tgt_lang}-dev/dev.src", 'r') as f:
            en_sentences = f.readlines()
            en_sentences = [line.rstrip() for line in en_sentences]

        with open(f"{args.data_root_dir}/wmt-qe-2021-data/{args.src_lang}-{args.tgt_lang}-dev/post-editing/{args.src_lang}-{args.tgt_lang}-dev/dev.src") as f:
            en_sentences_tok = f.readlines()
            en_sentences_tok = [line.rstrip() for line in en_sentences_tok]

        src_df = pd.DataFrame(data={'SRC': en_sentences, 'tokenized_SRC': en_sentences_tok})

    elif args.dataname.startswith('WMT20_HJQE_test'):
        # Original WMT20 data from https://github.com/facebookresearch/mlqe/tree/main/data
        # HJQE labels from https://github.com/ZhenYangIACAS/HJQE

        assert args.src_lang == 'en'
        en_sentences = pd.read_csv(
            f"{args.data_root_dir}/wmt-qe-2020-data/{args.src_lang}-{args.tgt_lang}-test20/test20.{args.src_lang}{args.tgt_lang}.df.short.tsv",
            sep='\t'
        )['original'].tolist()

        with open(f"{args.data_root_dir}/HJQE/{args.src_lang}-{args.tgt_lang}/test/test.tok.src") as f:
            en_sentences_tok = f.readlines()
            en_sentences_tok = [line.rstrip() for line in en_sentences_tok]

        src_df = pd.DataFrame(data={'SRC': en_sentences, 'tokenized_SRC': en_sentences_tok})

    elif args.dataname.startswith('WMT20_HJQE_dev'):
        # Original WMT20 data from https://github.com/facebookresearch/mlqe/tree/main/data
        # HJQE labels from https://github.com/ZhenYangIACAS/HJQE

        assert args.src_lang == 'en'
        en_sentences = pd.read_csv(
            f"{args.data_root_dir}/wmt-qe-2020-data/{args.src_lang}-{args.tgt_lang}/dev.{args.src_lang}{args.tgt_lang}.df.short.tsv",
            sep='\t'
        )['original'].tolist()

        with open(f"{args.data_root_dir}/HJQE/{args.src_lang}-{args.tgt_lang}/dev/dev.tok.src") as f:
            en_sentences_tok = f.readlines()
            en_sentences_tok = [line.rstrip() for line in en_sentences_tok]

        src_df = pd.DataFrame(data={'SRC': en_sentences, 'tokenized_SRC': en_sentences_tok})

    elif args.dataname == 'cherry-picked':
        assert args.src_lang == 'en'
        en_sentences = ['Women who are dieting can become iron deficient.',
                        'My new doctor got the medical degree from the KIT, which she worked hard for.']

        src_df = pd.DataFrame(data={'SRC': en_sentences})

    else:
        raise RuntimeError(f"Dataset {args.dataname} not available.")

    if args.dev:
        src_df = src_df[:5]

    src_df.to_csv(f"{args.output_dir}/src_df.csv")


if __name__ == "__main__":
    main()
