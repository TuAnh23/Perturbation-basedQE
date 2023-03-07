import argparse
import pandas as pd
import numpy as np
from utils import load_text_file, write_text_file
import re


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--analyse_output_path', type=str)
    parser.add_argument('--src_lang', type=str)
    parser.add_argument('--tgt_lang', type=str)

    args = parser.parse_args()
    print(args)

    # Collect the best dev score
    collect_dev_df = pd.DataFrame(columns=['MCC', 'alignment_tool', 'masking_type', 'unmasking_model'])
    for alignment_tool in ['Tercom', 'Levenshtein']:
        lines = load_text_file(
            f'{args.analyse_output_path}/WMT21_DA_dev_{args.src_lang}2{args.tgt_lang}/qe_hyperparam_tuning_{alignment_tool}.txt'
        )
        lines = lines[:-2]
        assert len(lines) == 15
        for line in lines:
            masking_type = re.search('mask_type: MultiplePerSentence_(.*), unmasking_models', line).group(1)
            unmasking_model = re.search('unmasking_models: (.*), QE_params', line).group(1)
            score = float(re.search('score: (.*)', line).group(1))
            collect_dev_df = pd.concat([collect_dev_df,
                                    pd.DataFrame(
                                        {0: {'MCC': score, 'alignment_tool': alignment_tool,
                                         'masking_type': masking_type, 'unmasking_model': unmasking_model}}
                                    ).transpose()],
                                   ignore_index=True)

    # Collect test score
    collect_test_df = pd.DataFrame(columns=['MCC', 'alignment_tool', 'masking_type', 'unmasking_model'])
    for alignment_tool in ['Tercom', 'Levenshtein']:
        for masking_type in ['content', 'allWords', 'allTokens']:
            for unmasking_model in ["bert-large-cased-whole-word-masking", "bert-large-cased", "distilbert-base-cased", "roberta-base", "bert-base-cased"]:
                lines = load_text_file(f"{args.analyse_output_path}/WMT21_DA_test_{args.src_lang}2{args.tgt_lang}/MultiplePerSentence_{masking_type}_{unmasking_model}/quality_estimation_{alignment_tool}.log")
                assert len(lines) == 7
                line = lines[-1]
                score = float(re.search('best score (.*), best params', line).group(1))
                collect_test_df = pd.concat([collect_test_df,
                                            pd.DataFrame(
                                                {0: {'MCC': score, 'alignment_tool': alignment_tool,
                                                     'masking_type': masking_type, 'unmasking_model': unmasking_model}}
                                            ).transpose()],
                                           ignore_index=True)
    best_dev = write_stats(
        collect_dev_df,
        f"{args.analyse_output_path}/WMT21_DA_dev_{args.src_lang}2{args.tgt_lang}/collect_results.txt", split='dev'
    )
    write_stats(
        collect_test_df,
        f"{args.analyse_output_path}/WMT21_DA_test_{args.src_lang}2{args.tgt_lang}/collect_results.txt",
        split='test', best_dev=best_dev
    )

def write_stats(df, outfile, split, best_dev=None):
    with open(outfile, 'w') as f:
        for attr in ['alignment_tool', 'masking_type', 'unmasking_model']:
            groups = df.groupby(attr, as_index=False)
            nr_rows = None
            for attr_value, group in groups:
                f.write(f"{attr_value}: " + "{:.3f}".format(np.mean(group['MCC'].to_numpy())) + u' \u00B1 ' + "{:.3f}".format(np.std(group['MCC'].to_numpy())) + '\n')
                if nr_rows is None:
                    nr_rows = group.shape[0]
                else:
                    assert group.shape[0] == nr_rows

        if split == 'dev':
            best_row_idx = df['MCC'].astype(float).idxmax()
            best_row = df.loc[best_row_idx]
            f.write('Best:\n')
            f.write(best_row.to_string())
            return best_row
        else:
            assert best_dev is not None
            f.write('Best:\n')
            f.write(df[(df['alignment_tool'] == best_dev['alignment_tool']) &
                       (df['masking_type'] == best_dev['masking_type']) &
                       (df['unmasking_model'] == best_dev['unmasking_model'])].to_string())


if __name__ == "__main__":
    main()
