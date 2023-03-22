"""
Run word-level quality estimation and output the predicted labels
Using supervised predictor-estimator model
https://github.com/Unbabel/OpenKiwi/releases/
https://unbabel.github.io/OpenKiwi/reproduce.html
https://unbabel.github.io/OpenKiwi/usage.html#training-and-pretraining

If using the wmt21 baseline model, add this line:
`self.config.model_name = "models/updated_models/xlm-roberta-base-finetuned"`
to line 149 in
/home/tdinh/miniconda3/envs/openkiwi/lib/python3.8/site-packages/kiwi/systems/encoders/xlmroberta.py
"""

from kiwi.lib.predict import load_system
import argparse
import pickle
import pandas as pd


def openkiwi_eval(model_path, tokenized_srcs, tokenized_trans, task):
    assert len(tokenized_srcs) == len(tokenized_trans)
    runner = load_system(model_path)
    predicted_labels = runner.predict(
        source=tokenized_srcs,
        target=tokenized_trans,
    )
    if task == 'trans_word_level_eval':
        return predicted_labels.target_tags_labels
    elif task == 'trans_word_level_eval':
        return predicted_labels.source_tags_labels
    else:
        raise RuntimeError(f"Unknown task {task}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--original_translation_output_dir', type=str,
                        help='Folder containing the translation output, including the log probabilities')
    parser.add_argument('--model_path', type=str)
    parser.add_argument('--task', type=str, default='trans_word_level_eval')
    parser.add_argument('--label_output_path', type=str)

    args = parser.parse_args()
    print(args)

    original_translation_df = pd.read_csv(f"{args.original_translation_output_dir}/translations.csv")
    pred_labels = openkiwi_eval(args.model_path, original_translation_df['tokenized_SRC'].tolist(),
                                original_translation_df['tokenized_SRC-Trans'].tolist(), args.task)

    with open(args.label_output_path, 'wb') as f:
        pickle.dump(pred_labels, f)


if __name__ == "__main__":
    main()
