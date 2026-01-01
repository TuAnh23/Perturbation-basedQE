from tune_quality_estimation import load_gold_labels, flatten_list
import argparse
import pickle
from sklearn.metrics import log_loss, matthews_corrcoef, recall_score, precision_score, accuracy_score, f1_score


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str)
    parser.add_argument('--data_root_path', type=str)
    parser.add_argument('--src_lang', type=str)
    parser.add_argument('--tgt_lang', type=str)
    parser.add_argument('--qe_pred_labels_path', type=str)
    parser.add_argument('--qe_pred_scores_path', type=str)
    parser.add_argument('--task', type=str, default='trans_word_level_eval')
    parser.add_argument('--output_path', type=str)

    args = parser.parse_args()
    print(args)

    gold_labels = load_gold_labels(args.dataset, args.data_root_path, args.src_lang, args.tgt_lang, args.task)
    with open(args.qe_pred_labels_path, 'rb') as f:
        pred_labels = pickle.load(f)
    with open(args.qe_pred_scores_path, 'rb') as f:
        pred_scores = pickle.load(f)

    labels = ['OK', 'BAD']
    mcc = matthews_corrcoef(y_true=flatten_list(gold_labels), y_pred=flatten_list(pred_labels))
    recall = recall_score(flatten_list(gold_labels),
                          flatten_list(pred_labels), labels=labels, pos_label='BAD')
    precision = precision_score(flatten_list(gold_labels),
                                flatten_list(pred_labels), labels=labels, pos_label='BAD')
    f1_bad = f1_score(flatten_list(gold_labels),
                      flatten_list(pred_labels), labels=labels, pos_label='BAD')
    f1_ok = f1_score(flatten_list(gold_labels),
                     flatten_list(pred_labels), labels=labels, pos_label='OK')
    y_true = [1 if x == "OK" else 0 for x in flatten_list(gold_labels)]
    y_pred = [1 - float(x) for x in flatten_list(pred_scores)]  # take (1-x) because the prob is prob of "BAD"
    bceloss = log_loss(y_true=y_true, y_pred=y_pred)

    with open(args.output_path, 'w') as f:
        f.write(f"Matthews_corrcoef: {mcc}\n")
        f.write(f"bceloss: {bceloss}\n")
        f.write(f"Recall wrt BAD: {recall}\n")
        f.write(f"Precision wrt BAD: {precision}\n")
        f.write(f"F1 wrt BAD: {f1_bad}\n")
        f.write(f"F1 wrt OK: {f1_ok}\n")


if __name__ == "__main__":
    main()
