import transformers
from transformers import AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoTokenizer
import argparse
import torch
from utils import write_text_file


def get_parser():
    """
    Generate a parameter parser
    """
    parser = argparse.ArgumentParser(description="Evaluate Translation for LLM's")
    parser.add_argument("--input_file", type=str, default="../../mustc_v3/en-de/data/tst2019/txt/tst2019.en")
    parser.add_argument("--target_file", type=str, default="../../mustc_v3/en-de/data/tst2019/txt/tst2019.de")
    parser.add_argument("--hyp_output_file", type=str, default="./tst2019.de")
    parser.add_argument("--tokenized_hyp_output_file", type=str)
    parser.add_argument("--logprobs_hyp_output_file", type=str)
    parser.add_argument("--shot", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--model", type=str, default="google/flan-t5-small")
    parser.add_argument("--cache_dir", type=str, default="/project/OML/skoneru/iwslt23/scripts/bloom/cache")
    return parser


def read_data(filepath):
    data = open(filepath, mode='r', encoding='utf-8', newline='\n').readlines()
    data = [x.rstrip() for x in data]
    return data


def write_data(data, output_file):
    with open(output_file, 'w') as (f):
        for item in data:
            sent = item.split(".")[0]
            f.write('%s\n' % sent)


def process_sent(gen_sent, prompt):
    prompt_len = len(prompt)
    gen_sent = gen_sent[prompt_len:]
    gen_sent = gen_sent.replace("\n", ".")
    return gen_sent


def divide_chunks(l, n):
    for i in range(0, len(l), n):
        yield l[i:i + n]


def context_prompt(src, tgt, shot):
    assert len(src) == len(tgt)
    cxt_prompt = []
    for i in range(len(src)):
        curr_src = src[i]
        prompt = ""
        for shot_id in range(shot):
            prev_src = src[i - shot_id - 1]
            prev_tgt = tgt[i - shot_id - 1]
            prompt += "Translate this into 1 . German:\n" + prev_src + "\n1." + prev_tgt
        prompt += "\n Translate this into 1. German:\n " + curr_src + "\n1."
        cxt_prompt.append(prompt)
        prompt = ""
        # cxt_prompt.append("Translate English into German\n English: " + prev_src + "\n German: " + prev_tgt +
        # "\nEnglish:  " + curr_src + "\nGerman: " )
        # cxt_prompt.append("Translate this into 1 . German:\n" + prev_src + "\n1." + prev_tgt + "\n Translate this
        # into 1. German:\n " + curr_src + "\n1." )
    assert len(cxt_prompt) == len(src)
    return cxt_prompt


def main(params):
    # os.environ['TRANSFORMERS_CACHE'] = '/project/OML/skoneru/iwslt23/cache/'
    transformers.set_seed(0)
    src = read_data(params.input_file)
    tgt = read_data(params.target_file)
    hyp_llm = []
    tokenized_hyp_llm = []
    logprobs_hyp_llm = []
    tokenizer = AutoTokenizer.from_pretrained(params.model)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # model = AutoModelForSeq2SeqLM.from_pretrained(params.model,
    # cache_dir="/project/OML/skoneru/iwslt23/scripts/bloom/cache/").to(device)
    model = AutoModelForSeq2SeqLM.from_pretrained(params.model,
                                                  load_in_8bit=True, device_map='auto')
    src_batches = list(divide_chunks(src, params.batch_size))
    tgt_batches = list(divide_chunks(tgt, params.batch_size))
    vocab = tokenizer.get_vocab()
    inv_vocab = {v: k for k, v in vocab.items()}
    bos_token = tokenizer.bos_token
    eos_token = tokenizer.eos_token
    pad_token = tokenizer.pad_token
    idx = 0
    for j in range(len(src_batches)):
        src_batch = src_batches[j]
        tgt_batch = tgt_batches[j]
        prompt = context_prompt(src_batch, tgt_batch, params.shot)
        inputs = tokenizer(prompt, return_tensors='pt', padding=True).to(device)
        outputs = model.generate(**inputs, max_new_tokens=256, output_scores=True, return_dict_in_generate=True)
        # Turn output logit score to log prob (apply softmax)
        sm = [torch.nn.functional.log_softmax(outputs['scores'][i], dim=1) for i in range(len(outputs['scores']))]

        tokenized_outputs = [[inv_vocab[tok_id.item()] for tok_id in sent]
                             for sent in outputs['sequences']]
        cleaned_tokenized_outputs = []
        log_probs = []
        for sent_idx in range(outputs['sequences'].shape[0]):
            cleaned_tokenized_outputs_per_sent = []
            log_probs_per_sent = []
            for tok_idx in range(outputs['sequences'][sent_idx].shape[0]):
                if (tokenized_outputs[sent_idx][tok_idx]
                        not in [bos_token, eos_token, pad_token]):
                    cleaned_tokenized_outputs_per_sent.append(tokenized_outputs[sent_idx][tok_idx])
                    log_probs_per_sent.append(
                        "{:.4f}".format(sm[tok_idx - 1][sent_idx][outputs['sequences'][sent_idx][tok_idx]].item()))
            cleaned_tokenized_outputs.append(cleaned_tokenized_outputs_per_sent)
            log_probs.append(log_probs_per_sent)
        cleaned_tokenized_outputs = [' '.join(x) for x in cleaned_tokenized_outputs]
        log_probs = [' '.join(x) for x in log_probs]

        hyps = tokenizer.batch_decode(outputs['sequences'], skip_special_tokens=True)

        hyp_llm.extend(hyps)
        tokenized_hyp_llm.extend(cleaned_tokenized_outputs)
        logprobs_hyp_llm.extend(log_probs)


    write_text_file(hyp_llm, params.hyp_output_file)
    write_text_file(tokenized_hyp_llm, params.tokenized_hyp_output_file)
    write_text_file(logprobs_hyp_llm, params.logprobs_hyp_output_file)

    return


if __name__ == '__main__':
    parser = get_parser()
    params = parser.parse_args()

    main(params)
