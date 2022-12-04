import argparse
import torch
import pandas as pd
import random
import logging
from transformers import T5ForConditionalGeneration, T5Tokenizer
import numpy as np


logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)

LOGGER = logging.getLogger(__name__)



def set_seed(seed=0):
    """Set the random seed for torch.
    Args:
        seed (int, optional): random seed. Default 0
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    # If CUDA is not available, this is silently ignored.
    torch.cuda.manual_seed_all(seed)


def main():
    def none_or_str(value: str):
        if value.lower() == 'none':
            return None
        return value

    parser = argparse.ArgumentParser(description='Translate with perturbation to source sentences.')
    parser.add_argument('--trans_direction', type=str, default="en2de")
    parser.add_argument('--beam', type=int, default=5)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--batch_size', type=int, default=5)
    parser.add_argument('--output_dir', type=str)
    parser.add_argument('--SRC_perturbed_type', type=none_or_str, help="[None|content|noun|verb|adjective|adverb]")
    args = parser.parse_args()
    print(args)

    set_seed(args.seed)

    LOGGER.info("Loading pretrained translation model")
    tokenizer = None
    device = None

    if args.trans_direction == 'en2de':
        src2tgt_model = torch.hub.load('pytorch/fairseq', 'transformer.wmt19.en-de', tokenizer='moses', bpe='fastbpe',
                                       checkpoint_file='model1.pt:model2.pt:model3.pt:model4.pt')
        src2tgt_model.eval()  # disable dropout
        src2tgt_model.cuda()  # move model to GPU
        src2tgt_model_type = 'fairseq'
    elif args.trans_direction == 'en2fr':
        src2tgt_model = torch.hub.load('pytorch/fairseq', 'transformer.wmt14.en-fr')
        src2tgt_model.eval()  # disable dropout
        src2tgt_model.cuda()  # move model to GPU
        src2tgt_model_type = 'fairseq'
    elif args.trans_direction == 'en2vi':
        # Use en-vi model at https://huggingface.co/NlpHUST/t5-en-vi-small
        if torch.cuda.is_available():
            device = torch.device("cuda")
            print('There are %d GPU(s) available.' % torch.cuda.device_count())
            print('We will use the GPU:', torch.cuda.get_device_name(0))
        else:
            print('No GPU available, using  CPU instead.')
            device = torch.device("cpu")

        src2tgt_model = T5ForConditionalGeneration.from_pretrained("NlpHUST/t5-en-vi-small")
        tokenizer = T5Tokenizer.from_pretrained("NlpHUST/t5-en-vi-small")
        src2tgt_model.to(device)
        src2tgt_model.eval()
        src2tgt_model_type = 'huggingface'
    else:
        raise RuntimeError(f"Direction {args.trans_direction} not available.")

    LOGGER.info("Loading in the SRC input")
    src_tgt_df = pd.read_csv(f"{args.output_dir}/input.csv", index_col=0)

    LOGGER.info("Translating ...")
    if args.SRC_perturbed_type is None:
        # Translate the original source sentences
        LOGGER.info("Translating original SRC sentences.")
        src_tgt_df["OriginalSRC-Trans"] = batch_translation(
            model=src2tgt_model,
            tokenizer=tokenizer,
            src_sentences=src_tgt_df['SRC'].tolist(),
            beam=args.beam, batch_size=args.batch_size,
            device=device, model_type=src2tgt_model_type
        )
    else:
        # Translate the perturbed sentences
        LOGGER.info("Translating perturbed SRC sentences.")
        src_tgt_df["SRC_perturbed-Trans"] = batch_translation(
            model=src2tgt_model,
            tokenizer=tokenizer,
            src_sentences=src_tgt_df["SRC_perturbed"].tolist(),
            beam=args.beam, batch_size=args.batch_size,
            device=device, model_type=src2tgt_model_type
        )

    LOGGER.info("Saving output")
    src_tgt_df.to_csv(f"{args.output_dir}/translations.csv")


def translate_single_batch_huggingface(model, tokenizer, src_sentences: list, beam: int, device) -> list:
    # Only compatible for huggingface T5 transformer models
    inputs = tokenizer(src_sentences, return_tensors="pt", padding=True).to(device)
    summary_ids = model.generate(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        max_length=128,
        num_beams=beam,
        repetition_penalty=2.5,
        length_penalty=1.0,
        early_stopping=True
    )
    outputs = tokenizer.batch_decode(summary_ids, skip_special_tokens=True)
    return outputs


def translate_single_batch_fairseq(model, src_sentences: list, beam: int) -> list:
    # Only compatible for fairseq models
    return model.translate(src_sentences, beam=beam)


def batch_translation(model, tokenizer, src_sentences: list, batch_size: int, beam: int, device, model_type) -> list:
    translations = []
    for i in range(0, len(src_sentences), batch_size):
        # Define the slice of sentences to translate in this batch
        start_slice = i
        if i + batch_size > len(src_sentences):
            stop_slice = len(src_sentences)
        else:
            stop_slice = i + batch_size

        # Translate and append to final translation list
        if model_type == 'fairseq':
            translations = translations + translate_single_batch_fairseq(model,
                                                                         src_sentences[start_slice:stop_slice],
                                                                         beam)
        elif model_type == 'huggingface':
            translations = translations + translate_single_batch_huggingface(model,
                                                                             tokenizer,
                                                                             src_sentences[start_slice:stop_slice],
                                                                             beam,
                                                                             device)
        else:
            raise RuntimeError(f'Model type {model_type} not found.')

    return translations


if __name__ == "__main__":
    main()
