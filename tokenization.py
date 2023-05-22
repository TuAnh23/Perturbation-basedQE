import argparse
import sys
import os
import subprocess
import tempfile
import re

from sacremoses import MosesTokenizer, MosesDetokenizer
import fugashi
import jieba

from utils import load_text_file, write_text_file


def jieba_tokenize(inlist):
    outlist = []
    for line in inlist:
        tokens = [tok[0] for tok in jieba.tokenize(line.strip())]
        out = ' '.join(tokens)
        out = re.sub('\s+', ' ', out)
        outlist.append(out.split())
    return outlist


def fugashi_tokenize(inlist):
    outlist = []
    tokenizer = fugashi.Tagger()
    for line in inlist:
        tokens = [word.surface for word in tokenizer(line.strip())]
        out = ' '.join(tokens)
        out = re.sub('\s+', ' ', out)
        outlist.append(out.split())
    return outlist


def flores_tokenize(language, inlist):
    print('Using flores tokenizer')
    indic_nlp_path='indic_nlp_resources'
    try:
        sys.path.extend([
            indic_nlp_path,
            os.path.join(indic_nlp_path, "src"),
        ])
        from indicnlp.tokenize import indic_tokenize
        from indicnlp.normalize.indic_normalize import IndicNormalizerFactory
    except:
        raise Exception(
            "Cannot load Indic NLP Library, make sure --indic-nlp-path is correct"
        )
    # create normalizer
    factory = IndicNormalizerFactory()
    normalizer = factory.get_normalizer(
        language, remove_nuktas=False,
    )
    factory = IndicNormalizerFactory()
    normalizer = factory.get_normalizer(
        language, remove_nuktas=False,
    )
    # normalize and tokenize
    outlist = []
    for line in inlist:
        line = normalizer.normalize(line)
        line = " ".join(indic_tokenize.trivial_tokenize(line, language))
        outlist.append(line.strip().split())
    return outlist


def moses_tokenize(lang, inlist):
    with tempfile.TemporaryDirectory() as tmp_folder:
        with open(f"{tmp_folder}/infile.txt", 'w') as f:
            for line in inlist:
                f.write(f"{line}\n")

        tokeniser_script = "../mosesdecoder/scripts/tokenizer/tokenizer.perl"
        perl_params = [tokeniser_script, '-l', lang, '-no-escape']
        with open(f"{tmp_folder}/outfile.txt", 'wb', 0) as fileout:
            with open(f"{tmp_folder}/infile.txt", 'r') as filein:
                subprocess.call(perl_params, stdin=filein, stdout=fileout)

        with open(f"{tmp_folder}/outfile.txt", 'r') as f:
            outlist = f.readlines()
            outlist = [line.strip().split() for line in outlist]

    return outlist


def moses_tokenize_python(lang, inlist):
    tokenizer = MosesTokenizer(lang=lang)
    outlist = [tokenizer.tokenize(x, escape=False, aggressive_dash_splits=False) for x in inlist]
    return outlist


def perform_tokenization(lang, inlist):
    if lang == 'zh':
        return jieba_tokenize(inlist)
    elif lang == 'ja':
        return fugashi_tokenize(inlist)
    elif lang == 'ne' or lang == 'si' or lang == 'ma' or lang == 'mr':
        return flores_tokenize(lang, inlist)
    else:
        return moses_tokenize(lang, inlist)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--text_file_path', type=str)
    parser.add_argument('--lang', type=str)
    parser.add_argument('--output_tok_path', type=str)

    args = parser.parse_args()
    print(args)

    lines = load_text_file(args.text_file_path)
    tok_list_lines = perform_tokenization(args.lang, lines)
    tok_lines = [' '.join(x) for x in tok_list_lines]
    write_text_file(tok_lines, args.output_tok_path)


if __name__ == "__main__":
    main()
