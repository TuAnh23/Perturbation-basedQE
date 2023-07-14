import spacy
import argparse
from spacy.tokens import Doc
from utils import load_text_file, write_text_file


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--tok_text_path', type=str)
    parser.add_argument('--lang', type=str)
    parser.add_argument('--out_path', type=str)

    args = parser.parse_args()
    print(args)

    if args.lang == 'de':
        spacy_model = spacy.load("de_core_news_sm")
    elif args.lang == 'en':
        spacy_model = spacy.load("en_core_web_sm")
    else:
        raise RuntimeError(f"lang {args.lang} not available.")

    tok_sentences = load_text_file(args.tok_text_path)
    pos_tags = pos_tag(tok_sentences, spacy_model)
    write_text_file(pos_tags, args.out_path)


class WhitespaceTokenizer(object):
    def __init__(self, vocab):
        self.vocab = vocab

    def __call__(self, text):
        words = text.split()
        # All tokens 'own' a subsequent space character in this tokenizer
        spaces = [True] * len(words)
        return Doc(self.vocab, words=words, spaces=spaces)


def pos_tag(tok_sentences, nlp):
    """
    Args:
        tok_sentences: string
        nlp: spacy_model
    Returns:
        POS of every word in the sentence
    """
    nlp.tokenizer = WhitespaceTokenizer(nlp.vocab)
    pos_tags = []
    for tok_sentence in tok_sentences:
        doc = nlp(tok_sentence)
        pos_tags_sent = [t.pos_ for t in doc]
        assert len(pos_tags_sent) == len(tok_sentence.split())
        pos_tags.append(' '.join(pos_tags_sent))
    return pos_tags


if __name__ == "__main__":
    main()
