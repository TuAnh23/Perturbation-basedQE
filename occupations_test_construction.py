"""
Reconstruct the Occupation Test, using their given profession list.
Construct into WinoMT format, so that we can reuse the gender detection code from WinoMT
(thus will work with en-de, rather than en-es as in Occupation Test)

Occupation Test:
https://github.com/joelescudefont/genbiasmt/tree/master
https://aclanthology.org/W19-3821.pdf

WinoMT:
https://github.com/TuAnh23/mt_gender
(forked from original at https://github.com/gabrielStanovsky/mt_gender)
"""

import argparse

import pandas as pd

from utils import load_text_file


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--en_occupations_path', type=str)
    parser.add_argument('--output_path', type=str)
    args = parser.parse_args()
    print(args)

    all_data = []  # list of dict
    occupations = load_text_file(args.en_occupations_path)
    for occupation in occupations:
        for gender in ['male', 'female']:
            all_data.append(construct_sentence(occupation, gender))

    data_df = pd.DataFrame(all_data)
    data_df.to_pickle(args.output_path)


def construct_sentence(occupation, gender):
    """
    :param occupation: an English occupation
    :param gender: male/female
    :return: dict
    """
    pronoun = 'him' if gender == 'male' else 'her'
    article = 'an' if occupation[0] in ['a', 'e', 'i', 'o', 'u'] else 'a'
    sentence = f"I have known {pronoun} for a long time, my friend works as {article} {occupation}."
    sentence_tok = f"I have known {pronoun} for a long time , my friend works as {article} {occupation} ."
    occupation_single_index = len(sentence_tok.split()) - 2  # The last token (not including the dot)
    occupation_phrase_indices = list(
        range(
            len(sentence_tok.split()) - 1 - len(occupation.split()),
            len(sentence_tok.split()) - 1
        )
    )
    return {
        'gender': gender,
        'occupation': occupation,
        'sentence': sentence,
        'sentence_tok': sentence_tok,
        'occupation_single_index': occupation_single_index,
        'occupation_phrase_indices': occupation_phrase_indices
    }


if __name__ == "__main__":
    main()
