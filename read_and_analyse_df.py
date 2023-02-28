"""
Read in an output translations df and optionally analyse it (by appending columns to the df itself)
"""

import pandas as pd
from difflib import SequenceMatcher
import os
import edist.sed as sed
from sacremoses import MosesTokenizer, MosesDetokenizer
import argparse
from utils import set_seed, str_to_bool


# Code taken from https://gitlab.ub.uni-bielefeld.de/bpaassen/python-edit-distances/-/blob/master/sed_demo.ipynb
def levenshtein(s1, s2):
    """
    Calculate edit distance between s1 and s2
    """
    return sed.standard_sed(s1, s2)


def changes_spread(opcodes):
    """
    The longest distance between two changes
    E.g., 111111111
            x    x
          112112121
          longest distance is between the two marks
    Args:
        opcodes: the transformation to turn string a to string b

    Returns:

    """
    start_change = -1
    end_change = -1
    for opcode in opcodes:
        if opcode[0] != 'equal':
            start_change = opcode[1]
            break
    for opcode in reversed(opcodes):
        if opcode[0] != 'equal':
            end_change = opcode[2]
            break
    return max(0, end_change - start_change)


def highlight_in_capital(sentence_tokenized, highlight_positions):
    """
    Params:
        sentence_tokenized: tokenzied sentence
        highlight_positions: list of 2-sized tuples: [(p1, p2), (p3,p4), ...]
            where we want to highlight sentence[p1:p2], sentence[p3:p4]
    """
    highlighted_sentence = []

    last = 0  # index of the last position added to the new sentence
    for (start, stop) in highlight_positions:
        highlighted_sentence.extend(
            sentence_tokenized[last:start] + \
            [w.upper() for w in sentence_tokenized[start:stop]]
        )
        last = stop
    if last < len(sentence_tokenized):
        highlighted_sentence.extend(
            sentence_tokenized[last:]
        )
    return ' '.join(highlighted_sentence)


def two_chunk_changed(original_tokenized, changed_tokenized, opcodes,
                      chunk_max_length=1, spacy_model=None, w2v_model=None):
    """
    Args:
        original_tokenized:
        changed_tokenized:
        opcodes:
        chunk_max_length:
        spacy_model:
        w2v_model:

    Returns:
        is_two_chunk_changed: whether this sentence has only two chunk changes within the max length
        chunk_distance: the distance between the two changed chunks
        is_same_subtree: whether the two changed chunks is in the same subtree (only when chunk_max_length=1, i.e.,
        2 words changed)
        changes_similarity: similarities of the two changes
    """

    is_two_chunk_changed = False
    chunk_distance = pd.NA
    is_same_subtree = pd.NA
    changes_similarity = pd.NA

    changes_types = [o[0] for o in opcodes]

    # If not exactly two changes, return
    if not (all(changes_type == 'replace' or changes_type == 'equal' for changes_type in changes_types) and \
            changes_types.count('replace') == 2):
        return is_two_chunk_changed, chunk_distance, is_same_subtree, changes_similarity

    # Find the positions of the two changed chunks
    i_replace = [i for i, change in enumerate(changes_types) if change == "replace"]

    # If two changed chunks not have length less than chunk_max_length, return
    if not (opcodes[i_replace[0]][2] - opcodes[i_replace[0]][1] <= chunk_max_length and \
            opcodes[i_replace[1]][2] - opcodes[i_replace[1]][1] <= chunk_max_length):
        return is_two_chunk_changed, chunk_distance, is_same_subtree, changes_similarity

    # At this point, this should be a valid two_chunk within length change
    is_two_chunk_changed = True

    # Check if there is indeed an equal chunks in between of the two changed chunk
    # Calculate the distance between two chunks = the equal chunk in between
    i_equal_in_between = (i_replace[1] + i_replace[0]) // 2
    assert opcodes[i_equal_in_between][0] == 'equal'
    chunk_distance = opcodes[i_equal_in_between][2] - opcodes[i_equal_in_between][1]

    if spacy_model is not None:
        # In the two_chunk_changed case when chunk_max_length=1, i.e., only two words are changed
        # comparing to the original translation
        # Check if the two changed words are in the same sub tree of the dependency tree
        if (opcodes[i_replace[0]][4] - opcodes[i_replace[0]][3] == 1 and \
                opcodes[i_replace[1]][4] - opcodes[i_replace[1]][3] == 1):
            # Find the ancestors and children of the two changed words
            doc = spacy_model(' '.join(changed_tokenized))
            token1, token2 = None, None
            family1, family2 = None, None
            for token in doc:
                if token.text == changed_tokenized[opcodes[i_replace[0]][3]]:
                    token1 = token.text
                    family1 = list(token.ancestors) + list(token.children)
                    family1 = [t.text for t in family1]
                elif token.text == changed_tokenized[opcodes[i_replace[1]][3]]:
                    token2 = token.text
                    family2 = list(token.ancestors) + list(token.children)
                    family2 = [t.text for t in family2]

            if token1 is None or token2 is None:
                is_same_subtree = pd.NA
            else:
                if token1 in family2 or token2 in family1:
                    is_same_subtree = True
                else:
                    is_same_subtree = False

    # Calculate the senmatic similarity of the two changed words (cosine similarity in [-1, 1])
    if w2v_model is not None:
        # Can only calculate when only two single tokens are changed
        if (opcodes[i_replace[0]][4] - opcodes[i_replace[0]][3] == 1 and
                opcodes[i_replace[1]][4] - opcodes[i_replace[1]][3] == 1 and
                opcodes[i_replace[0]][2] - opcodes[i_replace[0]][1] == 1 and
                opcodes[i_replace[1]][2] - opcodes[i_replace[1]][1] == 1):

            original_word_1 = original_tokenized[opcodes[i_replace[0]][1]]
            changed_word_1 = changed_tokenized[opcodes[i_replace[0]][3]]

            original_word_2 = original_tokenized[opcodes[i_replace[1]][1]]
            changed_word_2 = changed_tokenized[opcodes[i_replace[1]][3]]

            if original_word_1 in w2v_model.index_to_key and original_word_2 in w2v_model.index_to_key and \
                    changed_word_1 in w2v_model.index_to_key and changed_word_2 in w2v_model.index_to_key:
                changes_similarity = [{'original_word': original_word_1,
                                       'changed_word': changed_word_1,
                                       'semantic_similarity': w2v_model.similarity(original_word_1, changed_word_1)},
                                      {'original_word': original_word_2,
                                       'changed_word': changed_word_2,
                                       'semantic_similarity': w2v_model.similarity(original_word_2, changed_word_2)}]

    return is_two_chunk_changed, chunk_distance, is_same_subtree, changes_similarity


def highlight_changes(original_tokenized, changed_tokenized, opcodes):
    """
    Params:
        original_tokenized: tokenized original sentence
        changed_tokenized: tokenized changed sentence
        opcodes: changes to get from `original_tokenized` to `changed_tokenized`
    Returns:
        original_sentence and changed_sentence with the changes highlighted in capital
    """

    highlighted_original_sentence_positions = []
    highlighted_changed_sentence_positions = []

    for opcode in opcodes:
        tag, i1, i2, j1, j2 = opcode[0], opcode[1], opcode[2], opcode[3], opcode[4]

        if tag != 'equal':
            highlighted_original_sentence_positions.append((i1, i2))
            highlighted_changed_sentence_positions.append((j1, j2))

    original_sentence_highlighted = highlight_in_capital(
        sentence_tokenized=original_tokenized,
        highlight_positions=highlighted_original_sentence_positions
    )

    changed_sentence_highlighted = highlight_in_capital(
        sentence_tokenized=changed_tokenized,
        highlight_positions=highlighted_changed_sentence_positions
    )

    return original_sentence_highlighted, changed_sentence_highlighted


def calculate_change(original_tokenized, changed_tokenized):
    """
    Args:
        original_tokenized:
        changed_tokenized:

    Returns:
        opcodes: changes to get from `original_tokenized` to `changed_tokenized`.
                including the `equal` tag
                displayed in index
        changes: changes to get from `original_tokenized` to `changed_tokenized`.
                does not include the `equal` tag
                displayed in word
    """
    opcodes = SequenceMatcher(None, original_tokenized, changed_tokenized).get_opcodes()

    # Convert the opcodes (displayed by word index) to changes in words
    changes = []
    for opcode in opcodes:
        tag, i1, i2, j1, j2 = opcode[0], opcode[1], opcode[2], opcode[3], opcode[4]
        if tag != 'equal':
            changes.append((tag, ' '.join(original_tokenized[i1:i2]), ' '.join(changed_tokenized[j1:j2])))

    return opcodes, changes


def load_alignment(path_prefix, return_type='word'):
    """

    Args:
        path_prefix: prefix of the alignment file
        return_type: 'word' or 'index'

    Returns:

    """
    alignment_file_path = f"{path_prefix}_{return_type}_alignment.txt"
    if not os.path.isfile(alignment_file_path):
        raise RuntimeError("Alignment file not exist.")

    else:
        with open(alignment_file_path) as f:
            lines = [line.rstrip() for line in f]

        translation_alignment = []
        for line in lines:
            word_pairs = line.split()
            word_pairs = [word_pair.split(
                '<sep>' if return_type == 'word' else '-'
            ) for word_pair in word_pairs]
            word_pairs = dict(word_pairs)
            if return_type == 'index':
                word_pairs = dict(
                    (int(k), int(v)) for k, v in word_pairs.items()
                )
            translation_alignment.append(word_pairs)
        return translation_alignment


def add_reason_of_change(alignment, changes, perturbed_src_word):
    """
    Args:
        alignment:
        changes:
        perturbed_src_word:

    Returns:
        Whether each change is due to perturbation
    """
    if type(changes) != list:
        return pd.NA
    elif perturbed_src_word not in alignment.keys():
        changes[0]['change_type'] = None
        changes[1]['change_type'] = None
    elif alignment[perturbed_src_word] == changes[0]['changed_word'] \
            and alignment[perturbed_src_word] == changes[1]['changed_word']:
        # Both changes are due to perturbation --> weird --> pass
        changes[0]['change_type'] = None
        changes[1]['change_type'] = None
    elif alignment[perturbed_src_word] != changes[0]['changed_word'] \
            and alignment[perturbed_src_word] != changes[1]['changed_word']:
        # Both changes NOT due to perturbation --> weird --> pass
        changes[0]['change_type'] = None
        changes[1]['change_type'] = None
    elif alignment[perturbed_src_word] == changes[0]['changed_word']:
        changes[0]['change_type'] = "perturbed"
        changes[1]['change_type'] = "not_perturbed"
    elif alignment[perturbed_src_word] == changes[1]['changed_word']:
        changes[0]['change_type'] = "not_perturbed"
        changes[1]['change_type'] = "perturbed"

    return changes


def pos_tag_not_perturbed_change(changes, spacy_model):
    """

    Args:
        changes: a pair of changes
        spacy_model: for POS, of the target language

    Returns:
        POS of the changes that is not due to perturbation
    """
    if type(changes) != list:
        return pd.NA
    elif changes[0]['change_type'] == "not_perturbed":
        doc = spacy_model(changes[0]['changed_word'])
        return [t.pos_ for t in doc][0]
    elif changes[1]['change_type'] == "not_perturbed":
        doc = spacy_model(changes[1]['changed_word'])
        return [t.pos_ for t in doc][0]
    return pd.NA


def fix_tokenization(tokenized_sentence):
    # Only for WMT21_DA_en2de data
    # Some of the sentences is tokenized differently in the labeled data. I.e., the last dot is not tokenized
    # Fix in order to syncronize with the labeled data
    if tokenized_sentence[-1] != '.':
        str_sentence = ' '.join(tokenized_sentence)
        str_sentence = str_sentence[:-1] + ' .'
        return str_sentence.split()
    else:
        return tokenized_sentence


def read_output_df(df_root_path, data_root_path, dataset, src_lang, tgt_lang, mask_type, beam, replacement_strategy,
                   ignore_case=False, no_of_replacements=1, seed=0, ref_available=False,
                   tokenize_sentences=False, reformat_for_src_tgt_alignment=False,
                   analyse_feature=[], chunk_max_length=1,
                   spacy_model=None, w2v_model=None,
                   use_src_tgt_alignment=False, winoMT=False):
    """

    Args:
        tokenize_sentences: whether to tokenize src and trans sentences and add those columns to the df
        reformat_for_src_tgt_alignment: whether to output the reformatted src-trans sentences to be used by awesome-align
        analyse_feature: a subset of
            [
            'highlight_changes',   # Highlight the changes in the translation in capital
            'edit_distance',
            'change_spread',       # Longest distance between 2 changes
            'two_chunks_analysis'
            ]
        use_src_tgt_alignment: whether to read in the alignment output by awesome-align and append to the df
    Returns:
        output_df with optionally the feature column appended
    """
    if winoMT:
        path_prefix = f"{df_root_path}/winoMT_asmetric/wmt19_winoMT_perturbed"
        output_df = pd.read_csv(f'{df_root_path}/winoMT_asmetric/wmt19_winoMT_perturbed_format.csv', index_col=0)
    else:
        path_prefix = f"{df_root_path}/{dataset}/{replacement_strategy}/beam{beam}_perturb{mask_type}/" \
                      f"{no_of_replacements}replacements/seed{seed}/translations"
        output_df = pd.read_csv(f"{path_prefix}.csv", index_col=0)

        original_trans_path_prefix = \
            f"{df_root_path}/{dataset}/original/translations"
        original_trans = pd.read_csv(
            f"{original_trans_path_prefix}.csv", index_col=0
        )
        original_trans['SRC_original_idx'] = original_trans.index

    if tokenize_sentences:
        tgt_tokenizer = MosesTokenizer(lang=tgt_lang)
        src_tokenizer = MosesTokenizer(lang=src_lang)

        print("Tokenize everything ...")
        if not winoMT:
            # Tokenizing the original src-trans
            # For WMT21 QE data, use the provided tokens from the data for original SRC and trans
            # and use sacremoses to tokenize the perturbed SRC and trans
            if dataset.startswith("WMT21_DA"):
                # Load the tokens from data
                if dataset.startswith("WMT21_DA_test"):
                    tokenized_src_file = f"{data_root_path}/wmt-qe-2021-data/{src_lang}-{tgt_lang}-test21/test21.tok.src"
                    tokenized_trans_file = f"{data_root_path}/wmt-qe-2021-data/{src_lang}-{tgt_lang}-test21/test21.tok.mt"
                elif dataset.startswith("WMT21_DA_dev"):
                    tokenized_src_file = f"{data_root_path}/wmt-qe-2021-data/{src_lang}-{tgt_lang}-dev/post-editing/{src_lang}-{tgt_lang}-dev/dev.src"
                    tokenized_trans_file = f"{data_root_path}/wmt-qe-2021-data/{src_lang}-{tgt_lang}-dev/post-editing/{src_lang}-{tgt_lang}-dev/dev.mt"
                else:
                    raise RuntimeError
                with open(tokenized_src_file, 'r') as f:
                    tokenized_srcs = f.readlines()
                    tokenized_srcs = [tokenized_src.strip().split() for tokenized_src in tokenized_srcs]
                with open(tokenized_trans_file, 'r') as f:
                    tokenized_translations = f.readlines()
                    tokenized_translations = [tokenized_trans.strip().split() for tokenized_trans in tokenized_translations]
                # Put tokens to the df
                original_trans['tokenized_SRC-Trans'] = tokenized_translations
                original_trans['tokenized_SRC'] = tokenized_srcs
            else:
                original_trans['tokenized_SRC-Trans'] = original_trans['SRC-Trans'].apply(
                    lambda x: tgt_tokenizer.tokenize(x, escape=False, aggressive_dash_splits=False)
                )
                original_trans['tokenized_SRC'] = original_trans['SRC'].apply(
                    lambda x: src_tokenizer.tokenize(x, escape=False, aggressive_dash_splits=False)
                )
        # Tokenizing the perturbed src-trans
        output_df['tokenized_SRC_perturbed'] = output_df['SRC_perturbed'].apply(
            lambda x: tgt_tokenizer.tokenize(x, escape=False, aggressive_dash_splits=False)
        )
        output_df['tokenized_SRC_perturbed-Trans'] = output_df['SRC_perturbed-Trans'].apply(
            lambda x: tgt_tokenizer.tokenize(x, escape=False, aggressive_dash_splits=False)
        )

        if 'REF' in output_df.columns:
            output_df['tokenized_REF'] = output_df['REF'].apply(
                lambda x: tgt_tokenizer.tokenize(x, escape=False, aggressive_dash_splits=False)
            )

    if reformat_for_src_tgt_alignment and not winoMT:
        assert tokenize_sentences
        # Output reformatted file for the original src-trans
        with open(f"{original_trans_path_prefix}_reformatted.txt", 'w') as file:
            for index, row in original_trans.iterrows():
                tokenized_src = ' '.join((row[f'tokenized_SRC']))
                tokenized_tgt = ' '.join((row[f'tokenized_SRC-Trans']))
                file.write(f"{tokenized_src} ||| {tokenized_tgt}\n")
        # Output reformatted file for the perturbed src-trans
        with open(f"{path_prefix}_reformatted.txt", 'w') as file:
            for index, row in output_df.iterrows():
                tokenized_src = ' '.join(row[f'tokenized_SRC_perturbed'])
                tokenized_tgt = ' '.join(row[f'tokenized_SRC_perturbed-Trans'])
                file.write(f"{tokenized_src} ||| {tokenized_tgt}\n")

    if use_src_tgt_alignment:
        if not winoMT:
            original_trans['original_trans_alignment'] = load_alignment(original_trans_path_prefix, return_type='word')
            original_trans['original_trans_alignment_index'] = load_alignment(original_trans_path_prefix, return_type='index')
        output_df['perturbed_trans_alignment'] = load_alignment(path_prefix, return_type='word')
        output_df['perturbed_trans_alignment_index'] = load_alignment(path_prefix, return_type='index')

    # Join to get the translation of the original sentences as well
    # First drop duplicate columns
    output_df.drop('SRC', axis=1, inplace=True)
    output_df = pd.merge(output_df, original_trans, how='left', on='SRC_original_idx')

    if 'mustSHE' in dataset:
        output_df = output_df.merge(pd.read_csv(
            f"data/MuST-SHE_v1.2/MuST-SHE-v1.2-data/tsv/MONOLINGUAL.fr_v1.2.tsv",
            sep='\t')[['ID', 'CATEGORY']],
                                    how='left', left_on='SRC_original_idx', right_on='ID'
                                    )

    # Convert columns with sentences to str type
    cols = ['SRC', 'REF', 'SRC_perturbed', 'SRC_perturbed-Trans', 'SRC-Trans']
    if not ref_available:
        cols.remove('REF')
    output_df[cols] = output_df[cols].astype(str)

    if ignore_case:
        output_df[cols] = output_df[cols].applymap(lambda x: x.lower())

    # Reorder the columns
    first_cols = ['SRC', 'original_word', 'perturbed_word', 'SRC_perturbed', 'SRC-Trans', 'SRC_perturbed-Trans']
    cols = first_cols + [col for col in output_df.columns if col not in first_cols]
    output_df = output_df[cols]

    # print(f"Original df shape: {output_df.shape}")
    # output_df = output_df.dropna()
    # print(f"After dropping none-perturbed sentences: {output_df.dropna().shape}")

    # ------------------------------------------- Analyse features -------------------------------------------
    if len(analyse_feature) > 0:
        assert tokenize_sentences
        print('Calculating the changes between translations of original SRC and perturbed SRC ...')
        # Calculate the changes, i.e., how to get from the original trans sentence
        # to the changed trans sentence
        # https://docs.python.org/3/library/difflib.html
        output_df['opcodes'], output_df['changes'] = zip(*output_df.apply(
            lambda x: calculate_change(x['tokenized_SRC-Trans'],
                                       x['tokenized_SRC_perturbed-Trans']
                                       ), axis=1))

    # ---------------------------------------------------------------------------------------------------------
    if 'highlight_changes' in analyse_feature:
        print('Highlighting the changes ...')
        # Highlight the changes in the trans sentences
        output_df["SRC-Trans"], output_df['SRC_perturbed-Trans'] \
            = zip(*output_df.apply(
            lambda x: highlight_changes(
                x['tokenized_SRC-Trans'],
                x['tokenized_SRC_perturbed-Trans'],
                x['opcodes']), axis=1
        ))

    # ---------------------------------------------------------------------------------------------------------
    if 'edit_distance' in analyse_feature:
        print('Calculating the edit distance ...')
        if replacement_strategy == 'word2vec_similarity':
            # SRC difference is the number of occurances of the word we perturb
            output_df["SRC-edit_distance"] = output_df.apply(
                lambda x: x['tokenized_SRC-Trans'].count(x['original_word']), axis=1)
        else:
            output_df["SRC-edit_distance"] = 1
        output_df['Trans-edit_distance'] = output_df.apply(
            lambda x: levenshtein(x['tokenized_SRC-Trans'], x['tokenized_SRC_perturbed-Trans']), axis=1)

        #         output_df["#TransChanges-#SrcChanges"] = output_df['Trans-edit_distance'] - output_df[
        #         'SRC-edit_distance']

        output_df["#TransChanges/SentenceLength"] = \
            output_df['Trans-edit_distance'] / output_df['tokenized_SRC-Trans'].apply(lambda x: len(x))

        # Analyse on group of changes on the same sentence
        if no_of_replacements > 1:
            additional_col = output_df.groupby(by='SRC_masked_index', axis=0)[['Trans-edit_distance']].std()
            output_df = output_df.join(additional_col, rsuffix='--SD')

    # ---------------------------------------------------------------------------------------------------------
    if 'change_spread' in analyse_feature:
        output_df["ChangesSpread"] = output_df.apply(
            lambda x: changes_spread(x['opcodes']), axis=1)

        output_df["ChangesSpread/SentenceLength"] = \
            output_df["ChangesSpread"] / output_df['tokenized_SRC-Trans'].apply(lambda x: len(x))

    # ---------------------------------------------------------------------------------------------------------
    if 'two_chunks_analysis' in analyse_feature:
        print("Two-chunks changed analysis")
        # See if only two chunks within given max size are changed,
        # and do some analysis on this special case
        output_df['TwoChunksChanged'], output_df['ChunkDistance'], \
        output_df["is_same_subtree"], output_df['changes_similarity'] \
            = zip(*output_df.apply(
            lambda x: two_chunk_changed(x['tokenized_SRC-Trans'],
                                        x['tokenized_SRC_perturbed-Trans'],
                                        x['opcodes'],
                                        chunk_max_length=chunk_max_length,
                                        spacy_model=spacy_model,
                                        w2v_model=w2v_model), axis=1
        ))

        if use_src_tgt_alignment:
            # In the case where two changes occurs and the two similarities is calculated,
            # find out which change is due to the perturbation
            print("Find out changes directly caused by perturbation using alignment")
            output_df['changes_similarity'] = output_df.apply(
                lambda x: add_reason_of_change(
                    alignment=x['perturbed_trans_alignment'],
                    changes=x['changes_similarity'],
                    perturbed_src_word=x['perturbed_word']
                ),
                axis=1
            )

            if spacy_model is not None:
                # Add POS tagging of the not-perturbed change
                output_df['not_perturbed_TGT_change_type'] = output_df['changes_similarity'].apply(
                    lambda x: pos_tag_not_perturbed_change(x, spacy_model))

        # Analyse on group of changes on the same sentence
        if no_of_replacements > 1:
            additional_col = output_df.groupby(by='SRC_masked_index', axis=0)[['TwoChunksChanged']].sum()
            output_df = output_df.join(additional_col, rsuffix='--total')

    return output_df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--df_root_path', type=str)
    parser.add_argument('--data_root_path', type=str, default='data')
    parser.add_argument('--src_lang', type=str, default="en")
    parser.add_argument('--tgt_lang', type=str, default="de")
    parser.add_argument('--replacement_strategy', type=str, default='word2vec_similarity',
                        help='[word2vec_similarity|masking_language_model_{unmasking_model}]. '
                             'The later option is context-based.')
    parser.add_argument('--number_of_replacement', type=int, default=5,
                        help='The number of replacement for 1 SRC word')
    parser.add_argument('--dataname', type=str)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--beam', type=int, default=5)
    parser.add_argument('--mask_type', type=str)
    parser.add_argument('--output_dir', type=str)
    parser.add_argument('--winoMT', type=str_to_bool, default=False)
    parser.add_argument('--use_src_tgt_alignment', type=str_to_bool)
    parser.add_argument('--tokenize_sentences', type=str_to_bool)
    parser.add_argument('--analyse_feature', nargs="*", type=str, default=[])

    args = parser.parse_args()
    print(args)

    # spacy_model = spacy.load("de_core_news_sm")
    # Loading these models in is time consuming
    # German word2vec model Facebook https://fasttext.cc/docs/en/crawl-vectors.html (cc.de.300.bin)
    # de_model = load_facebook_model("../data/cc.de.300.bin").wv
    # vi_model = load_facebook_model("../data/cc.vi.300.bin").wv

    if args.winoMT:
        args.mask_type = 'pronoun'
        args.number_of_replacement = 1

    output = read_output_df(df_root_path=args.df_root_path, data_root_path=args.data_root_path,
                            dataset=f"{args.dataname}_{args.src_lang}2{args.tgt_lang}",
                            src_lang=args.src_lang, tgt_lang=args.tgt_lang, mask_type=args.mask_type,
                            beam=args.beam, replacement_strategy=args.replacement_strategy, ignore_case=False,
                            no_of_replacements=args.number_of_replacement, seed=args.seed,
                            spacy_model=None, w2v_model=None,
                            use_src_tgt_alignment=args.use_src_tgt_alignment, tokenize_sentences=args.tokenize_sentences,
                            winoMT=args.winoMT, analyse_feature=args.analyse_feature)

    output.to_pickle(f'{args.output_dir}/analyse_{args.dataname}_{args.src_lang}2{args.tgt_lang}_{args.mask_type}.pkl')


if __name__ == "__main__":
    main()
