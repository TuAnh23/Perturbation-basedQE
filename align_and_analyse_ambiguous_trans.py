"""
Used in the case of `MultiplePerSentence` perturbation. I.e., perturb multiple words in the sentece one by one, each
with different replacements
Containing functions to align the different translations and analyse on which words have unstable translation,...
"""

import pandas as pd
import nltk
import edist.sed as sed


def cast_to_index(string_index):
    """
    In an aligned tuple, the items could either be the index of a word, or the character '-' denoting nothing is aligned
    """
    # removes blank spaces
    string_index = string_index.strip()

    if string_index == '-':
        return pd.NA
    else:
        return int(string_index)


def edist_alignment(tokenized_sentence1, tokenized_sentence2):
    """
    Return the list of tuples of aligned indices
    """

    alignment = sed.standard_sed_backtrace(tokenized_sentence1, tokenized_sentence2)
    # Reformat the output from editst
    alignment = str(alignment).replace('[', '').replace(']', '').split(', ')
    alignment = [x.split('vs.') for x in alignment]
    alignment = [(cast_to_index(x[0]), cast_to_index(x[1])) for x in alignment]

    return alignment


def reorder_according_to_alignment(tokenized_sentence1, tokenized_sentence2, alignment):
    """
    Given the alignment tuples, reorder the second sentence to align to the first sentence
    """
    reordered_tokenized_sentence2 = [pd.NA] * len(tokenized_sentence1)
    for alignment_tuple in alignment:
        sentence1_idx, sentence2_idx = alignment_tuple
        if (not pd.isnull(sentence1_idx)) and (not pd.isnull(sentence2_idx)):
            reordered_tokenized_sentence2[sentence1_idx] = tokenized_sentence2[sentence2_idx]
    return reordered_tokenized_sentence2


def nltk_pos_tag(word):
    return nltk.pos_tag([word])[0][1]


def is_content_tag(nltk_pos):
    content_tags_prefix = ['NN', 'V', 'JJ', 'PRP']  # Noun, verb, adj, adv (RB, but removed), pronoun
    for prefix in content_tags_prefix:
        if nltk_pos.startswith(prefix):
            return True
    return False


def uniquify(a_list):
    """
    Add suffix to distinguish duplicated values in list
    """
    seen = set()

    for item in a_list:
        fudge = 1
        newitem = item

        while newitem in seen:
            fudge += 1
            newitem = "{}_{}".format(item, fudge)

        yield newitem
        seen.add(newitem)


def align_src_tgt_translations(sentence_df):
    """
    Single sentence, single perturbed word, different replacements.
    Align different translations of the perturbed sentences to the original SRC sentence.
    Args:
        sentence_df: the dataframe containing the different translations of
            (Single sentence, single perturbed word, different replacements)
    Returns:
        result_df: the column labels are the tokenized original SRC sentence
            the row labels are the different replacements of the perturbed word.
            The first row is the translation of the original version
    """
    # Convert everything to lowercase
    sentence_df = sentence_df.copy()
    sentence_df['SRC'] = sentence_df['SRC'].apply(lambda x: x.lower())
    sentence_df['original_trans_alignment'] = sentence_df['original_trans_alignment'].apply(
        lambda x: dict(
            (k.lower(), v.lower()) for k, v in x.items()
        )
    )
    sentence_df['perturbed_trans_alignment'] = sentence_df['perturbed_trans_alignment'].apply(
        lambda x: dict(
            (k.lower(), v.lower()) for k, v in x.items()
        )
    )

    original_word = sentence_df['original_word'].values[0]
    original_trans_alignment = sentence_df['original_trans_alignment'].values[0]

    original_src_tokenized = sentence_df['tokenized_SRC'].values[0]
    original_word_index = original_src_tokenized.index(original_word)

    result_df = pd.DataFrame(
        index=[original_word] + sentence_df['perturbed_word'].tolist(),  # original translation meant to be in the first row
        columns=original_src_tokenized,
        # columns=['[MASK]' if i == original_word_index else original_src_tokenized[i]
        #          for i in range(len(original_src_tokenized))]
    )

    # Add the original translation in the first row
    result_df.iloc[0] = original_trans_alignment
    result_df.iloc[0, original_word_index] = \
        original_trans_alignment[original_word] if original_word in original_trans_alignment.keys() else pd.NA

    # Add the perturbed translation
    result_df_i_row = 1
    for index, row in sentence_df.iterrows():
        perturbed_word = row['perturbed_word']
        perturbed_trans_alignment = row['perturbed_trans_alignment']
        result_df.iloc[result_df_i_row] = perturbed_trans_alignment
        result_df.iloc[result_df_i_row, original_word_index] = \
            perturbed_trans_alignment[perturbed_word] if perturbed_word in perturbed_trans_alignment.keys() else pd.NA

        result_df_i_row = result_df_i_row + 1

    # Fix columns with same name (due to word occurs twice in a sentence)
    result_df.columns = uniquify(result_df.columns)

    return result_df


def align_translations_tgt_only(sentence_df):
    """
    Single sentence, single perturbed word, different replacements.
    Align different translations of the perturbed sentences to the original translation.
    Args:
        sentence_df: the dataframe containing the different translations of
            (Single sentence, single perturbed word, different replacements)
    Returns:
        result_df: the column labels are the tokenized original translation
            the row labels are the different replacements of the perturbed word
    """
    original_word = sentence_df['original_word'].values[0]
    original_trans_tokenized = sentence_df['tokenized_SRC-Trans'].values[0]

    result_df = pd.DataFrame(
        index=[original_word] + sentence_df['perturbed_word'].tolist(), # original translation meant to be in the first row
        columns=original_trans_tokenized
    )

    # Add the original translation in the first row
    result_df.iloc[0] = original_trans_tokenized

    # Add the perturbed translation
    result_df_i_row = 1
    for index, row in sentence_df.iterrows():
        alignment = edist_alignment(original_trans_tokenized, row['tokenized_SRC_perturbed-Trans'])
        result_df.iloc[result_df_i_row] = reorder_according_to_alignment(
            original_trans_tokenized, row['tokenized_SRC_perturbed-Trans'], alignment
        )
        result_df_i_row = result_df_i_row + 1

    # Fix columns with same name (due to word occurs twice in a sentence)
    result_df.columns = uniquify(result_df.columns)

    return result_df


def align_translations(sentence_df, align_type="src-trans"):
    """
    Single sentence, single perturbed word, different replacements.
    Params:
        sentence_df: the dataframe containing the different translations of
            (Single sentence, single perturbed word, different replacements)
        align_type: "src-trans" align the translations with the source sentence, using awesome-align
                    "trans-only" align the translations with each other, using edit distance
    Returns:
        result_df: the aligned translations of different perturbed src
    """

    count_original_word = sentence_df['original_word'].value_counts()
    assert count_original_word.shape[0] == 1  # Because this function is for a single group

    if align_type == "src-trans":
        return align_src_tgt_translations(sentence_df)
    elif align_type == "trans-only":
        return align_translations_tgt_only(sentence_df)
    else:
        raise RuntimeError('Invalid alignment type')


def analyse_single_sentence_single_perturbed_word(sentence_perturbed_word_df, align_type="trans-only",
                                                  filter_content_word=False, return_word_index=False,
                                                  consistence_trans_portion_threshold=0.8,
                                                  uniques_portion_for_noiseORperturbed_threshold=0.4):
    """
    Single sentence, single perturbed word, different replacements.
    Analyse each word in the original SRC (if align_type=="src-trans")
        or in the original trans (if align_type=="trans-only")
    Whether each word is:
    - `perturbed_or_noise_words`: having many different translations. either this word is noise, or it is the perturbed
        word (different translations due to different replacements)
    - `words_with_unstable_trans`: having a few different translations
    - `words_with_consistent_trans`: word with the same translations in almost all perturbed sentences
    Params:
        sentence_perturbed_word_df: the dataframe containing the different translations of
            (Single sentence, single perturbed word, different replacements)
        align_type: "src-trans" align the translations with the source sentence, using awesome-align
                    "trans-only" align the translations with each other, using edit distance
        filter_content_word: whether to consider only the content words
        return_word_index: return word index instead of the word itself
        consistence_trans_portion_threshold
        uniques_portion_for_noiseORperturbed_threshold:  # nr_uniques_trans / nr_replacements_non_nan
    Returns:
        result = {'perturbed_or_noise_words': list of words,
                  'words_with_unstable_trans': {word: {trans:trans' frequency}},
                  'words_with_consistent_trans': {word: trans}
                  }
    """
    # In the case of SRC-TGT alignment, the original version is counted as a replacement
    nr_replacements = sentence_perturbed_word_df.shape[0] if align_type == "trans-only" \
        else sentence_perturbed_word_df.shape[0] + 1 if align_type == "src-trans" else None

    aligned_trans = align_translations(sentence_perturbed_word_df, align_type=align_type)

    result = {'perturbed_or_noise_words': [],
              'words_with_unstable_trans': {},
              'words_with_consistent_trans': {}
              }

    for col_idx, col in enumerate(aligned_trans.columns):
        word = col.split('_')[0]

        if filter_content_word and align_type == "trans-only":
            #             print('NLTK pos tag only available for English, skip filtering content words.')
            filter_content_word = False

        if (not filter_content_word) or is_content_tag(nltk_pos_tag(word)):
            count_unique_translated_words = aligned_trans[col].value_counts(sort=True, ascending=False)
            if len(count_unique_translated_words) == 0:
                # There is no directly aligned translation for this SRC word
                # So consider this a noisily translated word
                if return_word_index:
                    result['perturbed_or_noise_words'].append(col_idx)
                else:
                    result['perturbed_or_noise_words'].append(col)
                continue

            nr_unique_words = count_unique_translated_words.shape[0]
            # What is the portion of times that the most common translation occurs?
            # E.g., different trans are "die die die der das" --> 0.6

            portion_most_common_trans = count_unique_translated_words.iloc[0] / aligned_trans[col].count()
            most_common_trans = count_unique_translated_words.index[0]

            original_trans = col if align_type == "trans-only" \
                else aligned_trans.iloc[0][col] if align_type == "src-trans" else None

            # trans_eval: one of the three values:
            # ['words_with_consistent_trans', 'words_with_unstable_trans', 'perturbed_or_noise_words']

            if portion_most_common_trans > consistence_trans_portion_threshold:
                # If there is one dominant translation
                # and the translation of the original SRC is the same  as the dominant translation
                # then the translation of this word is consistent
                trans_eval = 'words_with_consistent_trans'
                # if original_trans == most_common_trans:
                #     trans_eval = 'words_with_consistent_trans'
                # else:
                #     trans_eval = 'words_with_unstable_trans'
            else:
                # The case where there is no dominant translation
                if nr_unique_words > uniques_portion_for_noiseORperturbed_threshold * aligned_trans[col].count():
                    # If number of unique translations are large,
                    # then this is the column of the perturbed word or noise
                    trans_eval = 'perturbed_or_noise_words'
                else:
                    trans_eval = 'words_with_unstable_trans'

            # Store result and returns
            if trans_eval == 'words_with_consistent_trans':
                if return_word_index:
                    result['words_with_consistent_trans'][col_idx] = count_unique_translated_words.index[0]
                else:
                    result['words_with_consistent_trans'][col] = count_unique_translated_words.index[0]
            elif trans_eval == 'words_with_unstable_trans':
                if return_word_index:
                    result['words_with_unstable_trans'][col_idx] = count_unique_translated_words.to_dict()
                else:
                    result['words_with_unstable_trans'][col] = count_unique_translated_words.to_dict()
            elif trans_eval == 'perturbed_or_noise_words':
                if return_word_index:
                    result['perturbed_or_noise_words'].append(col_idx)
                else:
                    result['perturbed_or_noise_words'].append(col)
            else:
                raise RuntimeError

    return result


def analyse_single_sentence(sentence_df,
                            align_type="trans-only",
                            filter_content_word=False,
                            return_word_index=False,
                            consistence_trans_portion_threshold=0.8,
                            uniques_portion_for_noiseORperturbed_threshold=0.4
                            ):
    """
    Single sentence, different perturbed words, different replacements.
    Params:
        sentence_perturbed_word_df: the dataframe containing the different translations of
            (Single sentence, different perturbed words, different replacements)
        align_type: "src-trans" align the translations with the source sentence, using awesome-align
                    "trans-only" align the translations with each other, using edit distance
        filter_content_word: whether to consider only the content words
        return_word_index: return word index instead of the word itself
    Returns:
        ambiguous_word: word that have clustered translation at least once (among different word position perturbation)
        no_effecting_words: perturbed words that does not effect the translation of the ambiguous_word
        effecting_words: perturbed words that effect the translation of the ambiguous_word
        result = {ambiguous_word:
            {'no_effecting_words': no_effect_words,
             'effecting_words': effect_words}
        }
    """
    count_original_sentence_idx = sentence_df['SRC_original_idx'].value_counts()
    assert count_original_sentence_idx.shape[0] == 1  # Because this function is for a single group

    groups_by_perturbed_word = sentence_df.groupby("SRC_masked_index", as_index=False)

    collect_results = {}
    original_words = [group_by_perturbed_word.iloc[0]['original_word']
                      for _, group_by_perturbed_word in groups_by_perturbed_word]
    groups_by_perturbed_word = [group_by_perturbed_word for _, group_by_perturbed_word in groups_by_perturbed_word]
    original_words = uniquify(original_words)
    for original_word, group_by_perturbed_word in zip(original_words, groups_by_perturbed_word):
        collect_results[original_word] = analyse_single_sentence_single_perturbed_word(group_by_perturbed_word,
                                                                                       align_type=align_type,
                                                                                       filter_content_word=filter_content_word,
                                                                                       return_word_index=return_word_index,
                                                                                       consistence_trans_portion_threshold=consistence_trans_portion_threshold,
                                                                                       uniques_portion_for_noiseORperturbed_threshold=uniques_portion_for_noiseORperturbed_threshold
                                                                                       )

    # For all words, find the perturbed words that makes its trans ambiguous,
    # and the perturbed words that makes its trans consistence
    if return_word_index:
        if align_type == "trans-only":
            # Original translation tokens/words
            words = list(range(0, len(sentence_df.iloc[0]['tokenized_SRC-Trans'])))
        elif align_type == "src-trans":
            # Original src tokens/words
            words = list(range(0, len(sentence_df.iloc[0]['tokenized_SRC'])))
        else:
            raise RuntimeError(f"Unknown align_type {align_type}")
    else:
        if align_type == "trans-only":
            # Original translation tokens/words
            words = uniquify(sentence_df.iloc[0]['tokenized_SRC-Trans'])
        elif align_type == "src-trans":
            # Original src tokens/words
            words = uniquify(sentence_df.iloc[0]['tokenized_SRC'])
        else:
            raise RuntimeError(f"Unknown align_type {align_type}")

    result = {}
    for word in words:
        no_effect_words = []
        effect_words = []

        for original_word, collected_result in collect_results.items():
            if word in collected_result['words_with_unstable_trans']:
                effect_words.append(original_word)
            elif word in collected_result['words_with_consistent_trans']:
                no_effect_words.append(original_word)

        result[word] = {'no_effecting_words': no_effect_words,
                        'effecting_words': effect_words}

    return result
