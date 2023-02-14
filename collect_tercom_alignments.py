"""
Code taken from Unbabel with minor modifications
https://github.com/Unbabel/word-level-qe-corpus-builder
"""
import parse_pra_xml
import pandas as pd


def format_output(in_tercom_xml, mt_original: list, pe_original: list):
    """

    Args:
        in_tercom_xml: xml output alignment file from tercom jar execution
        mt_original: list of tokenized original mt sentences
        pe_original: list of tokenized original pe sentences
    Returns:
        edit_alignments: list of list of tuples of aligned indices.
    """

    # Parse tercom HTML
    mt_tokens, pe_tokens, edits, hters = parse_pra_xml.parse_file(in_tercom_xml)

    # Sanity check: Original and tercom files match in number of tokens
    # Note that we will not use the tokenized tercom outputs only the alignments
    nr_sentences = len(mt_tokens)
    for index in range(nr_sentences):
        assert len(mt_original[index]) == len([x for x in mt_tokens[index] if x]), \
            "Lengths do  not match"
        assert len(pe_original[index]) == len([x for x in pe_tokens[index] if x]), \
            "Lengths do  not match"

    edit_alignments = []
    for sent_index, sent_edits in enumerate(edits):

        pe_original_index = 0
        mt_original_index = 0
        edit_alignments_sent = []
        sent_edit_actions = []
        for edit in sent_edits:

            # Store edit action
            sent_edit_actions.append(edit.o)

            if edit.o == 'C':

                # Sanity check
                # NOTE: Tercom ignores unless -s is used
                if (
                    mt_original[sent_index][mt_original_index].lower() !=
                    pe_original[sent_index][pe_original_index].lower()
                ):
                    raise Exception("Reading Tercom xml failed")

                edit_alignments_sent.append((pe_original_index, mt_original_index))
                pe_original_index += 1
                mt_original_index += 1

            elif edit.o == 'S':

                edit_alignments_sent.append((pe_original_index, mt_original_index))
                pe_original_index += 1
                mt_original_index += 1

            elif edit.o == 'I':

                edit_alignments_sent.append((pe_original_index, pd.NA))
                pe_original_index += 1

            elif edit.o == 'D':

                edit_alignments_sent.append((pd.NA, mt_original_index))
                mt_original_index += 1

            else:
                raise Exception("Uknown edit %s" % edit.o)

        edit_alignments.append(edit_alignments_sent)

    return edit_alignments
