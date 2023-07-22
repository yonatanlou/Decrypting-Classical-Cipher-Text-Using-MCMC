import re
import unicodedata
from collections import Counter

import numpy as np

from constants import BAD_HEBREW_CHARS


def remove_hebrew_special_chars(line):
    line = line.translate({1470: " "})
    normalized = unicodedata.normalize("NFKD", line)
    line = "".join([c for c in normalized if not unicodedata.combining(c)])

    temp = ""
    for char in line:
        if ord(char) not in BAD_HEBREW_CHARS:
            temp += char

    return temp


def remove_special_characters(line, regex_ignore):
    if regex_ignore:
        return line.upper()

    pattern = re.compile(regex_ignore)
    line = pattern.sub("", line.upper())
    return line


def process_text(
    filename,
    regex_ignore="[^\u0590-\u05FF\uFB1D-\uFB4F ]",
    is_hebrew=True,
    regularize=True,
):
    char_bigram_counts = Counter()
    char_unigram_counts = Counter()

    with open(filename, encoding="UTF-8") as f:
        counter = 0
        try:
            for line in f:
                counter += 1
                if counter % 5000 == 0:
                    print(f"{counter} lines read.")

                line = remove_special_characters(line, regex_ignore)

                if is_hebrew:
                    line = remove_hebrew_special_chars(line)

                build_frequency_counts(line, char_bigram_counts, char_unigram_counts)

        except Exception as e:
            print(e)

    char_bigram_counts[(" ", " ")] = 0

    i_c_map, c_i_map = create_index_maps(char_unigram_counts)

    transition_matrix = create_transition_matrix(
        char_bigram_counts, c_i_map, regularize
    )

    return {
        "char_bigram_counts": char_bigram_counts,
        "char_unigram_counts": char_unigram_counts,
        "bigram_freq_matrix": char_bigram_counts,
        "transition_matrix": transition_matrix,
        "character_index_map": c_i_map,
        "index_character_map": i_c_map,
    }


def build_frequency_counts(line, char_bigram_counts, char_unigram_counts):
    line_length = len(line)
    if line_length > 0:
        for i in range(line_length - 1):
            char_bigram_counts[(line[i], line[i + 1])] += 1
            char_unigram_counts[line[i]] += 1

        # Add last character in line
        char_unigram_counts[line[line_length - 1]] += 1


def create_index_maps(char_unigram_counts):
    sorted_chars = sorted(list(char_unigram_counts.items()), key=lambda x: x[0])
    i_c_map = dict(enumerate([q[0] for q in sorted_chars]))
    c_i_map = {v: k for k, v in i_c_map.items()}
    return i_c_map, c_i_map


def create_transition_matrix(char_bigram_counts, c_i_map, regularize=True):
    n = len(c_i_map)
    M = np.zeros((n, n))
    if regularize:
        M += 1

    for k in char_bigram_counts.keys():
        M[c_i_map[k[0]]][c_i_map[k[1]]] = char_bigram_counts[k]

    zero_rows = np.where(M.sum(axis=1) == 0.0)
    M[zero_rows, :] = 1
    row_sums = M.sum(axis=1)
    P = M / row_sums[:, np.newaxis]

    print(
        "{0} uniform row(s) inputted for characters {1}".format(
            zero_rows[0].size, [c_i_map[z] for z in zero_rows[0]]
        )
    )

    return P
