import pickle
import random
import re
from difflib import SequenceMatcher

from constants import LANGUAGES
from text_processing import process_text


def read_transition_mat(path, text_file, lang, regex_ignore):
    pickle_path = path + "pickles/" + text_file + ".pickle"
    try:
        with open(pickle_path, "rb") as handle:
            freq_matrix = pickle.load(handle)
    except FileNotFoundError:
        print(
            f"pickle file not found for {text_file}, writing pickle to: {pickle_path}"
        )
        freq_matrix = process_text(
            path + "text_files/" + text_file, regex_ignore=regex_ignore, is_hebrew=lang
        )
        with open(pickle_path, "wb") as handle:
            pickle.dump(freq_matrix, handle, protocol=pickle.HIGHEST_PROTOCOL)

    P = freq_matrix["transition_matrix"]
    F = freq_matrix["bigram_freq_matrix"]
    i_c_map = freq_matrix["index_character_map"]
    c_i_map = freq_matrix["character_index_map"]
    m = P.shape[0]
    return freq_matrix, P, F, i_c_map, c_i_map, m


def encrypt(plaintext, plaintext_alphabet, cipher_alphabet):
    assert set(list(plaintext)).issubset(
        set(list(plaintext_alphabet))
    ), "Plaintext must only contain characters in Plaintext alphabet"
    assert len(plaintext_alphabet) == len(
        cipher_alphabet
    ), "Cipher alphabet must have same number of characters as Plaintext alphabet"

    enc_key = dict(zip(list(plaintext_alphabet), list(cipher_alphabet)))
    acc = []
    for s in plaintext:
        acc += [enc_key[s]]
    ciphertext = "".join(acc)

    return {
        "cipher_alphabet": cipher_alphabet,
        "plaintext_alphabet": plaintext_alphabet,
        "ciphertext": ciphertext,
    }


def decrypt(ciphertext, plaintext_alphabet, cipher_alphabet):
    dec_key = dict(zip(list(cipher_alphabet), list(plaintext_alphabet)))
    acc = []
    for s in ciphertext:
        acc += [dec_key[s]]
    plaintext = "".join(acc)
    return {
        "cipher_alphabet": cipher_alphabet,
        "plaintext_alphabet": plaintext_alphabet,
        "plaintext": plaintext,
    }


def similarity(s1, s2):
    return SequenceMatcher(None, s1, s2).ratio()


def message_encrypton(lang, message):
    language = LANGUAGES.get(lang)
    regex_ignore = language.get("pattern")
    pattern = re.compile(regex_ignore)
    alphabet = language.get("alphabet")
    message_cleaned = pattern.sub("", message.upper())
    letters_of_current_message = "".join(list(set(list(message_cleaned))))
    letters_missing = set(list(alphabet)).difference(
        set(list(letters_of_current_message))
    )
    if letters_missing:
        print(
            f"When your message not consist all of the possible letteres from the alphabet, you may have inaccuracy prediction."
            f"in {message_cleaned} you dont have {letters_missing}."
        )
    ## CREATE CIPHER
    tmp = list(alphabet)
    random.shuffle(tmp)
    cipher_alphabet = "".join(tmp)
    message_enc = encrypt(message_cleaned, alphabet, cipher_alphabet)
    ciphertext = message_enc["ciphertext"]
    return alphabet, cipher_alphabet, ciphertext, message_cleaned, regex_ignore
