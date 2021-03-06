import unicodedata
import re
from collections import Counter
import random
import pickle
import os
import numpy as np
import seaborn as sns
import pandas as pd
from matplotlib import pyplot as plt

SEED_VALUE = 42
random.seed(SEED_VALUE)
np.random.seed(SEED_VALUE)
os.environ['PYTHONHASHSEED'] = str(SEED_VALUE)


def process_text(filename, regex_ignore='[^\u0590-\u05FF\uFB1D-\uFB4F ]', is_hebrew=True, regularize=True):
    char_bigram_counts = Counter()
    char_unigram_counts = Counter()

    with open(filename, encoding='UTF-8') as f:
        counter = 0
        try:
            for line in f:
                counter += 1
                if counter % 5000 == 0:
                    print(f"{counter} lines read.")
                if not regex_ignore:
                    pattern = re.compile(regex_ignore)
                    s = pattern.sub('', line.upper())
                    if is_hebrew:
                        s = s.translate({1470: " "})
                        normalized = unicodedata.normalize('NFKD', s)
                        s = "".join([c for c in normalized if not unicodedata.combining(c)])
                        temp = ""
                        for char in s:
                            if ord(char) not in [1524, 1523, 1522, 1521, 1520, 1518, 1515, 1480, 1472, 1475, 1478]:
                                temp += char
                        s = temp
                else:
                    s = line.upper()

                line_length = len(s)

                # building frequency dict for biagram and unigram.
                if line_length > 0:
                    for i in range(line_length - 1):
                        char_bigram_counts[(s[i], s[i + 1])] += 1
                        char_unigram_counts[s[i]] += 1

                    # Add last character in line
                    char_unigram_counts[s[line_length - 1]] += 1
        except Exception as e:
            print(e)

    char_bigram_counts[(' ', ' ')] = 0

    # Map each unique character from text to an index and vice-versa
    i_c_map = dict(enumerate([q[0] for q in sorted(list(char_unigram_counts.items()), key=lambda x: x[0])]))
    c_i_map = {v: k for k, v in i_c_map.items()}

    # Create first-order transition matrix
    n = len(c_i_map)
    M = np.zeros((n, n))
    if regularize:
        M += 1
    # build the frequency matrix for the biagram letters
    for k in char_bigram_counts.keys():
        M[c_i_map[k[0]]][c_i_map[k[1]]] = char_bigram_counts[k]

    # Replace any zero rows with a uniform distribution
    # i.e. characters that appear exactly once at the end of a corpus
    zero_rows = np.where(M.sum(axis=1) == 0.)
    M[zero_rows, :] = 1
    row_sums = M.sum(axis=1)
    P = M / row_sums[:, np.newaxis]

    print('{0} uniform row(s) inputed for characters {1}'.format(zero_rows[0].size,
                                                                 [i_c_map[z] for z in zero_rows[0]]))

    return {'char_bigram_counts': char_bigram_counts,
            'char_unigram_counts': char_unigram_counts,
            'bigram_freq_matrix': M,
            'transition_matrix': P,
            'character_index_map': c_i_map,
            'index_character_map': i_c_map,
            }


def read_transition_mat(path, text_file, is_pickle, is_hebrew, regex_ignore):
    if is_pickle:
        with open(path + text_file + ".pickle", 'rb') as handle:
            freq_matrix = pickle.load(handle)
    else:
        freq_matrix = process_text(path + text_file, regex_ignore=regex_ignore, is_hebrew=is_hebrew)
        with open(path + text_file + ".pickle", 'wb') as handle:
            pickle.dump(freq_matrix, handle, protocol=pickle.HIGHEST_PROTOCOL)

    P = freq_matrix['transition_matrix']
    F = freq_matrix['bigram_freq_matrix']
    i_c_map = freq_matrix['index_character_map']
    c_i_map = freq_matrix['character_index_map']
    m = P.shape[0]
    return freq_matrix, P, F, i_c_map, c_i_map, m


def encrypt(plaintext, plaintext_alphabet, cipher_alphabet):
    assert set(list(plaintext)).issubset(
        set(list(plaintext_alphabet))), "Plaintext must only contain characters in Plaintext alphabet"
    assert len(plaintext_alphabet) == len(
        cipher_alphabet), "Cipher alphabet must have same number of characters as Plaintext alphabet"

    enc_key = dict(zip(list(plaintext_alphabet), list(cipher_alphabet)))
    acc = []
    for s in plaintext:
        acc += [enc_key[s]]
    ciphertext = "".join(acc)

    return {'cipher_alphabet': cipher_alphabet,
            'plaintext_alphabet': plaintext_alphabet,
            'ciphertext': ciphertext,
            }


def decrypt(ciphertext, plaintext_alphabet, cipher_alphabet):
    dec_key = dict(zip(list(cipher_alphabet), list(plaintext_alphabet)))
    acc = []
    for s in ciphertext:
        acc += [dec_key[s]]
    plaintext = "".join(acc)
    return {'cipher_alphabet': cipher_alphabet,
            'plaintext_alphabet': plaintext_alphabet,
            'plaintext': plaintext,
            }


def probs_checker(f, char_index_map, cipher, transition_mtx, print_output=False):
    n = len(cipher)
    probs = np.zeros(n - 1)

    k = []
    v = []

    for i, j in f.items():
        k.append(i)
        v.append(j)

    dec = decrypt(cipher, "".join(v), "".join(k))['plaintext']
    for i in range(n - 1):
        char_a = dec[i]
        char_b = dec[i + 1]
        idx_a = char_index_map[char_a]
        idx_b = char_index_map[char_b]
        probs[i] = transition_mtx[idx_a][idx_b]

    # Ensures numerical stability
    probs_sorted = np.sort(probs, kind='quicksort')

    return {"score": np.sum(np.log(np.array(probs_sorted))), "attempt": dec}


def similarity(s1, s2):
    assert len(s1) == len(s2), "Both strings must be same length"
    n = len(s1)
    num_matches = sum([1 if a == b else 0 for a, b in zip(s1, s2)])
    return num_matches / n


def solve_mcmc(ciphertext, usual_alphabet, code_space, trans_mtx, char_index_mapping, message_cleaned, iters=10000):
    # Initialize with a random mapping
    f = dict(zip(list(code_space), list(usual_alphabet)))

    scores = [0.0] * iters
    similarity_scores = [0.0] * iters
    mappings = []
    accepted = 0
    for i in range(0, iters):

        mappings += [f]

        # Create proposal from f by random transposition of 2 letters
        r1, r2 = np.random.choice(list(code_space), 2, replace=True)
        f_proposal = f.copy()
        f_proposal[r1] = f[r2]
        f_proposal[r2] = f[r1]

        # Decrypt using the current and proposed mapping
        current = probs_checker(f, char_index_mapping, ciphertext, trans_mtx)
        f_prob = current['score']

        f_proposal_plaus = probs_checker(f_proposal, char_index_mapping, ciphertext, trans_mtx)
        f_proposal_prob = f_proposal_plaus['score']

        # Decide to accept new proposal
        u = random.uniform(0, 1)
        if f_proposal_prob > f_prob:
            f = f_proposal.copy()
            scores[i] = f_proposal_prob
            accepted += 1
            if (i // 5) == 0:
                print("iter:", i, f_proposal_plaus['attempt'][:300])
        if u < np.exp(f_proposal_prob - f_prob):
            f = f_proposal.copy()
            scores[i] = f_proposal_prob
            accepted += 1
            if i % 1000 == 0:
                print("iter:", i, f_proposal_plaus['attempt'][:300])
            # Print out progress
        else:
            scores[i] = f_prob

        plains = []
        ciphers = []
        for k in sorted(f.keys()):
            ciphers += [k]
            plains += [f[k]]
        f_key = ("".join(plains), "".join(ciphers))
        f_dec = decrypt(ciphertext, f_key[0], f_key[1])['plaintext']
        similarity_scores[i] = similarity(message_cleaned, f_dec)

    print("total acceptances: ", accepted)

    # Save best mapping
    best_f = mappings[np.argmax(scores)]
    best_score = max(scores)
    plains = []
    ciphers = []

    for k in sorted(best_f.keys()):
        ciphers += [k]
        plains += [best_f[k]]

    best_key = ("".join(plains), "".join(ciphers))
    best_attempt = decrypt(ciphertext, best_key[0], best_key[1])['plaintext']

    # Save best mapping
    best_f_sim = mappings[np.argmax(similarity_scores)]
    best_score_sim = max(similarity_scores)
    plains = []
    ciphers = []
    for k in sorted(best_f_sim.keys()):
        ciphers += [k]
        plains += [best_f[k]]
    best_key_sim = ("".join(plains), "".join(ciphers))
    best_attempt_sim = decrypt(ciphertext, best_key_sim[0], best_key_sim[1])['plaintext']

    print("prob_score:", best_score)
    print('similairty_score: ', best_score_sim)

    return {'num_iters': iters,
            'plaintext': best_attempt,
            'plaintextbysim': best_attempt_sim,
            'best_sim_score': best_score_sim,
            'best_score': best_score,
            'best_key': best_f,
            'best_sim_key': best_key_sim,
            'scores': scores,
            'sim_scores': similarity_scores,
            'total_acceptances': accepted
            }


def plot_scores(scores, similarity_scores):
    fig, axs = plt.subplots(1, 2, figsize=(20, 10))
    axs[0].plot(scores)
    axs[1].plot(similarity_scores)
    axs[0].set_title('score function')
    axs[1].set_title('similarity function')
    fig.show()
    plt.show()


def plot_transition_matrix(P, i_c_map, is_hebrew=True):
    data = pd.DataFrame(P)
    idx_range = range(1, 28) if is_hebrew else range(1, 27)
    data.columns = ["Space"] + [i_c_map[i] for i in idx_range]
    data.index = ["Space"] + [i_c_map[i] for i in idx_range]
    plt.figure(figsize=(20, 10))
    sns.heatmap(data, cmap="Greens")
    plt.show()


def run(path, text_file, is_hebrew, is_pickle):
    ##DEFS
    PATTERN_HEBREW = '[^\u0590-\u05FF\uFB1D-\uFB4F ]'
    PATTERN_ENGLISH = '[^A-Z ]'
    regex_ignore = PATTERN_HEBREW if is_hebrew else PATTERN_ENGLISH
    pattern = re.compile(regex_ignore)
    message_heb = "???????????? ???????????? ?????? ?????????? ?????????? ?????????? ?????????? ???????? ???????? ?????????? ???? ???? ???? ????, ???????? ?????????? ??????, ???????? ?????????????? ?????????? ???????????????? ?????????? ???????????? ???????????? ???????? ?????????? ???????????? ??????????"
    message_eng = "ENTER HAMLET TO BE OR NOT TO BE THAT IS THE QUESTION WHETHER TIS NOBLER IN THE MIND TO SUFFER THE SLINGS AND ARROWS OF OUTRAGEOUS FORTUNE OR TO TAKE ARMS AGAINST A SEA OF TROUBLES AND BY OPPOSING END"
    HEB_ALPHABET = "?????????????????????????????????????????????????????? "
    EN_ALPHABET = "ABCDEFGHIJKLMNOPQRSTUVWXYZ "
    ALPHABET = HEB_ALPHABET if is_hebrew else EN_ALPHABET

    ## CLEAN MESSAGE
    message = message_heb if is_hebrew else message_eng
    message_cleaned = pattern.sub('', message.upper())
    letters_of_current_message = "".join(list(set(list(message_cleaned))))
    letters_missing = set(list(ALPHABET)).difference(set(list(letters_of_current_message)))
    if letters_missing:
        print(
            f"in your message you missed the following letters: {letters_missing} which may cause inaccuracy prediction")

    ## CREATE CIPHER
    tmp = list(ALPHABET)
    random.shuffle(tmp)
    cipher_alphabet = "".join(tmp)
    message_enc = encrypt(message_cleaned, ALPHABET, cipher_alphabet)
    ciphertext = message_enc['ciphertext']
    ## Get transition matrix
    freq_matrix, P, F, i_c_map, c_i_map, m = read_transition_mat(path, text_file, is_pickle, is_hebrew, regex_ignore)

    plot_transition_matrix(P, i_c_map, is_hebrew)

    init = list(ALPHABET).copy()
    random.shuffle(init)
    mcmc_results = solve_mcmc(ciphertext, ALPHABET, init, P, c_i_map, message_cleaned, iters=50000)
    print(f"ciphertext:\n\t{ciphertext}\n")
    print(f"attempted decryption: \n{mcmc_results['plaintext']}\n")
    print(f"attempted decryption (By sim): \n{mcmc_results['plaintextbysim']}\n")

    print(f"original message: \n {message_cleaned}")
    ground_truth_score = probs_checker(dict(zip(cipher_alphabet, ALPHABET)), c_i_map, ciphertext, P)
    print('score of true key:', ground_truth_score['score'])
    print('similarity score - best prob key:', similarity(message_cleaned, mcmc_results['plaintext']))
    print('similarity score - best sim score:', similarity(message_cleaned, mcmc_results['plaintextbysim']))

    plot_scores(mcmc_results['scores'], mcmc_results['sim_scores'])


## Read Text
PICKLE = True
IS_HEBREW = True

PATH = os.path.dirname(os.path.realpath(__file__))+"/"
TEXT_FILE = 'haaretz.txt'
run(PATH, TEXT_FILE, IS_HEBREW, PICKLE)
