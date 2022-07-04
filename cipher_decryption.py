
import numpy as np
import re
import copy
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
import random
import pickle
import os
SEED_VALUE = 42
random.seed(SEED_VALUE)
np.random.seed(SEED_VALUE)
os.environ['PYTHONHASHSEED']=str(SEED_VALUE)


def encrypt(plaintext, plaintext_alphabet, cipher_alphabet, blacklist={}):
    assert set(list(plaintext)).difference(blacklist).issubset(
        set(list(plaintext_alphabet))), "Plaintext must only contain characters in Plaintext alphabet"
    assert len(plaintext_alphabet) == len(
        cipher_alphabet), "Cipher alphabet must have same number of characters as Plaintext alphabet"

    enc_key = dict(zip(list(plaintext_alphabet), list(cipher_alphabet)))
    acc = []
    for s in plaintext:
        if s in blacklist:
            acc += s
        else:
            acc += [enc_key[s]]
    ciphertext = "".join(acc)

    return {'cipher_alphabet': cipher_alphabet,
            'plaintext_alphabet': plaintext_alphabet,
            'ciphertext': ciphertext,
            }


def decrypt(ciphertext, plaintext_alphabet, cipher_alphabet, blacklist={}):
    dec_key = dict(zip(list(cipher_alphabet), list(plaintext_alphabet)))
    acc = []
    for s in ciphertext:

        if s in blacklist:
            acc += [s]
        else:
            acc += [dec_key[s]]
    plaintext = "".join(acc)
    return {'cipher_alphabet': cipher_alphabet,
            'plaintext_alphabet': plaintext_alphabet,
            'plaintext': plaintext,
            }


def process_text(filename, regex_ignore='[^A-Z .]', regularize=True):
    char_bigram_counts = Counter()
    char_unigram_counts = Counter()

    with open(filename, encoding='UTF-8') as f:
        counter = 0
        for line in f:
            counter += 1
            if counter % 5000 == 0: print(f"{counter} lines read.")
            if regex_ignore != None:
                pattern = re.compile(regex_ignore)
                s = pattern.sub('', line.upper())
                s = re.sub(r'[\u0591-\u05BD\u05BF-\u05C2\u05C4-\u05C7]', '', s) #to remove most of the NIKUD in hebrew
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

                # Map each unique character from text to an index and vice-versa
    i_c_map = dict(enumerate([q[0] for q in sorted(list(char_unigram_counts.items()), key=lambda x: x[0])]))
    c_i_map = {v: k for k, v in i_c_map.items()}

    # Create first-order transition matrix
    n = len(c_i_map)
    M = np.zeros((n, n))
    if regularize:
        M += 1
    #build the frequency matrix for the biagram letters
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


def plausibility(f, char_index_map, cipher, transition_mtx, blacklist={}, print_output=False):
    n = len(cipher)
    probs = np.zeros(n - 1)
    q = len(transition_mtx)
    k = []
    v = []

    for i, j in f.items():
        k += [i]
        v += [j]

    dec = decrypt(cipher, "".join(v), "".join(k), blacklist)['plaintext']
    for i in range(n - 1):

        # If bigram contains a character not part of the encryption, assign it a uniform transition prob
        if dec[i] in blacklist or dec[i + 1] in blacklist:
            probs[i] = 1 / q
        else:
            probs[i] = transition_mtx[c_i_map[dec[i]]][c_i_map[dec[i + 1]]]

        # Ensures numerical stability
        probs_sorted = np.sort(probs, kind='quicksort')

    return {"score": np.sum(np.log(np.array(probs))), "attempt": dec}


# Accuracy score
def similarity(s1, s2):
    assert len(s1) == len(s2), "Both strings must be same length"
    n = len(s1)
    num_matches = sum([1 if a == b else 0 for a, b in zip(s1, s2)])
    return num_matches / n


def solve_mcmc(ciphertext, usual_alphabet, code_space, trans_mtx, char_index_mapping, iters=2500, skip_chars={}):
    # Initialize with a random mapping
    f = dict(zip(list(code_space), list(usual_alphabet)))

    scores = [0.0] * iters
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
        current = plausibility(f, char_index_mapping, ciphertext, trans_mtx, blacklist=skip_chars)
        f_prob = current['score']

        # Print out progress
        if i % 500 == 0:
            print("iter:", i, current['attempt'][:300])

        f_proposal_prob = plausibility(f_proposal, char_index_mapping, ciphertext, trans_mtx, blacklist=skip_chars)[
            'score']

        # Decide to accept new proposal
        u = random.uniform(0, 1)
        if f_proposal_prob > f_prob:
            f = f_proposal.copy()
            scores[i] = f_proposal_prob
            accepted += 1
        elif u < np.exp(f_proposal_prob - f_prob):
            f = f_proposal.copy()
            scores[i] = f_proposal_prob
            accepted += 1
        scores[i] = f_prob

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
    best_attempt = decrypt(ciphertext, best_key[0], best_key[1], blacklist=skip_chars)['plaintext']

    print("score:", best_score)
    return {'num_iters': iters,
            'plaintext': best_attempt,
            'best_score': best_score,
            'best_key': best_f,
            'scores': scores,
            'total_acceptances': accepted
            }





IS_HEBREW =  True
IS_PICKLED = True
TEXT_FILE = 'wikipedia_1000000.txt'
# TEXT_FILE = 'war-and-peace.txt'
# TRAIN_SIZE = re.findall(r'\d+',TEXT_FILE)[0] #in lines
message = "המלחמה ברצועת עזה נגמרה לגמרי ושלום עולמי שוכן בארץ ישראל אף על פי כך, נדקר טיפוס אחד, המצב בשווקים הדרדר משמעותית בחודש האחרון בעקבות מצב החסה בשטחים"
# message = "ENTER HAMLET TO BE OR NOT TO BE THAT IS THE QUESTION WHETHER TIS NOBLER IN THE MIND TO SUFFER THE SLINGS AND ARROWS OF OUTRAGEOUS FORTUNE OR TO TAKE ARMS AGAINST A SEA OF TROUBLES AND BY OPPOSING END"


PATTERN_HEBREW = '[^\u0590-\u05FF\uFB1D-\uFB4F ]'
PATTERN_ENGLISH = '[^A-Z ]'
PATTERN_LANGUAGE = PATTERN_HEBREW if IS_HEBREW else PATTERN_ENGLISH
pattern = re.compile(PATTERN_LANGUAGE)
HEB_ALPHABET = "אבגדהוזחטיכךלמםנןסעפףצץקרשת "
EN_ALPHABET = "ABCDEFGHIJKLMNOPQRSTUVWXYZ "
ALPHABET = HEB_ALPHABET if IS_HEBREW else EN_ALPHABET

message_cleaned = pattern.sub('', message.upper())

# Generate random code space
letters_of_current_message = "".join(list(set(list(message_cleaned))))
letters_missing = set(list(ALPHABET)).difference(set(list(letters_of_current_message)))
if letters_missing:
    print(f"in your message you missed the following letters: {letters_missing} which may cause inaccuracy prediction")
alphabet = ALPHABET

tmp = list(alphabet)
random.shuffle(tmp)
cipher_alphabet = "".join(tmp)

# Display the true key
message_enc = encrypt(message_cleaned, alphabet, cipher_alphabet,
                      #                      blacklist={' ',}
                      )
ciphertext = message_enc['ciphertext']


if not IS_PICKLED:
    # Compute english bigram frequencies from a reference text
    freq_matrix = process_text("text_files/"+TEXT_FILE, regex_ignore=PATTERN_LANGUAGE)
    P = freq_matrix['transition_matrix']
    F = freq_matrix['bigram_freq_matrix']
    i_c_map = freq_matrix['index_character_map']
    c_i_map = freq_matrix['character_index_map']
    # Store data (serialize)
    with open(f'pickles/freq_matrix_{TEXT_FILE.split(".")[0]}.pickle', 'wb') as handle:
        pickle.dump(freq_matrix, handle, protocol=pickle.HIGHEST_PROTOCOL)
else:
    # Load data (deserialize)
    with open(f'pickles/freq_matrix_{TEXT_FILE.split(".")[0]}.pickle', 'rb') as handle:
        freq_matrix = pickle.load(handle)
        P = freq_matrix['transition_matrix']
        F = freq_matrix['bigram_freq_matrix']
        i_c_map = freq_matrix['index_character_map']
        c_i_map = freq_matrix['character_index_map']
        m = P.shape[0]





init = list(alphabet).copy()
random.shuffle(init)
soln = solve_mcmc(ciphertext, alphabet, init, P, c_i_map, iters=20000)
print(f"ciphertext:\n\t{ciphertext}\n")
print(f"attempted decryption: \n{soln['plaintext']}\n")

print(f"original message: \n {message_cleaned}")
ground_truth_score = plausibility(dict(zip(cipher_alphabet, alphabet)), c_i_map, ciphertext, P)
print('score of true key:', ground_truth_score['score'])
print('similarity score:', similarity(message_cleaned, soln['plaintext']))

plt.plot(soln['scores'])
plt.title('score function')
plt.show()

#TODO
# 1. Refactor all of the code:
# a. encapsulate the code
# b. refactor the variables names.
# c. Make a main file which will run the whole program more easily (with a function similar to: run(IS_HEBREW, TEXT_FILE, is_pickled, message):
# /
# 2. check if the algorithm is right (read the article before hand)
# 3. Create the final paper.