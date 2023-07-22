import os
import random

import numpy as np
from matplotlib import pyplot as plt

from utils import read_transition_mat, decrypt, similarity, message_encrypton


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def probs_checker(f, char_index_map, cipher, transition_mtx, print_output=False):
    n = len(cipher)
    probs = np.zeros(n - 1)

    k = []
    v = []

    for i, j in f.items():
        k.append(i)
        v.append(j)

    dec = decrypt(cipher, "".join(v), "".join(k))["plaintext"]
    for i in range(n - 1):
        char_a = dec[i]
        char_b = dec[i + 1]
        idx_a = char_index_map[char_a]
        idx_b = char_index_map[char_b]
        probs[i] = transition_mtx[idx_a][idx_b]

    # Ensures numerical stability
    probs_sorted = np.sort(probs, kind="quicksort")

    return {"score": np.sum(np.log(np.array(probs_sorted))), "attempt": dec}


def initialize_mapping(code_space, usual_alphabet):
    return dict(zip(list(code_space), list(usual_alphabet)))


def create_proposal(f, code_space):
    r1, r2 = np.random.choice(list(code_space), 2, replace=True)
    f_proposal = f.copy()
    f_proposal[r1] = f[r2]
    f_proposal[r2] = f[r1]
    return f_proposal


def decrypt_with_mapping(ciphertext, mapping, char_index_mapping, trans_mtx):
    return probs_checker(mapping, char_index_mapping, ciphertext, trans_mtx)


def accept_proposal(f, f_proposal, f_prob, f_proposal_prob):
    u = random.uniform(0, 1)
    if f_proposal_prob > f_prob:
        f = f_proposal.copy()
        return f, f_proposal_prob, True
    elif u < np.exp(f_proposal_prob - f_prob):
        f = f_proposal.copy()
        return f, f_proposal_prob, True
    else:
        return f, f_prob, False


def get_similarity_score(message_cleaned, f_dec):
    return similarity(message_cleaned, f_dec)


def get_best_mapping(ciphertext, scores, mappings):
    best_f = mappings[np.argmax(scores)]
    plains = []
    ciphers = []
    for k in sorted(best_f.keys()):
        ciphers += [k]
        plains += [best_f[k]]
    best_key = ("".join(plains), "".join(ciphers))
    return best_key, decrypt(ciphertext, best_key[0], best_key[1])["plaintext"]


def solve_mcmc(
    ciphertext,
    usual_alphabet,
    code_space,
    trans_mtx,
    char_index_mapping,
    message_cleaned,
    iters=10000,
):
    f = initialize_mapping(code_space, usual_alphabet)
    scores = [0.0] * iters
    similarity_scores = [0.0] * iters
    mappings = []
    accepted = 0
    last_accepted_iter = (
        -1000
    )  # Initialize with a value to prevent printing at the beginning

    for i in range(0, iters):
        mappings += [f]
        f_proposal = create_proposal(f, code_space)

        current = probs_checker(f, char_index_mapping, ciphertext, trans_mtx)
        f_prob = current["score"]
        f_proposal_plaus = probs_checker(
            f_proposal, char_index_mapping, ciphertext, trans_mtx
        )
        f_proposal_prob = f_proposal_plaus["score"]

        f, score, is_accepted = accept_proposal(f, f_proposal, f_prob, f_proposal_prob)
        scores[i] = score
        if is_accepted and (i - last_accepted_iter) >= 1000:
            last_accepted_iter = i
            print("iter:", i, f_proposal_plaus["attempt"][:300])

        plains = []
        ciphers = []
        for k in sorted(f.keys()):
            ciphers += [k]
            plains += [f[k]]
        f_key = ("".join(plains), "".join(ciphers))
        f_dec = decrypt(ciphertext, f_key[0], f_key[1])["plaintext"]
        similarity_scores[i] = get_similarity_score(message_cleaned, f_dec)

    best_key, best_attempt = get_best_mapping(ciphertext, scores, mappings)
    best_key_sim, best_attempt_sim = get_best_mapping(
        ciphertext, similarity_scores, mappings
    )

    return {
        "num_iters": iters,
        "plaintext": best_attempt,
        "best_text_by_similarity_score": best_attempt_sim,
        "best_sim_score": max(similarity_scores),
        "best_score": max(scores),
        "best_key": best_key,
        "best_sim_key": best_key_sim,
        "scores": scores,
        "sim_scores": similarity_scores,
        "total_acceptances": accepted,
    }


def plot_scores(scores, similarity_scores):
    fig, axs = plt.subplots(1, 2, figsize=(20, 10))
    axs[0].plot(scores)
    axs[1].plot(similarity_scores)
    axs[0].set_title("score function")
    axs[1].set_title("similarity function")
    fig.show()
    plt.show()


def summary(
    P,
    alphabet,
    c_i_map,
    cipher_alphabet,
    ciphertext,
    mcmc_results,
    message_cleaned,
    plot,
):
    print(f"ciphertext:\n{ciphertext}\n")
    print(f"attempted decryption: \n{mcmc_results['plaintext']}\n")
    print(
        f"attempted decryption (By similarity score): \n{mcmc_results['best_text_by_similarity_score']}\n"
    )
    print(f"original message: \n{message_cleaned}")
    ground_truth_score = probs_checker(
        dict(zip(cipher_alphabet, alphabet)), c_i_map, ciphertext, P
    )
    print("score of true key:", ground_truth_score["score"])
    print(
        "similarity score - best prob key:",
        similarity(message_cleaned, mcmc_results["plaintext"]),
    )
    print(
        "similarity score - best sim score:",
        similarity(message_cleaned, mcmc_results["best_text_by_similarity_score"]),
    )
    if plot:
        plot_scores(mcmc_results["scores"], mcmc_results["sim_scores"])


def run(path, text_file, lang, message, plot=False, iterations=10000):
    (
        alphabet,
        cipher_alphabet,
        ciphertext,
        message_cleaned,
        regex_ignore,
    ) = message_encrypton(lang, message)
    freq_matrix, P, F, i_c_map, c_i_map, m = read_transition_mat(
        path, text_file, lang, regex_ignore
    )

    init = list(alphabet).copy()
    random.shuffle(init)
    mcmc_results = solve_mcmc(
        ciphertext, alphabet, init, P, c_i_map, message_cleaned, iters=iterations
    )
    summary(
        P,
        alphabet,
        c_i_map,
        cipher_alphabet,
        ciphertext,
        mcmc_results,
        message_cleaned,
        plot,
    )
    return {"similarity": mcmc_results["sim_scores"], "scores": mcmc_results["scores"]}
