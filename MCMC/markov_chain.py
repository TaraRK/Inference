from __future__ import division
import sys
import random
import os 
import networkx as nx
import util

import json
import numpy as np
from matplotlib import pyplot as plt

def approx_markov_chain_steady_state(conditional_distribution, N_samples, iterations_between_samples):
    """
    Computes the steady-state distribution by simulating running the Markov
    chain. Collects samples at regular intervals and returns the empirical
    distribution of the samples.

    Inputs
    ------
    conditional_distribution : A dictionary in which each key is an state,
                               and each value is a Distribution over other
                               states.

    N_samples : the desired number of samples for the approximate empirical
                distribution

    iterations_between_samples : how many jumps to perform between each collected
                                 sample

    Returns
    -------
    An empirical Distribution over the states that should approximate the
    steady-state distribution.
    """
    distribution = util.Distribution()

    def selectRandomState():
        return random.choice(list(conditional_distribution.keys()))

    def getSampleAfterJumps(jumps):
        state = selectRandomState()
        for _ in range(jumps):
            if random.random() < 0.9 and conditional_distribution[state]:
                next_state = getNextState(state)
                state = next_state if next_state else state
            else:
                state = selectRandomState()
        return state

    def getNextState(current_state):
        next_states = conditional_distribution[current_state]
        total = sum(next_states.values())
        threshold = random.random() * total
        cumulative = 0
        for state, prob in next_states.items():
            cumulative += prob
            if cumulative >= threshold:
                return state
        return None

    all_samples = [getSampleAfterJumps(iterations_between_samples) for _ in range(N_samples)]
    for state in conditional_distribution.keys():
        distribution[state] = all_samples.count(state)

    distribution.normalize()
    return distribution

# Part D degree ranking

def approx_markov_chain_steady_state_by_degree(transition_matrix):
    """
    Estimates the importance of states in a Markov chain by considering both
    in-degree and out-degree of each state in the transition matrix.
    
    Parameters
    ----------
    transition_matrix : dict
        A dictionary representing the conditional distribution of states, 
        where each key is a state and its value is a distribution over other states.
    
    Returns
    -------
    A normalized util.Distribution representing the importance of each state.
    """
    state_importance = util.Distribution()

    # Collect all transitions in a flat list
    transitions = []
    for outgoing_links in transition_matrix.values():
        transitions.extend(outgoing_links)

    # Calculate in-degree and out-degree for each state
    for state, outgoing in transition_matrix.items():
        in_degree = sum(1 for link in transitions if link == state)
        out_degree = len(outgoing)
        state_importance[state] = in_degree + out_degree

    state_importance.normalize()
    return state_importance

def run_pagerank_degree(data_filename):
    """
    Runs the PageRank algorithm, and returns the empirical
    distribution of the samples.

    Inputs
    ------
    data_filename : a file with the weighted directed graph on which to run the Markov Chain

    Returns
    -------
    An empirical Distribution over the states that should approximate the
    steady-state distribution.
    """
    conditional_distribution = get_graph_distribution(data_filename)

    steady_state = approx_markov_chain_steady_state_by_degree(conditional_distribution)

    pages = conditional_distribution.keys()
    top = sorted( (((steady_state[page]), page) for page in pages), reverse=True )

    values_to_show = min(20, len(steady_state))
    print("Top %d pages from empirical distribution:" % values_to_show)
    for i in range(0, values_to_show):
        print("%0.6f: %s" %top[i])
    return steady_state

def get_graph_distribution(filename):
    G = nx.read_gml(filename)
    d = nx.to_dict_of_dicts(G)
    cond_dist = util.Distribution({k: util.Distribution({k_: v_['weight'] for k_,v_ in v.items()}) for k,v in d.items()})
    return cond_dist

def run_pagerank(data_filename, N_samples, iterations_between_samples):
    """
    Runs the PageRank algorithm, and returns the empirical
    distribution of the samples.

    Inputs
    ------
    data_filename : a file with the weighted directed graph on which to run the Markov Chain

    N_samples : the desired number of samples for the approximate empirical
                distribution

    iterations_between_samples : how many jumps to perform between each collected
                                 sample

    Returns
    -------
    An empirical Distribution over the states that should approximate the
    steady-state distribution.
    """
    conditional_distribution = get_graph_distribution(data_filename)

    steady_state = approx_markov_chain_steady_state(conditional_distribution,
                            N_samples,
                            iterations_between_samples)

    pages = conditional_distribution.keys()
    top = sorted( (((steady_state[page]), page) for page in pages), reverse=True )

    values_to_show = min(20, len(steady_state))
    print("Top %d pages from empirical distribution:" % values_to_show)
    for i in range(0, values_to_show):
        print("%0.6f: %s" %top[i])
    return steady_state

if __name__ == '__main__':
    if len(sys.argv) != 4:
        print("Usage: python markovChain.py <data file> <samples> <iterations between samples>")
        sys.exit(1)
    data_filename = sys.argv[1]
    N_samples = int(sys.argv[2])
    iterations_between_samples = int(sys.argv[3])

    page_rank = run_pagerank(data_filename, N_samples, iterations_between_samples)
    page_degree = run_pagerank_degree(data_filename)

    def generate_part_A_graph():
        kl_divergence = lambda p, q: sum([p[x] * np.log2(p[x]/q[x]) for x in p.keys()])
        divergences = []
        samples_to_test = [128, 256, 512, 1024, 2048, 4096, 8192]
        for num_samples in samples_to_test:
            p, q = run_pagerank(data_filename, num_samples, iterations_between_samples), run_pagerank(data_filename, num_samples, iterations_between_samples)
            divergence = kl_divergence(p, q)
            print(num_samples, divergence)
            divergences.append(divergence)
        plt.plot(samples_to_test, divergences, marker="o")
        plt.xlabel("Samples Used")
        plt.ylabel("KL-Divergences Between Emerical Distributions Formed From The Same Number of Samples")
        plt.title("Samples Used vs Divergence")
        plt.savefig("./results/Samples Used vs Divergence")
        plt.show()
    
    def save_part_B_rankings(page_rank):
        with open("./results/part_b.json", "w") as f:
            json.dump(page_rank, f)
    
    def generate_part_D_graph(page_rank: util.Distribution, page_degree: util.Distribution):
        # This is a plot of the page_rank scores vs the page_degree scores
        # pages = page_rank.keys()
        # page_rank_rankings = sorted( (((page_rank[page]), page) for page in pages), reverse=True )
        # page_degree_rankings = sorted( (((page_degree[page]), page) for page in pages), reverse=True )
        # fig = plt.figure()
        # ax1 = fig.add_subplot(111)
        # ax1.scatter(range(len(page_degree_rankings)), [page_rank_rankings[i][0] for i in range(len(page_rank_rankings))], s=10, c='b', marker="s", label='Page Rank')
        # ax1.scatter(range(len(page_degree_rankings)), [page_degree_rankings[i][0] for i in range(len(page_degree_rankings))], s=10, c='r', marker="o", label='Page Degree')
        # ax1.legend(loc='upper left')
        # plt.xlabel("Nth highest scoring")
        # plt.ylabel("Score")
        # plt.savefig("./figures/Page Rank vs Page Degree Scores")
        # plt.show()

        pages = page_rank.keys()
        page_rank_rankings = sorted( (((page_rank[page]), page) for page in pages), reverse=True )
        # page_degree_rankings = sorted( (((page_degree[page]), page) for page in pages), reverse=True )
        fig = plt.figure()
        ax1 = fig.add_subplot(111)
        x = [page_rank_rankings[i][0] for i in range(len(page_rank_rankings))]
        y = [page_degree[page_rank_rankings[i][1]] for i in range(len(page_rank_rankings))]
        ax1.scatter(x, y, s=10, c='b')
        # ax1.axline([min(x), min(y)], [max(x), max(y)])
        a, b = np.polyfit(x, y, 1)
        ax1.plot(x, a*np.array(x)+b)
        # labels = [page_rank_rankings[i][1] for i in range(len(page_rank_rankings))]
        # for i, label in enumerate(labels):
        #     ax1.annotate(label, (x[i], y[i]), fontsize=4)
        plt.xlabel("Page Rank Score")
        plt.ylabel("Page Degree Score")
        plt.title("Page Rank vs Page Degree Scores")
        plt.savefig("./results/Page Rank vs Page Degree Scores")
        plt.show()

    # generate_part_A_graph()
    # save_part_B_rankings(page_rank)
    generate_part_D_graph(page_rank, page_degree)
