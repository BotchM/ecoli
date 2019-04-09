#!/usr/bin/python3

import sys
import random
import math

# Outputs a random integer, according to a multinomial
# distribution specified by probs.
def rand_multinomial(probs):
    # Make sure probs sum to 1
    assert(abs(sum(probs) - 1.0) < 1e-5)
    rand = random.random()
    for index, prob in enumerate(probs):
        if rand < prob:
            return index
        else:
            rand -= prob
    return 0

# Outputs a random key, according to a (key,prob)
# iterator. For a probability dictionary
# d = {"A": 0.9, "C": 0.1}
# call using rand_multinomial_iter(d.items())
def rand_multinomial_iter(iterator):
    rand = random.random()
    for key, prob in iterator:
        if rand < prob:
            return key
        else:
            rand -= prob
    return 0

class HMM():

    def __init__(self):
        self.num_states = 2
        self.prior = [0.5, 0.5]
        self.transition = [[0.999, 0.001], [0.01, 0.99]]
        self.emission = [{"A": 0.291, "T": 0.291, "C": 0.209, "G": 0.209},
                         {"A": 0.169, "T": 0.169, "C": 0.331, "G": 0.331}]

    # Generates a sequence of states and characters from
    # the HMM model.
    # - length: Length of output sequence
    def sample(self, length):
        sequence = []
        states = []
        rand = random.random()
        cur_state = rand_multinomial(self.prior)
        for i in range(length):
            states.append(cur_state)
            char = rand_multinomial_iter(self.emission[cur_state].items())
            sequence.append(char)
            cur_state = rand_multinomial(self.transition[cur_state])
        return sequence, states

    # Generates a emission sequence given a sequence of states
    def generate_sequence(self, states):
        sequence = []
        for state in states:
            char = rand_multinomial_iter(self.emission[state].items())
            sequence.append(char)
        return sequence

    # Computes the (natural) log probability of sequence given a sequence of states.
    # calculates the probability of a particular sequence of states and characters.
    # log(P(sequence, states))
    def logprob(self, sequence, states):
        ###########################################
        #Start your code
        p = math.log(self.prior[0])

        i = 0
        while i < len(sequence):
            seq = sequence[i]
            state = states[i]
            prev_state = states[i - 1]
            tran = self.transition[prev_state][state]
            emmis = self.emission[state][seq]

            if i > 0:
                p += math.log(tran)
            p += math.log(emmis)
            i += 1
        return p
        # End your code
        ###########################################


    # Outputs the most likely sequence of states given an emission sequence
    # - sequence: String with characters [A,C,T,G]
    # return: list of state indices, e.g. [0,0,0,1,1,0,0,...]
    '''
    def viterbi(HMM, Time, evidence)
    m = max value matrix
    prev = store all the previous paths
    D = domain of x
    '''
    def viterbi(self, sequence):
        ###########################################
        # Start your code
        M = [[0 for k in range(len(sequence))] for t in range(2)]
        prev = [[0 for k in range(len(sequence))] for t in range(2)]
        path = []
        pos = 0

        # calculate prev positions of each state and get path
        for t in range(0,len(sequence)):
            for i in range(2):
                for j in range(2):
                    #In a 2D matrix make sure to simultaneously check for each values
                    # at the respective index
                    # prob1 at current state (0) and prob2 at state (1)
                    prob1 = M[j % 1][t-1] + math.log(self.transition[j % 1][i]) + math.log(self.emission[i][sequence[t]])
                    prob2 = M[1][t-1] + math.log(self.transition[1][i]) + math.log(self.emission[i][sequence[t]])

                    # compare every probability from both sections to get the max values
                    if(max(prob1, prob2)) == prob2:
                        M[i][t] = prob2
                        prev[i][t] = j + 1 % 1
                    else:
                        M[i][t] = max(prob1, prob2)
                        prev[i][t] = j % 1

            #assign the state of the max value at the last postion of matrix
            if M[0][len(sequence) - 1] < M[1][len(sequence) - 1]:
                pos = 1

        # Store all the states according to the prev matrix
        for i in range(len(sequence)):
            path.append(pos)
            pos = prev[pos][len(sequence) - i - 1]

        states = path[::-1]
        return states
        # End your code
        ###########################################


def read_sequence(filename):
    with open(filename, "r") as f:
        return f.read().strip()

def write_sequence(filename, sequence):
    with open(filename, "w") as f:
        f.write("".join(sequence))

def write_output(filename, logprob, states):
    with open(filename, "w") as f:
        f.write(str(logprob))
        f.write("\n")
        for state in range(2):
            f.write(str(states.count(state)))
            f.write("\n")
        f.write("".join(map(str, states)))
        f.write("\n")

hmm = HMM()

file = sys.argv[1]
sequence = read_sequence(file)
viterbi = hmm.viterbi(sequence)
logprob = hmm.logprob(sequence, viterbi)
name = "my_"+file[:-4]+'_output.txt'
write_output(name, logprob, viterbi)
