from __future__ import print_function

import sys
import os
import argparse
import requests
import xml.etree.ElementTree as ET
from collections import Counter
from possible_observations import possible_observations as AMINO_CHARS
import numpy as np

import pandas as pd

url = 'http://pdbtm.enzim.hu/data/pdbtmall'
ALPHA = 0.04


class Chain:
    def __init__(self, id, num_transmembrane_segments, type):
        self.id = id
        self.num_transmembrane_segments = num_transmembrane_segments
        self.type = type
        self.seq = ""
        self.regions = []


class Region:
    def __init__(self, seq_start, seq_end, pdb_start, pdb_end, type):
        self.seq_start = seq_start
        self.seq_end = seq_end
        self.pdb_start = pdb_start
        self.pdb_end = pdb_end
        self.type = type


def get_database(prefix='.'):
    if not prefix.endswith('/'):
        prefix += '/'
    print('Fetching database...', file=sys.stderr)
    r = requests.get(url, stream=True)
    print('Saving database...', file=sys.stderr)
    f = open('%s/pdbtmall' % prefix, 'w')
    for line in r.iter_lines():
        decoded_line = line.decode("utf-8")
        if line:
            f.write(decoded_line)
    r.close()
    f.close()


def build_database(fn, prefix):
    print('Unpacking database...', file=sys.stderr)
    f = open(fn)
    db = f.read()
    f.close()
    firstline = 1
    header = ''
    entries = []
    pdbids = []
    for l in db.split('\n'):
        if firstline:
            header += l
            firstline -= 1
            continue
        if 'PDBTM>' in l:
            continue
        if l.startswith('<?'):
            continue
        if l.startswith('<pdbtm'):
            a = l.find('ID=') + 4
            b = a + 4
            pdbids.append(l[a:b])
            entries.append(header)
        entries[-1] += '\n' + l
    if not prefix.endswith('/'):
        prefix += '/'
    if not os.path.isdir(prefix):
        os.mkdir(prefix)
    for entry in zip(pdbids, entries):
        f = open(prefix + entry[0] + '.xml', 'w')
        f.write(entry[1])
        f.close()


def read_chains(prefix='.'):
    chains = []
    if not prefix.endswith('/'):
        prefix += '/'
    tree = ET.parse('%s/pdbtmall' % prefix)
    root = tree.getroot()
    for chain in root.iter('{http://pdbtm.enzim.hu}CHAIN'):
        to_append = True
        chain_obj = Chain(id=chain.attrib['CHAINID'], num_transmembrane_segments=chain.attrib['NUM_TM'],
                          type=chain.attrib['TYPE'])
        for child in chain:
            tag = child.tag.split("{http://pdbtm.enzim.hu}", 1)[1]
            if tag == 'SEQ':
                chain_obj.seq = child.text
                seq = chain_obj.seq.replace(" ", "")
                if not all(c in AMINO_CHARS for c in seq):
                    to_append = False
                    break
            if tag == 'REGION' and child.attrib['type']:
                region_obj = Region(seq_start=child.attrib['seq_beg'], seq_end=child.attrib['seq_end'],
                                    pdb_start=child.attrib['pdb_beg'], pdb_end=child.attrib['pdb_end'],
                                    type=child.attrib['type'])
                chain_obj.regions.append(region_obj)
        if to_append:
            chains.append(chain_obj)
    return chains


def get_alpha_helix_subsequences(chains):
    res = []
    for chain in chains:
        if chain.type == "alpha":
            seq = chain.seq
            start_seq_positions = []
            end_seq_positions = []
            for region in chain.regions:
                if region.type == "H":
                    start_seq_positions.append(int(region.seq_start))
                    end_seq_positions.append(int(region.seq_end))
            start_seq_positions.sort()
            end_seq_positions.sort()
            seq = seq.replace(" ", "")
            for i in range(len(start_seq_positions)):
                alpha_helix_subseq = seq[start_seq_positions[i]-1:end_seq_positions[i]]
                res.append(alpha_helix_subseq)
    return res


def get_alpha_sequences(chains):
    res = []
    for chain in chains:
        if chain.type == "alpha":
            seq = chain.seq
            seq = seq.replace(" ", "")
            res.append(seq)
    return res



def find_motifs(seqs, k):
    """
    The function finds the possible motifs of a certain length in a list of sequences and the number of occurrences of each
    motif in all sequences.
    :param seqs: The list of sequences in string type.
    :param k: The length of the motifs searched.
    :return: A list of tuples with the seed and it's number of occurrences sorted by number of occurrences.
    """
    motifs = []
    motifs_count = Counter()
    for seq in seqs:
        motifs_count.update([seq[i:i + k] for i in range(len(seq) - k + 1)])
    for num in sorted(motifs_count.keys()):
        motifs.append((num, motifs_count[num]))
    return sorted(motifs, key=lambda tup: tup[1], reverse=True)


def build_e(seed, alpha):
    """
    The function creates the emmisions matrix.
    :param seed: the matrix is build.
    :param alpha: Softening parameter α: Set the initial emission probabilities for the motif states
    :return: 'log_e': The emissions matrix in log scale, 'ind' : the names of the states(B + M's by motif length).
    """
    with np.errstate(divide="ignore"):
        ind = ['B']
        len_amino_chars = len(AMINO_CHARS)
        e = [[1/len_amino_chars] * len_amino_chars]
        log_e = np.log(np.float_(e)).tolist()
        dic = {AMINO_CHARS[i]: i for i in range(len_amino_chars)}
        for i, letter in enumerate(seed):
            row = [alpha] * len_amino_chars
            row[dic[letter]] = 1 - (len_amino_chars-1) * alpha
            log_e.insert(2 + i, np.log(np.float_(row)))
            ind.insert(2 + i, 'M' + str(i + 1))
    log_e = pd.DataFrame(log_e, index=ind, columns=AMINO_CHARS)
    return log_e, ind



def build_t(p, ind):
    """
    The function receives the transition probability and builds the matrix.
    :param p: The probability from B to M1 and (1-p) from B back to B.
    :param ind: The names of the states.
    :return: A dataFrame containing the transitions matrix where the columns and rows are named after the states.
    """
    with np.errstate(divide="ignore"):
        k = len(ind)
        trans = pd.DataFrame(np.log(float(0)), ind, ind)
        lp = np.log(p)
        trans.loc['B']['B'] = np.log(1 - p)
        trans.loc['B']['M1'] = lp
        for i in range(1, k):
            if i == k - 1:
                trans.loc['M' + str(k - 1)]['B'] = np.log(float(1))
            else:
                trans.loc['M' + str(i)]['M' + str(i + 1)] = np.log(float(1))
    return trans


def get_seeds(seqs):
    """
    The function finds the most common seeds for several lengths and calculated the emissions and transitions matrices
     for each seed.
    :param seqs: The sequences to be learned.
    :param letters: The possible letters in the sequences ABC
    :return: Returns a dictionary where the keys are the k's(lengths of the seeds) and the value for each key is a list
     of tuples = (seed, emissions, transitions)
    """
    k_lengths = list(range(6, 20))
    final_tuples = {}
    motifs_dic = {}

    # find most common seeds
    seeds = [find_motifs(seqs, k) for k in k_lengths]
    for i in k_lengths:
        motifs_dic[i] = seeds[i - 6][0:5]

    # calculate emissions and transitions
    global_possible_occurrences = [sum([len(seq) - k + 1 for seq in seqs]) for k in k_lengths]
    for key in motifs_dic.keys():
        key_tuples = []
        for m in motifs_dic[key]:
            seed = m[0]
            emissions, ind = build_e(seed, ALPHA)
            p = m[1] / global_possible_occurrences[key - 6]
            transitions = build_t(p, ind)
            key_tuples.append((seed, emissions, transitions))
        final_tuples[key] = key_tuples

    return final_tuples


def get_alpha_helix_subseq_len_dist(alpha_helix_subsequences):
    num_alpha_helix_subsequences = len(alpha_helix_subsequences)
    alpha_helix_subsequences_lengths = {}

    for s in alpha_helix_subsequences:
        len_s = len(s)
        if len_s not in alpha_helix_subsequences_lengths.keys():
            alpha_helix_subsequences_lengths[len_s] = 1
        else:
            alpha_helix_subsequences_lengths[len_s] += 1

    max_alpha_helix_subsequence_len = max(alpha_helix_subsequences_lengths.keys())

    alpha_helix_subsequences_len_dist = [0] * max_alpha_helix_subsequence_len
    for key, val in alpha_helix_subsequences_lengths.items():
        alpha_helix_subsequences_len_dist[key - 1] = val / num_alpha_helix_subsequences

    return alpha_helix_subsequences_len_dist


def main():
    parser = argparse.ArgumentParser(
        description='Manages PDBTM databases. Automatically fetches the PDBTM database if no options are '
                    'specified. Run without any arguments, dbtool will retrieve the PDBTM database, '
                    'store it in pdbtm, and unpack it.')
    parser.add_argument('-d', '--db', default='pdbtmall', help='name of db file')
    parser.add_argument('-b', '--build-db', action='store_true', help='rebuild database from an existing pdbtmsall file (available at http://pdbtm.enzim.hu/data/pdbtmall)')
    parser.add_argument('directory', nargs='?', default='pdbtm', help='directory to store database in')
    parser.add_argument('-f', '--force-refresh', action='store_true', help='overwrite of existing db')
    args = parser.parse_args()

    if args.build_db:
        build_database(args.db, args.directory)
    else:
        if not os.path.isdir(args.directory):
            os.mkdir(args.directory)
        if args.force_refresh or not os.path.isfile('%s/%s' % (args.directory, args.db)):
            get_database(args.directory)
        build_database('%s/%s' % (args.directory, args.db), args.directory)

    chains = read_chains(args.directory)
    print(len(chains))
    print('Getting alpha helix subsequences...', file=sys.stderr)
    alpha_helix_subsequences = get_alpha_helix_subsequences(chains)
    print(len(alpha_helix_subsequences))

    alpha_sequences = get_alpha_sequences(chains)

    alpha_helix_subseq_dist = get_alpha_helix_subseq_len_dist(alpha_helix_subsequences)

    for i in range(len(alpha_helix_subseq_dist)):
        print("Len = " + str(i+1) + " Percent=" + str(round(alpha_helix_subseq_dist[i], 3)))

    # finds the possible letters in ABC
    letters = set()
    for seq in alpha_helix_subsequences:
        letters.update(seq)

    # Roee Liberman
    final_tuples = get_seeds(alpha_helix_subsequences)
    print(final_tuples)

    print('Done', file=sys.stderr)


if __name__ == '__main__':
    main()