import numpy as np
import matplotlib.pyplot as plt

BACKGROUND = 'B'
MOTIF = 'M'
MOTIF_MIN_LEN = 1


class resultsCompare:
    def __init__(self):
        pass

    def plot_results(self, results):
        vals = np.matrix(results).T
        self.display_number_of_motifs_diff(vals)

    def display_number_of_motifs_diff(self, vals):
        region_num_diff = np.abs(vals[0, :] - vals[1, :])
        fig1, ax1 = plt.subplots()
        ax1.set_title('Number Of Motifs Difference')
        ax1.boxplot(region_num_diff.T)
        plt.show()

    def compare_found_expected(self, found, expected):
        found_regions_count = self.find_region_count(found)
        expected_regions_count = self.find_region_count(expected)
        match_percent = self.find_match_percent(expected, found)

        return (found_regions_count, expected_regions_count, match_percent)

    def find_match_percent(self, expected, found):
        match_count = 0
        max_len = max(len(expected), len(found))  # len should be the same at most cases
        min_len = min(len(expected), len(found))
        for i in range(min_len):
            if found[i] == expected[i]:
                match_count += 1
        return match_count / max_len

    def find_region_count(self, tested_seq):
        curr_letter = tested_seq[0]
        regions = []
        sizable_region_count = 0
        region_len = 0
        for i in range(len(tested_seq)):
            if tested_seq[i] == curr_letter:
                if curr_letter == MOTIF:
                    region_len += 1
                continue
            curr_letter = tested_seq[i]
            if curr_letter == BACKGROUND and region_len >= MOTIF_MIN_LEN:
                sizable_region_count += 1
            region_len = 0
        return sizable_region_count


if __name__ == '__main__':
    found = "BBBBMMMMMMBBBBBBBBBBB"

    expected = "BBBBMMMMMMBBBBBMMMMMBB"

    data_values = [
        ("BBBBMMMMBBBB", "BBBBMMMMBBBB"),
        ("BBBBMMMMBBBBMMMMBBBB", "BBBBMMMMBBBBBBBBBBBB"),
        ("BBBBMMMMBBBBMMMMBBBB", "BBBBMMMMBBBBBBBBBBBB"),
        ("BBBBMMMMBBBBMMMMBBBB", "BBBBMMMMBBBBBBBBBBBB"),
        ("BBBBMMMMBBBBMMMMBBBB", "BBBBMMMMBBBBBBBBBBBB"),
        ("BBBBMMMMBBBBMMMMBBBB", "BBBBMMMMBBBBBBBBBBBB"),
        ("BBBBMMMMBBBBMMMMBBBB", "BBBBMMMMBBBBBBBBBBBB"),
        ("BBBBMMMMBBBBMMMMBBBB", "BBBBMMMMBBBBBBBBBBBB"),
        ("BBBMMMBBBMMMBBBMMMBBBMMMBB", "BBBBBBBBBBBBBBBBBBBBBBBBBB"),
        ("BBBMMMBBBMMMBBBMMMBBB", "BBBBBBBBBBBBBBBBBBBBB"),
        ("BBBBBBBBBBBBBBBBBBBBBB", "BBBMMMBBBBBMMMBBMMMBBB")
    ]


    def split(word):
        return [char for char in word]


    comparer = resultsCompare()
    results = []
    for i in data_values:
        results.append(comparer.compare_found_expected(split(i[0]), split(i[1])))
    comparer.plot_results(results)
