#!/usr/bin/env python3

"""This extracts data from irisData.txt into a pickle file"""

import pickle

data = []

with open('irisData.txt', 'r') as ifh:
    for line in ifh:
        s = line.split(',')
        data.append({'sepLength': s[0],
                     'sepWidth': s[1],
                     'petLength': s[2],
                     'petWidth': s[3],
                     'type': s[4]})

with open('irisData.pickle', 'wb') as out:
    pickle.dump(data, out)
