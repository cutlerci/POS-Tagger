import re
import sys
from sys import argv

from collections import defaultdict

tagDict = defaultdict(int)
wordVsTagDict = defaultdict(dict)
highestProb = 0


def tagged(param):
    high_prob = 0
    selected_tag = ""
    for x in wordVsTagDict[param]:
        if (wordVsTagDict[param][x] / tagDict[x]) > high_prob:
            high_prob = wordVsTagDict[param][x] / tagDict[x]
            selected_tag = x
    return selected_tag


# Open and read the input test file and store it as a list of items.
with open(argv[1], 'r') as file:
    trainingData = file.read().split()
    file.close()

# Open and read the input file that should be tagged and store it as a list of items.
with open(argv[2], 'r') as file:
    dataToTag = file.read().split()
    file.close()

for index in range(0, len(trainingData)):
    if trainingData[index] == "[" or trainingData[index] == "]":
        continue
    else:
        phrase = (re.sub(r"([^\\])(/)(.)", r"\1***\3", trainingData[index])).split("***")
        word = phrase[0]
        tag = phrase[1]
        tagDict[tag] += 1  # This updates the Frequency we have seen the specific tag
        # This updates the frequency we have seen the tag associated to a specific word
        wordVsTagDict[word][tag] = (wordVsTagDict.setdefault(word, {})).setdefault(tag, 0) + 1
previous_tag = ""
index = 0
mostProbableTag = "NN"
for word in dataToTag:
    if word == "[":
        sys.stdout.write("\n[ ")
        continue
    elif word == "]":
        sys.stdout.write("]\n")
        continue
    else:
        if not(len(wordVsTagDict[word]) == 0):
            for items in wordVsTagDict[word]:
                if (wordVsTagDict[word][items]/tagDict[items]) > highestProb:
                    highestProb = wordVsTagDict[word][items]/tagDict[items]
                    mostProbableTag = items

            string = word + "/" + mostProbableTag + " "
            sys.stdout.write(string)
            highestProb = 0
            previous_tag = mostProbableTag
        else:
            if re.search(r"[0-9]", word):
                string = word + "/CD "
                previous_tag = "CD"
            elif len(word) > 2 and word[-2] == "l" and word[-1] == "y":
                string = word + "/RB "
                previous_tag = "RB"
            elif word[0].isupper() and word[-1] == "s":
                string = word + "/NNPS "
                previous_tag = "NNPS"
            elif word[0].isupper() and word[-1] != "s":
                string = word + "/NNP "
                previous_tag = "NNP"
            elif (not(word[0].isupper())) and word[-1] == "s":
                string = word + "/NNS "
                previous_tag = "NNS"
            elif previous_tag == "RB" or previous_tag == "TO":
                if len(word) > 3 and word[-3] == "i" and word[-2] == "n" and word[-1] == "g":
                    string = word + "/VBG "
                    previous_tag = "VBG"
                #else:
                 #   string = word + "/VB "
                  #  previous_tag = "VB"
            else:
                string = word + "/NN "
                previous_tag = "NN"
            sys.stdout.write(string)
    index += 1
