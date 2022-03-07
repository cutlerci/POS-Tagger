# coding: utf-8
#
# Name: Charles Ian Cutler
# Date: 03/07/2022
# Class: CMSC 416 Introduction to Natural Language Processing
# The following program is for the third programming assignment in the course
# CMSC 416, Intro to Natural Language Processing, at Virginia Commonwealth University
#
# This program was created with the intention of learning how part of speech tagging works to associate
# tags with words. These tags represent the part of speech that the specific word is. Tags are usually
# found in a tag-set which contains a wide array of different tags for varying specifications of different
# parts of speech. We were tasked to create our own POS tagger and have it tag a text file.
# After which we grade it to see how accurate it was by using a provided key.
#
# The program is broken into two parts the first is tagger.py which should be run from the command line
# with exactly two arguments. The first argument should be a text file that contains a corpus that has been
# tagged with the parts of speech manually. This serves as the data to train the model.
# The second argument should also be a text file. Specifically it should contain the text that should be tagged
# by the part-of-speech [POS] tagger. The output is written to the standard output and can be directed towards
# a text file. It should be run at the command prompt as follows with the output piped into a file names as
# chosen by the user:
#
# COMMAND PROMPT SAMPLE: >>> python3 tagger.py pos-mod-key.txt pos-test.txt > pos-test-with-tags.txt
#
# The second part of the program is scorer.py which should be run from the command line with exactly
# two arguments. The first argument should be a text file that contains a corpus that has been
# tagged with the parts of speech, intended to be the output of tagger.py . The second argument should
# also be a text file. Specifically it should contain the key by which the input text file should be graded
# against for accuracy of tagging. The output is written to the standard output and can be directed towards
# a text file. It should be run at the command prompt as follows with the output piped into a file names as
# chosen by the user:
#
# COMMAND PROMPT SAMPLE: >>> python3 scorer.py pos-test-with-tags.txt pos-test-key.txt > output.txt
#
# After the two parts have been run the output.txt should contain an accuracy score and a confusion matrix.
# The accuracy score is a percentage of how accurate the tagged file is when compared to the key.
# The confusion matrix contains all the tags along the top and left side. The tags along the top
# represent the tag that was intended to be seen. The tags along the left side represent the tags that were actually
# seen. If a cell contains a nonzero number, that number represents the number of times we saw the tag in
# the row name position when we were expecting the tag in the column name position. A perfect score would thus
# result in non-zero values appearing only in the diagonal from top left to bottom right cells.
#
# Below is an example output when run where I have pasted the contents of output.txt which I did
# truncate but can be seen in full on the github or by running the program yourself!
#
# <><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>
# >>> python3 tagger.py pos-mod-key.txt pos-test.txt > pos-test-with-tags.txt
# >>> python3 scorer.py pos-test-with-tags.txt pos-test-key.txt > output.txt
#
# Accuracy score is : 88.04026467689708 %
# ╔══════╦══════╦══════╦══════╦════╦════╦══════╦══════╦═════╦═════╦════╦═════╦══════╦══════╦══════╦══════╦═════╦═════╦
# ║      ║  CC  ║  CD  ║  DT  ║ EX ║ FW ║  IN  ║  JJ  ║ JJR ║ JJS ║ LS ║  MD ║  NN  ║ NNS  ║ NNP  ║ NNPS ║ PDT ║ POS ║
# ╠══════╬══════╬══════╬══════╬════╬════╬══════╬══════╬═════╬═════╬════╬═════╬══════╬══════╬══════╬══════╬═════╬═════╬
# ║  CC  ║ 1286 ║  0   ║  0   ║ 0  ║ 0  ║  0   ║  0   ║  0  ║  0  ║ 0  ║  0  ║  2   ║  0   ║  0   ║  0   ║  0  ║  0  ║
# ╠══════╬══════╬══════╬══════╬════╬════╬══════╬══════╬═════╬═════╬════╬═════╬══════╬══════╬══════╬══════╬═════╬═════╬
# ║  CD  ║  0   ║ 1837 ║  0   ║ 0  ║ 0  ║  0   ║  53  ║  0  ║  0  ║ 0  ║  0  ║  7   ║  11  ║  26  ║  1   ║  0  ║  0  ║
# ╠══════╬══════╬══════╬══════╬════╬════╬══════╬══════╬═════╬═════╬════╬═════╬══════╬══════╬══════╬══════╬═════╬═════╬
# ║  DT  ║  8   ║  0   ║ 4632 ║ 0  ║ 0  ║  0   ║  0   ║  0  ║  0  ║ 0  ║  0  ║  0   ║  0   ║  5   ║  0   ║  1  ║  0  ║
# ╠══════╬══════╬══════╬══════╬════╬════╬══════╬══════╬═════╬═════╬════╬═════╬══════╬══════╬══════╬══════╬═════╬═════╬
# ║  EX  ║  0   ║  0   ║  0   ║ 57 ║ 0  ║  0   ║  0   ║  0  ║  0  ║ 0  ║  0  ║  0   ║  0   ║  0   ║  0   ║  0  ║  0  ║
# ╠══════╬══════╬══════╬══════╬════╬════╬══════╬══════╬═════╬═════╬════╬═════╬══════╬══════╬══════╬══════╬═════╬═════╬
# ║  FW  ║  0   ║  0   ║  0   ║ 0  ║ 0  ║  0   ║  0   ║  0  ║  0  ║ 0  ║  0  ║  0   ║  0   ║  1   ║  0   ║  0  ║  0  ║
# ╠══════╬══════╬══════╬══════╬════╬════╬══════╬══════╬═════╬═════╬════╬═════╬══════╬══════╬══════╬══════╬═════╬═════╬
# ║  IN  ║  0   ║  0   ║  0   ║ 0  ║ 0  ║ 5316 ║  2   ║  0  ║  0  ║ 0  ║  0  ║  2   ║  0   ║  0   ║  0   ║  0  ║  0  ║
# ╠══════╬══════╬══════╬══════╬════╬════╬══════╬══════╬═════╬═════╬════╬═════╬══════╬══════╬══════╬══════╬═════╬═════╬
# ║  JJ  ║  0   ║  0   ║  2   ║ 0  ║ 0  ║  10  ║ 2397 ║  0  ║  0  ║ 0  ║  0  ║ 129  ║  16  ║  87  ║  5   ║  0  ║  0  ║
# ╠══════╬══════╬══════╬══════╬════╬════╬══════╬══════╬═════╬═════╬════╬═════╬══════╬══════╬══════╬══════╬═════╬═════╬
# ║ JJR  ║  0   ║  0   ║  0   ║ 0  ║ 0  ║  0   ║  0   ║  98 ║  0  ║ 0  ║  0  ║  0   ║  0   ║  1   ║  0   ║  0  ║  0  ║
# ╠══════╬══════╬══════╬══════╬════╬════╬══════╬══════╬═════╬═════╬════╬═════╬══════╬══════╬══════╬══════╬═════╬═════╬
# ║ JJS  ║  0   ║  0   ║  0   ║ 0  ║ 0  ║  0   ║  1   ║  0  ║  84 ║ 0  ║  0  ║  0   ║  0   ║  0   ║  0   ║  0  ║  0  ║
# ╠══════╬══════╬══════╬══════╬════╬════╬══════╬══════╬═════╬═════╬════╬═════╬══════╬══════╬══════╬══════╬═════╬═════╬
# ║  LS  ║  0   ║  89  ║  0   ║ 0  ║ 0  ║  0   ║  0   ║  0  ║  0  ║ 4  ║  0  ║  0   ║  0   ║  0   ║  0   ║  0  ║  0  ║
# ╠══════╬══════╬══════╬══════╬════╬════╬══════╬══════╬═════╬═════╬════╬═════╬══════╬══════╬══════╬══════╬═════╬═════╬
# ║  MD  ║  0   ║  0   ║  0   ║ 0  ║ 0  ║  0   ║  0   ║  0  ║  0  ║ 0  ║ 582 ║  3   ║  0   ║  0   ║  0   ║  0  ║  0  ║
# ╠══════╬══════╬══════╬══════╬════╬════╬══════╬══════╬═════╬═════╬════╬═════╬══════╬══════╬══════╬══════╬═════╬═════╬
# ║  NN  ║  0   ║  2   ║  0   ║ 0  ║ 10 ║  2   ║ 719  ║  16 ║  9  ║ 0  ║  0  ║ 6535 ║  5   ║  19  ║  0   ║  0  ║  0  ║
# ╠══════╬══════╬══════╬══════╬════╬════╬══════╬══════╬═════╬═════╬════╬═════╬══════╬══════╬══════╬══════╬═════╬═════╬
# ║ NNS  ║  0   ║  0   ║  0   ║ 0  ║ 0  ║  2   ║  31  ║  0  ║  0  ║ 0  ║  0  ║  36  ║ 3222 ║  14  ║  8   ║  0  ║  0  ║
# ╠══════╬══════╬══════╬══════╬════╬════╬══════╬══════╬═════╬═════╬════╬═════╬══════╬══════╬══════╬══════╬═════╬═════╬
# ║ NNP  ║  0   ║  7   ║  0   ║ 0  ║ 6  ║  11  ║ 111  ║  1  ║  2  ║ 0  ║  2  ║ 110  ║  8   ║ 5431 ║  0   ║  0  ║  0  ║
# ╠══════╬══════╬══════╬══════╬════╬════╬══════╬══════╬═════╬═════╬════╬═════╬══════╬══════╬══════╬══════╬═════╬═════╬
# ║ NNPS ║  0   ║  0   ║  0   ║ 0  ║ 4  ║  1   ║  1   ║  0  ║  0  ║ 0  ║  0  ║  1   ║  90  ║ 333  ║  28  ║  0  ║  0  ║
# ╠══════╬══════╬══════╬══════╬════╬════╬══════╬══════╬═════╬═════╬════╬═════╬══════╬══════╬══════╬══════╬═════╬═════╬
# ║ PDT  ║  0   ║  0   ║  45  ║ 0  ║ 0  ║  0   ║  44  ║  0  ║  0  ║ 0  ║  0  ║  11  ║  0   ║  0   ║  0   ║  17 ║  0  ║
# ╠══════╬══════╬══════╬══════╬════╬════╬══════╬══════╬═════╬═════╬════╬═════╬══════╬══════╬══════╬══════╬═════╬═════╬
# ║ POS  ║  0   ║  0   ║  0   ║ 0  ║ 0  ║  0   ║  0   ║  0  ║  0  ║ 0  ║  0  ║  0   ║  0   ║  0   ║  0   ║  0  ║ 551 ║
# ╠══════╬══════╬══════╬══════╬════╬════╬══════╬══════╬═════╬═════╬════╬═════╬══════╬══════╬══════╬══════╬═════╬═════╬
# ║ PRP  ║  0   ║  0   ║  0   ║ 0  ║ 0  ║  0   ║  0   ║  0  ║  0  ║ 0  ║  0  ║  0   ║  0   ║  0   ║  0   ║  0  ║  0  ║
# ╠══════╬══════╬══════╬══════╬════╬════╬══════╬══════╬═════╬═════╬════╬═════╬══════╬══════╬══════╬══════╬═════╬═════╬
# ║ PP$  ║  0   ║  0   ║  0   ║ 0  ║ 0  ║  0   ║  0   ║  0  ║  0  ║ 0  ║  0  ║  0   ║  0   ║  0   ║  0   ║  0  ║  0  ║
# ╠══════╬══════╬══════╬══════╬════╬════╬══════╬══════╬═════╬═════╬════╬═════╬══════╬══════╬══════╬══════╬═════╬═════╬
# ║  RB  ║  1   ║  0   ║  0   ║ 0  ║ 0  ║  15  ║ 137  ║  0  ║  0  ║ 0  ║  0  ║  20  ║  0   ║  20  ║  0   ║  0  ║  0  ║
# ╠══════╬══════╬══════╬══════╬════╬════╬══════╬══════╬═════╬═════╬════╬═════╬══════╬══════╬══════╬══════╬═════╬═════╬
# ║ RBR  ║  0   ║  0   ║  0   ║ 0  ║ 0  ║  0   ║  6   ║  83 ║  0  ║ 0  ║  0  ║  0   ║  0   ║  0   ║  0   ║  0  ║  0  ║
# ╠══════╬══════╬══════╬══════╬════╬════╬══════╬══════╬═════╬═════╬════╬═════╬══════╬══════╬══════╬══════╬═════╬═════╬
# ║ RBS  ║  0   ║  0   ║  0   ║ 0  ║ 0  ║  0   ║  0   ║  0  ║  33 ║ 0  ║  0  ║  0   ║  0   ║  0   ║  0   ║  0  ║  0  ║
# ╠══════╬══════╬══════╬══════╬════╬════╬══════╬══════╬═════╬═════╬════╬═════╬══════╬══════╬══════╬══════╬═════╬═════╬
# ║  RP  ║  0   ║  0   ║  0   ║ 0  ║ 0  ║ 195  ║  5   ║  0  ║  0  ║ 0  ║  0  ║  0   ║  0   ║  0   ║  0   ║  0  ║  0  ║
# ╠══════╬══════╬══════╬══════╬════╬════╬══════╬══════╬═════╬═════╬════╬═════╬══════╬══════╬══════╬══════╬═════╬═════╬
# ║ SYM  ║  78  ║  0   ║  0   ║ 0  ║ 0  ║  0   ║  0   ║  0  ║  0  ║ 0  ║  0  ║  0   ║  0   ║  0   ║  0   ║  0  ║  0  ║
# ╠══════╬══════╬══════╬══════╬════╬════╬══════╬══════╬═════╬═════╬════╬═════╬══════╬══════╬══════╬══════╬═════╬═════╬
# ║  TO  ║  0   ║  0   ║  0   ║ 0  ║ 0  ║  0   ║  12  ║  0  ║  0  ║ 0  ║  0  ║  11  ║  0   ║  0   ║  0   ║  0  ║  0  ║
# ╠══════╬══════╬══════╬══════╬════╬════╬══════╬══════╬═════╬═════╬════╬═════╬══════╬══════╬══════╬══════╬═════╬═════╬
# ║  UH  ║  0   ║  0   ║  37  ║ 0  ║ 0  ║  0   ║  0   ║  0  ║  0  ║ 0  ║  0  ║  0   ║  0   ║  0   ║  0   ║  0  ║  0  ║
# ╠══════╬══════╬══════╬══════╬════╬════╬══════╬══════╬═════╬═════╬════╬═════╬══════╬══════╬══════╬══════╬═════╬═════╬
# ║  VB  ║  0   ║  0   ║  0   ║ 0  ║ 0  ║  0   ║  12  ║  0  ║  0  ║ 0  ║  0  ║ 331  ║  0   ║  1   ║  0   ║  0  ║  0  ║
# ╠══════╬══════╬══════╬══════╬════╬════╬══════╬══════╬═════╬═════╬════╬═════╬══════╬══════╬══════╬══════╬═════╬═════╬
# ║ VBD  ║  0   ║  0   ║  0   ║ 0  ║ 0  ║  0   ║  27  ║  0  ║  0  ║ 0  ║  0  ║  27  ║  0   ║  0   ║  0   ║  0  ║  0  ║
# ╠══════╬══════╬══════╬══════╬════╬════╬══════╬══════╬═════╬═════╬════╬═════╬══════╬══════╬══════╬══════╬═════╬═════╬
# ║ VBG  ║  0   ║  0   ║  0   ║ 0  ║ 0  ║  0   ║  32  ║  0  ║  0  ║ 0  ║  0  ║ 197  ║  0   ║  0   ║  0   ║  0  ║  0  ║
# ╠══════╬══════╬══════╬══════╬════╬════╬══════╬══════╬═════╬═════╬════╬═════╬══════╬══════╬══════╬══════╬═════╬═════╬
# ║ VBN  ║  0   ║  0   ║  0   ║ 0  ║ 0  ║  0   ║ 111  ║  0  ║  0  ║ 0  ║  0  ║  5   ║  0   ║  0   ║  0   ║  0  ║  0  ║
# ╠══════╬══════╬══════╬══════╬════╬════╬══════╬══════╬═════╬═════╬════╬═════╬══════╬══════╬══════╬══════╬═════╬═════╬
# ║ VBP  ║  0   ║  0   ║  0   ║ 0  ║ 0  ║  0   ║  11  ║  0  ║  0  ║ 0  ║  0  ║ 152  ║  0   ║  0   ║  0   ║  0  ║  0  ║
# ╠══════╬══════╬══════╬══════╬════╬════╬══════╬══════╬═════╬═════╬════╬═════╬══════╬══════╬══════╬══════╬═════╬═════╬
# ║ VBZ  ║  0   ║  0   ║  0   ║ 0  ║ 0  ║  0   ║  0   ║  0  ║  0  ║ 0  ║  0  ║  1   ║ 161  ║  1   ║  0   ║  0  ║  0  ║
# ╠══════╬══════╬══════╬══════╬════╬════╬══════╬══════╬═════╬═════╬════╬═════╬══════╬══════╬══════╬══════╬═════╬═════╬
# ║ WDT  ║  0   ║  0   ║ 103  ║ 0  ║ 0  ║ 314  ║  0   ║  0  ║  0  ║ 0  ║  0  ║  0   ║  0   ║  0   ║  0   ║  0  ║  0  ║
# ╠══════╬══════╬══════╬══════╬════╬════╬══════╬══════╬═════╬═════╬════╬═════╬══════╬══════╬══════╬══════╬═════╬═════╬
# ║  WP  ║  0   ║  0   ║  0   ║ 0  ║ 0  ║  0   ║  0   ║  0  ║  0  ║ 0  ║  0  ║  0   ║  0   ║  0   ║  0   ║  0  ║  0  ║
# ╠══════╬══════╬══════╬══════╬════╬════╬══════╬══════╬═════╬═════╬════╬═════╬══════╬══════╬══════╬══════╬═════╬═════╬
# ║ WP$  ║  0   ║  0   ║  0   ║ 0  ║ 0  ║  0   ║  0   ║  0  ║  0  ║ 0  ║  0  ║  0   ║  0   ║  0   ║  0   ║  0  ║  0  ║
# ╠══════╬══════╬══════╬══════╬════╬════╬══════╬══════╬═════╬═════╬════╬═════╬══════╬══════╬══════╬══════╬═════╬═════╬
# ║ WRB  ║  0   ║  0   ║  0   ║ 0  ║ 0  ║  0   ║  0   ║  0  ║  0  ║ 0  ║  0  ║  0   ║  0   ║  0   ║  0   ║  0  ║  0  ║
# ╠══════╬══════╬══════╬══════╬════╬════╬══════╬══════╬═════╬═════╬════╬═════╬══════╬══════╬══════╬══════╬═════╬═════╬
# ║  #   ║  0   ║  0   ║  0   ║ 0  ║ 0  ║  0   ║  0   ║  0  ║  0  ║ 0  ║  0  ║  0   ║  0   ║  0   ║  0   ║  0  ║  0  ║
# ╠══════╬══════╬══════╬══════╬════╬════╬══════╬══════╬═════╬═════╬════╬═════╬══════╬══════╬══════╬══════╬═════╬═════╬
# ║  $   ║  0   ║  0   ║  0   ║ 0  ║ 0  ║  0   ║  0   ║  0  ║  0  ║ 0  ║  0  ║  0   ║  0   ║  0   ║  0   ║  0  ║  0  ║
# ╠══════╬══════╬══════╬══════╬════╬════╬══════╬══════╬═════╬═════╬════╬═════╬══════╬══════╬══════╬══════╬═════╬═════╬
# ║  .   ║  0   ║  0   ║  0   ║ 0  ║ 0  ║  0   ║  0   ║  0  ║  0  ║ 0  ║  0  ║  0   ║  0   ║  0   ║  0   ║  0  ║  0  ║
# ╠══════╬══════╬══════╬══════╬════╬════╬══════╬══════╬═════╬═════╬════╬═════╬══════╬══════╬══════╬══════╬═════╬═════╬
# ║  ,   ║  0   ║  0   ║  0   ║ 0  ║ 0  ║  0   ║  0   ║  0  ║  0  ║ 0  ║  0  ║  0   ║  0   ║  0   ║  0   ║  0  ║  0  ║
# ╠══════╬══════╬══════╬══════╬════╬════╬══════╬══════╬═════╬═════╬════╬═════╬══════╬══════╬══════╬══════╬═════╬═════╬
# ║  :   ║  0   ║  0   ║  0   ║ 0  ║ 0  ║  0   ║  0   ║  0  ║  0  ║ 0  ║  0  ║  0   ║  0   ║  0   ║  0   ║  0  ║  0  ║
# ╠══════╬══════╬══════╬══════╬════╬════╬══════╬══════╬═════╬═════╬════╬═════╬══════╬══════╬══════╬══════╬═════╬═════╬
# ║  (   ║  0   ║  0   ║  0   ║ 0  ║ 0  ║  0   ║  0   ║  0  ║  0  ║ 0  ║  0  ║  0   ║  0   ║  0   ║  0   ║  0  ║  0  ║
# ╠══════╬══════╬══════╬══════╬════╬════╬══════╬══════╬═════╬═════╬════╬═════╬══════╬══════╬══════╬══════╬═════╬═════╬
# ║  )   ║  0   ║  0   ║  0   ║ 0  ║ 0  ║  0   ║  0   ║  0  ║  0  ║ 0  ║  0  ║  0   ║  0   ║  0   ║  0   ║  0  ║  0  ║
# ╠══════╬══════╬══════╬══════╬════╬════╬══════╬══════╬═════╬═════╬════╬═════╬══════╬══════╬══════╬══════╬═════╬═════╬
# ║  "   ║  0   ║  0   ║  0   ║ 0  ║ 0  ║  0   ║  0   ║  0  ║  0  ║ 0  ║  0  ║  0   ║  0   ║  0   ║  0   ║  0  ║  0  ║
# ╠══════╬══════╬══════╬══════╬════╬════╬══════╬══════╬═════╬═════╬════╬═════╬══════╬══════╬══════╬══════╬═════╬═════╬
# ║  '   ║  0   ║  0   ║  0   ║ 0  ║ 0  ║  0   ║  0   ║  0  ║  0  ║ 0  ║  0  ║  0   ║  0   ║  0   ║  0   ║  0  ║  0  ║
# ╠══════╬══════╬══════╬══════╬════╬════╬══════╬══════╬═════╬═════╬════╬═════╬══════╬══════╬══════╬══════╬═════╬═════╬
# ║  “   ║  0   ║  0   ║  0   ║ 0  ║ 0  ║  0   ║  0   ║  0  ║  0  ║ 0  ║  0  ║  0   ║  0   ║  0   ║  0   ║  0  ║  0  ║
# ╠══════╬══════╬══════╬══════╬════╬════╬══════╬══════╬═════╬═════╬════╬═════╬══════╬══════╬══════╬══════╬═════╬═════╬
# ║ APO  ║  0   ║  0   ║  0   ║ 0  ║ 0  ║  0   ║  0   ║  0  ║  0  ║ 0  ║  0  ║  0   ║  0   ║  0   ║  0   ║  0  ║  0  ║
# ╠══════╬══════╬══════╬══════╬════╬════╬══════╬══════╬═════╬═════╬════╬═════╬══════╬══════╬══════╬══════╬═════╬═════╬
# ║  ”   ║  0   ║  0   ║  0   ║ 0  ║ 0  ║  0   ║  0   ║  0  ║  0  ║ 0  ║  0  ║  0   ║  0   ║  0   ║  0   ║  0  ║  0  ║
# ╠══════╬══════╬══════╬══════╬════╬════╬══════╬══════╬═════╬═════╬════╬═════╬══════╬══════╬══════╬══════╬═════╬═════╬
# ║  ``  ║  0   ║  0   ║  0   ║ 0  ║ 0  ║  0   ║  0   ║  0  ║  0  ║ 0  ║  0  ║  0   ║  0   ║  0   ║  0   ║  0  ║  0  ║
# ╠══════╬══════╬══════╬══════╬════╬════╬══════╬══════╬═════╬═════╬════╬═════╬══════╬══════╬══════╬══════╬═════╬═════╬
# ║  ''  ║  0   ║  0   ║  0   ║ 0  ║ 0  ║  0   ║  0   ║  0  ║  0  ║ 0  ║  0  ║  0   ║  0   ║  0   ║  0   ║  0  ║  0  ║
# ╠══════╬══════╬══════╬══════╬════╬════╬══════╬══════╬═════╬═════╬════╬═════╬══════╬══════╬══════╬══════╬═════╬═════╬
# ║ PRP$ ║  0   ║  0   ║  0   ║ 0  ║ 0  ║  0   ║  0   ║  0  ║  0  ║ 0  ║  0  ║  0   ║  0   ║  0   ║  0   ║  0  ║  0  ║
# ╚══════╩══════╩══════╩══════╩════╩════╩══════╩══════╩═════╩═════╩════╩═════╩══════╩══════╩══════╩══════╩═════╩═════╩
# <><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>
#
# The tagger program works by first parsing in teh command line arguments.
# The program opens and reads the input training file and stores it as a list of items. Then it opens and reads
# the input file that should be tagged and stores it as a list of items. After this the program iterates through
# all the tagged training words, skipping the phrasal brackets, and storing the frequencies of a tag associated
# with a given word. Then for every item in the file of data to be tagged a tag is assigned, properly handling
# phrasal brackets when encountered. The program attempts to fit the most probable tag to words that have been
# "learned" in the training data. Otherwise it attempts to match the word to a feature rule to assign a likely
# tag before simply assign the tag of noun to unknown words.
#
# The scorer program works by first parsing in the command line arguments.
# Then the number of items that are in the key are counted, excluding the phrasal brackets.
# The program then evaluates the number of tags that are tagged correctly. It does this in a loop
# that iterates through every item in the key, skipping over the phrasal brackets and stopping when
# either the input file has reached its end or every object was checked. At then end it send to the standard
# output an accuracy score as well as a confusion matrix.
#
# I evaluated the accuracy of the tagger in phases, removing and adding feature rules to see their individual
# contributions to the accuracy of the program the results are as follows:
#   Accuracy Score with all features   >> 88.31 %
#   Independent Ablations                                          [remove only the feature listed and record accuracy]
# 	    Feature 1 					   >> 87.61 %          (-0.70) [removing feature resulted in a decrease in accuracy]
# 	    Feature 2					   >> 88.16 %          (-0.15)                      "
# 	    Feature 3					   >> 88.28 %          (-0.03)                      "
# 	    Feature 4					   >> 84.85 %          (-3.46)                      "
# 	    Feature 5					   >> 87.24 %          (-1.06)                      "
# 	    Feature 6					   >> 88.27 %          (-0.04)                      "
# 	    Feature 7					   >> 88.04 %          (-0.27)                      "
#   Accuracy Score with base
#    setting of "Most Likely Tag" 	   >> 82.59 %
#
# I used resources provided by Dr. Bridget McInnes of Virgina Commonwealth University to understand how to
# calculate the probability of a tag based on the frequency of that tag being seen with a given word.
# Additionally, I used a guide provided by her to learn how to take inputs from the command line and to
# iterate over dictionaries in python 3.
#
# Additionally, I used the website "GEEKS FOR GEEKS", https://www.geeksforgeeks.org/ to further my understanding of
# dictionaries,collections classes, the prettytable library, and writing to the standard output in python.
#
# No code was copied from either source.
#
# This code is property of CHARLES I CUTLER,
# student at Virginia Commonwealth University.

import sys
import re
from sys import argv
from prettytable import PrettyTable, ALL, DOUBLE_BORDER


# The following function is used to return the appropriate integer number that represents
# the index at which the tag list of values within the tag counts dictionary should be increased
def index_for_confusion_table(param):
    if param == "CC":
        return 1
    elif param == "CD":
        return 2
    elif param == "DT":
        return 3
    elif param == "EX":
        return 4
    elif param == "FW":
        return 5
    elif param == "IN":
        return 6
    elif param == "JJ":
        return 7
    elif param == "JJR":
        return 8
    elif param == "JJS":
        return 9
    elif param == "LS":
        return 10
    elif param == "MD":
        return 11
    elif param == "NN":
        return 12
    elif param == "NNS":
        return 13
    elif param == "NNP":
        return 14
    elif param == "NNPS":
        return 15
    elif param == "PDT":
        return 16
    elif param == "POS":
        return 17
    elif param == "PRP":
        return 18
    elif param == "PP$":
        return 19
    elif param == "RB":
        return 20
    elif param == "RBR":
        return 21
    elif param == "RBS":
        return 22
    elif param == "RP":
        return 23
    elif param == "SYM":
        return 24
    elif param == "TO":
        return 25
    elif param == "UH":
        return 26
    elif param == "VB":
        return 27
    elif param == "VBD":
        return 28
    elif param == "VBG":
        return 29
    elif param == "VBN":
        return 30
    elif param == "VBP":
        return 31
    elif param == "VBZ":
        return 32
    elif param == "WDT":
        return 33
    elif param == "WP":
        return 34
    elif param == "WP$":
        return 35
    elif param == "WRB":
        return 36
    elif param == "#":
        return 37
    elif param == "$":
        return 38
    elif param == ".":
        return 39
    elif param == ",":
        return 40
    elif param == ":":
        return 41
    elif param == "(":
        return 42
    elif param == ")":
        return 43
    elif param == "\"":
        return 44
    elif param == "\'":
        return 45
    elif param == "“":
        return 46
    elif param == "APO":
        return 47
    elif param == "”":
        return 48
    elif param == "``":
        return 49
    elif param == "''":
        return 50
    elif param == "PRP$":
        return 51
    else:
        return -1


# "tags" contains all the Tags that might appear in the data when evaluating the accuracy of
# a text file against its respective key
tags = ["CC", "CD", "DT", "EX", "FW", "IN", "JJ", "JJR", "JJS", "LS", "MD", "NN", "NNS", "NNP", "NNPS",
        "PDT", "POS", "PRP", "PP$", "RB", "RBR", "RBS", "RP", "SYM", "TO", "UH", "VB", "VBD", "VBG",
        "VBN", "VBP", "VBZ", "WDT", "WP", "WP$", "WRB", "#", "$", ".", ",", ":", "(", ")", "\"", "\'",
        "“", "APO", "”", "``", "''", "PRP$"]
# tagCounts contains a dictionary for each tag where each value is a list
# The lists contain integers which represent the number of times a specific tag has been seen where
# the index within the list represents a specific tag. These lists are uses to generate the confusion matrix.
tagCounts = {
    "CC": ["CC", 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
           0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    "CD": ["CD", 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
           0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    "DT": ["DT", 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
           0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    "EX": ["EX", 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
           0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    "FW": ["FW", 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
           0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    "IN": ["IN", 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
           0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    "JJ": ["JJ", 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
           0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    "JJR": ["JJR", 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    "JJS": ["JJS", 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    "LS": ["LS", 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
           0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    "MD": ["MD", 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
           0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    "NN": ["NN", 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
           0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    "NNS": ["NNS", 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    "NNP": ["NNP", 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    "NNPS": ["NNPS", 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    "PDT": ["PDT", 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    "POS": ["POS", 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    "PRP": ["PRP", 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    "PP$": ["PP$", 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    "RB": ["RB", 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
           0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    "RBR": ["RBR", 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    "RBS": ["RBS", 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    "RP": ["RP", 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
           0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    "SYM": ["SYM", 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    "TO": ["TO", 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
           0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    "UH": ["UH", 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
           0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    "VB": ["VB", 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
           0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    "VBD": ["VBD", 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    "VBG": ["VBG", 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    "VBN": ["VBN", 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    "VBP": ["VBP", 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    "VBZ": ["VBZ", 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    "WDT": ["WDT", 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    "WP": ["WP", 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
           0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    "WP$": ["WP$", 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    "WRB": ["WRB", 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    "#": ["#", 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
          0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    "$": ["$", 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
          0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    ".": [".", 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
          0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    ",": [",", 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
          0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    ":": [":", 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
          0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    "(": ["(", 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
          0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    ")": [")", 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
          0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    "\"": ["\"", 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
           0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    "\'": ["\'", 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
           0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    "“": ["“", 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
          0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    "APO": ["APO", 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    "”": ["”", 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
          0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    "``": ["``", 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
           0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    "''": ["''", 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
           0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    "PRP$": ["PRP$", 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
}

total_tags_in_key = 0
correct_tags = 0
flag = 0

x = PrettyTable()
x.set_style(DOUBLE_BORDER)
x.hrules = ALL
header = tags.copy()
header.insert(0, " ")
x.field_names = header

incIndex = 0

# Open and read the input test file and store it as a list of items.
with open(argv[1], 'r') as file:
    string = file.read()
    string = re.sub(r"\\/", r"", string)
    string = re.sub(r"\[]", r"", string)
    pos_test_tagged = string.split()
    file.close()

# Open and read the input key file and store it as a string
with open(argv[2], 'r') as file:
    string = file.read()
    string = re.sub(r"\\/", r"", string)
    pos_key = string.split()
    file.close()

# Get the total number of items in the key
for item in pos_key:
    if item == "[" or item == "]":
        continue
    else:
        total_tags_in_key += 1

# Evaluate the number of correctly tagged objects
for index in range(0, len(pos_key)):
    # If at any point the key still has items to evaluate but the
    # input file that is being evaluated has reach the end we break the loop.
    if flag == len(pos_test_tagged):
        break

    # If there is a bracket in the key file that separates phrasal structures, ignore it.
    if pos_key[index] == "[" or pos_key[index] == "]":
        continue

    # Otherwise, check that the word in the key and the input file match
    else:
        # If they match then add 1 to the corresponding position in the tagCounts tag list
        # an Example would be at Key "CC" and then in the list "CC" has a value of index 1
        # so the list at index 1 would be increased by one value.
        if pos_test_tagged[index].split("/")[1] == pos_key[index].split("/")[1].split("|")[0]:
            incIndex = index_for_confusion_table(pos_test_tagged[index].split("/")[1])
            tagCounts[pos_test_tagged[index].split("/")[1]][incIndex] += 1
            correct_tags += 1
            flag += 1

        # Otherwise, increment the corresponding location that represents
        # the tag that was supposed to be tag vs the tag we actually saw
        else:
            incIndex = index_for_confusion_table(pos_key[index].split("/")[1].split("|")[0])
            tagCounts[pos_test_tagged[index].split("/")[1]][incIndex] += 1
            correct_tags += 0
            flag += 1

# Write the Accuracy to the Standard Output
sys.stdout.write("\nAccuracy score is : " + str((correct_tags/total_tags_in_key)*100) + " %\n")

# Produce and write the confusion matrix to the Standard Output
for values in tagCounts.values():
    x.add_row(values)
sys.stdout.write(x.get_string())
