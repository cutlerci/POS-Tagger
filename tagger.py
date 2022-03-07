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
# After which the text files are read in and stored as a single string variable.
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

import re
import sys
from sys import argv

from collections import defaultdict

tagDict = defaultdict(int)
wordVsTagDict = defaultdict(dict)
highestProb = 0


# This function returns the most probable tag of a passed in word
def tagged(param):
    high_prob = 0
    selected_tag = ""
    for x in wordVsTagDict[param]:
        if (wordVsTagDict[param][x] / tagDict[x]) > high_prob:
            high_prob = wordVsTagDict[param][x] / tagDict[x]
            selected_tag = x
    return selected_tag


# Open and read the input training file and store it as a list of items.
with open(argv[1], 'r') as file:
    trainingData = file.read().split()
    file.close()

# Open and read the input file that should be tagged and store it as a list of items.
with open(argv[2], 'r') as file:
    dataToTag = file.read().split()
    file.close()

# Iterates through all the tagged training words, skipping the phrasal brackets, and
# storing the frequencies of a tag associated with a given word.
for index in range(0, len(trainingData)):
    if trainingData[index] == "[" or trainingData[index] == "]":
        continue
    else:
        phrase = (re.sub(r"([^\\])(/)(.)", r"\1***\3", trainingData[index])).split("***")
        word = phrase[0]
        tag = phrase[1]

        # This updates the Frequency we have seen the specific tag
        tagDict[tag] += 1

        # This updates the frequency we have seen the tag associated to a specific word
        wordVsTagDict[word][tag] = (wordVsTagDict.setdefault(word, {})).setdefault(tag, 0) + 1

previous_tag = ""
index = 0
mostProbableTag = "NN"

for word in dataToTag:
    # Correctly process left phrasal brackets found in the test file.
    if word == "[":
        sys.stdout.write("\n[ ")
        continue

    # Correctly process right phrasal brackets found in the test file.
    elif word == "]":
        sys.stdout.write("]\n")
        continue

    # Handle words to be tagged in the test file.
    else:
        # If the word is one that we have already seen before when the program was trained then select
        # the most probable tag, that is, the tag that has the highest probability fo being associated
        # with the given word.
        if not(len(wordVsTagDict[word]) == 0):
            for items in wordVsTagDict[word]:
                if (wordVsTagDict[word][items]/tagDict[items]) > highestProb:
                    highestProb = wordVsTagDict[word][items]/tagDict[items]
                    mostProbableTag = items

            string = word + "/" + mostProbableTag + " "
            sys.stdout.write(string)
            highestProb = 0
            previous_tag = mostProbableTag

        # Otherwise, try and match it to a rule or just decide that the word is a noun.
        else:
            # Feature 1 -- Any word that contains digits becomes a Counting Digit tag
            if re.search(r"[0-9]", word):
                string = word + "/CD "
                previous_tag = "CD"

            # Feature 2 -- Any word that ends in -ly becomes an Adverb tag
            elif len(word) > 2 and word[-2] == "l" and word[-1] == "y":
                string = word + "/RB "
                previous_tag = "RB"

            # Feature 3 -- Any word that starts with a capital letter and ends in a -s becomes a Proper Noun Plural tag
            elif word[0].isupper() and word[-1] == "s":
                string = word + "/NNPS "
                previous_tag = "NNPS"

            # Feature 4 -- Any word that starts with a capital letter and
            # does not end in -s becomes a Proper Noun Singular tag
            elif word[0].isupper() and word[-1] != "s":
                string = word + "/NNP "
                previous_tag = "NNP"

            # Feature 5 -- Any word that does not start with a capital letter and
            # does end in -s becomes a noun plural tag
            elif (not(word[0].isupper())) and word[-1] == "s":
                string = word + "/NNS "
                previous_tag = "NNS"

            elif previous_tag == "RB" or previous_tag == "TO":
                # Feature 6 -- Any word that is preceded by an adverb or an infinitival and ends in -ing
                # becomes a Verb Present tag
                if len(word) > 3 and word[-3] == "i" and word[-2] == "n" and word[-1] == "g":
                    string = word + "/VBG "
                    previous_tag = "VBG"

                # Feature 7 -- Otherwise any word that is preceded by an adverb or infinitival
                # becomes a Verb
                else:
                    string = word + "/VB "
                    previous_tag = "VB"

            # Otherwise, we just say it is a noun
            else:
                string = word + "/NN "
                previous_tag = "NN"
            sys.stdout.write(string)
    index += 1
