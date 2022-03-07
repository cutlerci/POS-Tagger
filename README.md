![VCU Logo](https://ocpe.vcu.edu/media/ocpe/images/logos/bm_CollEng_CompSci_RF2_hz_4c.png)

# Cutlerci-CMSC416-Project 3
My name is Charles Ian Cutler, CIC, currently enrolled in the College of Engineering at Virginia Commonwealth Univeristy. 
This repository is for a project in Virginia Commonwealth University course CMSC 416, Introduction Natural Language Processing, Spring 2022.
## Repository Files
1) tagger.py -- The Part of Speech tagger program.
2) scorer.py -- The accuracy evalutuing program.
## POS-Tagger Description
The following program is for the third programming assignment in the course CMSC 416, Intro to Natural Language Processing, at Virginia Commonwealth University

This program was created with the intention of learning how part of speech tagging works to associate tags with words. These tags represent the part of speech that the specific word is. Tags are usually found in a tag-set which contains a wide array of different tags for varying specifications of different parts of speech. We were tasked to create our own POS tagger and have it tag a text file. After which we grade it to see how accurate it was by using a provided key.

The program is broken into two parts the first is tagger.py which should be run from the command line with exactly two arguments. The first argument should be a text file that contains a corpus that has been tagged with the parts of speech manually. This serves as the data to train the model. The second argument should also be a text file. Specifically it should contain the text that should be tagged by the part-of-speech [POS] tagger. The output is written to the standard output and can be directed towards a text file. It should be run at the command prompt as follows with the output piped into a file names as chosen by the user:

COMMAND PROMPT SAMPLE: >>> python3 tagger.py pos-mod-key.txt pos-test.txt > pos-test-with-tags.txt

The second part of the program is scorer.py which should be run from the command line with exactly two arguments. The first argument should be a text file that contains a corpus that has been tagged with the parts of speech, intended to be the output of tagger.py . The second argument should also be a text file. Specifically it should contain the key by which the input text file should be graded against for accuracy of tagging. The output is written to the standard output and can be directed towards a text file. It should be run at the command prompt as follows with the output piped into a file names as chosen by the user:

COMMAND PROMPT SAMPLE: >>> python3 scorer.py pos-test-with-tags.txt pos-test-key.txt > output.txt

After the two parts have been run the output.txt should contain an accuracy score and a confusion matrix. The accuracy score is a percentage of how accurate the tagged file is when compared to the key. The confusion matrix contains all the tags along the top and left side. The tags along the top represent the tag that was intended to be seen. The tags along the left side represent the tags that were actually seen. If a cell contains a nonzero number, that number represents the number of times we saw the tag in the row name position when we were expecting the tag in the column name position. A perfect score would thus result in non-zero values appearing only in the diagonal from top left to bottom right cells.

Below is an example output when run where I have pasted the contents of output.txt

 <><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>
 >>> python3 tagger.py pos-mod-key.txt pos-test.txt > pos-test-with-tags.txt
 >>> python3 scorer.py pos-test-with-tags.txt pos-test-key.txt > output.txt

# SEE OUTPUT.txt for sample output
 <><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>

The tagger program works by first parsing in teh command line arguments. The program opens and reads the input training file and stores it as a list of items. Then it opens and reads the input file that should be tagged and stores it as a list of items. After this the program iterates through all the tagged training words, skipping the phrasal brackets, and storing the frequencies of a tag associated with a given word. Then for every item in the file of data to be tagged a tag is assigned, properly handling phrasal brackets when encountered. The program attempts to fit the most probable tag to words that have been "learned" in the training data. Otherwise it attempts to match the word to a feature rule to assign a likely tag before simply assign the tag of noun to unknown words.

The scorer program works by first parsing in the command line arguments.Then the number of items that are in the key are counted, excluding the phrasal brackets. The program then evaluates the number of tags that are tagged correctly. It does this in a loop that iterates through every item in the key, skipping over the phrasal brackets and stopping when either the input file has reached its end or every object was checked. At then end it send to the standard output an accuracy score as well as a confusion matrix.

I evaluated the accuracy of the tagger in phases, removing and adding feature rules to see their individual contributions to the accuracy of the program the results are as follows:

Accuracy Score with all features   >> 88.31 %
Independent Ablations                                          [remove only the feature listed and record accuracy]
 	    Feature 1 					   >> 87.61 %          (-0.70) [removing feature resulted in a decrease in accuracy]
 	    Feature 2					   >> 88.16 %          (-0.15)                      "
 	    Feature 3					   >> 88.28 %          (-0.03)                      "
 	    Feature 4					   >> 84.85 %          (-3.46)                      "
 	    Feature 5					   >> 87.24 %          (-1.06)                      "
 	    Feature 6					   >> 88.27 %          (-0.04)                      "
 	    Feature 7					   >> 88.04 %          (-0.27)                      "
Accuracy Score with base
    setting of "Most Likely Tag" 	   >> 82.59 %

I used resources provided by Dr. Bridget McInnes of Virgina Commonwealth University to understand how to calculate the probability of a tag based on the frequency of that tag being seen with a given word. Additionally, I used a guide provided by her to learn how to take inputs from the command line and to iterate over dictionaries in python 3.

Additionally, I used the website "GEEKS FOR GEEKS", https://www.geeksforgeeks.org/ to further my understanding of dictionaries,collections classes, the prettytable library, and writing to the standard output in python.

No code was copied from either source.

This code is property of CHARLES I CUTLER, student at Virginia Commonwealth University.
