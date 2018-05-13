import nltk
import math
from nltk import FreqDist
import os

#Given lists of txt files, it generates a dictionary containing a list of the 2500 most common 
#words and their frequency throughout all the txt files.
def genTotalWordDictionary(nsTestFiles, nsTrainFiles, sTestFiles, sTrainFiles):
    #Generate a mega document from all the emails
    megaDoc = catFiles(nsTestFiles, "data/nonspam-test/")
    megaDoc = megaDoc + catFiles(nsTrainFiles, "data/nonspam-train/")
    megaDoc = megaDoc + catFiles(sTestFiles, "data/spam-test/")
    megaDoc = megaDoc + catFiles(sTrainFiles, "data/spam-train/")

    #Find the 2500 most common words in the mega document
    wordList = megaDoc.split()
    fdist = FreqDist(wordList)
    topWords = fdist.most_common(2500)

    #Fill Dictionary
    dict = {}
    for word in topWords:
        dict[word[0]] = fdist[word[0]] 

    return dict

#Given a list of text files and a relative path to those file, it concatenates the contents
#of all the files into one big string
def catFiles(files, path):
    catFiles = ""
    #For each file in the given path, concatenate its contents onto one another
    for fileName in files:
        file = open(path + fileName)
        contents = file.read()
        catFiles = catFiles +  contents + "\r\n"
        file.close()

    return catFiles

#Given lists of txt files and a dictionary of common words, it generates a dictionary
#containing files names linked to their dictionary of common words
def genFileDictionary(nsTestFiles, nsTrainFiles, sTestFiles, sTrainFiles, commonDictionary):
    fileDictionary = {}
    
    #Fill dictionary with data from every text file
    fileDictionary.update(genFileWordDictionary(nsTestFiles, "data/nonspam-test/", commonDictionary))
    fileDictionary.update(genFileWordDictionary(nsTrainFiles, "data/nonspam-train/", commonDictionary))
    fileDictionary.update(genFileWordDictionary(sTestFiles, "data/spam-test/", commonDictionary))
    fileDictionary.update(genFileWordDictionary(sTrainFiles, "data/spam-train/", commonDictionary))

    return fileDictionary

#Given a list of txt files, a path to those files, and a dictionary of common words, it
#generates a dictionary of file names linked to their dictionary of common words
def genFileWordDictionary(fileList, path, commonDictionary):
    fileDictionary = {}
    
    #For each file in the given path, create a dictionary of words that exist in the dictionary of common words
    #then fill the fileDictionary with each file and their dictiionary of common words 
    for fileName in fileList:
        wordDictionary = {}
        wordList = []
        file = open(path + fileName)
        contents = file.read()
        wordList = contents.split()
        fdist = FreqDist(wordList)
        for word in fdist:
            if word in commonDictionary:
                wordDictionary[word] = fdist[word]
        fileDictionary[fileName] = wordDictionary
        file.close()
    
    return fileDictionary

#Given a dictionary of words, it collects the vocabulary
def getVocab(wordDictionary):
    #Collect vocabulary words
    vocab = []
    for word in wordDictionary:
        vocab.append(word)

    return vocab

#Given a dictionary of files linked to a dictionary of frequency of words in each 
#corresponding file and some training files, generate a dictionary containing 
#classes linked to a dictionary of words and their frequencies
def genClassWordsDictionary(fileDictionary, sTrainFiles, nsTrainFiles):
    classWordsDictionary = {}
    spamWords = {}
    nonSpamWords = {}

    #For every file, if it is a train file, add the frequencies of each word
    #to their corresponding class
    for file in fileDictionary:
        if file in sTrainFiles:
            currDictionary = fileDictionary[file]
            for word in currDictionary:
                if word in spamWords:
                    spamWords[word] = spamWords[word] + currDictionary[word]
                else:
                    spamWords[word] = currDictionary[word]
        elif file in nsTrainFiles:
            currDictionary = fileDictionary[file]
            for word in currDictionary:
                if word in nonSpamWords:
                    nonSpamWords[word] = nonSpamWords[word] + currDictionary[word]
                else:
                    nonSpamWords[word] = currDictionary[word]

    classWordsDictionary["spam"] = spamWords
    classWordsDictionary["nonspam"] = nonSpamWords
    return classWordsDictionary

#Given the frequency of words in a given class and a vocabulary set, create a 
#dictionary where each class is linked to a dictionary of the probabilities of words
#in the vocabulary in that class
def buildProbDictionary(classWordsDict, vocab):
    probDictionary = {}
    spam = {}
    nonSpam = {}
    spamDenom = len(vocab)
    nonSpamDenom = len(vocab)

    #Calculate both denominators
    for word in classWordsDict["spam"]:
        spamDenom = spamDenom + classWordsDict["spam"][word]
    for word in classWordsDict["nonspam"]:
        nonSpamDenom = nonSpamDenom + classWordsDict["nonspam"][word]
    
    #For each word in the vocabulary, calculate its probability in each class
    for word in vocab:
        if word in classWordsDict["spam"]: 
            spam[word] = float(classWordsDict["spam"][word] + 1) / float(spamDenom)
        else:
            spam[word] = float(1) / float(spamDenom)
        if word in classWordsDict["nonspam"]:
            nonSpam[word] = float(classWordsDict["nonspam"][word] + 1) / float(nonSpamDenom)
        else:
            nonSpam[word] = float(1) / float(nonSpamDenom)

    probDictionary["spam"] = spam
    probDictionary["nonSpam"] = nonSpam
    return probDictionary

#Given a file and a dictionary of probabilities, guess the class that the file belongs to.
#Return 1 if it is nonspam, 0 if it is spam
def guessClass(testFile, path, probDictionary):
    currFile = open(path + testFile)
    words = currFile.read().split()
    totalNonSpamProb = math.log(float(probDictionary["nonspamPrior"]), 2)
    totalSpamProb = math.log(float(probDictionary["spamPrior"]), 2)
    
    currFile.close()
    #For each word in the given file, add its log space probability to the 
    #running sum based on the probabilities in the probDictionary
    for word in words:
	if (word in probDictionary["nonSpam"]):
	    totalNonSpamProb = totalNonSpamProb + math.log(probDictionary["nonSpam"][word], 2)
	if (word in probDictionary["spam"]):
		totalSpamProb = totalSpamProb + math.log(probDictionary["spam"][word], 2)
    if (totalNonSpamProb > totalSpamProb):
	return 1 #The model guessed the file is nonspam
    else:
	return 0 #The model guessed the file is spam

def trainModel(nsTestFiles, nsTrainFiles, sTestFiles, sTrainFiles):
    #Calculate Prior Probabilities
    totalFiles = len(nsTrainFiles) + len(sTrainFiles)
    spamProb = float(len(sTrainFiles)) / float(totalFiles)
    nonspamProb = float(len(nsTrainFiles)) / float(totalFiles)

    wordDictionary = genTotalWordDictionary(nsTestFiles, nsTrainFiles, sTestFiles, sTrainFiles)            #Dictionary of the 2500 most frequent words
    fileDictionary = genFileDictionary(nsTestFiles, nsTrainFiles, sTestFiles, sTrainFiles, wordDictionary) #Dictionary containing file names linked to their dictionary of common words
    classWordsDictionary = genClassWordsDictionary(fileDictionary, sTrainFiles, nsTrainFiles)              #Dictionary containing classes linked to a dictionary of words and their frequencies
    vocab = getVocab(wordDictionary) #Vocabulary of the 2500 most frequent words

    #Calculate parameter probabilites
    probDictionary = buildProbDictionary(classWordsDictionary, vocab);
    probDictionary["spamPrior"] = spamProb
    probDictionary["nonspamPrior"] = nonspamProb
    return probDictionary

#Given a model and some labeled test files, calculate the precision, recall, and f-score of the model
def testModel(model, sTestFiles, nsTestFiles):
    truePos = 0
    falsePos = 0
    trueNeg = 0
    falseNeg = 0

    #Test the model against files labeled as nonspam
    for testFile in nsTestFiles:
        if (guessClass(testFile, "data/nonspam-test/", model) == 1):
            truePos = truePos + 1
        else:
            falsePos = falsePos + 1
    #Test the model against files labeled as spam
    for testFile in sTestFiles:
        if (guessClass(testFile, "data/spam-test/", model) == 0):
            trueNeg = trueNeg + 1
        else:
            falseNeg = falseNeg + 1

    print("\n          2 x 2 Contingency Table         ")
    print("__________________________________________")
    print("             |   Correct   | Not Correct |")
    print("_____________|_____________|_____________|")
    print("   Non-Spam  |     " + str(truePos) + "     |      " + str(falsePos) + "      |")
    print("_____________|_____________|_____________|")
    print("   Spam      |     " + str(trueNeg) + "     |      " + str(falseNeg) + "      |")
    print("_____________|_____________|_____________|")

    #Calculate and print precision, recall, and f-score
    prec = float(truePos) / (float(truePos) + float(falsePos))
    recall = float(truePos) / (float(truePos) + float(falseNeg))
    fScore = 2*((prec * recall) / (prec + recall))
    print("\nPrecision: " + str(prec * 100) + "%")
    print("Recall: " + str(recall * 100) + "%")
    print("F-score: " + str(fScore * 100) + "%")

def main():
    #Generate lists of txt files in the data directory  
    nsTestFiles = os.listdir("data/nonspam-test/")
    nsTrainFiles = os.listdir("data/nonspam-train/")
    sTestFiles = os.listdir("data/spam-test/")
    sTrainFiles = os.listdir("data/spam-train/")

    model = trainModel(nsTestFiles, nsTrainFiles, sTestFiles, sTrainFiles)
    
    testModel(model, sTestFiles, nsTestFiles)

main()
