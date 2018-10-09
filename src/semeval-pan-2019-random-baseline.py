#!/usr/bin/env python

"""Random baseline for the PAN19 hyperpartisan news detection task"""
# Version: 2018-09-24

# Parameters:
# --inputDataset=<directory>
#   Directory that contains the articles XML file with the articles for which a prediction should be made.
# --outputDir=<directory>
#   Directory to which the predictions will be written. Will be created if it does not exist.

from __future__ import division

import getopt
import os
import random
import sys
import xml.sax

from article import Article

# import pickle

random.seed(42)
runOutputFileName = "prediction.txt"
articles = {}



def parse_options():
    """Parses the command line options."""
    try:
        long_options = ["inputDataset=", "outputDir="]
        opts, _ = getopt.getopt(sys.argv[1:], "d:o:", long_options)
    except getopt.GetoptError as err:
        print(str(err))
        sys.exit(2)

    inputDataset = "undefined"
    outputDir = "undefined"

    for opt, arg in opts:
        if opt in ("-d", "--inputDataset"):
            inputDataset = arg
        elif opt in ("-o", "--outputDir"):
            outputDir = arg
        else:
            assert False, "Unknown option."
    if inputDataset == "undefined":
        sys.exit("Input dataset, the directory that contains the articles XML file, is undefined. Use option -d or --inputDataset.")
    elif not os.path.exists(inputDataset):
        sys.exit("The input dataset folder does not exist (%s)." % inputDataset)

    if outputDir == "undefined":
        sys.exit("Output path, the directory into which the predictions should be written, is undefined. Use option -o or --outputDir.")
    elif not os.path.exists(outputDir):
        os.mkdir(outputDir)

    return (inputDataset, outputDir)


########## SAX ##########

class HyperpartisanNewsRandomPredictor(xml.sax.ContentHandler):
    def __init__(self, outFile):
        xml.sax.ContentHandler.__init__(self)
        self.outFile = outFile
        self.previousId = "null"

    def startElement(self, name, attrs):
        if name == "article":
            articleId = attrs.getValue("id") # id of the article for which hyperpartisanship should be predicted
            self.previousId = articleId
            if articleId not in articles.keys():
                articles[articleId] = Article(articleId)
            if "hyperpartisan" in attrs.keys():
                articles[articleId].hyperpartisan = attrs.getValue("hyperpartisan")
            if "bias" in attrs.keys():
                articles[articleId].bias = attrs.getValue("bias")
            if "title" in attrs.keys():
                articles[articleId].title = attrs.getValue("title")

            if "labeled-by" in attrs.keys():
                articles[articleId].labeled = attrs.getValue("labeled-by")
            if "published-at" in attrs.keys():
                articles[articleId].published_at = attrs.getValue("published-at")
            prediction = random.choice(["true", "false"]) # random prediction
            confidence = random.random() # random confidence value for prediction
            # output format per line: "<article id> <prediction>[ <confidence>]"
            #   - prediction is either "true" (hyperpartisan) or "false" (not hyperpartisan)
            #   - confidence is an optional value to describe the confidence of the predictor in the prediction---the higher, the more confident
            self.outFile.write(articleId + " " + prediction + " " + str(confidence) + "\n")
        if name == "p":
            articles[self.previousId].count_paragraphs += 1
        if name == "q":
            articles[self.previousId].count_quotes += 1
        if name == "a":
            articles[self.previousId].count_urls += 1

    def characters(self, content):
        try:
            if self.previousId != "null":
                articles[self.previousId].text.append(content)
        except AttributeError:
            print(self.previousId)


########## MAIN ##########


def main(inputDataset, outputDir):
    """Main method of this module."""

    with open(outputDir + "/" + runOutputFileName, 'w') as outFile:
        for file in os.listdir(inputDataset):
            if file.endswith(".xml"):
                with open(inputDataset + "/" + file) as inputRunFile:
                    print(file)
                    xml.sax.parse(inputRunFile, HyperpartisanNewsRandomPredictor(outFile))

    # pickle.dump(articles, open("articles.p", "wb"))
    print("The predictions have been written to the output folder.")


if __name__ == '__main__':
    main(*parse_options())

