import os 
import pandas as pd
from collections import Counter

from textblob import TextBlob
import spacy

nlp = spacy.load("en_core_web_sm") #if you get error here download the model using -> "python -m spacy download en_core_web_sm"
class SentimentPhrases:
    def __init__(self,dataset):
        try:
            parent_dir = os.path.dirname(__file__)
            dataset_path = os.path.join(os.path.join(parent_dir,"dataset"),dataset)
            self.dataframe = pd.read_csv(dataset_path)
        except:
            self.dataframe = pd.read_csv(dataset)

    def blobPolarity(self,text):
        return TextBlob(text).sentiment.polarity
    def getSentiment(self,score):
        if score < 0:
            return "Negative"
        elif score == 0:
            return "Neutral"
        else:
            return "Positive"
    def sentimentAnalysis(self):
        self.dataframe["TextBlob-Polarity"] = self.dataframe["text"].apply(self.blobPolarity)
        self.dataframe["TextBlob-Sentiment"] = self.dataframe["TextBlob-Polarity"].apply(self.getSentiment)
    def extract_key_phrases(self,text):
        doc = nlp(text)
        key_phrases = Counter()
        for chunk in doc.noun_chunks:
            key_phrases[chunk.text] += 1
        total_phrases = sum(key_phrases.values())
        # Calculate relevance score for each key phrase
        key_phrases_with_scores = [(phrase, freq / total_phrases) for phrase, freq in key_phrases.items()]
        return key_phrases_with_scores
    def getScoreDict(self,text):
        key_phrases_with_scores = self.extract_key_phrases(text)
        relevance_score_list={}
        key_phrase_list=[]
        for key_phrase, relevance_score in key_phrases_with_scores:
            key_phrase_list.append(key_phrase)
            relevance_score_list[key_phrase] = relevance_score
        return relevance_score_list
    def getPhraseName(self,score):
        return [i for i in score]

    def keyPhrases(self):
        # Function to extract noun phrases with their frequencies
        self.dataframe["score"] = self.dataframe["text"].apply(self.getScoreDict)

        self.dataframe["key_phrases"] = self.dataframe["score"].apply(self.getPhraseName)
        # print(self.dataframe)
    def getPrediction(self):
        self.sentimentAnalysis()
        self.keyPhrases()
        return self.dataframe

if __name__=="__main__":
        
    obj = SentimentPhrases("interviewtext.csv")
    print(obj.getPrediction())