import pandas as pd
import numpy as np
import nltk
import re
from textblob import TextBlob
from textblob import Word
nltk.download("punkt")
nltk.download("stopwords")
from nltk.tokenize import word_tokenize
from nltk.stem.porter import PorterStemmer
import string
from nltk.corpus import stopwords
class Wrangler:
    def __init__(self):
        self.ANXIOUS_TWEETS_PATH = "data/Anxious_Tweets.csv"
        self.LONELY_TWEETS_PATH = "data/Lonely_Tweets.csv"
        self.NORMAL_TWEETS_PATH = "data/Normal_Tweets.csv"
        self.SENTIMENT_TWEETS_PATH = "data/sentiment_tweets3.csv"
        self.STRESSED_TWEETS_PATH = "data/Stressed_Tweets.csv"
        self.SUICIDE_DETECTION_PATH = "data/Suicide_Detection.csv"
    def getLonelyTweets(self):
        df = pd.read_csv(self.LONELY_TWEETS_PATH)
        lonely_col = []
        for i in range(len(df)):
            lonely_col.append("lonely")
        df["label"] = lonely_col
        df.columns = ["id", "text", "label"]
        df.drop('id', inplace=True, axis=1)
        return df
    def getAnxiousTweets(self):
        df = pd.read_csv(self.ANXIOUS_TWEETS_PATH)
        anxious_col = []
        for i in range(len(df)):
            anxious_col.append("anxious")
        df["label"] = anxious_col
        df.columns = ["id", "text", "label"]
        df.drop('id', inplace=True, axis=1)
        return df
    def getStressedTweets(self):
        df = pd.read_csv(self.STRESSED_TWEETS_PATH)
        stressed_col = []
        for i in range(len(df)):
            stressed_col.append("stressed")
        df["label"] = stressed_col
        df.columns = [ "text", "label"]
        # df.drop('id', inplace=True, axis=1)
        return df
    def getNormalTweets(self):
        df = pd.read_csv(self.NORMAL_TWEETS_PATH)
        normal_col = []
        for i in range(len(df)):
            normal_col.append("normal")
        df["label"] = normal_col
        df.columns = ["id", "text", "label"]
        df.drop('id', inplace=True, axis=1)
        return df
    def getDepressedTweets(self):
        df = pd.read_csv(self.SENTIMENT_TWEETS_PATH)
        dep_dict = {0: "not depressed", 1: "depressed"}
        df.columns = ["id", "text", "label"]
        df["label"].replace(dep_dict, inplace=True)
        df.drop('id', inplace=True, axis=1)
        return df
    def getSuicidalTweets(self):
        df = pd.read_csv(self.SUICIDE_DETECTION_PATH)
        df.columns = ["id", "text", "label"]
        df.drop('id', inplace=True, axis=1)
        return df 
    def textToLower(self,df):
        df["text"] = df["text"].apply(str.lower)
    def applyReduceLetterRepeats(self,df):
        # rx = re.compile(r'([^\W\d_])\1{2,}')
        rx = re.compile(r'(\w)\1{2,}')
        df["text"] = df["text"].apply(lambda x: [self.reduceLetterRepeats(y,rx) for y in x])
        # df["text"] =  df["text"].str.replace(rx, )
    def reduceLetterRepeats(self,text,rx):
        # return re.sub(r'[^\W\d_]+', lambda t: Word(rx.sub(r'\1\1', t.group())).correct() if rx.search(t.group()) else t.group(), text)
        return re.sub(r'(\w)\1{2,}', lambda t: Word(rx.sub(r'\1\1', t.group())).correct() , text)

    def removePunctuation(self,df):
        table = str.maketrans('', '', string.punctuation)
        df["text"] = df["text"].apply(lambda x: [y.translate(table) for y in x])
    def removeStopWord(self,df,stop_word):
        df["text"] = df["text"].apply(lambda x: [y for y in x if y not in stop_word])
    def tokenizeText(self,df):
        df["text"] = df["text"].apply(word_tokenize)
    def stemText(self,df,stemmer):
        df["text"] = df["text"].apply(lambda x: [stemmer.stem(y) for y in x])

        

if __name__ == "__main__":
    wrang = Wrangler()
    lonelydf = wrang.getLonelyTweets()
    anxiousdf = wrang.getAnxiousTweets()
    stresseddf = wrang.getStressedTweets()
    normaldf = wrang.getNormalTweets()
    depdf = wrang.getDepressedTweets()
    suidf = wrang.getSuicidalTweets()
    df = pd.concat([lonelydf, anxiousdf, stresseddf, normaldf, depdf, suidf])
    stop_words = set(stopwords.words("english"))
    print(stop_words)
    wrang.textToLower(df)
    wrang.tokenizeText(df)
    print("about to reduce repeats")
    wrang.applyReduceLetterRepeats(df)
    print("removing stop word")
    wrang.removeStopWord(df,stop_words)
    stem = PorterStemmer()
    wrang.stemText(df,stem)
    df.dropna(inplace=True)
    df.to_csv("data/text_dataset.csv", encoding="utf-8", index=False)

