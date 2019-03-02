from flask import Flask,jsonify
import tweepy
import re
import calendar
import time
from nltk.tokenize import TweetTokenizer,word_tokenize
from nltk.corpus import sentiwordnet as sn
import nltk
app = Flask(__name__)

@app.route("/analyze/<keyword>", methods=['GET'])
def analyze(keyword):
    consumer_key = "mYf39MsctHYfzdqna2kLu28K5"
    consumer_secret = "HexjZTEwS8r8swe40clOrQaISCPN7jzoKVflLvGXqEGRvVpTuh"

    access_token = "1068448554-8K0mSRfBzkAh3mu1K6dPodhEK4d7ncIrWI1y4S8"
    access_token_secret = "0g8PRIseg53l8ISa4p55tFGl98WomOSgnnxT0NuLZOJCy"

    auth  = tweepy.OAuthHandler(consumer_key,consumer_secret)
    auth.set_access_token(access_token,access_token_secret)

    api  = tweepy.API(auth)

    public_tweets = api.search(keyword,lang='en',count='100')
    #public_tweets += api.search(keyword,lang='en',count='100')
    tweets = []
    public_tweets = [
    "Watching the sopranos again from start to finish!",
    "Finding out i have to go to the  dentist tomorrow",
    "I want to go outside and chalk but I have no chalk",
    "I HATE PAPERS AH #AH #HATE",
    "My mom wasn't mad",
    "Do people have no Respect for themselves or you know others peoples homes",
]
    for tweet in public_tweets:
        #print(tweet.text)
        #trim = str(tweet.created_at)
        #t  = calendar.timegm(time.strptime(trim,"%Y-%m-%d  %H:%M:%S"))
        #tweets.append([posTagging(tokenize(remove_punct(tweet.text))),tweet.created_at])
        #tweets.append([knowledgeBaseValidation(tokenize(remove_punct(tweet.text))),tweet.text,t])
        tweets.append([knowledgeBaseValidation(tokenize(remove_punct(tweet))),tweet])
    return jsonify({"tweets":tweets})

def remove_punct(text):
    punctuations ="!\"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~'"
    #remove urls from tweet
    text = re.sub('https?://(?:[-\w.]|(?:%[\da-fA-F]{2}))+','',text) 
    #remove punctuations from tweets
    text  = "".join([char for char in text if char not in punctuations])
    #Remove numbers from tweets
    text = re.sub('[0-9]+', '', text)
    #remove RT(Retweet) mark from tweets
    text = text.replace("RT","")
    #Convert to lower text
    text = text.lower()
    return text

def tokenize(text):
    tknzr = TweetTokenizer()
    tokenz = tknzr.tokenize(text)
    return tokenz

def posTagging(tokenz):
    return nltk.pos_tag(tokenz)

def knowledgeBaseValidation(text):
    classArr= []
    for word in text:
        syns = sn.senti_synsets(word)
        pos = 0
        neg = 0
        for j in syns:
            pos += j.pos_score()
            neg+= j.neg_score()
            break
        if(pos == 0):
            if(neg < -0.1 and neg > -0.5 ):
                classArr.append(2)
            elif(neg >= -1 and neg <= -0.5):
                classArr.append(3)
        else:
            if(pos > 0.1 and pos < 0.5 ):
                classArr.append(1)
            elif(pos >= 0.5 and pos <= 1):
                classArr.append(0)
    if(len(classArr)==0):
        return "null"
    else:
        return max(set(classArr),key=classArr.count)

if __name__ == '__main__':
    app.run(debug=True)