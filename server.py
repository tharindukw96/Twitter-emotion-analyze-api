from flask import Flask,jsonify
import tweepy
import re
import calendar
import time
from nltk.tokenize import TweetTokenizer,word_tokenize
from nltk.corpus import sentiwordnet as sn
import nltk
import matplotlib.pyplot as plt
import pandas as pd
import os
os.environ['KERAS_BACKEND'] = 'theano'
from emotion_predictor import EmotionPredictor
from flask_cors import CORS, cross_origin

app = Flask(__name__)

cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'


@app.route("/analyze/<keyword>", methods=['GET'])
@cross_origin()
def analyze(keyword):
    consumer_key = "mYf39MsctHYfzdqna2kLu28K5"
    consumer_secret = "HexjZTEwS8r8swe40clOrQaISCPN7jzoKVflLvGXqEGRvVpTuh"

    access_token = "1068448554-8K0mSRfBzkAh3mu1K6dPodhEK4d7ncIrWI1y4S8"
    access_token_secret = "0g8PRIseg53l8ISa4p55tFGl98WomOSgnnxT0NuLZOJCy"

    auth  = tweepy.OAuthHandler(consumer_key,consumer_secret)
    auth.set_access_token(access_token,access_token_secret)

    api  = tweepy.API(auth)

    public_tweets = api.search(keyword,lang='en',count='100')
    public_tweets += api.search(keyword,lang='en',count='100')
    public_tweets += api.search(keyword,lang='en',count='200')
    tweets = []
    times = []
    for tweet in public_tweets:
        trim = str(tweet.created_at)
        t  = calendar.timegm(time.strptime(trim,"%Y-%m-%d  %H:%M:%S"))
        tweets.append(remove_punct(tweet.text))
        times.append(t)
    #Time dataframe
    d = {'Time':times}
    #print(d)
    df = pd.DataFrame(data=d)
    # Pandas presentation options
    pd.options.display.max_colwidth = 150   # show whole tweet's content
    pd.options.display.width = 200          # don't break columns
    # pd.options.display.max_columns = 7      # maximal number of columns


    # Predictor for Ekman's emotions in multiclass setting.
    model = EmotionPredictor(classification='ekman', setting='mc', use_unison_model=True)
    predictions = model.predict_classes(tweets)
    predictions = pd.concat([predictions, df], axis=1, sort=False)
    #m = predictions.groupby(["Time","Emotion"]).size().reset_index(name='counts')

    
    #print(df)
    #############################
    m = predictions.groupby(["Time","Emotion"]).size().unstack(fill_value=0).stack().reset_index(name='counts')
    #print(m)
    df = m
    joy = df.loc[df['Emotion'] == 'Joy'].filter(items=['Time', 'counts'])
    anger = df.loc[df['Emotion'] == 'Anger'].filter(items=['Time', 'counts'])
    fear = df.loc[df['Emotion'] == 'Fear'].filter(items=['Time', 'counts'])
    sadness = df.loc[df['Emotion'] == 'Sadness'].filter(items=['Time', 'counts'])
    surprise = df.loc[df['Emotion'] == 'Surprise'].filter(items=['Time', 'counts'])
    disgust = df.loc[df['Emotion'] == 'Disgust'].filter(items=['Time', 'counts'])
   
    #m.to_csv('results.csv', index=True, header=True)
    #Read the dataframe and divided into lists
    #df.plot(kind='scatter',x='Time',y='0')
    #plt.show()
    #print(m[m.columns[2]],m[m.columns[1]])
    
    #print(predictions, '\n')
    #print(times)

    return jsonify({"joy":joy.to_json(orient='values'),"anger":anger.to_json(orient='values'),"fear":fear.to_json(orient='values'),"sadness":sadness.to_json(orient='values'),"surprise":surprise.to_json(orient='values'),"disgust":disgust.to_json(orient='values')})

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