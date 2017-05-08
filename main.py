#!/usr/bin/env python
# -*- coding: utf-8 -*-

import tweepy
import json
import string
import os
import argparse
import time
import math
import re

# NOTE: register with Twitter API and fill out the below fields
# twitter authentication
auth = tweepy.OAuthHandler('', '')
auth.set_access_token('', '')

api = tweepy.API(auth)

# global vars for sentiment analysis
WORDS = {}
POSITIVE_VOCABULARY_SIZE = 0
NEGATIVE_VOCABULARY_SIZE = 0
TOTAL_POSITIVE_WORD_COUNT = 0
TOTAL_NEGATIVE_WORD_COUNT = 0
SENTIMENT_RANKING = []

# global vars for cross entropy
tri_entropy_counts = [0, 0, 0, 0, 0] # 0-.2, .2-.4, .4-.6, .6-.8, .8-1
bi_entropy_counts = [0, 0, 0, 0, 0]

# characters that will be considered punctuation to remove.
# note the hashtag character # will not be removed.
PUNCTUATION = '!"$%&\'()*+,-./:;<=>?@[\\]^_`{|}~'


## ---------- Querying functions ----------

def getAllReplys(tweet_id):
    while True:
        try:
            new_tweets = api.user_timeline(q=query, count=count, max_id=str(last_id - 1))
            if not new_tweets:
                print "no more new tweets"
                break

            searched_tweets += len(new_tweets)
            for tweet in new_tweets:
               outfile.write(json.dumps(tweet._json))
               outfile.write('\n')

            last_id = new_tweets[-1].id

        except tweepy.TweepError as e:
            print "hit rate limit...sleeping for 15 minutes", e
            time.sleep(15 * 60 + 1)

def getUserTimeline(screen_name):
    results = []
    last_id = 0
    while True:
        try:
            if last_id is 0:
                new_tweets = api.user_timeline(id=screen_name, include_rts=1, count=200)
            else:
                new_tweets = api.user_timeline(id=screen_name, include_rts=1, count=200, max_id=str(last_id - 1))

            if not new_tweets:
                print "no more new tweets for", screen_name
                break
            else:
                print new_tweets[0].id, " - ", new_tweets[-1].id

            for tweet in new_tweets:
               results.append(json.dumps(tweet._json))

            last_id = new_tweets[-1].id

        except tweepy.TweepError as e:
            ratelimited = True
            if e[0]:
                g = e[0][0]
                if g["code"] is 131:
                    print "internal error, ignoring"
                    ratelimited = False
            if ratelimited:
                print "hit rate limit...sleeping for 15 minutes", e
                time.sleep(15 * 60 + 1)

    return results

# return a string of contiguous tweet texts
def getUserCorpus(screen_name):
    text = ""
    with open("./timelines/" + screen_name) as inputfile:
        for t in inputfile:
            tObj = json.loads(json.loads(t))
            text += tObj['text'] + " "
    return text.lower()

# return an array of tweet texts
def getUserCorpusSiloed(screen_name):
    tweets = []
    with open("./timelines/" + screen_name) as inputfile:
        for t in inputfile:
            tObj = json.loads(json.loads(t))
            # convert to lower case and replace with NAME
            # text = re.sub(r'(?<=^|(?<=[^a-zA-Z0-9-\.]))#([A-Za-z]+[A-Za-z0-9-]+)', 'NAME', tObj['text'].lower())
            text = re.sub(r'(^|[^@\w])@(\w{1,15})\b', ' NAME ', tObj['text'].replace('\n', ' ').lower())
            # remove punctuation
            exclude = set(PUNCTUATION)
            tt = ''
            for ch in text:
                if ch not in exclude:
                    tt += ch
                else:
                    tt += " "
            tweets.append((tt, tObj['id_str']))
    return tweets


## ---------- Cross entropy functions ----------

def calculateCrossEntropy(tw):
    screen_name = tw['user']['screen_name']
    corpus = getUserCorpusSiloed(screen_name)
    words = tw['text'].lower().split()

    tri_probs = generateTrigramProbabilities(generateTrigrams(screen_name, corpus))
    bi_probs = generateBigramProbabilities(generateBigrams(screen_name, corpus))

    tri_entropy = 0.0
    bi_entropy = 0.0

    w0 = words[0]
    w1 = words[1]
    words = words[2:]

    for w2 in words:
        tri_entropy += (tri_probs[w0][w1][w2] * math.log(tri_probs[w0][w1][w2], 2))
        bi_entropy +=  (bi_probs[w0][w1] * math.log(bi_probs[w0][w1], 2))

        w0 = w1
        w1 = w2

    bi_entropy += (bi_probs[w0][w1] * math.log(bi_probs[w0][w1], 2)) # capture last bigram

    tri_entropy = -(tri_entropy / len(words))
    bi_entropy =  -(bi_entropy / len(words))

    return (bi_entropy, tri_entropy)


def calculateAllCrossEntropies(tweetArr, screen_name):
    corpus = getUserCorpusSiloed(screen_name)

    tri_probs = generateTrigramProbabilities(generateTrigrams(screen_name, corpus))
    bi_probs = generateBigramProbabilities(generateBigrams(screen_name, corpus))

    tri_res = []
    bi_res = []

    for tw in tweetArr:
        # convert to lower case and replace with NAME
        # text = re.sub(r'(?<=^|(?<=[^a-zA-Z0-9-\.]))#([A-Za-z]+[A-Za-z0-9-]+)', 'NAME', tw['text'].lower())
        text = re.sub(r'(^|[^@\w])@(\w{1,15})\b', ' NAME ', tw['text'].replace('\n', ' ').lower())
        # remove punctuation
        exclude = set(PUNCTUATION)
        tt = ''
        for ch in text:
            if ch not in exclude:
                tt += ch
            else:
                tt += " "

        words = tt.split()

        tri_entropy = 0.0
        bi_entropy = 0.0

        if len(words) < 3: continue

        w0 = words[0]
        w1 = words[1]
        words = words[2:]

        for w2 in words:
            tri_entropy += (tri_probs[w0][w1][w2] * math.log(tri_probs[w0][w1][w2], 2))
            bi_entropy +=  (bi_probs[w0][w1] * math.log(bi_probs[w0][w1], 2))

            w0 = w1
            w1 = w2

        bi_entropy += (bi_probs[w0][w1] * math.log(bi_probs[w0][w1], 2)) # capture last bigram

        tri_entropy = -(tri_entropy / len(words))
        bi_entropy =  -(bi_entropy / len(words))

        tri_res += [(tri_entropy, tw['text'], tt)]
        bi_res += [(bi_entropy, tw['text'], tt)]

        pigeonholeEntropy(tri_entropy, True)
        pigeonholeEntropy(bi_entropy, False)

    tri_res = sorted(tri_res)
    bi_res = sorted(bi_res)
    print len(tri_res), "trigrams;", len(bi_res), "bigrams"
    print tri_entropy_counts, bi_entropy_counts
    print "trigrams:", tri_res[0:10], "  ...  ", tri_res[-10:]
    print "bigrams:", bi_res[0:10], "  ...  ", bi_res[-10:]


def pigeonholeEntropy(val, istri):
    global tri_entropy_counts, bi_entropy_counts

    i = 0
    if val >= 0.8:
        i = 4
    elif val >= 0.6:
        i = 3
    elif val >= 0.4:
        i = 2
    elif val >= 0.2:
        i = 1

    if istri:
        tri_entropy_counts[i] += 1
    else:
        bi_entropy_counts[i] += 1


def generateTrigrams(screen_name, tweets):
    trigrams = {}

    for tweet in tweets:
        words = tweet[0].split()

        if len(words) > 2:
            w0 = words[0]
            w1 = words[1]
            words = words[2:]

            for w2 in words:
                if w0 not in trigrams:
                    trigrams[w0] = {}
                if w1 not in trigrams[w0]:
                    trigrams[w0][w1] = {}
                if w2 not in trigrams[w0][w1]:
                    trigrams[w0][w1][w2] = 0

                trigrams[w0][w1][w2] += 1
                w0 = w1
                w1 = w2

    return trigrams

def generateTrigramProbabilities(counts):
    SMOOTHING = 0.1
    probs = {}

    for w0, val0 in counts.iteritems():
        probs[w0] = {}
        for w1, val1 in val0.iteritems():
            probs[w0][w1] = {}
            countSum = 0.0
            for w2, val2 in val1.iteritems():
                countSum += (val2 + SMOOTHING)
            for w2, val2 in val1.iteritems():
                if countSum > 0:
                    probs[w0][w1][w2] = (val2 + SMOOTHING)/countSum
                else:
                    probs[w0][w1][w2] = 0

    return probs

def generateBigrams(screen_name, tweets):
    bigrams = {}

    for tweet in tweets:
        words = tweet[0].split()

        if len(words) > 1:
            w0 = words[0]
            words = words[1:]

            for w1 in words:
                if w0 not in bigrams:
                    bigrams[w0] = {}
                if w1 not in bigrams[w0]:
                    bigrams[w0][w1] = 0

                bigrams[w0][w1] += 1
                w0 = w1

    return bigrams

def generateBigramProbabilities(counts):
    SMOOTHING = 0.1
    probs = {}

    for w0, val0 in counts.iteritems():
        probs[w0] = {}
        countSum = 0.0
        for w1, val1 in val0.iteritems():
            countSum += (val1 + SMOOTHING)
        for w1, val1 in val0.iteritems():
            if countSum > 0:
                probs[w0][w1] = (val1 + SMOOTHING)/countSum
            else:
                probs[w0][w1] = 0

    return probs


## ---------- Bayes functions ----------

def calculateLeaveOneOut(tw):
    screen_name = tw['user']['screen_name']
    corpus = getUserCorpus(screen_name)
    arr = corpus.split()
    class_words = {}

    for word in arr:
        if word in class_words:
            class_words[word] += 1
        else:
            class_words[word] = 0
    class_word_count = len(arr)
    class_vocab_size = len(class_words)

    print "corpus size:", class_word_count

    tw_text = tw['text'].split()
    # tw_words = {}
    # for word in tw_text:
    #     if word in tw_words:
    #         tw_words[word] += 1
    #     else:
    #         tw_words[word] = 0

    SMOOTH = 0.1
    likelihood = 1.0
    for word in tw_text:
        if word in class_words:
            word_count = class_words[word]
        else:
            word_count = 0
        likelihood = likelihood * ((word_count + SMOOTH) / (class_word_count + class_vocab_size * SMOOTH))

    return likelihood


## ---------- Sentiment functions ----------

def trainSentiment(screen_name):
    global WORDS, POSITIVE_VOCABULARY_SIZE, NEGATIVE_VOCABULARY_SIZE, TOTAL_POSITIVE_WORD_COUNT, TOTAL_NEGATIVE_WORD_COUNT, SENTIMENT_RANKING

    # train sentiment models
    for filename in os.listdir("./tweets/positive"):
        with open("./tweets/positive/" + filename) as inputfile:
            for tweet in inputfile:
                consumeSentimentTweet(tweet, True)

    for filename in os.listdir("./tweets/negative"):
        with open("./tweets/negative/" + filename) as inputfile:
            for tweet in inputfile:
                consumeSentimentTweet(tweet, False)

    # remove lower-occuring words
    temp = {}
    for word, obj in WORDS.iteritems():
        if obj['count'] > 5:
            temp[word] = obj
    WORDS = temp

    # find usual sentiment of user timeline
    corpus = getUserCorpusSiloed(screen_name)
    positive_count = 0
    negative_count = 0
    for tweet in corpus:
        if bayesianSentimentClassify(tweet):
            positive_count += 1
        else:
            negative_count += 1
    ratio = positive_count / negative_count
    print "sentiment profile:", positive_count, negative_count, ratio
    SENTIMENT_RANKING = sorted(SENTIMENT_RANKING)
    print SENTIMENT_RANKING[0:10], "  ...  ", SENTIMENT_RANKING[-10:]

    for o in SENTIMENT_RANKING:
        print o


def consumeSentimentTweet(tweet, positive):
    global WORDS, POSITIVE_VOCABULARY_SIZE, NEGATIVE_VOCABULARY_SIZE, TOTAL_POSITIVE_WORD_COUNT, TOTAL_NEGATIVE_WORD_COUNT

    tw = json.loads(tweet)
    # convert to lower case and replace with NAME
    text = re.sub(r'(^|[^@\w])@(\w{1,15})\b', ' NAME ', tw['text'].replace('\n', ' ').lower())
    # text = re.sub(r'(?<=^|(?<=[^a-zA-Z0-9-\.]))#([A-Za-z]+[A-Za-z0-9-]+)', 'NAME', tw['text'].lower())
    # remove punctuation
    exclude = set(PUNCTUATION)
    tt = ''
    for ch in text:
        if ch not in exclude:
            tt += ch
        else:
            tt += " "

    words = tt.split()

    for word in words:
        if word in WORDS:
            WORDS[word]['count'] += 1
        else:
            WORDS[word] = {
                "count": 1,
                "positiveCount": 0,
                "negativeCount": 0
            }
            if positive:
                POSITIVE_VOCABULARY_SIZE += 1
            else:
                NEGATIVE_VOCABULARY_SIZE += 1
        if positive:
            WORDS[word]['positiveCount'] += 1
            TOTAL_POSITIVE_WORD_COUNT += 1
        else:
            WORDS[word]['negativeCount'] += 1
            TOTAL_NEGATIVE_WORD_COUNT += 1

    return WORDS

# Calculate probabilty of string against both negative and positive
# corpus
def bayesianSentimentClassify(tweet):
    global SENTIMENT_RANKING
    PRIOR = 0.5
    positiveProbability = PRIOR * calculateTextLikelihood(tweet[0], True)
    negativeProbability = PRIOR * calculateTextLikelihood(tweet[0], False)
    ratio = positiveProbability / negativeProbability
    SENTIMENT_RANKING += [(ratio, tweet[1])]
    return positiveProbability > negativeProbability

# Use calculateWordLikelihood to sum the total log likelihood of
# a given string, against given positive or negative
def calculateTextLikelihood(string, givenPositive):
    likelihood = 0.0
    arr = string.split()
    for word in arr:
        likelihood += calculateWordLikelihood(word, givenPositive)
    return likelihood

def calculateWordLikelihood(word, givenPositive):
    SMOOTHING = 0.1

    word_count = 0
    if givenPositive:
        total_word_count = TOTAL_POSITIVE_WORD_COUNT
        vocabulary_size = POSITIVE_VOCABULARY_SIZE
    else:
        total_word_count = TOTAL_NEGATIVE_WORD_COUNT
        vocabulary_size = NEGATIVE_VOCABULARY_SIZE

    if word in WORDS:
        if givenPositive:
            word_count = WORDS[word]["positiveCount"]
        else:
            word_count = WORDS[word]["negativeCount"]

    return math.log((word_count + SMOOTHING) / (total_word_count + vocabulary_size * SMOOTHING))


# arg 1: path of replies to analyze
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('function') # get_profiles (from thread) | reaction (from tweet_id)
    parser.add_argument('arg1') # path to thread | tweet_id
    args = parser.parse_args()

    if args.function == "get_profiles":
        i = -1
        with open(args.arg1) as inputfile:
            with open("viable_tweets", 'w') as viable:
                for tweet in inputfile:
                    i += 1
                    if i is 0: continue

                    t = json.loads(tweet)
                    uid = t["user"]["screen_name"]
                    timeline_path = "./timelines/" + uid

                    # don't collect profiles we already have
                    if os.path.isfile(timeline_path):
                        viable.write(tweet)
                        viable.write('\n')
                        continue

                    user = api.get_user(uid)
                    print user.statuses_count
                    # don't collect profiles with less than x tweets
                    if user.statuses_count < 20000: continue
                    viable.write(tweet)
                    viable.write('\n')

                    with open(timeline_path, 'w') as outfile:
                        for res in getUserTimeline(uid):
                            outfile.write(json.dumps(res))
                            outfile.write('\n')

    elif args.function == 'reactions':
        with open(args.arg1) as inputfile:
            for tweet in inputfile:
                if len(tweet) < 10: continue
                tw = json.loads(tweet)
                reaction(tw)

    elif args.function == 'reaction':
        tw = api.get_status(args.arg1)
        reaction(tw._json)

    elif args.function == 'entropies':
        with open("./timelines/" + args.arg1) as inputfile:
            tweetArr = []
            for tweet in inputfile:
                if len(tweet) < 10: continue
                tw = json.loads(json.loads(tweet))
                tweetArr.append(tw)
        calculateAllCrossEntropies(tweetArr, args.arg1)

    elif args.function == 'sentiment':
        trainSentiment(args.arg1)


    else:
        print "bad command"



def reaction(tw):
    screen_name = tw.get("user").get("screen_name")
    print "author:", screen_name

    likelihood = calculateLeaveOneOut(tw)
    print "likelihood:", likelihood

    entropy = calculateCrossEntropy(tw)
    print "entropy (bi, tri):", entropy



if __name__ == '__main__':
    main()