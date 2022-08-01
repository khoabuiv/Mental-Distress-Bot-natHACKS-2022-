#!/usr/bin/python
import praw
import re
import random

# Quotes taken from: http://www.imdb.com/character/ch0007553/quotes
marvin_quotes = \
    [
        "I can help!",
        "Hang in there, you got this!",

    ]

reddit = praw.Reddit('bot1')

subreddit = reddit.subreddit("Mental_Health_Voices")

for comment in subreddit.stream.comments():
    print(comment.body)
    if re.search("Marvin Help", comment.body, re.IGNORECASE):
        marvin_reply = "Marvin the helper Robot says: " + \
            random.choice(marvin_quotes)
        comment.reply(marvin_reply)
        print(marvin_reply)
