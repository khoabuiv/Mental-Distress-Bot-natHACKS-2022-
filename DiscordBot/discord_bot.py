import discord
import os
import requests
import json
import random
from model import ClassifierModel
from hatesonar import Sonar
sonar = Sonar()
client = discord.Client()

sad_words = ["sad", "depressed", "unhappy", "angry", "miserable"]
negative_emotions = set(["lonely", "anxious", "depressed", "stressed"])
starter_encouragements = [
    "Cheer up!",
    "Hang in there.",
    "You are a great person / bot!"
]


def get_quote():
    response = requests.get("https://zenquotes.io/api/random")
    json_data = json.loads(response.text)
    quote = json_data[0]['q'] + " -" + json_data[0]['a']
    return(quote)


@client.event
async def on_ready():
    print('We have logged in as {0.user}'.format(client))


@client.event
async def on_message(message):
    if message.author == client.user:
        return

    msg = message.content

    if message.content.startswith('$inspire'):
        quote = get_quote()
        await message.channel.send(quote)
    #TODO- Figure out this dm
    potential_hate_speech = await sonar.ping(text=msg)
    if "top_class"  in potential_hate_speech and potential_hate_speech["top_class"] == "hate_speech":
        return
    if ClassifierModel.predict_label(msg) == "suicide":
        return
    if ClassifierModel.predict_label(msg) in negative_emotions:
        return
    if any(word in msg for word in sad_words):
        await message.channel.send(random.choice(starter_encouragements))

client.run('TOKEN')
