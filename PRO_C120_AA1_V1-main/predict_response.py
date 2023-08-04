import nltk
import json
import pickle
import numpy as np
import random
import tensorflow
from data_preprocessing import get_stem_words


ignore_words = ['?', '!',',','.', "'s", "'m"]
model_load = tensorflow.keras.models.load_model("./chatbot_model.h5")
intents = json.loads(open("./intents.json").read())
r_words = pickle.load(open("./words.pkl", "rb"))
t_words = pickle.load(open("./classes.pkl", "rb"))

def w_process(text_input):
    tokened_word = nltk.word_tokenize(text_input)
    stem_token = get_stem_words(tokened_word, ignore_words)
    stem_token = sorted(list(set(stem_token)))
    w_bag1 = []
    w_bagmaster = []
    for rwordsanalize in r_words:
        if rwordsanalize in stem_token:
            w_bag1.append(1)
        else :
            w_bag1.append(0)
    w_bagmaster.append(w_bag1)
    return(w_bagmaster)

def text_p(text_input):
    w_processcall = w_process(text_input)
    w_psend = model_load.predict(w_processcall)
    p_select = np.argmax(w_psend[0])
    return(p_select)

def bot_response(text_input):
    text_pcall= text_p(text_input)
    class_save = t_words[text_pcall]
    for i_response in intents["intents"]:
        if i_response ["tag"] == class_save:
            r_response = random.choice(i_response["responses"])
            return(r_response)

print("Hi, Welcome im your new digital assistant")

while True:
    text_input = input("Type here your message")
    print("User:", text_input)
    bot_input = bot_response(text_input)
    print("bot:", bot_input)