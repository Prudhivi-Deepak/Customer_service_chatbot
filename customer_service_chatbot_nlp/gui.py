import nltk
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
import pickle
import numpy as np
from keras.models import load_model
import json
import random
model = load_model('chatbot_model1.h5')
intents = json.loads(open('intents1.json').read())
classes = pickle.load(open('classes1.pkl','rb'))
print(intents['intents'][0])
print(classes)


def bagofwords(sentence, words):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    bag = [0]*len(words) 
    for s in sentence_words:
        for i,w in enumerate(words):
            if w == s: 
                bag[i] = 1
                
    return(np.array(bag))

def predict_intent(sentence, model):
    print("sentence : ",sentence)
    words = pickle.load(open('words1.pkl','rb'))
    p = bagofwords(sentence, words)
    res = model.predict(np.array([p]))[0]
    print("res",res)
    results = [[i,r] for i,r in enumerate(res) if r>0.25]
    print(classes)
    print(results)
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({"intent": classes[r[0]], "probability": str(r[1])})
    print(return_list)
    return return_list

def chatbot_response(text):
    ints = predict_intent(text, model)
    print("ints",ints)
    tag = ints[0]['intent']
    list_of_intents = intents['intents']
    result=""
    for i in list_of_intents:
        if(i['tag']== tag):
            result = random.choice(i['responses'])
            break
    return result

from flask import Flask, render_template, flash, redirect, request, url_for, send_file, session

app = Flask(__name__)
app.secret_key = "dhutrr"


@app.route('/')
def hello_world():
    session.permanent = True 
    session["all"]=[]
    return render_template("chatbot.html")

@app.route('/chatbot',methods=["POST","GET"])
def admin1():
    print("before session : ",session["all"])
    msg=""
    newlist=[]
    if request.method == 'POST':
        message=request.form['message']
        print(message)
        response = chatbot_response(message)
        print("Response :",response)
        dict1 = {
            "message1" : message,
             "response1" : response
        }
        session["all"].insert(0,dict1)
        print("After session all : ",session["all"])
    return render_template("chatbot.html",response = response,history = session["all"])

if __name__ == "__main__":
    app.run(debug=True)
