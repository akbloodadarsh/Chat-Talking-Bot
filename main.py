import nltk
import os
import time
from tkinter import * 
import speech_recognition as sr
from gtts import gTTS
from nltk.stem.lancaster import LancasterStemmer
stemmer = LancasterStemmer()
import pickle
import numpy 
import tensorflow
import random
import json
from keras.models import load_model
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.optimizers import SGD
# This module is imported so that we can play the converted audio 
from playsound import playsound
def speak(text):
	# Passing the text and language to the engine,here we have marked slow=False. 
	# Which tells the module that the converted audio should have a high speed 
	speech = gTTS(text = text, lang = 'en', slow = False)
	tts = gTTS(text=text, lang="en")
	# Saving the converted audio in mp3 format 
	speech.save("voice.mp3") 
	# Playing the converted file 
	playsound("voice.mp3") 

"""
def get_audio():
	r = sr.Recognizer()
	with sr.Microphone() as source:
		audio = r.listen(source)
		said = ""
		try:
			said = r.recognize_google(audio)
			print(said)
		except Exception as e:
			print("An Exception occured: "+ str(e))
	return said
"""

with open("intents.json") as file:
	data = json.load(file)
try:
	with open("data.pickle","rb") as f: 
		words,labels,training,output = pickle.load(f)
except:
	words = []
	labels = []
	docs_x = []
	docs_y = []

	for intent in data["intents"]:
		for pattern in intent["patterns"]:
			wrds = nltk.word_tokenize(pattern)
			words.extend(wrds)
			docs_x.append(wrds)
			docs_y.append(intent["tag"])
			
		if intent["tag"] not in labels:
			labels.append(intent["tag"])

	words = [stemmer.stem(w.lower()) for w in words if w != "?"]
	words = sorted(list(set(words)))		
	labels = sorted(labels)

	training = []
	output = []

	out_empty = [0 for _ in range(len(labels))]

	for x, doc in enumerate(docs_x):
		bag=[]
		wrds = [stemmer.stem(w) for w in doc]
		for w in words:
			if w in wrds:
				bag.append(1)
			else:
				bag.append(0)

		output_row = out_empty[:]
		output_row[labels.index(docs_y[x])] = 1

		training.append(bag)
		output.append(output_row)

	training = numpy.array(training)
	output = numpy.array(output)

	with open("data.pickle","wb") as f:
		pickle.dump((words,labels,training,output), f)

try:
	model = load_model("model.h5")

except: 
	model = Sequential()
	model.add(Dense(128, input_shape=(len(training[0]),), activation='relu'))
	model.add(Dropout(0.5))
	model.add(Dense(64, activation='relu'))
	model.add(Dropout(0.5))
	model.add(Dense(len(output[0]), activation='softmax'))

	# Compile model. Stochastic gradient descent with Nesterov accelerated gradient gives good results for this model
	sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
	model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

	model.fit(numpy.array(training),numpy.array(output), epochs=1000, batch_size=8, verbose=1)
	model.save("model.h5")

def bag_of_words(s,words):
	bag = [0 for _ in range(len(words))]
	s_words = nltk.word_tokenize(s)
	s_words = [stemmer.stem(word.lower()) for word in s_words]

	for se in s_words:
		for i,w in enumerate(words):
			if w == se:
				bag[i] = 1
	return numpy.array(bag)


def chat(inp,talk_text):
	#inp = input(get_audio())
	inp = inp.lower()
	results = bag_of_words(inp,words)
	results = model.predict(numpy.array([results]))[0]
	results_index = numpy.argmax(results)
	tag = labels[results_index]
	if results[results_index] > 0.7:
		for tg in data["intents"]:
			if tg['tag'] == tag:
				responses = random.choice(tg['responses'])
				if talk_text == 1: speak(responses)
				return responses
	else :
		txt = "I didn't get that, try again."
		if talk_text == 1: speak("I didn't get that, try again.")
		return txt

def text_back():
	txt = text_input.get()
	msg_list.insert(END,txt)
	msg_list.insert(END,chat(txt,0))

def speak_back():
	txt = text_input.get()
	msg_list.insert(END,txt)
	msg_list.insert(END,chat(txt,1))

mainwindow = Tk()
mainwindow.title("YOU")
Label(mainwindow, text="TALK BOT", bg="black", fg="white").pack(side=TOP, fill=X, padx=2, pady=2)
messages_frame = Frame(mainwindow)
scrollbar = Scrollbar(messages_frame)  # To navigate through past messages.
# Following will contain the messages.
msg_list = Listbox(messages_frame, height=15, width=50, yscrollcommand=scrollbar.set)
scrollbar.pack(side=RIGHT, fill=Y,padx=2,pady=2)
msg_list.pack(side=LEFT, fill=BOTH,padx=2,pady=2)
msg_list.pack(padx=2,pady=2)
messages_frame.pack()
text_input = StringVar()  # For the messages to be sent.
Label(mainwindow, text ='Type your text here', font = "50", bg="black", fg="white").pack(fill=X) 
text_input_box = Entry(mainwindow, textvariable=text_input)
text_input_box.pack(padx=2,pady=2)
text_reply_button = Button(mainwindow, text="Text Back", command=text_back, bg="black", fg="white")
text_reply_button.pack(side=LEFT,padx=2,pady=2)
talk_reply_button = Button(mainwindow, text="Speak Back", command=speak_back, bg="black", fg="white")
talk_reply_button.pack(side=LEFT,padx=2,pady=2)
mainwindow.mainloop()