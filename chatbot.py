#this is self learning chatbot program
#install nltk,newspaper3k
from newspaper import Article
import random
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
import numpy as np 
import warnings
import playsound
import os
from gtts import gTTS
#ignoring the warnings
warnings.filterwarnings('ignore')

#download the package from nltk
nltk.download('punkt')
nltk.download('wordnet')

#get article url
article =Article("https://www.mayoclinic.org/diseases-conditions/chronic-kidney-disease/symptoms-causes/syc-20354521")
article.download()
article.parse()
article.nlp()
corpous = article.text
print(corpous)

#tokenisation
text = corpous
sent_tokens =  nltk.sent_tokenize(text) #convert the text into a list of sentences
print(sent_tokens)

#create a dictionary to remove punctiantions
remove_punct_dict=dict( (ord(punct),None)for punct in string.punctuation)#ord reprsent no.
print(string.punctuation)
print(remove_punct_dict)

#create a function to return a list of lemmatized lower case wordsafter removing punctuations
def LemNormalize(text):
  return nltk.word_tokenize(text.lower().translate(remove_punct_dict))
print(LemNormalize(text))

#keyword matching
#greeting inputs
g=["hi","hello","hola","greetings","hey"]
#list of g response back to the user
gr=["howdy","hi","hey","what's good","hello","hey there"]
#function to return a random greeting resonse to a user greeting
def greeting(sentence):
  #if the user's input is greeting, thenm return a randomly chosen greeting response
  for word in sentence.split():
    if word.lower() in g:
      return random.choice(gr)

#genterate the response
def response(user_res):
  #user response
  #user_res="What is chronic kidney disease"
  user_res = user_res.lower()#make respnse in lower
  #print(user_res)
   #robo response
  rr=''
  #append the user response
  sent_tokens.append(user_res)
  #print(sent_tokens)
  #create tfidfvectorizer object
  TfidfVec=TfidfVectorizer(tokenizer = LemNormalize,stop_words='english')

  #conert the text to a matrix of tf-idf features
  tfidf = TfidfVec.fit_transform(sent_tokens)
  #print(tfidf)
  #get the measure of similarity 
  vals =  cosine_similarity(tfidf[-1],tfidf)
  #print(vals)
  #GET  the index of the most similar text/sentence to the users response
  idx=vals.argsort()[0][-2]#beacuse -1 is the user response itself

  #reduce the dimensionality of vals
  flat = vals.flatten()
  #sort the list in asc
  flat.sort()
  #get the most similar score to the response
  score = flat[-2]#beacuse -1 is the user response itself
  #print(score)
  #if variavble score is 0 then  here si no text similar
  if(score==0):
    rr =rr+"I apologise ,I don't understand"
  else:
    rr=rr+sent_tokens[idx]
  #remove the users response from sentence tokens list
  sent_tokens.remove(user_res)
  return rr
def sound(audio_string):
    tts=gTTS(text=audio_string,lang='en')
    r = random.randint(1,10000000)
    audio_file= 'audio-' +str(r) + '.mp3'
    tts.save(audio_file)
    playsound.playsound(audio_file)
    print(audio_string)
    os.remove(audio_file)

flag = True
print('Ella: I am doctor bot . T will answer your queries about chronic kidney desease. If you want to exit type Bye')
while(flag==True):
  user_res= input()
  user_res=user_res.lower()
  if(user_res !='bye'):
    if(user_res =="thanks" or user_res == "thankyou"):
      flag = False
      sound("Ella:You are welcome!")
    else:
      if(greeting(user_res)!=None):
        sound("Ella: "+greeting(user_res))
      else:
        sound("Ella: "+response(user_res))
  else:
    flat=False
    sound("Ella: Chat with you later")