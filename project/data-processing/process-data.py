# -*- coding: utf-8 -*-
"""
Created on Fri Oct  8 16:59:42 2021

@author: Yasel
"""

## Data Cleaning 

## Remove Punctuations:
import string
punctuation_string = string.punctuation
punctuation_string = punctuation_string.translate({ord(i): None for i in '@'})

def remove_punct(text,punctuation_string):
    text  = "".join([char for char in text if char not in punctuation_string])
    text = re.sub('[0-9]+', '', text)
    return text

## Tokenization :
def tokenization(text):
    text = re.split('\W+', text)
    return text

## Word Replacement :
replaced_words = [("hmmyou",""),("sry","sorry"),("inlove","in love"),("thats",""),("wanna",""),
                  ("soo","so"),("inlove","in love"),("amazingwell","amazing well"),
                  ("messagesorry","message sorry"),("½",""),("tomorrowneed","tomorrow need"),
                  ("tomorrowis","tomorrow is"),("amusedtime","amused time"),("weekendor","weekend or"),
                  ("competitionhope","competition hope"),("partypicnic","party picnic"),
                  ("ahmazing","amazing"),("wont","will not"),("didnt","did not"),("dont","do not"),
                  ("lookin","looking"),("u","you"),("youre","you are"),("nite","night"),("isnt","is not"),
                  ("k",""),("is",""),("doesnt","does not"),("l",""),("x",""),("c",""),("ur","your"),
                  ("e",""),("yall","you all"),("he",""),("us",""),("okim","ok i am"),("jealousi","jealous"),
                  ("srry","sorry"),("itll","it will"),("vs",""),("weeknend","weekend"),("w",""),
                  ("yr","year"),("youve","you have"),("havent","have not"),("iï",""),("gonna","going to"),
                  ("gimme","give me"),("ti",""),("ta",""),("thru","through"),("th",""),("imma","i am going to"),
                  ("wasnt","was not"),("arent","are not"), ("bff","best friend forever"),("sometimesdid","sometimes did"),
                  ("waitt","wait"),("bday","birthday"),("toobut","too but"),("showerand","shower and"),
                  ("innit","is not it"),("surgury","surgery"),("soproudofyo","so proud of you"),("p",""),
                  ("couldnt","could not"),("dohforgot","forgot"),("rih","right"),("b",""),("bmovie","movie"),
                  ("pleaseyour","please your"),("tonite","tonight"),("grea","great"),("se",""),("soonso","soon so"),
                  ("gettin","getting"),("blowin","blowing"),("coz","because"),("thanks","thank"),("st",""),("rd",""),
                  ("gtta","have got to"),("gotta","have got to"),("anythingwondering","anything wondering"),
                  ("annoyedy","annoyed"),("p",""),("beatiful","beautiful"),("multitaskin","multitasking"),
                  ("nightmornin","night morning"),("thankyou","thank you"),("iloveyoutwoooo","i love you two"),
                  ("tmwr","tomorrow"),("wordslooks","words looks"),("ima","i am going to"),("liek","like"),("mr",""),
                  ("allnighter","all nighter"),("tho","though"),("ed",""),("fyou",""),("footlong","foot long"),
                  ("placepiggy","place piggy"),("semiflaky","semi flaky"),("gona","going to"),("tmr","tomorrow"),
                  ("ppl","people"),("n",""),("dis","this"),("dun","done"),("houseee","house"),("havee","have"),
                  ("studyingwhew","studying whew"),("awwyoure","aww you are"),("softyi","softy"),
                  ("weddingyou","wedding you"),("hassnt","has not"),("lowerleft","lower left"),("anywayss","anyway"),
                  ("adoarble","adorable"),("blogyeahhhh","blog yeahhhh"),("billsim","bills i am"),("ps",""),
                  ("cheescake","cheesecake"),("morningafternoonnight","morning after noon night"),
                  ("allstudying","all studying"),("ofcoooursee","of course"),("jst","just"),("shes","she is"),
                  ("sonicswhich","sonics which"),("ouchwaited","ouch waited"),("itll","it will"),("orreply","or reply"),
                  ("somethin","something"),("fridayand","friday and"),("outta","out of"),("herenever","here never")
                 ] 

def replace_words(text,replaced_words):
    ind = -1 
    for word in text:
        ind +=1
        for k in range(len(replaced_words)):
            if word == replaced_words[k][0]:
                text[ind] = replaced_words[k][1]
            elif "http" in word:
                text[ind] = ""
            elif "@" in word:
                text[ind] = ""
            elif "www." in word:
                text[ind] = ""
            elif "Â" in word: 
                text[ind] = ""
            elif "Ã" in word: 
                text[ind] = ""
            elif "½" in word:
                text[ind] = ""
    return text



import nltk
nltk.download('stopwords')

## Remove stopwords :
stopword = nltk.corpus.stopwords.words('english')
print("stopword:\n",stopword)
print("\n\n There are some words that we want to keep, for example 'no', 'nor','not'\n")
words_to_keep = ["not","no","nor"]
stopword = [elem for elem in stopword if not elem in words_to_keep]
stopword.extend(["im","theyre","ive","p","alot","er",""]) # Other stopwords to remove
print("stopword:\n",stopword,"\n")

def remove_stopwords(text,stopword):
    text = [word for word in text if word not in stopword]
    return text


## Stemming 
ps = nltk.PorterStemmer()

def stemming(text):
    text = [ps.stem(word) for word in text]
    return text

## Lemmatization
wn = nltk.WordNetLemmatizer()

def lemmatizer(text):
    text = [wn.lemmatize(word) for word in text]
    return text

def process(df,dir) :
  df['review_punct'] = df['reviews'].apply(lambda x: remove_punct(x,punctuation_string))
  print(df.shape)
  df['review_tokenized'] = df['review_punct'].apply(lambda x: tokenization(x.lower()))
  print(df.shape)
  df['review_tokenized'] = df['review_tokenized'].apply(lambda x: replace_words(x,replaced_words))
  print(df.shape)
  df['review_nonstop'] = df['review_tokenized'].apply(lambda x: remove_stopwords(x,stopword))
  print(df.shape)
  df['review_stemmed'] = df['review_nonstop'].apply(lambda x: stemming(x))
  print(df.shape)
  #df['review_lemmatized'] = df['review_nonstop'].apply(lambda x: lemmatizer(x))
  #print(df.head(10))
  df['review_stemmed'] = df['review_stemmed'].apply(lambda x: ' '.join(str(e) for e in x))
  print(df.shape)
  df.to_csv("alldata{}.csv".format(dir))