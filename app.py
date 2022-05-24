from flask import Flask,request, render_template
#Import fundamentals
import numpy as np
import pandas as pd
import re
import pickle

# Import word_tokenize and stopwords from nltk
import nltk
nltk.download('omw-1.4')
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer#, PorterStemmercmdcd
from nltk.tag import pos_tag

app = Flask(__name__)
#app=Flask(__name__,template_folder='template')

enstopwords = set(stopwords.words('english'))

def process_text(feedback):

    # clean the words
    clean = word_tokenize(feedback)

    # Remove the English stop words
    clean = [token for token in clean if token not in enstopwords]

    # Remove non-alphabetic characters and keep the words contains three or more letters
    clean = [token for token in clean if token.isalpha() and len(token) > 2]

    clean = ' '.join(clean)

    return clean

def NormalizeWithPOS(text):
    # Lemmatization & Stemming according to POS tagging

    word_list = word_tokenize(text)
    rev = []
    lemmatizer = WordNetLemmatizer()
    #stemmer = PorterStemmer()
    for word, tag in pos_tag(word_list):
        if tag.startswith('J'):
            w = lemmatizer.lemmatize(word, pos='a')
        elif tag.startswith('V'):
            w = lemmatizer.lemmatize(word, pos='v')
        elif tag.startswith('N'):
            w = lemmatizer.lemmatize(word, pos='n')
        elif tag.startswith('R'):
            w = lemmatizer.lemmatize(word, pos='r')
        else:
            w = word
        #w = stemmer.stem(w)
        rev.append(w)
    feedback = ' '.join(rev)
    return feedback


@app.route('/', methods=["GET", "POST"])
@app.route('/dashboard', methods=["GET", "POST"])
def dashboard():

    # load the dataset
    tweets = pd.read_csv("studentFeebackOnServicesLMS_wlabel.csv")

    # Visualize the Label counts
    neutral = tweets.loc[tweets['Label'] == 'Neutral'].count()
    positive = tweets.loc[tweets['Label'] == 'Positive'].count()
    negative = tweets.loc[tweets['Label'] == 'Negative'].count()
    data = {'Label': 'Count', 'Neutral': neutral[0], 'Positive': positive[0], 'Negative': negative[0]}

    # Initialize a Tf-idf Vectorizer
    vectorizer_lms = pickle.load(open('vectorizer_lms.pkl', 'rb')) #Vectorizer for lms
    vectorizer_course = pickle.load(open('vectorizer_course.pkl', 'rb'))  # Vectorizer for course

    if request.method == "POST":

        feedback = request.form.get("feedback")
        clean = process_text(str.lower(feedback))
        clean = NormalizeWithPOS(clean)
        preprocess_lms = vectorizer_lms.transform([clean])
        preprocess_course = vectorizer_course.transform([clean])

        #load the model
        lg_lms = pickle.load(open('lg_lms_model.pkl', 'rb')) #Model for LMS
        lg_course = pickle.load(open('lg_course_model.pkl', 'rb'))  # Model for Course

        prediction_lms = lg_lms.predict(preprocess_lms)
        prediction_course = lg_course.predict(preprocess_course)


        return render_template('Dashboard.html',feedback= feedback, p1=prediction_lms[0], data=data)

    return render_template('Dashboard.html', data=data)

if __name__ == "__main__":
    app.run(debug=True)