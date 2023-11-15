#!/usr/bin/env python
# coding: utf-8

# # Creating a Flask API 
# 
# This is the last leg of the project, and we will build a flask API from the classifier built in the previous exercise. The model created in the previous phase had an accuracy around 90% and it will be transformed into a flask application that takes around 150 characters of text and predicts the sentiment of the text. Additional Features include profanity filter and hate speech detection. The API will be tested on Flask Development environment. 

# In[1]:


#importing the required packages
from flask import Flask, request, jsonify
import pickle
from profanityfilter import ProfanityFilter
import spacy
from flask_restful import Resource, Api
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer


# In[2]:


# Mapping numerical labels to emotions
label_to_emotion = {
    0: 'anger',
    1: 'fear',
    2: 'joy',
    3: 'love',
    4: 'sadness',
    5: 'surprise'
}


# In[3]:


# Loading the pickled model
with open('best_xgb_classifier.pkl', 'rb') as model_file:
    model = pickle.load(model_file)


# In[4]:


# Profanity filter setup
profanity_filter = ProfanityFilter()


# In[5]:


# Loading Spacy English model
nlp = spacy.load("en_core_web_sm")


# In[6]:


#defining app
app = Flask(__name__)
api = Api(app)


# In[7]:


# Endpoint for sentiment analysis



class SentimentAnalysis(Resource):
    def post(self):
        try:
            data = request.get_json()
            text = data['text']

   
            if not (10 <= len(text) <= 150):
                return jsonify({'error': 'Text length should be between 10 and 150 characters.'})

           
            doc = nlp(text)
            tokens = [token.text for token in doc]
            processed_text = ' '.join(tokens)

   
            filtered_text = profanity_filter.censor(processed_text)

          
            if profanity_filter.is_profane(filtered_text):
                result = {'text': text, 'sentiment_label': 'Hate Speech'}
            else:
            
                prediction = model.predict([filtered_text])[0]

            
                emotion = label_to_emotion.get(prediction, 'Unknown')
                result = {'text': text, 'sentiment_label': emotion}

           
            return jsonify(result)

        except Exception as e:
            return jsonify({'error': str(e)})


# In[8]:


# Adding the resource to the API with the desired endpoint
api.add_resource(SentimentAnalysis, '/analyze_text')


# In[9]:


# Running the APP

if __name__ == '__main__':
    app.run(port=1234, debug=True,use_reloader=False)


# In[ ]:




