import re,numpy as np,pandas as pd
import numpy,json
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
import streamlit as st

with open("C:/Users/Umang/Downloads/wordind.json", 'r') as file:
    wordind_data = json.load(file)
word_ind = wordind_data['wordind']
with open("C:/Users/Umang/Downloads/indword.json", 'r') as file:
    indword_data = json.load(file)
ind_word = indword_data['indword']



# Load the model
model = load_model("C:/Users/Umang/Downloads/Next_Word.h5")


def predict_next_word(input_text, max_seq_len):
    input_text = input_text.lower()
    input_text = re.sub(r'[^\w\s\']', '', input_text)
    input_tokens=[]
    for i in input_text.split():
        input_tokens.append(word_ind[i])
    input_tokens = pad_sequences([input_tokens], maxlen=max_seq_len, padding='post')
    prediction = model.predict(input_tokens)[0]
    next_word_idx = np.argmax(prediction)
    print(next_word_idx)
    next_word=ind_word[str(next_word_idx)]
    return next_word





data=pd.read_csv("C:/Users/Umang/Downloads/archive (8)/Conversation.csv")
data['question'] = data['question'].str.lower().apply(lambda x: re.sub(r'[^\w\s\']', '', x))
data['answer'] = data['answer'].str.lower().apply(lambda x: re.sub(r'[^\w\s\']', '', x))
new_train = data['question']
new_target = data['answer']
data = pd.concat([new_train, new_target], axis=0)
data = data.reset_index(drop=True)
print(len(data))

max_seq_len=18


user_input = st.text_input("Enter text here:")
suggestions = predict_next_word(user_input,max_seq_len) # Your function to generate suggestions

if suggestions:
    st.write("Next Word Suggestion:")
    st.write(suggestions)













