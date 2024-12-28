import streamlit as st
import sklearn
import helper
import pickle
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt_tab')

model = pickle.load(open("Models/model.pkl", 'rb'))
vectorizer = pickle.load(open("Models/vectorizer.pkl", 'rb'))


st.title("Sentiment Analysis Application USING ML")
st.text("Enter your review below:")

text = st.text_input("Please Enter Your Review")

if st.button("Predict"):
    if text.strip() == "":
        st.error("Please enter a review before predicting.")
    else:
        token = helper.preprocessing_step(text)
        vectorized_data = vectorizer.transform([token])
        prediction = model.predict(vectorized_data)

        if prediction == 1:
            st.success("Sentiment: Positive Review.")
        else:
            st.error("Sentiment: Negative Review.")

        if st.button("Clear"):
            st.experimental_rerun()
