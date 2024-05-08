import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

# Load your dataset
datas = pd.read_csv("sn1500.csv")
# Assuming datas is your dataset
X = datas.new_com
y = datas.Expresion

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2)

# Create the model pipeline
tvec = TfidfVectorizer()
clf2 = LogisticRegression(solver='lbfgs')
model = Pipeline([('vectorizer', tvec), ('classifier', clf2)])

# Fit the model
model.fit(X_train, y_train)

# Define the Streamlit app
def main():
    st.title("Sentiment Analysis with Logistic Regression")
    st.subheader("Enter a sentence to predict its sentiment:")
    user_input = st.text_input("")

    # Make prediction when user submits input
    if st.button("Predict"):
        prediction = model.predict([user_input])[0]
        if prediction == 0:
            st.write("Predicted Sentiment: Sad")
        elif prediction == 4:
            st.write("Predicted Sentiment: Happy")
        else:
            st.write("Predicted Sentiment: It's not making sense")

if __name__ == "__main__":
    main()
