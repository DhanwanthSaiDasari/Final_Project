import streamlit as st
import pandas as pd
import pickle
import re
from nltk.corpus import stopwords
import nltk
from nltk.stem import SnowballStemmer
nltk.download('stopwords')
# Load the pre-trained model
model = pickle.load(open('voting_model1.sav', 'rb'))

preprocessor = pickle.load(open('final_preprocessor1.sav', 'rb'))
# Load the dataset
data = pd.read_csv('fake_job_postings.csv')

cols = ['title', 'company_profile', 'description', 'requirements', 'telecommuting', 
        'has_company_logo', 'has_questions', 'employment_type', 'required_experience', 
        'required_education', 'industry']

def clean(text):
    stop=set(stopwords.words("english"))
    text=text.lower()
    obj=re.compile(r"<.*?>")                     #removing html tags
    text=obj.sub(r" ",text)
    obj=re.compile(r"https://\S+|http://\S+")    #removing url
    text=obj.sub(r" ",text)
    obj=re.compile(r"[^\w\s]")                   #removing punctuations
    text=obj.sub(r" ",text)
    obj=re.compile(r"\d{1,}")                    #removing digits
    text=obj.sub(r" ",text)
    obj=re.compile(r"_+")                        #removing underscore
    text=obj.sub(r" ",text)
    obj=re.compile(r"\s\w\s")                    #removing single character
    text=obj.sub(r" ",text)
    obj=re.compile(r"\s{2,}")                    #removing multiple spaces
    text=obj.sub(r" ",text)
    stemmer = SnowballStemmer("english")
    text=[stemmer.stem(word) for word in text.split() if word not in stop]
    
    return " ".join(text)

# Create a function to preprocess the input data
def preprocess_data(input_df):
    text_features=['title','company_profile','description','requirements']
    for i in text_features:
        input_df[i]=input_df[i].apply(clean)
    # Select the relevant columns
    X = input_df[cols]
    # Transform the data using the preprocessor
    X_transformed = preprocessor.transform(X)
    # Return the transformed data
    return X_transformed

# Create a function to make predictions
def predict(input_df):
    # Preprocess the input data
    X_transformed = preprocess_data(input_df)
    # Make predictions using the voting classifier
    y_pred = model.predict(X_transformed)
    # Return the predictions
    return y_pred

def change(x):
    if x=='No':
        return 0
    else:
        return 1

# Create a Streamlit app
def main():
    with st.form("my_form"):
    # Set the app title
        st.title("Fake Job Posting Detector")

    # Create a form for the user to input data
        st.header("Enter Job Posting Details")
    # Display input fields for the job posting features

        title = st.text_input("Title")
        company_profile = st.text_area("Company Profile")
        description = st.text_area("Description")
        requirements = st.text_area("Requirements")
        telecommuting = change(st.radio("Telecommuting", ['Yes','No']))
        has_company_logo = change(st.radio("Has Company Logo", ['Yes','No']))
        has_questions = change(st.radio("Has Questions", ['Yes','No']))
        employment_type = st.selectbox("Employment Type", data['employment_type'].dropna().unique())
        required_experience = st.selectbox("Required Experience", data['required_experience'].dropna().unique())
        required_education = st.selectbox("Required Education", data['required_education'].dropna().unique())
        industry = st.selectbox("Industry", data['industry'].dropna().unique())
    # Create a dictionary with the input features
        input_data = {
            'title': title,
            'company_profile': company_profile,
            'description': description,
            'requirements': requirements,
            'telecommuting': telecommuting,
            'has_company_logo': has_company_logo,
            'has_questions': has_questions,
            'employment_type': employment_type,
            'required_experience': required_experience,
            'required_education': required_education,
            'industry': industry
        }
    # Create a DataFrame with the input data
        input_df = pd.DataFrame([input_data])
    #predict
        check=''
    #button
        if st.form_submit_button():
            check=predict(input_df)
            if check==[0]:
                st.success("This is a Real job post")
            else:
                st.success("This is a Fraudulent job post")

main()
