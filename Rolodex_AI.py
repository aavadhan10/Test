import streamlit as st
import pandas as pd
import openai
import faiss
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize
from config import OPENAI_API_KEY
from dotenv import load_dotenv


# Initialize OpenAI API
openai.api_key = OPENAI_API_KEY

# Load CSV data
@st.cache_data
def load_data(file_path):
    data = pd.read_csv(file_path)
    return data

# Create vector database
@st.cache_resource
def create_vector_db(data):
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(data.astype(str).apply(lambda x: ' '.join(x), axis=1))
    X = normalize(X)
    index = faiss.IndexFlatL2(X.shape[1])
    index.add(X.toarray())
    return index, vectorizer

data = load_data('Matter_Bio.csv')
index, vectorizer = create_vector_db(data)


# Function to query GPT with context from vector DB
def query_gpt_with_data(question, data, index, vectorizer):
    try:
        # Extract practice area from the question
        practice_area = question.split("for")[-1].strip()
        
        # Vectorize the practice area
        practice_area_vec = vectorizer.transform([practice_area])
        practice_area_vec = normalize(practice_area_vec)

        # Find the most similar rows in the data
        D, I = index.search(practice_area_vec.toarray(), k=5)
        relevant_data = data.iloc[I[0]]

        if "contact information" in question.lower():
            # Return the contact information in a table format
            return relevant_data.to_dict(orient='records')
        else:
            # Create a prompt that includes the relevant data and the user's question
            prompt = f"Given the following data on top lawyers:\n{relevant_data.to_string()}\nWho are the top lawyers for {practice_area}?"

            # Call the GPT-3.5-turbo model using the new API
            response = openai.Completion.create(
                model="gpt-3.5-turbo",
                prompt=prompt,
                max_tokens=150
            )
            return response.choices[0].text.strip()
    except Exception as e:
        st.error(f"Error querying GPT: {e}")
        return "An error occurred while processing your request."

# Streamlit app layout
st.title("Find the best lawyer for your needs")

st.write("Ask questions about the top lawyers in a specific practice area:")

user_input = st.text_input("Your question: (e.g., 'What are the top lawyers for corporate law' or 'What is the contact information for the top lawyers for corporate law')")

if user_input:
    data = load_data('Matter_Bio.csv')
    if not data.empty:
        index, vectorizer = create_vector_db(data)
        if index is not None and vectorizer is not None:
            answer = query_gpt_with_data(user_input, data, index, vectorizer)
            if isinstance(answer, list):
                st.write("Contact Information for Best Lawyers:")
                st.table(answer)
            else:
                st.write("Answer:")
                st.write(answer)
        else:
            st.error("Failed to create vector database.")
    else:
        st.error("Failed to load data.")
