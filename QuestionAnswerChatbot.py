import streamlit as st
from transformers import pipeline


# Load Hugging Face question-answering model
@st.cache_resource
def load_qa_model():
    return pipeline("question-answering", model="mjpsm/Togo")

qa_model = load_qa_model()

# Streamlit App
st.title("Hugging Face Q&A Chatbot")

# Text input for context
st.subheader("Provide Context")
context = st.text_area("Enter the context here:", placeholder="Type a paragraph for the chatbot to base its answers on...")

# Text input for question
st.subheader("Ask a Question")
question = st.text_input("Enter your question:", placeholder="Ask something based on the context provided above...")

# Button to get the answer
if st.button("Get Answer"):
    if context and question:
        with st.spinner("Thinking..."):
            result = qa_model(question=question, context=context)
            answer = result['answer']
        st.success(f"Answer: {answer}")
    else:
        st.error("Please provide both context and a question.")
