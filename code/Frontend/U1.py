import streamlit as st
import json
import sys
from streamlit_lottie import st_lottie

sys.path.append('/Users/anureddy/Desktop/SEM03/Pratikums/patho-RAG/code')
from generate_text import generate_response
st.set_page_config(layout="wide")
def main():
    def load_lottiefile(filepath: str):
        with open(filepath, "r") as f:
            return json.load(f)

    lottie_coding = load_lottiefile('/Users/anureddy/Desktop/SEM03/Pratikums/patho-RAG/code/Frontend/animate.json')
    font_family = "Rockwell"
    color = "#007bff"
    # Centering container
    with st.container():
        st.markdown(f"<h1 style='text-align: center; font-family: {font_family}; color: {color};'>PathoRAG</h1>", unsafe_allow_html=True)

        # Attempting to center using Streamlit columns with more control
        col1, col2, col3 = st.columns([1, 2, 1])  # Adjust the ratio if needed

        with col2:
            st_lottie(lottie_coding, speed=1, reverse=False, loop=True, quality='low', height=300, width=850, key='PathoRag')



    st.header("Question Answering System", divider='blue')

    # Input form
    with st.form(key='question_form'):
        question = st.text_input("Enter your question:")
        submit_button = st.form_submit_button(label='Submit')
        if submit_button and question:
            # Call your generate_response function
            response = generate_response(question)
            if response:
                # Displaying the response
                st.subheader("Response")
                st.write(response)
            else:
                st.write("No response generated.")

if __name__ == "__main__":
    main()