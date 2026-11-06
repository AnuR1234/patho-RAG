import streamlit as st
import json 
import sys
from streamlit_lottie import st_lottie
sys.path.append('/Users/anureddy/Desktop/SEM03/Pratikums/patho-RAG/code')
# Import your patho_rag and generate_response function from your script
from generate_text import generate_response
def main():
    st.title("PathoRAG")
    def load_lottiefile(filepath:str):
        with open(filepath,"r") as f:
            return json.load(f)
    lottie_coding = load_lottiefile('/Users/anureddy/Desktop/SEM03/Pratikums/patho-RAG/code/Frontend/animate.json')
    st.header("Question Answering System",divider='rainbow')
    
    st_lottie(
        lottie_coding,speed=1,
        reverse=False,
        loop=True,
        quality='low',
        #renderer='svg',
        height=None,
        width=None,
        key='PathoRag'
    )
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