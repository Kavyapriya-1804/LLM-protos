import streamlit as st
import chains

def poem_generator_app():
    """
    Generates Poem Generator App with streamlit for getting user input and displaying output
    """

    st.title("Lets generate a poem ! 👋")

    with st.form("poem_generator"):
        topic = st.text_input(
          "Enter a topic for the poem:"
        )
        submitted = st.form_submit_button("Submit")

        if submitted:
            response = chains.generate_poem(topic)
            st.info(response)

poem_generator_app()
