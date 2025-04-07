import streamlit as st
import pipelines

def quiz_generator_app():
    """
    Generates Quiz Generator App with streamlit for getting user input and displaying output
    """

    st.title("Lets generate a Quiz ! 👋")

    with st.form("quiz_generator"):
        topic = st.text_input(
          "Enter a topic for the quiz:"
        )
        number_of_questions = st.text_input(
          "No of questions:"
        )
        difficulty = st.select_slider(
                "Set the difficulty level: ",
                options=[
                    "Easy",
                    "Medium",
                    "Difficult",
                    "Expert"
                ],
        )
        submitted = st.form_submit_button("Submit")

        if submitted:
            response = pipelines.generate_quiz(topic, number_of_questions, difficulty)
            st.info(response)

quiz_generator_app()
