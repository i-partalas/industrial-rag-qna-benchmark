import streamlit as st


def display(reset_wizard):
    st.header("View the Evaluation Results")
    st.write(
        "Once you upload or generate an evaluation set, results will be displayed here."
    )
    st.button("Restart", on_click=reset_wizard)
