import time

import streamlit as st


def display(prev_step, next_step):
    st.header("Run the Application")
    button_footers = st.columns([1.1, 1.1, 1.1, 6.7])
    if button_footers[1].button("Run"):
        with st.spinner():
            time.sleep(5)
            st.success("Application started successfully!")
            button_footers[2].button("Next", on_click=next_step, disabled=False)
    button_footers[0].button("Back", on_click=prev_step)
