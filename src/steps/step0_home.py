import streamlit as st


def display(next_step):
    st.header(
        body="Welcome to the Evaluation App",
        help="Benchmarking both proprietary and open-sourced LLMs hosted on diverse platforms.",
    )
    st.write(
        "This app allows you to evaluate RAG performance of proprietary and open-sourced LLMs."
    )
    st.image("res/rag_eval_pipeline.png")
    if st.button("Start"):
        next_step()
        st.rerun()
