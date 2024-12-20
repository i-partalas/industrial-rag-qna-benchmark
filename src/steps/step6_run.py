import streamlit as st

from preprocessing.preprocessing import PDFProcessor


def disable_button():
    st.session_state.button_disabled = True


def display(prev_step, next_step):
    st.header("Run the Application")
    button_footers = st.columns([1.1, 1.1, 1.1, 6.7])
    # TODO: Add "Stop" button
    if button_footers[1].button(
        "Run", disabled=st.session_state.button_disabled, on_click=disable_button
    ):
        with st.status("Running the application..."):
            # Run preprocessing
            st.write("Chunking PDF content into elements...")
            pdf_processor = PDFProcessor(st.session_state.uploaded_pdfs)
            docs = pdf_processor.chunk_files_to_docs()  # noqa: F841

            st.write("Creating synthetic data...")
            st.write("Generating answers with LLM...")
            st.write("Evaluating answers...")
            st.success("Application ran successfully!")
            button_footers[2].button("Next", on_click=next_step, disabled=False)
    button_footers[0].button("Back", on_click=prev_step)
