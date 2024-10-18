import streamlit as st


def display(prev_step, next_step):
    st.header("Upload the PDF files")
    uploaded_pdfs = st.file_uploader(
        "Upload the PDF file(s) you want to interact with:",
        type="pdf",
        accept_multiple_files=True,
    )
    button_footers = st.columns([1, 1, 8])
    button_footers[0].button("Back", on_click=prev_step)

    # Validation: Ensure at least one PDF file is uploaded
    button_footers[1].button("Next", on_click=next_step)
    if uploaded_pdfs:
        st.session_state.uploaded_pdfs = uploaded_pdfs
    #     button_footers[1].button("Next", on_click=next_step)
    # else:
    #     st.warning("Please upload at least one PDF file to proceed.")
    #     button_footers[1].button("Next", on_click=next_step, disabled=True)
