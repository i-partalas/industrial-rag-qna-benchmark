import streamlit as st


def display(prev_step, next_step):
    st.header("Upload or Create an Evaluation Dataset")

    method_1 = "Upload your own evaluation dataset"
    method_2 = "Generate a synthetic evaluation dataset based on uploaded PDF file(s)"
    choice = st.radio(
        label="Choose the method to proceed with the evaluation dataset:",
        options=(method_1, method_2),
        # TODO: Complete the help msg
        help="For the structure and format of the file, please advise ...",
    )

    dataset_uploaded = False
    if choice == method_1:
        uploaded_file = st.file_uploader(
            "Upload your evaluation dataset file:", type="xlsx"
        )
        if uploaded_file:
            st.success(f"Uploaded your evaluation dataset: {uploaded_file.name}")
            dataset_uploaded = True
    elif choice == method_2:
        uploaded_pdfs = st.file_uploader(
            "Upload your PDF file(s):", type="pdf", accept_multiple_files=True
        )
        if uploaded_pdfs:
            st.success(
                f"Uploaded {len(uploaded_pdfs)} PDF file(s) for synthetic dataset generation"
            )
            dataset_uploaded = True

    button_footers = st.columns([1, 1, 8])
    button_footers[0].button("Back", on_click=prev_step)

    button_footers[1].button("Next", on_click=next_step)
    # if dataset_uploaded:
    #     button_footers[1].button("Next", on_click=next_step)
    # else:
    #     st.warning("Please upload a dataset to proceed.")
    #     button_footers[1].button("Next", on_click=next_step, disabled=True)
