import streamlit as st


def display(prev_step, next_step):
    st.header("Upload or Create a Test Dataset")

    method_1 = "Upload your own test dataset"
    method_2 = "Generate a synthetic test dataset based on uploaded PDF file(s)"
    choice = st.radio(
        label="Choose the method to proceed with the test dataset:",
        options=(method_1, method_2),
        help="Ensure the mandatory columns 'question' and 'ground_truth' are included exactly as here specified.",
    )

    testset_choice = False
    if choice == method_1:
        uploaded_testset = st.file_uploader(
            "Upload your test dataset file:", type="xlsx"
        )
        if uploaded_testset:
            testset_choice = True
            st.session_state.uploaded_testset = uploaded_testset
            st.success(f"Uploaded test dataset: {uploaded_testset.name}")
    elif choice == method_2:
        testset_choice = True

    button_footers = st.columns([1, 1, 8])
    button_footers[0].button("Back", on_click=prev_step)

    button_footers[1].button("Next", on_click=next_step)
    # if testset_choice:
    #     button_footers[1].button("Next", on_click=next_step)
    # else:
    #     st.warning("Please select one of the available options.")
    #     button_footers[1].button("Next", on_click=next_step, disabled=True)
