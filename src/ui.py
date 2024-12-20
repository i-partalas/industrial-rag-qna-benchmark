import streamlit as st

import utils.suppress_warnings  # noqa: F401
from steps import (
    step0_home,
    step1_upload,
    step2_llms,
    step3_metrics,
    step4_dataset,
    step5_hyperparams,
    step6_run,
    step7_results,
)


def display_step_indicator():
    steps = [
        "Step 1: Home",
        "Step 2: Upload Data",
        "Step 3: Select LLMs",
        "Step 4: Select Metrics",
        "Step 5: Upload/Create Dataset",
        "Step 6: Configure Hyperparameters",
        "Step 7: Run the Application",
        "Step 8: View the Results",
    ]

    # Current step (0-based index)
    current_step = st.session_state.step

    # Display steps in the sidebar
    st.sidebar.header("Navigation")
    for i, step in enumerate(steps):
        # Highlight the current step
        if i == current_step:
            st.sidebar.write(f"**➡️ {step}**")
        # Mark completed steps
        elif i < current_step:
            st.sidebar.write(f"✅ {step}")
        # Indicate forthcoming steps
        else:
            st.sidebar.write(f"🔜 {step}")


def prepare_session_state():
    if "button_disabled" not in st.session_state:
        st.session_state.button_disabled = False
    if "step" not in st.session_state:
        st.session_state.step = 0
    # Preprocessing session variables
    if "uploaded_pdfs" not in st.session_state:
        st.session_state.uploaded_pdfs = None
    # LLMs session variables (proprietary)
    if "proprietary_platform" not in st.session_state:
        st.session_state.proprietary_platform = "OpenAI"
    if "openai_api_key" not in st.session_state:
        st.session_state.openai_api_key = None
    if "openai_llm_name" not in st.session_state:
        st.session_state.openai_llm_name = None
    if "openai_embedding_model_name" not in st.session_state:
        st.session_state.openai_embedding_model_name = None
    if "openai_endpoint" not in st.session_state:
        st.session_state.openai_endpoint = None
    if "openai_api_version" not in st.session_state:
        st.session_state.openai_api_version = None
    # LLMs session variables (open-sourced)
    if "opensource_platform" not in st.session_state:
        st.session_state.opensource_platform = "HuggingFace"
    if "opensource_api_key" not in st.session_state:
        st.session_state.opensource_api_key = None
    if "opensource_llm_name" not in st.session_state:
        st.session_state.opensource_llm_name = None
    if "opensource_embedding_model_name" not in st.session_state:
        st.session_state.opensource_embedding_model_name = None
    # LLMs session variables (general)
    if "generation_temperature" not in st.session_state:
        st.session_state.generation_temperature = 0.2
    if "evaluation_temperature" not in st.session_state:
        st.session_state.evaluation_temperature = 0.2


def next_step():
    st.session_state.step += 1


def prev_step():
    st.session_state.step -= 1


def reset_wizard():
    st.session_state.step = 0


def run_gui():
    # Initialize session variables
    prepare_session_state()

    # Show step indicator in the sidebar
    display_step_indicator()

    # Step Flow
    if st.session_state.step == 0:
        step0_home.display(next_step)
    elif st.session_state.step == 1:
        step1_upload.display(prev_step, next_step)
    elif st.session_state.step == 2:
        step2_llms.display(prev_step, next_step)
    elif st.session_state.step == 3:
        step3_metrics.display(prev_step, next_step)
    elif st.session_state.step == 4:
        step4_dataset.display(prev_step, next_step)
    elif st.session_state.step == 5:
        step5_hyperparams.display(prev_step, next_step)
    elif st.session_state.step == 6:
        step6_run.display(prev_step, next_step)
    elif st.session_state.step == 7:
        step7_results.display(reset_wizard)


if __name__ == "__main__":
    run_gui()
