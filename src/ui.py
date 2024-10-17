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

# Initialize session state to track progress between steps and form inputs
if "step" not in st.session_state:
    st.session_state.step = 0


def next_step():
    st.session_state.step += 1


def prev_step():
    st.session_state.step -= 1


def reset_wizard():
    st.session_state.step = 0


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
