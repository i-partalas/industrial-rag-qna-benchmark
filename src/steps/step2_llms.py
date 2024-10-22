import streamlit as st


def display(prev_step, next_step):
    st.header("Select the LLMs to Evaluate")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Proprietary LLMs")
        platform_choice = st.radio(
            "Select the Platform:",
            ("OpenAI Platform", "AzureOpenAI Platform"),
            key="proprietary_platform",
        )
        platform_name_pr = platform_choice.split()[0]
        openai_api_key = st.text_input(
            f"{platform_name_pr} API Key",
            placeholder=f"Enter your {platform_name_pr} API key",
            type="password",
        )
        openai_llm_name = st.text_input(
            "Proprietary LLM Name", placeholder="Enter the LLM name"
        )
        openai_embedding_model_name = st.text_input(
            "Proprietary Embedding Model Name",
            placeholder="Enter the embedding model name",
        )
        # If OpenAI is chosen, disable "endpoint" and "api_version" fields
        disabled = True if platform_name_pr == "OpenAI" else False

        openai_endpoint = st.text_input(
            "Endpoint", placeholder="Enter the API endpoint", disabled=disabled
        )
        openai_api_version = st.text_input(
            "API Version", placeholder="Enter the API version", disabled=disabled
        )

    with col2:
        st.subheader("Open-Sourced LLMs")
        platform_choice_os = st.radio(
            "Select the Platform:",
            ("HuggingFace Platform", "Ollama Platform"),
            key="opensource_platform",
            help="Further platforms are to be implemented, such as Ollama.",
        )
        platform_name_os = platform_choice_os.split()[0]
        if platform_name_os == "HuggingFace":
            huggingface_api_key = st.text_input(
                f"{platform_name_os} API Key",
                placeholder=f"Enter your {platform_name_os} API key",
                type="password",
            )
            huggingface_llm_name = st.text_input(
                "Open-Sourced LLM Name or ID", placeholder="Enter the LLM name or ID"
            )
            huggingface_embedding_model_name = st.text_input(
                "Open-Sourced Embedding Model Name",
                placeholder="Enter the embedding model name",
            )
        elif platform_name_os == "Ollama":
            st.warning(
                (
                    f"The {platform_name_os} Platform is not yet implemented. "
                    "Please select another platform."
                )
            )

    # Validation: Ensure required fields for the selected platform are filled
    proprietary_fields_filled = all(
        [
            openai_api_key,
            openai_llm_name,
            openai_embedding_model_name,
            openai_endpoint,
            openai_api_version,
        ]
    )
    opensource_fields_filled = (
        platform_name_os == "HuggingFace"
        and all(
            [
                huggingface_api_key,
                huggingface_llm_name,
                huggingface_embedding_model_name,
            ]
        )
    ) or platform_name_os == "Ollama"

    button_footers = st.columns([1, 1, 8])
    button_footers[0].button("Back", on_click=prev_step)

    button_footers[1].button("Next", on_click=next_step)
    # if proprietary_fields_filled and opensource_fields_filled:
    #     button_footers[1].button("Next", on_click=next_step)
    # else:
    #     st.warning("Please fill out all required fields to proceed.")
    #     button_footers[1].button("Next", on_click=next_step, disabled=True)
