import streamlit as st


def display(prev_step, next_step):
    st.header("Select the LLMs to Evaluate")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Proprietary LLMs")
        platform_choice_pr = st.radio(
            label="Select the Platform:",
            options=("OpenAI", "AzureOpenAI"),
            key="proprietary_platform",
        )
        openai_api_key = st.text_input(
            label=f"{platform_choice_pr} API Key",
            placeholder=f"Enter your {platform_choice_pr} API key",
            type="password",
            key="openai_api_key",
        )
        openai_llm_name = st.text_input(
            label="Proprietary LLM Name",
            placeholder="Enter the LLM name",
            key="openai_llm_name",
        )
        openai_embedding_model_name = st.text_input(
            label="Proprietary Embedding Model Name",
            placeholder="Enter the embedding model name",
            key="openai_embedding_model_name",
        )
        # If OpenAI is chosen, disable "endpoint" and "api_version" fields
        disabled = True if platform_choice_pr == "OpenAI" else False

        openai_endpoint = st.text_input(
            label="Endpoint",
            placeholder="Enter the API endpoint",
            disabled=disabled,
            key="openai_endpoint",
        )
        openai_api_version = st.text_input(
            label="API Version",
            placeholder="Enter the API version",
            disabled=disabled,
            key="openai_api_version",
        )

    with col2:
        st.subheader("Open-Sourced LLMs")
        platform_choice_os = st.radio(
            label="Select the Platform:",
            options=("HuggingFace", "Ollama"),
            key="opensource_platform",
            help="Further platforms are to be implemented, such as Ollama.",
        )
        if platform_choice_os == "HuggingFace":
            huggingface_api_key = st.text_input(
                label=f"{platform_choice_os} API Key",
                placeholder=f"Enter your {platform_choice_os} API key",
                type="password",
                key="opensource_api_key",
            )
            huggingface_llm_name = st.text_input(
                label="Open-Sourced LLM Name or ID",
                placeholder="Enter the LLM name or ID",
                key="opensource_llm_name",
            )
            huggingface_embedding_model_name = st.text_input(
                label="Open-Sourced Embedding Model Name",
                placeholder="Enter the embedding model name",
                key="opensource_embedding_model_name",
            )
            # If huggingface_embedding_model_name is not provided, fallback to OpenAI
            if not huggingface_embedding_model_name:
                st.info(
                    f"No {platform_choice_os} embedding model specified. "
                    f"The {platform_choice_pr} embedding model will be used as a fallback."
                )

        elif platform_choice_os == "Ollama":
            st.warning(
                (
                    f"The {platform_choice_os} Platform is not yet implemented. "
                    "Please select another platform."
                )
            )
    # Validation: Ensure required fields for the selected platform are filled
    # Proprietary fields
    proprietary_fields = [
        openai_api_key,
        openai_llm_name,
        openai_embedding_model_name,
        openai_endpoint,
        openai_api_version,
    ]
    # Remove endpoint and api_version from validation if OpenAI is the platform
    if platform_choice_pr == "OpenAI":
        for field in (openai_endpoint, openai_api_version):
            proprietary_fields.remove(field)
    # Fields used in validation
    proprietary_fields_filled = all(proprietary_fields)

    # Open-sourced fields
    opensource_fields = [
        huggingface_api_key,
        huggingface_llm_name,
        huggingface_embedding_model_name,
    ]
    # Remove open-sourced embedding model from validation if no value inserted
    if not huggingface_embedding_model_name:
        opensource_fields.remove(huggingface_embedding_model_name)
    # Fields used in validation
    opensource_fields_filled = (
        platform_choice_os == "HuggingFace" and all(opensource_fields)
    ) or platform_choice_os == "Ollama"

    # Roaming buttons
    button_footers = st.columns([1, 1, 8])
    button_footers[0].button("Back", on_click=prev_step)
    button_footers[1].button("Next", on_click=next_step)
    # if proprietary_fields_filled and opensource_fields_filled:
    #     button_footers[1].button("Next", on_click=next_step)
    # else:
    #     st.warning("Please fill out all required fields to proceed.")
    #     button_footers[1].button("Next", on_click=next_step, disabled=True)
