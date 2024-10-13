import time

import streamlit as st

# Initialize session state to track progress between steps and form inputs
if "step" not in st.session_state:
    st.session_state.step = 0


def next_step():
    st.session_state.step += 1


def prev_step():
    st.session_state.step -= 1


def reset_wizard():
    st.session_state.step = 0


# Step 0: Home
if st.session_state.step == 0:
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


# Step 1: Upload Data
elif st.session_state.step == 1:
    st.header("Upload the PDF files")
    uploaded_pdfs = st.file_uploader(
        "Upload the PDF file(s) you want to interact with:",
        type="pdf",
        accept_multiple_files=True,
    )
    button_footers = st.columns([1, 1, 8])
    button_footers[0].button("Back", on_click=prev_step)

    # Validation: Ensure at least one PDF file is uploaded
    if uploaded_pdfs:
        button_footers[1].button("Next", on_click=next_step)
    else:
        st.warning("Please upload at least one PDF file to proceed.")
        button_footers[1].button("Next", on_click=next_step, disabled=True)

# Step 2: Select LLMs
elif st.session_state.step == 2:
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
        openai_endpoint = st.text_input(
            "Endpoint", placeholder="Enter the API endpoint"
        )
        openai_api_version = st.text_input(
            "API Version", placeholder="Enter the API version"
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
                f"The {platform_name_os} Platform is not yet implemented. Please select another platform."
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
    if proprietary_fields_filled or opensource_fields_filled:
        button_footers[1].button("Next", on_click=next_step)
    else:
        st.warning("Please fill out all required fields to proceed.")
        button_footers[1].button("Next", on_click=next_step, disabled=True)

# Step 3: Select Metrics
elif st.session_state.step == 3:
    st.header("Select the Evaluation Metrics")

    metric_families = {
        "Intrinsic": ["Perplexity"],
        "Lexical": ["BLEU", "ROUGE-1", "ROUGE-2", "ROUGE-L", "METEOR"],
        "Embedding-based": ["BERTScore", "SEMScore", "Answer Semantic Similarity"],
        "LLM-assisted": {
            "Generation-related": [
                "Answer Correctness",
                "Answer Relevancy",
                "Coherence",
                "Hallucination",
                "Faithfulness",
            ],
            "Retrieval-related": [
                "Contextual Precision",
                "Contextual Recall",
                "Contextual Relevancy",
            ],
        },
    }
    # Track if any metrics are selected
    all_metrics_selected = False

    for category, metrics in metric_families.items():
        if category == "LLM-assisted":
            st.subheader(f"{category} Metrics")

            generation_related_metrics = st.multiselect(
                "Select Generation-related Metrics:",
                metrics["Generation-related"],
                key=f"{category}_generation_related_metrics",
            )

            retrieval_related_metrics = st.multiselect(
                "Select Retrieval-related Metrics:",
                metrics["Retrieval-related"],
                key=f"{category}_retrieval_related_metrics",
            )

            if generation_related_metrics or retrieval_related_metrics:
                all_metrics_selected = True

        else:
            ppl_msg = "Please bare in mind that 'Perplexity' can be calculated only for Open-Sourced LLMs."
            help_msg = ppl_msg if category == "Intrinsic" else None
            st.subheader(f"{category} Metrics", help=help_msg)
            selected_metrics = st.multiselect(
                f"Select {category} Metrics:", metrics, key=f"{category}_metrics"
            )
            if selected_metrics:
                all_metrics_selected = True

    button_footers = st.columns([1, 1, 8])
    button_footers[0].button("Back", on_click=prev_step)
    if all_metrics_selected:
        button_footers[1].button("Next", on_click=next_step)
    else:
        button_footers[1].button("Next", on_click=next_step, disabled=True)
        st.warning("Please select at least one metric to proceed.")

# Step 4: Upload/Create Evaluation Set
elif st.session_state.step == 4:
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

    if dataset_uploaded:
        button_footers[1].button("Next", on_click=next_step)
    else:
        st.warning("Please upload a dataset to proceed.")
        button_footers[1].button("Next", on_click=next_step, disabled=True)

# Step 5: Configure Hyperparameters
elif st.session_state.step == 5:
    st.header("Configure the RAG Hyperparameters")

    # Section 1: Preprocessing
    st.subheader("Preprocessing Hyperparameters")
    max_text_length = st.number_input(
        "Maximum Characters in Text Length:",
        min_value=100,
        max_value=5000,
        value=4000,
        step=100,
        help="Cuts off new sections after reaching a length of n characters. (This is a hard maximum.)",
    )
    new_after_n_chars = st.number_input(
        "Soft Maximum Characters for New Sections:",
        min_value=100,
        max_value=5000,
        value=3800,
        step=100,
        help="Applies only when the chunking strategy is specified. Cuts off new sections after reaching a length of n characters. (This is a soft maximum.)",
    )
    combine_text_under_n_chars = st.number_input(
        "Combine Text under n Characters:",
        min_value=0,
        max_value=5000,
        value=2000,
        step=100,
        help="Combine small sections. In certain documents, partitioning may identify a list-item or other short paragraph as a Title element even though it does not serve as a section heading. This can produce chunks substantially smaller than desired. This behavior can be mitigated using the combine_text_under_n_chars argument. Setting this to 0 will disable section combining.",
    )

    # Section 2: Indexing
    # TODO
    st.subheader("Indexing Hyperparameters")

    # Section 3: Retrieval
    st.subheader("Retrieval Hyperparameters")
    retrieval_model = st.selectbox(
        "Retrieval Model:",
        options=["BM25", "Dense Embedding", "Hybrid"],
        index=1,
        help="Choose the retrieval model: BM25, dense embedding-based, or a hybrid approach.",
    )
    top_k_retrievals = st.number_input(
        "Top-K Retrieved Documents:",
        min_value=1,
        max_value=100,
        value=4,
        step=1,
        help="Number of top documents to retrieve for answering a query.",
    )
    # Section 4: Generation
    st.subheader("Generation Hyperparameters")
    model_temperature = st.slider(
        "Model Temperature:",
        min_value=0.0,
        max_value=1.0,
        value=0.2,
        step=0.05,
        help="Controls the randomness of predictions. Lower values make the model more deterministic.",
    )
    max_tokens = st.number_input(
        "Maximum Tokens:",
        min_value=1,
        max_value=1024,
        value=150,
        step=10,
        help="Maximum number of tokens to generate in a response.",
    )

    # Section 5: Evaluation
    st.subheader("Evaluation Hyperparameters")
    eval_model_temperature = st.slider(
        "Model Temperature for Evaluation:",
        min_value=0.0,
        max_value=1.0,
        value=0.2,
        step=0.05,
        help="Controls the randomness of predictions. Lower values make the model more deterministic.",
    )

    # Ensure hyperparameters are filled
    button_footers = st.columns([1, 1, 8])
    if all(
        [
            max_text_length,
            new_after_n_chars,
            combine_text_under_n_chars,
            top_k_retrievals,
            model_temperature,
            max_tokens,
        ]
    ):
        button_footers[1].button("Next", on_click=next_step)
    else:
        st.warning("Please fill in all hyperparameters.")
        button_footers[1].button("Next", on_click=next_step, disabled=True)

    button_footers[0].button("Back", on_click=prev_step)

# Step 6: Run the Evaluation
elif st.session_state.step == 6:
    st.header("Run the Evaluation")
    button_footers = st.columns([1.1, 1.1, 1.1, 6.7])
    if button_footers[1].button("Run"):
        with st.spinner():
            time.sleep(5)
            st.success("Evaluation started successfully!")
            button_footers[2].button("Next", on_click=next_step, disabled=False)
    button_footers[0].button("Back", on_click=prev_step)

# Step 7: View Results
elif st.session_state.step == 7:
    st.header("View the Evaluation Results")
    st.write(
        "Once you upload or generate an evaluation set, results will be displayed here."
    )
    st.button("Restart", on_click=reset_wizard)
