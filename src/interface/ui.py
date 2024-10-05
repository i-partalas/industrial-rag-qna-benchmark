"""
Run with 'streamlit run ./src/interface/ui.py'
"""

import streamlit as st

st.title(
    body="Benchmarking LLMs on Industrial Data",
    help="Benchmarking both proprietary and open-sourced LLMs hosted on diverse platforms.",
)

# Create tabs for multiple pages
tabs = st.tabs(
    [
        "Home",
        "Data",
        "LLMs",
        "Metrics",
        "Evaluation Set",
        "Hyperparameters",
        "Evaluation",
        "Results",
    ]
)

# Tab: Home Tab
with tabs[0]:
    st.header("Welcome to the Evaluation App")
    st.write(
        "This app allows you to evaluate RAG performance of proprietary and open-sourced LLMs."
    )
    st.image("res/rag_eval_pipeline.png")

# Tab: Upload Data
with tabs[1]:
    uploaded_pdfs = st.file_uploader(
        "Upload the PDF file(s) you want to interact with:",
        type="pdf",
        accept_multiple_files=True,
    )

# Tab: Select LLMs
with tabs[2]:
    # Create two columns: Proprietary and Open-Sourced LLMs
    col1, col2 = st.columns(2)

    # Proprietary LLMs - Column 1
    with col1:
        st.subheader("Proprietary LLMs")

        # Radio box to select platform
        platform_choice = st.radio(
            "Select the Platform:",
            ("OpenAI Platform", "AzureOpenAI Platform"),
            key="proprietary_platform",
        )

        # Prompt user for platform credentials
        st.write("Please provide your credentials below:")
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

    # Open-Sourced LLMs - Column 2
    with col2:
        st.subheader("Open-Sourced LLMs")

        # Radio box with only one option
        platform_choice_os = st.radio(
            "Select the Platform:",
            ("HuggingFace Platform", "Ollama Platform"),
            key="opensource_platform",
            help="Further platforms are to be implemented, such as Ollama.",
        )

        # Handle the platform selection
        platform_name_os = platform_choice_os.split()[0]
        if platform_name_os == "HuggingFace":
            # Prompt user for platform credentials
            st.write("Please provide your credentials below:")
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
            # Display a message to indicate that this platform is not yet implemented
            st.warning(
                f"The {platform_name_os} Platform is not yet implemented. Please select another platform."
            )

# Tab: Select Metrics
with tabs[3]:
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

    # Iterate through each metric family and create a dropdown list for selecting metrics
    for category, metrics in metric_families.items():
        # Handling LLM-assisted metrics separately as it has sub-categories
        if category == "LLM-assisted":
            st.subheader(f"{category} Metrics")

            # Multiselect for Generation-related metrics
            generation_related_metrics = st.multiselect(
                "Select Generation-related Metrics:",
                metrics["Generation-related"],
                key=f"{category}_generation_related_metrics",
            )

            # Multiselect for Retrieval-related metrics
            retrieval_related_metrics = st.multiselect(
                "Select Retrieval-related Metrics:",
                metrics["Retrieval-related"],
                key=f"{category}_retrieval_related_metrics",
            )

            # Display selected metrics for confirmation or further usage
            if generation_related_metrics:
                st.write(
                    f"Selected Generation-related Metrics: {', '.join(generation_related_metrics)}"
                )

            if retrieval_related_metrics:
                st.write(
                    f"Selected Retrieval-related Metrics: {', '.join(retrieval_related_metrics)}"
                )

        # Handling other categories
        else:
            ppl_msg = "Please bare in mind that 'Perplexity' can be calculated only for Open-Sourced LLMs."
            help_msg = ppl_msg if category == "Intrinsic" else None
            st.subheader(f"{category} Metrics", help=help_msg)

            selected_metrics = st.multiselect(
                f"Select {category} Metrics:", metrics, key=f"{category}_metrics"
            )

            # Display selected metrics for confirmation or further usage
            if selected_metrics:
                st.write(f"Selected {category} Metrics: {', '.join(selected_metrics)}")

# Tab: Upload/Create Evaluation Set
# TODO: Complete the help msg
with tabs[4]:
    method_1 = "Upload your own evaluation dataset"
    method_2 = "Generate a synthetic evaluation dataset based on uploaded PDF file(s)"
    choice = st.radio(
        label="Choose the method to proceed with the evaluation dataset:",
        options=(method_1, method_2),
        help="For the structure and format of the file, please advise ...",
    )

    if choice == method_1:
        # Upload evaluation dataset in Excel format
        uploaded_file = st.file_uploader(
            "Upload your evaluation dataset file:", type="xlsx"
        )
        if uploaded_file is not None:
            st.success(f"Uploaded your evaluation dataset: {uploaded_file.name}")

    elif choice == method_2:
        # Upload PDF(s) for an artificially created evaluation dataset
        uploaded_pdfs = st.file_uploader(
            "Upload your PDF file(s):", type="pdf", accept_multiple_files=True
        )
        if uploaded_pdfs:
            st.success(
                f"Uploaded {len(uploaded_pdfs)} PDF file(s) for synthetic dataset generation"
            )

# Tab: RAG Hyperparameters
with tabs[5]:
    st.header("Configure RAG Hyperparameters")

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
    # TODO
    st.subheader("Evaluation Hyperparameters")
    eval_model_temperature = st.slider(
        "Model Temperature for Evaluation:",
        min_value=0.0,
        max_value=1.0,
        value=0.2,
        step=0.05,
        help="Controls the randomness of predictions. Lower values make the model more deterministic.",
    )

    # Button to save configurations
    if st.button("Save Hyperparameters"):
        st.success("RAG Hyperparameters saved successfully!")

# Tab: Evaluate
with tabs[6]:
    st.button("Run Evaluation", type="primary", use_container_width=True)

# Tab: View Results
with tabs[7]:
    st.write(
        "Once you upload or generate an evaluation set, results will be displayed here."
    )
