import streamlit as st


def display(prev_step, next_step):
    st.header("Configure the RAG Hyperparameters")

    # Section 1: Preprocessing
    st.subheader("Preprocessing Hyperparameters")
    max_text_length = st.number_input(
        "Maximum Characters in Text Length:",
        min_value=100,
        max_value=5000,
        value=4000,
        step=100,
        help=(
            "Cuts off new sections after reaching a length of n characters. "
            "(This is a hard maximum.)"
        ),
    )
    new_after_n_chars = st.number_input(
        "Soft Maximum Characters for New Sections:",
        min_value=100,
        max_value=5000,
        value=3800,
        step=100,
        help=(
            "Applies only when the chunking strategy is specified. "
            "Cuts off new sections after reaching a length of n characters. "
            "(This is a soft maximum.)"
        ),
    )
    combine_text_under_n_chars = st.number_input(
        "Combine Text under n Characters:",
        min_value=0,
        max_value=5000,
        value=2000,
        step=100,
        help=(
            "Combine small sections. In certain documents, partitioning may identify "
            "a list-item or other short paragraph as a Title element even though it "
            "does not serve as a section heading. "
            "This can produce chunks substantially smaller than desired. "
            "This behavior can be mitigated using the combine_text_under_n_chars argument. "
            "Setting this to 0 will disable section combining."
        ),
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
        key="generation_temperature",
        help=(
            "Controls the randomness of predictions. "
            "Lower values make the model more deterministic."
        ),
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
        key="evaluation_temperature",
        help=(
            "Controls the randomness of predictions. "
            "Lower values make the model more deterministic."
        ),
    )

    # Ensure hyperparameters are filled
    button_footers = st.columns([1, 1, 8])
    if all(
        [
            max_text_length,
            new_after_n_chars,
            combine_text_under_n_chars,
            retrieval_model,
            top_k_retrievals,
            model_temperature,
            max_tokens,
            eval_model_temperature,
        ]
    ):
        button_footers[1].button("Next", on_click=next_step)
    else:
        st.warning("Please fill in all hyperparameters.")
        button_footers[1].button("Next", on_click=next_step, disabled=True)

    button_footers[0].button("Back", on_click=prev_step)
