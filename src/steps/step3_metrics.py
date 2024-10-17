import streamlit as st


def display(prev_step, next_step):
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
            ppl_msg = (
                "Please bare in mind that 'Perplexity' "
                "can be calculated only for Open-Sourced LLMs."
            )
            help_msg = ppl_msg if category == "Intrinsic" else None
            st.subheader(f"{category} Metrics", help=help_msg)
            selected_metrics = st.multiselect(
                f"Select {category} Metrics:", metrics, key=f"{category}_metrics"
            )
            if selected_metrics:
                all_metrics_selected = True

    button_footers = st.columns([1, 1, 8])
    button_footers[0].button("Back", on_click=prev_step)

    button_footers[1].button("Next", on_click=next_step)
    # if all_metrics_selected:
    #     button_footers[1].button("Next", on_click=next_step)
    # else:
    #     button_footers[1].button("Next", on_click=next_step, disabled=True)
    #     st.warning("Please select at least one metric to proceed.")
