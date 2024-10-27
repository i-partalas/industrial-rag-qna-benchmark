from typing import Literal

import streamlit as st
from langchain_openai import (
    AzureChatOpenAI,
    AzureOpenAI,
    AzureOpenAIEmbeddings,
    ChatOpenAI,
    OpenAI,
    OpenAIEmbeddings,
)

from src.utils.logger import logger

# TODO: Add logic to distinguish between llm and chat_llm provided model name
# (param: openai_llm_name)

PLATFORM = st.session_state.proprietary_platform


def get_embed_model() -> AzureOpenAIEmbeddings | OpenAIEmbeddings:
    """
    Initializes and returns an embedding model, either AzureOpenAIEmbeddings or
    OpenAIEmbeddings, based on the selected platform and configured session state parameters.

    :return: An instance of AzureOpenAIEmbeddings or OpenAIEmbeddings.
    """
    embed_model = None
    try:
        # Define parameters based on platform
        model_params = {
            "model": st.session_state.openai_embedding_model_name,
            "api_key": st.session_state.openai_api_key,
        }
        # Add Azure-specific parameters if PLATFORM is AzureOpenAI
        if PLATFORM == "AzureOpenAI":
            model_params.update(
                {
                    "azure_endpoint": st.session_state.openai_endpoint,
                    "openai_api_version": st.session_state.openai_api_version,
                }
            )
            embed_model = AzureOpenAIEmbeddings(**model_params)
        else:
            embed_model = OpenAIEmbeddings(**model_params)
    except Exception as e:
        logger.error(f"Failed to initialize the embedding model: {e}")
        st.error(
            "Embedding model initialization failed. Please ensure your credentials are correct."
        )
    return embed_model


def get_proprietary_llm(
    task: Literal["generation", "evaluation"] = "generation"
) -> AzureOpenAI | OpenAI:
    """
    Initializes and returns a proprietary LLM based on the platform
    configuration, supporting both text generation and evaluation tasks.

    :param task: Specifies the type of task, either 'generation' or 'evaluation'.
    :return: An instance of AzureOpenAI or OpenAI configured for the specified task.
    """
    temperature = (
        st.session_state.generation_temperature
        if task == "generation"
        else st.session_state.evaluation_temperature
    )
    llm_model = None
    try:
        # Common parameters for both Azure and OpenAI
        model_params = {
            "temperature": temperature,
            "api_key": st.session_state.openai_api_key,
        }
        # Set Azure-specific or OpenAI-specific parameters
        if PLATFORM == "AzureOpenAI":
            model_params.update(
                {
                    "deployment_name": st.session_state.openai_llm_name,
                    "azure_endpoint": st.session_state.openai_endpoint,
                    "api_version": st.session_state.openai_api_version,
                }
            )
            llm_model = AzureOpenAI(**model_params)
        else:
            model_params.update({"model": st.session_state.openai_llm_name})
            llm_model = OpenAI(**model_params)
    except Exception as e:
        logger.error(f"Failed to initialize the LLM model: {e}")
        st.error(
            "LLM model initialization failed. Please ensure your credentials are correct."
        )
    return llm_model


def get_chat_model(
    task: Literal["generation", "evaluation"] = "generation"
) -> AzureChatOpenAI | ChatOpenAI:
    """
    Initializes and returns a chat model based on the platform
    configuration, supporting text generation and evaluation tasks.

    :param task: Specifies the type of task, either 'generation' or 'evaluation'.
    :return: An instance of AzureChatOpenAI or ChatOpenAI configured for the specified task.
    """
    temperature = (
        st.session_state.generation_temperature
        if task == "generation"
        else st.session_state.evaluation_temperature
    )
    chat_model = None
    try:
        # Common parameters for both Azure and OpenAI
        model_params = {
            "temperature": temperature,
            "api_key": st.session_state.openai_api_key,
        }
        # Platform-specific initialization
        if PLATFORM == "AzureOpenAI":
            model_params.update(
                {
                    "azure_deployment": st.session_state.openai_llm_name,
                    "azure_endpoint": st.session_state.openai_endpoint,
                    "api_version": st.session_state.openai_api_version,
                }
            )
            chat_model = AzureChatOpenAI(**model_params)
        else:
            model_params.update({"model": st.session_state.openai_llm_name})
            chat_model = ChatOpenAI(**model_params)
    except Exception as e:
        logger.error(f"Failed to initialize the chat model: {e}")
        st.error(
            "Chat model initialization failed. Please ensure your credentials are correct."
        )
    return chat_model
