from langchain_openai import (
    AzureChatOpenAI,
    AzureOpenAI,
    AzureOpenAIEmbeddings,
    ChatOpenAI,
    OpenAI,
    OpenAIEmbeddings,
)

azure_embed_model = AzureOpenAIEmbeddings(
    model=settings.azure_embed_model,
    azure_endpoint=settings.azure_endpoint,
    api_key=settings.azure_api_key,
    api_version=settings.azure_api_version,
)

azure_lm = AzureOpenAI(
    deployment_name=settings.azure_gpt35,
    azure_endpoint=settings.azure_endpoint,
    api_key=settings.azure_api_key,
    api_version=settings.azure_api_version,
    temperature=0.2,
)

azure_chat_lm = AzureChatOpenAI(
    azure_deployment=settings.azure_gpt4,
    azure_endpoint=settings.azure_endpoint,
    openai_api_key=settings.azure_api_key,
    openai_api_version=settings.azure_api_version,
    temperature=0.2,
)

openai_embed_model = OpenAIEmbeddings(
    model=settings.openai_embed_model,
    api_key=settings.openai_api_key,
    openai_api_version=settings.openai_api_version,
)

openai_lm = OpenAI(
    model=settings.openai_gpt35,
    openai_api_key=settings.openai_api_key,
    temperature=0.2,
)

openai_chat_lm = ChatOpenAI(
    model=settings.openai_gpt4,
    openai_api_key=settings.openai_api_key,
    temperature=0.2,
)
