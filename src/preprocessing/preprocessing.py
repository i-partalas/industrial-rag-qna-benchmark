import json
from pathlib import Path
from typing import Any, Dict, List

from langchain_community.document_loaders import DirectoryLoader, UnstructuredFileLoader
from langchain_core.documents import Document
from streamlit.runtime.uploaded_file_manager import UploadedFile
from tqdm import tqdm

from config.setup_paths import CHUNKS_REPORT, PDF_DIR
from utils.helper_functions import determine_max_concurrency
from utils.logger import logger


class PDFProcessor:
    def __init__(
        self,
        files: List[UploadedFile],
        pdf_dir: Path = PDF_DIR,
        report_path: Path = CHUNKS_REPORT,
    ) -> None:
        """
        Initializes the PDFProcessor with the list of uploaded files,
        and optional paths for the PDF directory and the report file.

        :param files: A list of UploadedFile instances.
        :param pdf_dir: The directory where PDFs will be saved.
        :param report_path: The file path where the JSON report will be saved.
        """
        self.files = files
        self.pdf_dir = pdf_dir
        self.report_path = report_path

        logger.info("Starting to process PDF files.")
        self._save_files()

    def _get_filepath(self, file: UploadedFile) -> Path:
        """
        Generates the file path for a given uploaded file within the PDF directory.

        :param file: An instance of UploadedFile representing the uploaded file.
        :return: The complete file path where the file will be saved.
        """
        return self.pdf_dir / file.name

    def _save_files(self) -> None:
        """
        Saves the uploaded files to the designated PDF directory.
        """
        logger.info("Saving uploaded files to the directory.")

        for file in tqdm(self.files, desc="Saving PDF files", unit=" file"):
            try:
                filepath = self._get_filepath(file)
                with open(filepath, "wb") as f:
                    f.write(file.getbuffer())
                logger.info(f"File {file.name} saved successfully.")
            except Exception as e:
                logger.error(f"Failed to save file {file.name}: {e}")

    def _save_docs(self, docs: List[Document]) -> None:
        """
        Saves the list of Document instances to a JSON file.

        :param docs: A list of Document instances.
        """
        try:
            json_docs = [doc.to_json() for doc in docs]
            with open(self.report_path, "w") as f:
                json.dump(json_docs, f, indent=4)
            logger.info(f"Documents saved to {self.report_path}.")
        except Exception as e:
            logger.error(f"Failed to save documents to {self.report_path}: {e}")

    def _create_custom_loader(self) -> type:
        """
        Creates a custom file loader class with 'elements' mode as the default for chunking.

        :return: A subclass of UnstructuredFileLoader with custom settings.
        """
        logger.info("Creating custom file loader.")

        class CustomFileLoader(UnstructuredFileLoader):
            def __init__(self, *args, **kwargs):
                kwargs["mode"] = "elements"
                super().__init__(*args, **kwargs)

        return CustomFileLoader

    def _get_loader_configuration(self) -> Dict[str, Any]:
        """
        Provides configuration settings for the DirectoryLoader used for chunking the PDFs.

        :return: A dictionary containing loader configuration settings.
        """
        logger.info("Fetching loader configuration settings.")
        return {
            "extract_images_in_pdf": False,
            "strategy": "hi_res",
            "languages": ["eng"],
            "chunking_strategy": "by_title",
            "max_characters": 4000,
            "new_after_n_chars": 3800,
            "combine_text_under_n_chars": 2000,
        }

    def _load_documents(
        self, loader_cls: UnstructuredFileLoader, loader_kwargs: Dict[str, Any]
    ) -> List[Document]:
        """
        Loads documents from the PDF directory using the provided loader class and settings.

        :param loader_cls: The loader class to be used for loading documents.
        :param loader_kwargs: A dictionary of arguments for configuring the loader.
        :return: A list of loaded Document instances.
        """
        logger.info("Initializing DirectoryLoader to load documents.")
        loader = DirectoryLoader(
            path=self.pdf_dir,
            loader_cls=loader_cls,
            loader_kwargs=loader_kwargs,
            show_progress=True,
            use_multithreading=True,
            max_concurrency=determine_max_concurrency(),
        )

        try:
            docs = loader.load()
            logger.info("Documents loaded successfully.")
            return docs
        except Exception as e:
            logger.error(f"Failed to load documents: {e}")
            return []

    def chunk_files2docs(self) -> List[Document]:
        """
        Orchestrates the process of splitting PDF files into smaller document chunks.

        :return: A list of chunked Document instances.
        """
        logger.info("Starting the chunking process.")
        custom_loader_cls = self._create_custom_loader()
        loader_kwargs = self._get_loader_configuration()

        docs = self._load_documents(custom_loader_cls, loader_kwargs)
        if docs:
            self._save_docs(docs)

        logger.info("Chunking process completed.")
        return docs
