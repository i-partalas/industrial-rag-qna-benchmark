import os
from typing import Dict, List
from uuid import uuid4

import pandas as pd
from langchain_core.documents import Document
from tqdm import tqdm
from unstructured.documents.elements import Element
from unstructured.partition.pdf import partition_pdf


class PDFChunker:
    def __init__(self, evalset_path: str, pdf_path: str) -> None:
        """
        Initializes the PDFChunker with paths to an Excel file containing references
        and a directory containing PDF files to be processed.

        :param evalset_path: Path to an Excel file with gold-standard references.
        :param pdf_path: Path to the directory containing PDF files.
        """
        self.evalset_df = self._read_data(evalset_path)
        self.pdf_path = pdf_path

    def _read_data(self, path: str) -> pd.DataFrame:
        df = pd.read_excel(path)
        df.columns = df.columns.str.lower()
        df = df.fillna("")
        return df

    def _extract_queried_pdf_names(self) -> Dict[str, str]:
        """
        Extracts the filenames that are queried based on the names listed
        in the evalset DataFrame.

        :return: A dictionary mapping PDF filenames to their full file paths.
        """
        fname_to_fpath = {}
        queried_pdf_names = self.evalset_df["doc_name"].tolist()
        for root, _, filenames in os.walk(self.pdf_path, topdown=True):
            for filename in filenames:
                if filename.endswith(".pdf") and filename in queried_pdf_names:
                    fname_to_fpath[filename] = os.path.join(root, filename)
        return fname_to_fpath

    def prepare_evalset_df(self) -> None:
        """
        Updates the evalset DataFrame with file paths for the PDF documents listed
        in the reference sheet. Also simplifies the DataFrame by removing unnecessary
        columns and duplicates.
        """
        fname_to_fpath = self._extract_queried_pdf_names()
        self.evalset_df["filepath"] = self.evalset_df["doc_name"].map(fname_to_fpath)
        self.evalset_df.drop(
            self.evalset_df.columns[[0, 1, 2, 3, -2]], axis=1, inplace=True
        )
        self.evalset_df.drop_duplicates(
            subset=["filepath"], inplace=True, ignore_index=True
        )

    def _insert_data_to_df(
        self, row_idx: int, chunks: List[Element], lang: str
    ) -> None:
        """
        Inserts chunk data into the evalset DataFrame at the specified row.

        :param row_idx: The index of the row in the DataFrame to update.
        :param chunks: The list of document elements extracted from the PDF.
        :param lang: The language of the document.
        """
        lang = lang.lower()
        cols = ["texts", "ids"]
        for col in cols:
            if col not in self.evalset_df.columns:
                self.evalset_df[col] = None

        chunk_texts = set([chunk.text for chunk in chunks])
        chunk_ids = [str(uuid4()) for _ in chunk_texts]
        self.evalset_df.at[row_idx, "texts"] = chunk_texts
        self.evalset_df.at[row_idx, "ids"] = chunk_ids

    def _unstructured_chunk_pdf(self, filepath: str) -> List[Document]:
        """
        Chunks a single PDF file into elements.

        :param filepath: Path to the PDF file.
        :return: A list of document elements extracted from the PDF.
        """
        chunks: List[Element] = partition_pdf(
            filename=filepath,
            extract_images_in_pdf=False,
            strategy="hi_res",
            infer_table_structure=True,
            languages=["deu", "eng"],
            chunking_strategy="by_title",
            max_characters=4000,
            new_after_n_chars=3800,
            combine_text_under_n_chars=2000,
        )
        docs: List[Document] = [
            Document(page_content=chunk.text, metadata=chunk.metadata.to_dict())
            for chunk in chunks
        ]
        return docs

    def chunk_pdfs(self) -> None:
        """
        Processes each PDF file listed in the evalset DataFrame, chunks it,
        and stores the chunk data back into the DataFrame.
        """
        for index, row in tqdm(
            self.evalset_df.iterrows(), desc="Processing PDFs", unit=" PDF"
        ):
            filepath = row["filepath"]
            lang = row["language"]
            docs = self._unstructured_chunk_pdf(filepath)
            self._insert_data_to_df(index, docs, lang)
