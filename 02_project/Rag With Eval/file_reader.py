from pymupdf import open
import re
from typing import List
from pathlib import Path

def extract_paragraphs(pdf_path: str) -> List[str]:
    """
    Extracts paragraphs from a PDF file.

    Args:
        pdf_path (str): The path to the PDF file.
    """

    doc = open(pdf_path)
    paragraphs = []

    for page in doc:
        blocks = page.get_text("blocks")

        blocks = sorted(blocks,key=lambda x: (x[1], x[0]))

        for block in blocks:
            text = block[4]
            #replace the line breaks with a space and remove leading/trailing whitespace
            text = re.sub(r'\r', " ",text).strip()

            if text:
                paragraphs.append(text)
    return paragraphs
