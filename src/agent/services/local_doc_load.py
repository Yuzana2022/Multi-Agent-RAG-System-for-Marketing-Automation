import logging
from typing import List, Optional

from pdfminer.high_level import extract_pages
from pdfminer.layout import LTTextContainer
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class LocalDocResult(BaseModel):
    title: str = Field(..., description="The title of the local document")
    content: str = Field(..., description="The main body content extracted from the local document")


class AllLocalDocResults(BaseModel):
    results: List[LocalDocResult] = Field(default_factory=list)


def extract_text_from_pdf(
    filename: str,
    page_numbers: Optional[List[int]] = None,
    min_line_length: int = 10,
) -> AllLocalDocResults:
    output_results = AllLocalDocResults()
    valid_lines: List[str] = []

    for i, page_layout in enumerate(extract_pages(filename)):
        if page_numbers is not None and i not in page_numbers:
            continue

        for element in page_layout:
            if isinstance(element, LTTextContainer):
                for text_line in element.get_text().split("\n"):
                    clean_text = text_line.strip()
                    if len(clean_text) >= min_line_length:
                        if clean_text.endswith("-"):
                            valid_lines.append(clean_text.rstrip("-"))
                        else:
                            valid_lines.append(clean_text + " ")

    full_content = "".join(valid_lines).strip()

    if full_content:
        output_results.results.append(LocalDocResult(title=filename, content=full_content))
        logger.debug("Extracted %d characters from %s", len(full_content), filename)
    else:
        logger.warning("No content extracted from %s", filename)

    return output_results
