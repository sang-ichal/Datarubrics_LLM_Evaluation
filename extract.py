import os
import argparse
from pypdf import PdfReader


def extract_text_from_pdf(
    pdf_path, output_path=None, start_page_i=0, end_page_i=None, extraction_mode="plain"
):
    """
    Extract text from PDF file.

    Args:
        pdf_path (str): Path to PDF file
        output_path (str, optional): Path to output text file. If None, uses same name as PDF with .txt extension in /txt folder
        start_page_i (int, optional): Starting page index. Defaults to 0.
        end_page_i (int, optional): Ending page index. If None, extracts all pages. Defaults to None.
        extraction_mode (str, optional): Text extraction mode. Defaults to "layout".
    """
    # Create reader
    reader = PdfReader(pdf_path)

    # Set end page if not specified
    if end_page_i is None:
        end_page_i = len(reader.pages)

    # Generate default output path if not specified
    if output_path is None:
        pdf_name = os.path.splitext(os.path.basename(pdf_path))[0]
        output_dir = os.path.join(os.path.dirname(os.path.dirname(pdf_path)), "txt")
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f"{pdf_name}.txt")

    # Extract text from pages
    extracted_text = []
    for i in range(start_page_i, end_page_i):
        page = reader.pages[i]
        text = page.extract_text(extraction_mode=extraction_mode)
        extracted_text.append(text)

    # Write to output file
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n\n".join(extracted_text))


def main():
    parser = argparse.ArgumentParser(description="Extract text from PDF files")
    parser.add_argument("--input_path", help="Path to PDF file")
    parser.add_argument("--output_path", help="Path to output text file")
    parser.add_argument(
        "--start_page_i", type=int, default=0, help="Starting page index (0-based)"
    )
    parser.add_argument(
        "--end_page_i", type=int, default=None, help="Ending page index (0-based)"
    )
    parser.add_argument(
        "--mode",
        default="plain",
        choices=["plain", "layout"],
        help="Text extraction mode",
    )

    args = parser.parse_args()

    extract_text_from_pdf(
        args.input_path, args.output_path, args.start_page_i, args.end_page_i, args.mode
    )


if __name__ == "__main__":
    main()
