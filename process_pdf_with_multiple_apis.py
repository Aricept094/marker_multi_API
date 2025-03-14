#!/usr/bin/env python3
import argparse
import os
import sys
import tempfile
from pathlib import Path
from typing import List

import PyPDF2
from tqdm import tqdm

# Add the current directory to the path so that we can import marker
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from marker.converters.pdf import PdfConverter
from marker.output import save_output
from marker.renderers.markdown import MarkdownRenderer
from marker.models import create_model_dict


def split_pdf(input_path: str, output_dir: str, pages_per_chunk: int = 200) -> List[str]:
    """Split a PDF file into chunks of specified size and save them to output_dir."""
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Open the input PDF
    with open(input_path, 'rb') as file:
        pdf_reader = PyPDF2.PdfReader(file)
        total_pages = len(pdf_reader.pages)
        
        print(f"Total pages in PDF: {total_pages}")
        
        # Calculate number of chunks
        num_chunks = (total_pages + pages_per_chunk - 1) // pages_per_chunk
        chunk_paths = []
        
        # Create each chunk
        for chunk_num in tqdm(range(num_chunks), desc="Splitting PDF"):
            start_page = chunk_num * pages_per_chunk
            end_page = min((chunk_num + 1) * pages_per_chunk, total_pages)
            
            # Create a new PDF writer
            pdf_writer = PyPDF2.PdfWriter()
            
            # Add pages to the writer
            for page_num in range(start_page, end_page):
                pdf_writer.add_page(pdf_reader.pages[page_num])
            
            # Save the chunk
            output_path = os.path.join(output_dir, f"chunk_{chunk_num + 1}.pdf")
            with open(output_path, 'wb') as output_file:
                pdf_writer.write(output_file)
            
            chunk_paths.append(output_path)
    
    return chunk_paths


def process_pdf_chunk(pdf_path: str, api_keys: List[str], output_dir: str, use_cuda: bool = False) -> str:
    """Process a PDF chunk using marker with multiple API keys."""
    # Get filename without extension
    filename = os.path.basename(pdf_path)
    base_filename = os.path.splitext(filename)[0]
    
    # Create output directory for this chunk
    chunk_output_dir = os.path.join(output_dir, base_filename)
    os.makedirs(chunk_output_dir, exist_ok=True)
    
    # Load models/artifacts
    device = "cuda" if use_cuda else None
    artifacts = create_model_dict(device=device)
    
    # Configure the converter with multiple API keys
    config = {
        "use_llm": True,
        "gemini_api_keys": api_keys,
        "gemini_api_key": api_keys[0]  # Add the first key as the default singular key
    }
    
    # Enable CUDA if requested
    if use_cuda:
        print("Enabling CUDA/GPU acceleration...")
        config["TORCH_DEVICE"] = "cuda"
    
    converter = PdfConverter(
        artifacts,
        config=config
    )
    
    # Process the PDF
    document = converter(pdf_path)
    
    # Create an output handler
    renderer = MarkdownRenderer()
    rendered_output = renderer(document)
    
    # Save the results
    save_output(rendered_output, chunk_output_dir, base_filename)
    
    return chunk_output_dir


def main():
    parser = argparse.ArgumentParser(description="Process large PDFs with marker using multiple Gemini API keys")
    parser.add_argument("pdf_path", help="Path to the PDF file to process")
    parser.add_argument("--api-keys", required=True, nargs="+", help="List of Gemini API keys to use")
    parser.add_argument("--output-dir", default="./output", help="Directory to save the output")
    parser.add_argument("--pages-per-chunk", type=int, default=200, help="Number of pages per chunk")
    parser.add_argument("--use-cuda", action="store_true", help="Enable CUDA/GPU acceleration")
    
    args = parser.parse_args()
    
    # Create a temporary directory for the PDF chunks
    with tempfile.TemporaryDirectory() as temp_dir:
        # Split the PDF into chunks
        print(f"Splitting PDF into chunks of {args.pages_per_chunk} pages...")
        chunk_paths = split_pdf(args.pdf_path, temp_dir, args.pages_per_chunk)
        
        # Process each chunk
        print(f"Processing {len(chunk_paths)} chunks with {len(args.api_keys)} API keys...")
        for i, chunk_path in enumerate(chunk_paths):
            print(f"Processing chunk {i+1}/{len(chunk_paths)}: {chunk_path}")
            output_path = process_pdf_chunk(chunk_path, args.api_keys, args.output_dir, use_cuda=args.use_cuda)
            print(f"Chunk {i+1} output saved to: {output_path}")
        
        print(f"All chunks processed. Results saved to: {args.output_dir}")


if __name__ == "__main__":
    main() 