#!/bin/bash

# input_path: source PDF file to extract text from
# output_path: destination text file to write extracted content
# start_page_i: first page to extract (0-based index)
# end_page_i: last page to extract (0-based index) 
# mode: extraction mode - 'layout', 'plain'


python extract.py \
    --input_path pdf/2410.12705v4.pdf \
    --output_path txt/2410.12705v4.txt \
    --start_page_i 0 \
    --end_page_i 7 \
    --mode plain
