from preprocess import MedicalTextPreprocessor
import fitz
import re
import nltk
from os import path, makedirs

def extract_text_from_pdf(pdf_path):
    with fitz.open(pdf_path) as doc:
        pages_to_include = list(range(10, 94))

        selected_blocks = []
        for page_num in pages_to_include:
            page = doc.load_page(page_num)  # Load page by number
            selected_blocks.append([b[4] for b in page.get_text("blocks")])

        selected_blocks = [x for xs in selected_blocks for x in xs]
        return selected_blocks


def merge_blocks(blocks):
    new_blocks = [[]]
    section_number = 0
    for b in blocks:
        if re.match("^\d{1,2}\.\d{1,2} [A-Z](\w| )*", b) or re.match("^\d{1,2} [A-Z](\w| )*", b):
            new_blocks.append([])
            section_number += 1
        new_blocks[section_number].append(b)
    return new_blocks


guide_blocks = extract_text_from_pdf("ehab368.pdf")
merged_blocks = merge_blocks(guide_blocks)
print(merged_blocks)

preprocessor = MedicalTextPreprocessor()

from nltk.tokenize import sent_tokenize

sections = [sent_tokenize(''.join(b)) for b in merged_blocks]
processed_sections = [preprocessor.batch_preprocess(s) for s in sections]

processed_guide = [" ".join(ps) for ps in processed_sections]

if not path.exists('sections'):
    makedirs('sections')

for (i, block) in enumerate(processed_guide):
    with open(f'sections/{i}.txt', "w+", encoding="utf-8") as file:
        file.write(block)
