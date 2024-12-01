from preprocess import MedicalTextPreprocessor
import fitz
import nltk

def extract_text_from_pdf(pdf_path):
    with fitz.open(pdf_path) as doc:
        pages_to_include = list(range(10, 94))

        selected_text = []
        for page_num in pages_to_include:
            page = doc.load_page(page_num)  # Load page by number
            selected_text.append(page.get_text())

        return "\n".join(selected_text)


guide_text = extract_text_from_pdf("ehab368.pdf")

preprocessor = MedicalTextPreprocessor()

from nltk.tokenize import sent_tokenize

sections = sent_tokenize(guide_text)
processed_sections = preprocessor.batch_preprocess(sections)

processed_guide = " ".join(processed_sections)
with open("processed_guide.txt", "w", encoding="utf-8") as file:
    file.write(processed_guide)