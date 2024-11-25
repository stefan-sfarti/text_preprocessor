from preprocess import MedicalTextPreprocessor
import fitz
import nltk

# Step 1: Setup
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')


# Step 2: Load the guide (with excluded pages)
def extract_text_from_pdf(pdf_path):
    with fitz.open(pdf_path) as doc:
        # Define pages to include: 10-93 (assuming pages start at 0, so 10 is page 11)
        pages_to_include = list(range(10, 94))  # Pages 10 to 93 (0-based indexing)

        # Extract text only from the selected pages
        selected_text = []
        for page_num in pages_to_include:
            page = doc.load_page(page_num)  # Load page by number
            selected_text.append(page.get_text())

        return "\n".join(selected_text)


# Load the guide while excluding first 10 pages and last pages (94-126)
guide_text = extract_text_from_pdf("ehab368.pdf")

# Step 3: Preprocessing
preprocessor = MedicalTextPreprocessor()

# Step 4: Split and preprocess
from nltk.tokenize import sent_tokenize

sections = sent_tokenize(guide_text)
processed_sections = preprocessor.batch_preprocess(sections)

# Step 5: Save output
processed_guide = " ".join(processed_sections)
with open("processed_guide.txt", "w", encoding="utf-8") as file:
    file.write(processed_guide)
