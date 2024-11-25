from preprocess import MedicalTextPreprocessor
import fitz
import nltk



# Step 1: Setup
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')

# Step 2: Load the guide
def extract_text_from_pdf(pdf_path):
    with fitz.open(pdf_path) as doc:
        return "\n".join(page.get_text() for page in doc)

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