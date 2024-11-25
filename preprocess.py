import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import sent_tokenize
from collections import Counter


class MedicalTextPreprocessor:
    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        # Add medical-specific stopwords
        self.medical_stop_words = {
            'mg', 'ml', 'mcg', 'mmol', 'mmhg', 'patients', 'treatment',
            'therapy', 'dose', 'daily', 'versus', 'figure', 'table',
            'reference', 'study', 'studies', 'trial', 'trials'
        }
        self.stop_words.update(self.medical_stop_words)

        # Preserve medical units and measurements
        self.preserve_patterns = [
            r'\d+\.?\d*\s*(?:mg|mcg|ml|kg|mmol|mmHg|μg|ng|g|%)',  # Units
            r'class [I|V]+[a-zA-Z]?',  # Classification patterns
            r'grade [1-4]',  # Medical grades
            r'stage [A-D]',  # Disease stages
            r'type [1-2]'  # Disease types
        ]

    def preserve_medical_entities(self, text):
        """Preserve important medical measurements and classifications"""
        preserved_entities = []
        for pattern in self.preserve_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                preserved_entities.append((match.span(), match.group()))

        return preserved_entities

    def clean_text(self, text):
        """Main cleaning function"""
        # Store preserved entities
        preserved = self.preserve_medical_entities(text)

        # Convert to lowercase except preserved entities
        text = text.lower()

        # Remove URLs
        text = re.sub(r'https?://\S+|www\.\S+', '', text)

        # Remove HTML tags
        text = re.sub(r'<.*?>', '', text)

        # Remove references in brackets [1], [2,3], etc.
        text = re.sub(r'\[\d+(?:,\s*\d+)*\]', '', text)

        # Remove parenthetical abbreviations like (HF), (LVEF)
        text = re.sub(r'\s*\([A-Z]+\)', '', text)

        # Standardize measurements
        text = re.sub(r'(\d+)(?:\s*)(mg|mcg|ml|kg|mmol)', r'\1 \2', text)

        # Restore preserved entities
        for (start, end), entity in preserved:
            text = text[:start] + entity + text[end:]

        return text

    def remove_punctuation(self, text, preserve_units=True):
        """Remove punctuation while preserving units and decimals"""
        if preserve_units:
            # Temporarily replace decimal points in numbers
            text = re.sub(r'(\d+)\.(\d+)', r'\1DECIMAL\2', text)

        # Remove punctuation
        text = text.translate(str.maketrans('', '', string.punctuation))

        if preserve_units:
            # Restore decimal points
            text = text.replace('DECIMAL', '.')

        return text

    def lemmatize_text(self, text):
        """Lemmatize text while preserving medical terms"""
        words = text.split()
        lemmatized_words = []

        for word in words:
            # Skip lemmatization for measurements and preserved patterns
            if any(re.match(pattern, word, re.IGNORECASE) for pattern in self.preserve_patterns):
                lemmatized_words.append(word)
            else:
                lemmatized_words.append(self.lemmatizer.lemmatize(word))

        return ' '.join(lemmatized_words)

    def remove_stopwords(self, text):
        """Remove stopwords while preserving medical context"""
        words = text.split()
        return ' '.join([word for word in words if word.lower() not in self.stop_words])

    def standardize_medical_terms(self, text):
        """Standardize common medical abbreviations"""
        medical_abbrev = {
            'af': 'atrial fibrillation',
            'mi': 'myocardial infarction',
            'hf': 'heart failure',
            'lvef': 'left ventricular ejection fraction',
            'ace': 'angiotensin converting enzyme',
            'arb': 'angiotensin receptor blocker',
            'cv': 'cardiovascular'
        }

        words = text.split()
        standardized = []
        for word in words:
            lower_word = word.lower()
            if lower_word in medical_abbrev:
                standardized.append(medical_abbrev[lower_word])
            else:
                standardized.append(word)

        return ' '.join(standardized)

    def remove_headers(self, text):
        """Remove headers based on patterns found in the document."""
        # Split the text into lines
        lines = text.split('\n')
        cleaned_lines = []

        for line in lines:
            # Skip lines matching header patterns
            if (
                    re.match(r'^\s*ESC Guidelines\s*$', line, re.IGNORECASE) or  # Match 'ESC Guidelines'
                    re.match(r'^\s*\d{1,5}\s*$', line) or  # Match page numbers
                    re.match(r'^\s*\d{1,5}\s*ESC Guidelines\s*$', line, re.IGNORECASE) or  # Combined format
                    re.match(r'^\s*ESC Guidelines\s*\d{1,5}\s*$', line, re.IGNORECASE)  # Alternate combined format
            ):
                continue  # Skip this line

            # Add the line if it doesn't match the header patterns
            cleaned_lines.append(line)

        return '\n'.join(cleaned_lines)

    def preprocess(self, text):
        """Complete preprocessing pipeline"""
        text = self.remove_headers(text)
        # Clean the text
        text = self.clean_text(text)

        # Standardize medical terms
        text = self.standardize_medical_terms(text)

        # Remove punctuation
        text = self.remove_punctuation(text)

        # Lemmatize
        text = self.lemmatize_text(text)

        # Remove stopwords
        text = self.remove_stopwords(text)

        return text

    def batch_preprocess(self, texts):
        """Process multiple texts"""
        return [self.preprocess(text) for text in texts]