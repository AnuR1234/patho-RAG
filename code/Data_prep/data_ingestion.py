import fitz  # PyMuPDF
import os
from itertools import count

class Document:
    _id_generator = count(100000)
    def __init__(self, article_text, metadata, unique_id=None):
        self.article_text = article_text  # Renamed from page_content to article_text
        self.metadata = metadata
        # Adjust metadata to use 'article-title' instead of 'book_title'
        self.metadata['article-title'] = self.metadata.pop('book_title', 'Unknown Title')
        # Removed 'page_number' from metadata
        if 'page_number' in self.metadata:
            del self.metadata['page_number']
        self.unique_id = unique_id if unique_id is not None else next(self._id_generator)

    def __repr__(self):
        # Adjusted the representation to match the new format
        return f"Document(unique_id='article {self.unique_id}', article-text='{self.article_text[:30]}...', metadata={self.metadata})"

    def to_dict(self):
        # Converts the Document instance to a dictionary, maintaining the new naming
        return {
            'id': 'article ' + str(self.unique_id),  # Prefix 'article ' added to the id
            'article-text': self.article_text,
            'metadata': self.metadata
        }

def process_pdf(pdf_path, start_page):
    try:
        pdf = fitz.open(pdf_path)
        documents = []

        for page_num in range(start_page - 1, pdf.page_count):
            page = pdf[page_num]
            text = page.get_text()
            metadata = {
                'source': pdf_path,
                'book_title': os.path.basename(pdf_path).replace('.pdf', '').replace('_', ' ').replace('-', ' '),
                # 'page_number' key is now omitted based on requirements
            }
            documents.append(Document(text, metadata))

        pdf.close()
        return documents
    except Exception as e:
        print(f"Error processing the PDF: {e}")
        return []

def load_pdfs(pdf_details):
    all_documents = []
    for pdf_path, start_page in pdf_details.items():
        documents = process_pdf(pdf_path, start_page)
        all_documents.extend(documents)
    return all_documents
