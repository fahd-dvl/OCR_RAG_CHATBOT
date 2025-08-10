from docx import Document
import os

def extract_text_from_docx(file_path):
    text_content = []
    doc = Document(file_path)
    for paragraph in doc.paragraphs:
        text = paragraph.text.strip()
        if text:
            text_content.append(text)
    return text_content

input_file = '../data/uploads/dataset_partie2.docx'
output_file = '../data/extracted_text/extracted_text2.txt'

os.makedirs(os.path.dirname(output_file), exist_ok=True)
paragraphs = extract_text_from_docx(input_file)

with open(output_file, 'w', encoding='utf-8') as f:
    f.write('\n\n'.join(paragraphs))

print(f"Text extracted and saved to {output_file}")
