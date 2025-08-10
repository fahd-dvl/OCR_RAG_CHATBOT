import zipfile
import os

def extract_images_from_docx(file_path, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    with zipfile.ZipFile(file_path, 'r') as docx_zip:
        for file_name in docx_zip.namelist():
            if file_name.startswith('word/media/'):
                image_data = docx_zip.read(file_name)
                image_name = os.path.basename(file_name)
                image_path = os.path.join(output_dir, image_name)
                with open(image_path, 'wb') as f:
                    f.write(image_data)
    print(f"All images extracted to: {output_dir}")


input_file = '../data/uploads/dataset_partie1.docx'
output_dir = '../data/extracted_images/extracted_images1'

extract_images_from_docx(input_file, output_dir)
