import re
import os
import shutil
import uuid


def empty_folder(dossier):
    for root, dirs, files in os.walk(dossier, topdown=False):
        for name in files:
            file_path = os.path.join(root, name)
            os.remove(file_path)
        for name in dirs:
            dir_path = os.path.join(root, name)
            shutil.rmtree(dir_path)


def empty_chroma_folder():
    empty_folder("data")


def clean_text(text):
    return re.sub(r'[^\x00-\x7Féàèù@]+', '', text)


# If needed
def generate_unique_id(existing_ids):
    new_id = str(uuid.uuid4())
    while new_id in existing_ids:
        new_id = str(uuid.uuid4())
    return new_id