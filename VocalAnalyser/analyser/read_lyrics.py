import json
import re

class ReadLyrics:
    def __init__(self, file_name: str, key_word: str):
        self.dest = file_name
        self.key = key_word

    def load(self, source_file: str):
        try:
            with open(source_file, 'r', encoding='utf-8') as f:
                text = f.read()
            
            cleaned_text = re.findall(r'\b\w+\b', text, re.UNICODE)
            data = {self.key: cleaned_text}

            with open(self.dest, 'w', encoding='utf-8') as json_f:
                json.dump(data, json_f, indent=4, ensure_ascii=False)
                
        except FileExistsError:
            print(f"Error: The file '{source_file}' does not exist.")
        except Exception as e:
            print(f"An unexpected error occurred: {e}")


