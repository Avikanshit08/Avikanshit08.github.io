import re

class CodeAwareTextSplitter:
    def __init__(self, chunk_size=1500, chunk_overlap=200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
    def split_text(self, text: str):
        parts = re.split(r'(\n# FILE:|\nclass |\ndef )', text)
        chunks = []
        buffer = ""
        for part in parts:
            if len(buffer) + len(part) > self.chunk_size:
                if buffer:
                    chunks.append(buffer)
                buffer = buffer[-self.chunk_overlap:] + part
            else:
                buffer += part
        if buffer:
            chunks.append(buffer)
        return chunks