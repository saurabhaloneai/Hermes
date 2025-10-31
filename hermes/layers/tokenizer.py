from pathlib import Path
from tokenizer import Tokenizer
from huggingface_hub import snapshot_download

class Qwen3Tokenizer():
    def __init__(self, tokenizer_file_path="tokenizer.json", repo_id=None):
        if not Path(tokenizer_file_path).is_file() and repo_id and snapshot_download:
            snapshot_download(repo_id=repo_id, local_dir=Path(tokenizer_file_path).parent)
        self.tokenizer = Tokenizer.from_file(tokenizer_file_path)
     
    def encode(self, prompt):
        messages = [{"role": "user", "content": prompt}]
        formatted_prompt = self.format_qwen_chat(messages)
        return self.tokenizer.encode(formatted_prompt).ids
                     
    def decode(self, token_ids):
        return self.tokenizer.decode(token_ids, skip_special_tokens=False)
         
    @staticmethod
    def format_qwen_chat(messages):
        prompt = ""
        for msg in messages:
            prompt += f"<|im_start|>{msg['role']}\n{msg['content']}<|im_end|>\n"
        prompt += "<|im_start|>assistant<|think>\n\n<|/think>\n\n"
        return prompt