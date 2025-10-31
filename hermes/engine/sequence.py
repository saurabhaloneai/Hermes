from enum import Enum, auto 
from itertools import count 

class SequenceStatus(Enum):
    WAITING = auto()
    RUNNING = auto()
    FINISHED = auto()

class Sequence:
    _counter = count()
    block_size = 256

    def __init__(self, pormpt_tokens, temperature=0.0, max_tokens=1024, ignore_eos=False):
        self.seq_id = next(Sequence._counter)
        self.status = SequenceStatus.WAITING
        self.tokens = list(prompt_tokens)
        self.num_promt_tokens = len(self.prompt_tokens)
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.ignore_eos = ignore_eos
        self.block_table = []
        self.num_cached_tokens = 0

    def __len__(self):
        return len(self.tokens)

    def __getitem__(self, key):
        return self.tokens[key]
    
    @property
    def last_token(self):
        return self.tokens[-1]
    
    @property
    def num_completion_tokens(self):
        return len(self.tokens) - self.num_prompt_tokens
    
    @property
    def prompt_tokens(self):
        return self.tokens[:self.num_prompt_tokens]
    
    @property
    def completion_tokens(self):
        return self.tokens[self.num_prompt_tokens:]
    
    @property
    def num_blocks(self):
        return (len(self.tokens) + self.block_size - 1) // self.block_size
    
    @property
    def num_cached_blocks(self):
        return self.num_cached_tokens // self.block_size
    
    @property
    def last_block_size(self):
        remainder = len(self.tokens) % self.block_size
        return remainder if remainder > 0 else self.block_size

    def get_block(self, block_idx):
        start = block_idx * self.block_size
        end  = min(start + self.block_size, len(self.tokens))
        return self.tokens[start:end]

    def append_token(self, token_id: int):
        self.tokens.append(token_id)
    
    @property
    def is_finished(self):
        return self.status == SequenceStatus.FINISHED