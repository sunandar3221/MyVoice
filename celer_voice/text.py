"""
Text processing and normalization module for CelerVoice.
Optimized for Indonesian and English text phonetics/graphemes.
"""

import re

# Indonesian number-to-words dictionary
ONES = ["", "satu", "dua", "tiga", "empat", "lima", "enam", "tujuh", "delapan", "sembilan"]
TENS = ["", "sepuluh", "dua puluh", "tiga puluh", "empat puluh", "lima puluh", 
        "enam puluh", "tujuh puluh", "delapan puluh", "sembilan puluh"]
TEENS = ["sepuluh", "sebelas", "dua belas", "tiga belas", "empat belas", "lima belas",
         "enam belas", "tujuh belas", "delapan belas", "sembilan belas"]

def number_to_indonesian(n: int) -> str:
    """Convert an integer (0 to 999,999) to Indonesian words."""
    if n == 0:
        return "nol"
    if n < 0:
        return "minus " + number_to_indonesian(-n)
    
    parts = []
    if n >= 1000:
        thousands = n // 1000
        n %= 1000
        if thousands == 1:
            parts.append("seribu")
        else:
            parts.append(number_to_indonesian(thousands) + " ribu")
            
    if n >= 100:
        hundreds = n // 100
        n %= 100
        if hundreds == 1:
            parts.append("seratus")
        else:
            parts.append(ONES[hundreds] + " ratus")
            
    if 10 <= n <= 19:
        parts.append(TEENS[n - 10])
    else:
        if n >= 20:
            tens = n // 10
            parts.append(TENS[tens])
            n %= 10
        if n > 0:
            parts.append(ONES[n])
            
    return " ".join(parts).strip()


def normalize_indonesian_text(text: str) -> str:
    """Normalize text: convert digits, lower case, clean invalid characters."""
    text = text.lower()
    
    # Replace numbers with spoken words
    def replace_num(match):
        val = int(match.group(0))
        if val <= 999999:
            return " " + number_to_indonesian(val) + " "
        return " " + " ".join([number_to_indonesian(int(d)) for d in match.group(0)]) + " "
        
    text = re.sub(r'\d+', replace_num, text)
    
    # Expand common abbreviations
    abbreviations = {
        r'\bdll\.?': 'dan lain-lain',
        r'\bdkk\.?': 'dan kawan-kawan',
        r'\btsb\.?': 'tersebut',
        r'\byg\b': 'yang',
        r'\bdgn\b': 'dengan',
        r'\bdr\b': 'dari',
        r'\butk\b': 'untuk',
        r'\bsdh\b': 'sudah',
        r'\btdk\b': 'tidak',
        r'\bkmrn\b': 'kemarin',
        r'\bbgt\b': 'banget',
        r'\bak\b': 'aku',
        r'\bsy\b': 'saya',
    }
    for pattern, expansion in abbreviations.items():
        text = re.sub(pattern, expansion, text)
        
    # Clean whitespace and symbols
    text = re.sub(r'[\r\n\t]+', ' ', text)
    text = re.sub(r'[^a-z0-9\s\.\,\!\?\-]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


# Vocabulary definition
PAD_TOKEN = "<pad>"
UNK_TOKEN = "<unk>"
BOS_TOKEN = "<bos>"
EOS_TOKEN = "<eos>"

SPECIAL_TOKENS = [PAD_TOKEN, UNK_TOKEN, BOS_TOKEN, EOS_TOKEN]
CHARACTERS = list("abcdefghijklmnopqrstuvwxyz .!?,-\'")

VOCAB = SPECIAL_TOKENS + CHARACTERS
CHAR_TO_ID = {c: i for i, c in enumerate(VOCAB)}
ID_TO_CHAR = {i: c for i, c in enumerate(VOCAB)}

PAD_ID = CHAR_TO_ID[PAD_TOKEN]
UNK_ID = CHAR_TO_ID[UNK_TOKEN]
BOS_ID = CHAR_TO_ID[BOS_TOKEN]
EOS_ID = CHAR_TO_ID[EOS_TOKEN]


class TextTokenizer:
    """Ultra-lightweight character-level tokenizer for speech synthesis."""
    
    def __init__(self):
        self.vocab = VOCAB
        self.vocab_size = len(VOCAB)
        self.pad_id = PAD_ID
        self.bos_id = BOS_ID
        self.eos_id = EOS_ID
        self.unk_id = UNK_ID
        
    def text_to_ids(self, text: str, add_special_tokens: bool = True) -> list[int]:
        normalized = normalize_indonesian_text(text)
        ids = []
        if add_special_tokens:
            ids.append(self.bos_id)
        for ch in normalized:
            ids.append(CHAR_TO_ID.get(ch, self.unk_id))
        if add_special_tokens:
            ids.append(self.eos_id)
        return ids
        
    def ids_to_text(self, ids: list[int]) -> str:
        tokens = []
        for i in ids:
            if i in [self.pad_id, self.bos_id, self.eos_id]:
                continue
            tokens.append(ID_TO_CHAR.get(i, ""))
        return "".join(tokens)
