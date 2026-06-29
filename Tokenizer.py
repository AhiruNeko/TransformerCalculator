import re

CHARS = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
         '+', '-', '*', '/', '^', 'sqrt', '=',
         '(', ')', '[NAN]', '[BOS]', '[EOS]', '[PAD]', '[Q]', '[A]']


class Tokenizer:
    def __init__(self):
        self.chars = CHARS
        self.stoi = {ch: i for i, ch in enumerate(self.chars)}
        self.itos = {i: ch for i, ch in enumerate(self.chars)}
        sorted_chars = sorted(self.chars, key=len, reverse=True)
        escaped_chars = [re.escape(ch) for ch in sorted_chars]
        self.pattern = re.compile("(" + "|".join(escaped_chars) + ")")

    def tokenize(self, text):
        return self.pattern.findall(text)

    def encode(self, text, add_special_tokens=True, max_len=None):
        if '=' in text and not text.endswith('='):
            question, answer_part = text.split('=', 1)
            answer = "=" + answer_part

            q_tokens = self.tokenize(question)
            a_tokens = self.tokenize(answer)

            q_ids = [self.stoi[ch] for ch in q_tokens if ch in self.stoi]
            a_ids = [self.stoi[ch] for ch in a_tokens if ch in self.stoi]

            ids = [self.stoi['[Q]']] + q_ids + [self.stoi['[A]']] + a_ids
        else:
            tokens = self.tokenize(text)
            ids = [self.stoi[ch] for ch in tokens if ch in self.stoi]
        if add_special_tokens:
            ids = [self.stoi['[BOS]']] + ids + [self.stoi['[EOS]']]
        if max_len is not None:
            if len(ids) < max_len:
                ids += [self.stoi['[PAD]']] * (max_len - len(ids))
            else:
                ids = ids[:max_len]

        return ids

    def decode(self, ids):
        return "".join(
            [self.itos[i] for i in ids if i not in [self.stoi['[PAD]'], self.stoi['[BOS]'], self.stoi['[EOS]']]])

    def __call__(self, text, **kwargs):
        return self.encode(text, **kwargs)
