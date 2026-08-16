class CharTokenizer:
    def __init__(self, text: str) -> None:
        chars = sorted(list(set(text)))

        self._stoi: dict[str, int] = {
            char: token_id
            for token_id, char in enumerate(chars)
        }

        self._itos: dict[int, str] = {
            token_id: char
            for char, token_id in self._stoi.items()
        }

    @property
    def vocab_size(self) -> int:
        return len(self._stoi)

    def encode(self, text: str) -> list[int]:
        token_ids: list[int] = []

        for token in text:
            if token not in self._stoi:
                raise ValueError(f"Token {token} not in vocabulary")

            token_ids.append(self._stoi[token])

        return token_ids

    def decode(self, token_ids: list[int]) -> str:
        text: list[str] = []

        for token_id in token_ids:
            if token_id not in self._itos:
                raise ValueError(f"Token {token_id} not in vocabulary")

            text.append(self._itos[token_id])

        return "".join(text)