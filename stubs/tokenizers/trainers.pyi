class Trainer: ...

class BpeTrainer(Trainer):
    def __init__(
        self,
        vocab_size: int = ...,
        min_frequency: int = ...,
        show_progress: bool = ...,
        special_tokens: list[str] = ...,
        initial_alphabet: list[str] = ...,
    ) -> None: ...
