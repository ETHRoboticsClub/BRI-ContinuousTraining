import torch


def channel_gain(x: torch.Tensor, log_gain: torch.Tensor) -> torch.Tensor:
    """Scale each channel of x [C, T] by exp(log_gain[c])."""
    return x * log_gain.exp()[:, None]


class Augment:
    """Random augmentation of one training example [8, T], drawn anew on every call.

    `gain` is the std of the log of an independent gain per channel. `rotate` and `shift` are the largest
    rotation (whole electrodes) and fractional shift (electrodes) around the armband's ring. 0 turns each off.
    The draws come from the object's own generator, so the global torch RNG (shuffling, dropout) is untouched.
    """

    def __init__(self, gain: float = 0.0, rotate: int = 0, shift: float = 0.0) -> None:
        if gain < 0 or rotate < 0 or shift < 0:
            raise ValueError(f"augmentation strengths must be >= 0, got gain={gain}, rotate={rotate}, shift={shift}")
        self.gain, self.rotate, self.shift = gain, rotate, shift
        self.generator = torch.Generator()

    @staticmethod
    def from_params(params: dict) -> "Augment | None":
        """Augment from the train params' aug_* keys, or None when every strength is 0."""
        aug = Augment(params["aug_gain"], params["aug_rotate"], params["aug_shift"])
        return aug if aug.gain or aug.rotate or aug.shift else None

    def manual_seed(self, seed: int) -> None:
        self.generator.manual_seed(seed)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if self.gain:
            x = channel_gain(x, self.gain * torch.randn(x.shape[0], generator=self.generator))
        return x
