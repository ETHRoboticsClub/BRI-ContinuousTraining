import math

import torch

# EMG channel indices in order around the armband. The paper (Colot et al., Section II-A) has the 8 electrodes evenly
# spaced around the forearm. The order is from the data: in 18 of 20 subjects, EMG 1 -> 2 -> ... -> 8 -> 1 is the
# ring of all 2520 that maximizes the summed |correlation| between neighbours (5th and 12th in the other two).
RING_ORDER = tuple(range(8))


def rotate(x: torch.Tensor, r: int, order=RING_ORDER) -> torch.Tensor:
    """Move the channel at ring position i of x [8, T] to ring position i + r (mod 8)."""
    idx = torch.tensor(order)
    out = torch.empty_like(x)
    out[idx] = x[idx].roll(r, 0)
    return out


def shift(x: torch.Tensor, u: float, order=RING_ORDER) -> torch.Tensor:
    """Rotate x [8, T] by a fractional u electrodes around the ring, linearly interpolating between neighbours."""
    r = math.floor(u)
    f = u - r
    return (1 - f) * rotate(x, r, order) + f * rotate(x, r + 1, order)


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
        if self.rotate:
            x = rotate(x, int(torch.randint(-self.rotate, self.rotate + 1, (), generator=self.generator)))
        if self.shift:
            x = shift(x, self.shift * (2 * torch.rand((), generator=self.generator).item() - 1))
        if self.gain:
            x = channel_gain(x, self.gain * torch.randn(x.shape[0], generator=self.generator))
        return x
