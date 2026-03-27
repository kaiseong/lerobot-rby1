import pytest
import torch

from lerobot.policies.pi05.modeling_pi05 import get_image_valid_mask, reduce_action_losses
from lerobot.policies.pi05.processor_pi05 import Pi05PrepareStateTokenizerProcessorStep
from lerobot.processor import create_transition, TransitionKey
from lerobot.utils.constants import OBS_STATE


def test_pi05_prepare_state_tokenizer_skips_invalid_state_dims():
    state = torch.linspace(-1.0, 1.0, 22)
    state_is_pad = torch.tensor([True] * 6 + [False] * 8 + [True] * 8, dtype=torch.bool)
    transition = create_transition(
        observation={OBS_STATE: state, f"{OBS_STATE}_dim_is_pad": state_is_pad},
        complementary_data={"task": ["pick_cube"]},
    )

    step = Pi05PrepareStateTokenizerProcessorStep()
    processed = step(transition)
    prompt = processed[TransitionKey.COMPLEMENTARY_DATA]["task"][0]
    state_tokens = prompt.split("State: ", 1)[1].split(";\nAction:", 1)[0].split()
    assert len(state_tokens) == 8


def test_get_image_valid_mask_uses_per_sample_missing_flags():
    batch = {"observation.images.left_is_missing": torch.tensor([True, False])}
    mask = get_image_valid_mask(batch, "observation.images.left", batch_size=2, device=torch.device("cpu"))
    assert mask.tolist() == [False, True]


def test_reduce_action_losses_ignores_invalid_action_dims():
    losses = torch.tensor([[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]])
    action_dim_is_pad = torch.tensor([[False, True, False]])

    loss, per_sample_loss, loss_per_dim = reduce_action_losses(losses, action_dim_is_pad=action_dim_is_pad)

    assert loss.item() == pytest.approx(3.5)
    assert per_sample_loss.tolist() == pytest.approx([3.5])
    assert loss_per_dim.tolist() == pytest.approx([2.5, 0.0, 4.5])
