import pytest
import torch

from lerobot.policies.pi05.configuration_pi05 import PI05Config
from lerobot.policies.pi05.modeling_pi05 import (
    apply_action_prefix,
    build_prefix_step_mask,
    build_runtime_action_prefix,
    get_image_valid_mask,
    reduce_action_losses,
)
from lerobot.policies.rtc.configuration_rtc import RTCConfig
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


def test_reduce_action_losses_ignores_prefix_steps():
    losses = torch.tensor(
        [
            [[10.0, 10.0], [1.0, 2.0], [3.0, 4.0]],
        ]
    )
    action_step_is_pad = torch.tensor([[True, False, False]])

    loss, per_sample_loss, loss_per_dim = reduce_action_losses(losses, action_step_is_pad=action_step_is_pad)

    assert loss.item() == pytest.approx(2.5)
    assert per_sample_loss.tolist() == pytest.approx([2.5])
    assert loss_per_dim.tolist() == pytest.approx([2.0, 3.0])


def test_build_runtime_action_prefix_pads_to_prefix_length_and_action_dim():
    prev_chunk_left_over = torch.tensor(
        [
            [[1.0, 2.0], [3.0, 4.0]],
        ]
    )

    prefix, prefix_mask = build_runtime_action_prefix(prev_chunk_left_over, prefix_length=4, max_action_dim=3)

    assert prefix.shape == (1, 4, 3)
    assert prefix_mask.tolist() == [[True, True, False, False]]
    assert prefix[0, 0].tolist() == pytest.approx([1.0, 2.0, 0.0])
    assert prefix[0, 1].tolist() == pytest.approx([3.0, 4.0, 0.0])
    assert prefix[0, 2].tolist() == pytest.approx([0.0, 0.0, 0.0])


def test_apply_action_prefix_overwrites_only_valid_steps():
    x_t = torch.zeros((1, 4, 2))
    action_prefix = torch.tensor([[[1.0, 1.0], [2.0, 2.0], [9.0, 9.0], [9.0, 9.0]]])
    action_prefix_mask = torch.tensor([[True, True, False, False]])

    out = apply_action_prefix(x_t, action_prefix, action_prefix_mask)

    assert out[0, 0].tolist() == pytest.approx([1.0, 1.0])
    assert out[0, 1].tolist() == pytest.approx([2.0, 2.0])
    assert out[0, 2].tolist() == pytest.approx([0.0, 0.0])
    assert out[0, 3].tolist() == pytest.approx([0.0, 0.0])


def test_build_prefix_step_mask_marks_prefix_steps():
    mask = build_prefix_step_mask(batch_size=2, chunk_size=5, prefix_length=3, device=torch.device("cpu"))
    assert mask.tolist() == [
        [True, True, True, False, False],
        [True, True, True, False, False],
    ]


def test_pi05_config_validates_action_prefix_constraints():
    with pytest.raises(ValueError, match="action_prefix_length must satisfy"):
        PI05Config(
            chunk_size=20,
            n_action_steps=18,
            use_action_prefix_conditioning=True,
            action_prefix_length=4,
        )


def test_pi05_config_rejects_rtc_with_action_prefix_conditioning():
    with pytest.raises(ValueError, match="cannot be enabled together with rtc_config.enabled"):
        PI05Config(
            chunk_size=20,
            n_action_steps=10,
            use_action_prefix_conditioning=True,
            action_prefix_length=4,
            rtc_config=RTCConfig(enabled=True),
        )
