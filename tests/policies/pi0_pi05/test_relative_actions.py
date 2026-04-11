import types

import torch

from lerobot.policies.pi05.modeling_pi05 import PI05Policy
from lerobot.processor import TransitionKey, create_transition
from lerobot.processor.relative_action_processor import (
    AbsoluteActionsProcessorStep,
    RelativeActionsProcessorStep,
    to_absolute_actions,
    to_relative_actions,
)
from lerobot.utils.constants import OBS_STATE


def test_relative_action_roundtrip_for_chunk_actions():
    actions = torch.randn(2, 5, 4)
    state = torch.randn(2, 4)
    mask = [True, True, True, True]

    relative = to_relative_actions(actions, state, mask)
    recovered = to_absolute_actions(relative, state, mask)

    torch.testing.assert_close(recovered, actions)


def test_relative_action_exclude_joints_masks_by_partial_name():
    step = RelativeActionsProcessorStep(
        enabled=True,
        exclude_joints=["gripper"],
        action_names=[
            "right_joint_1.pos",
            "right_gripper.pos",
            "left_joint_1.pos",
            "left_gripper.pos",
        ],
    )
    assert step._build_mask(4) == [True, False, True, False]


def test_relative_and_absolute_processors_roundtrip_and_cache_state():
    relative_step = RelativeActionsProcessorStep(enabled=True)
    absolute_step = AbsoluteActionsProcessorStep(enabled=True, relative_step=relative_step)

    state = torch.tensor([[1.0, 2.0, 3.0]])
    actions = torch.tensor([[[2.0, 3.0, 4.0], [3.0, 4.0, 5.0]]])
    transition = create_transition(observation={OBS_STATE: state}, action=actions)

    relative_transition = relative_step(transition)
    assert relative_step._last_state is not None

    recovered_transition = absolute_step(relative_transition)
    torch.testing.assert_close(recovered_transition[TransitionKey.ACTION], actions)


def test_relative_processor_caches_state_during_inference_without_actions():
    relative_step = RelativeActionsProcessorStep(enabled=True)
    state = torch.tensor([[1.0, 2.0, 3.0]])
    transition = create_transition(observation={OBS_STATE: state})

    processed = relative_step(transition)

    assert TransitionKey.ACTION not in processed
    torch.testing.assert_close(relative_step._last_state, state)


def test_pi05_local_prefix_delay_estimation_uses_env_dt_and_clips():
    dummy_policy = types.SimpleNamespace(
        config=types.SimpleNamespace(use_action_prefix_conditioning=True, action_prefix_length=4),
        _estimated_env_dt=0.05,
    )

    assert PI05Policy._estimate_local_prefix_delay_steps(dummy_policy, 0.0) == 0
    assert PI05Policy._estimate_local_prefix_delay_steps(dummy_policy, 0.09) == 1
    assert PI05Policy._estimate_local_prefix_delay_steps(dummy_policy, 0.24) == 4

    dummy_policy._estimated_env_dt = None
    assert PI05Policy._estimate_local_prefix_delay_steps(dummy_policy, 0.24) == 0
