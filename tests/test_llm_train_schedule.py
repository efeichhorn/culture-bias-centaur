from __future__ import annotations

from src.llm_train import LengthGroupedBatchSampler, training_schedule


def test_max_steps_overrides_epoch_length_and_can_repeat_the_corpus() -> None:
    assert training_schedule(
        dataloader_batches=480,
        gradient_accumulation_steps=4,
        num_train_epochs=1.0,
        max_steps=360,
    ) == (120, 360, 3)


def test_fixed_steps_span_partial_passes_for_larger_corpora() -> None:
    assert training_schedule(960, 4, 1.0, 360) == (240, 360, 2)
    assert training_schedule(1440, 4, 1.0, 360) == (360, 360, 1)
    assert training_schedule(2880, 4, 1.0, 360) == (720, 360, 1)


def test_epoch_schedule_still_supports_fractional_epochs() -> None:
    assert training_schedule(480, 4, 1.5, -1) == (120, 180, 2)


def test_length_grouped_sampler_reshuffles_repeated_passes_reproducibly() -> None:
    first = LengthGroupedBatchSampler(list(range(64)), batch_size=4, seed=17)
    second = LengthGroupedBatchSampler(list(range(64)), batch_size=4, seed=17)
    first_epoch = list(first)
    second_epoch = list(first)
    assert first_epoch != second_epoch
    assert first_epoch == list(second)
    assert second_epoch == list(second)
