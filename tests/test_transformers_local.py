import pytest

from modelforge_tw.engines.transformers_local import _validate_context_budget


def test_transformers_context_budget_accepts_exact_fit():
    _validate_context_budget(
        prompt_tokens=1856,
        max_new_tokens=192,
        context_window_tokens=2048,
    )


@pytest.mark.parametrize(
    ("prompt_tokens", "max_new_tokens", "context_window_tokens"),
    [
        (1857, 192, 2048),
        (1, 2048, 2048),
        (-1, 192, 2048),
        (1, 0, 2048),
        (1, 192, 0),
    ],
)
def test_transformers_context_budget_fails_closed(
    prompt_tokens, max_new_tokens, context_window_tokens
):
    with pytest.raises(ValueError, match="token budgets|context window"):
        _validate_context_budget(
            prompt_tokens=prompt_tokens,
            max_new_tokens=max_new_tokens,
            context_window_tokens=context_window_tokens,
        )
