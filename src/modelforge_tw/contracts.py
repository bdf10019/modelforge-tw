from typing import Literal

OptimizedPromptVersion = Literal[
    "optimized-v1",
    "optimized-v2",
    "optimized-v3",
    "optimized-v4",
    "optimized-v5",
    "optimized-v6",
]
PromptVersion = Literal["minimal-v1"] | OptimizedPromptVersion
SafetyPolicyVersion = Literal["risk-precedence-v2", "risk-precedence-v3"]
