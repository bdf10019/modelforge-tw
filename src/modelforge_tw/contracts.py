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
EvaluationSchemaVersion = Literal["1.0", "1.1", "1.2", "1.3", "1.4", "1.5"]
EvaluationContractVersion = Literal[
    "release-metrics-v2",
    "release-metrics-v3",
    "release-metrics-v4",
    "release-metrics-v5",
]
