from modelforge_tw.prompts import OPTIMIZED_PROMPT_V2_VERSION, messages_for_version
from modelforge_tw.schemas import TriageRequest


def test_optimized_v2_contains_explicit_severity_rubric():
    messages = messages_for_version(
        TriageRequest(incident_text="馬達重載時反覆偏熱，卸載後恢復。"),
        OPTIMIZED_PROMPT_V2_VERSION,
    )
    system = messages[0]["content"]
    assert "medium 是反覆或間歇" in system
    assert "警告字樣本身不等於 high" in system
    assert "單次火花仍是 critical" in system
