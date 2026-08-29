from modelforge_tw.prompts import (
    OPTIMIZED_PROMPT_V2_VERSION,
    OPTIMIZED_PROMPT_V3_VERSION,
    OPTIMIZED_PROMPT_V4_VERSION,
    OPTIMIZED_PROMPT_V5_VERSION,
    messages_for_version,
)
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


def test_optimized_v3_contains_external_calibration_rubric_and_output_limits():
    messages = messages_for_version(
        TriageRequest(incident_text="刀具磨耗接近警戒值，但失效旗標為 0。"),
        OPTIMIZED_PROMPT_V3_VERSION,
    )
    system = messages[0]["content"]
    assert "刀具是 mechanical" in system
    assert "過應力或公差是 process" in system
    assert "失效旗標觸發或讀值越過警戒值是 high" in system
    assert "不得只因『失效』二字升為 critical" in system
    assert "evidence_spans 最多 2 項" in system
    assert "safe_actions 只選最主要的 1 項" in system


def test_optimized_v4_prioritizes_insufficient_evidence_and_abnormal_signals():
    messages = messages_for_version(
        TriageRequest(incident_text="只有溫度與磨耗讀值，缺少基準、門檻、趨勢或異常現象。"),
        OPTIMIZED_PROMPT_V4_VERSION,
    )
    system = messages[0]["content"]
    assert "第一步先判斷證據是否足夠" in system
    assert "就必須輸出 unknown" in system
    assert "status 為 normal 或 unknown 的通用訊號名稱不得改變事件類型" in system


def test_optimized_v5_requires_unknown_when_event_type_is_explicitly_indeterminate():
    messages = messages_for_version(
        TriageRequest(incident_text="沒有基準、門檻與趨勢，無法判定事件類型。"),
        OPTIMIZED_PROMPT_V5_VERSION,
    )
    system = messages[0]["content"]
    assert "event_type 必須是 unknown" in system
    assert "severity 也必須是 unknown" in system
    assert "不得自行套用未提供的數值門檻" in system
