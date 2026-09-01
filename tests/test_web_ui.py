from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_web_ui_localizes_contract_values_without_hiding_raw_metadata():
    script = (PROJECT_ROOT / "src" / "modelforge_tw" / "web" / "app.js").read_text(encoding="utf-8")
    for label in (
        "機械異常",
        "電氣異常",
        "溫度異常",
        "壓力異常",
        "通訊異常",
        "製程異常",
        "事件類型待確認",
        "危急",
        "隔離設備並升級處理",
        "停止相關作業並升級處理",
        "補充量測與現場證據",
    ):
        assert label in script
    assert "output_contract" in script
    assert "event_type: output.event_type" in script
    assert "safe_actions: output.safe_actions" in script


def test_web_ui_renders_untrusted_values_with_text_content_only():
    script = (PROJECT_ROOT / "src" / "modelforge_tw" / "web" / "app.js").read_text(encoding="utf-8")
    assert ".innerHTML" not in script
    assert "item.textContent = value" in script
    assert 'document.querySelector("#summary").textContent' in script


def test_web_ui_assets_share_an_explicit_cache_version():
    html = (PROJECT_ROOT / "src" / "modelforge_tw" / "web" / "index.html").read_text(
        encoding="utf-8"
    )
    assert "styles.css?v=0.1.1" in html
    assert "app.js?v=0.1.1" in html
    assert "favicon.svg?v=0.1.1" in html
    assert (PROJECT_ROOT / "src" / "modelforge_tw" / "web" / "favicon.svg").is_file()
