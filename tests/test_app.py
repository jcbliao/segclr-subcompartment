from pathlib import Path
import pytest


def test_browser_editor_renders():
    pytest.importorskip("streamlit")
    from streamlit.testing.v1 import AppTest
    app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "app.py"))
    app.run(timeout=30)
    assert not app.exception
    assert app.title[0].value == "Explore and edit subcompartment labels"
    assert len(app.get("download_button")) == 2
