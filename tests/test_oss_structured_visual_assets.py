from __future__ import annotations

import json
from pathlib import Path

from memslides.tools import structured_visuals as sv


def test_table_asset_emits_svg_metadata_and_cjk_raster_hints(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(sv, "vlc", None)

    result = sv.render_table_asset_impl(
        rows=[
            {
                "指标": "长文本指标说明需要自动换行以避免表格溢出",
                "Score": "91.2",
                "Evidence": "Native text plus raster-safe SVG fallback.",
            },
            {"指标": "鲁棒性", "Score": "88.0", "Evidence": "Revision rounds completed."},
        ],
        columns=["指标", "Score", "Evidence"],
        caption="中文评估表",
        footnote="Generated for OSS structured visual parity.",
        output_mode="both",
        workspace=tmp_path,
    )

    assert result["kind"] == "table"
    assert result["contains_cjk"] is True
    assert result["visual_type"] == "table"
    assert result["preferred_pptx_export"] == "raster"
    assert result["recommended_width"] == 960
    assert result["recommended_height"] > 120
    assert result["layout"]["wrapped_cells"]["指标"] >= 1
    assert result["rendered_paths"]["svg"].endswith(".svg")
    assert result["primary_path"].endswith(".svg")
    assert result["warnings"]
    svg_text = Path(result["svg_path"]).read_text(encoding="utf-8")
    assert 'data-visual-kind="table"' in svg_text
    assert 'data-preferred-pptx-export="raster"' in svg_text
    meta = json.loads(Path(result["meta_path"]).read_text(encoding="utf-8"))
    assert meta["contains_cjk"] is True
    assert meta["preferred_pptx_export"] == "raster"


def test_flowchart_asset_supports_labeled_edges_and_layout_metadata(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(sv, "vlc", None)

    result = sv.render_flowchart_asset_impl(
        nodes=["输入", "Encode", "Score", "Review", "Output"],
        edges=[
            "输入 -- raw text -> Encode",
            "Encode -- attention -> Score",
            "Score -- inspect -> Review",
            "Review -- accepted -> Output",
        ],
        diagram_kind="pipeline",
        title="修订管线",
        output_format="both",
        workspace=tmp_path,
    )

    assert result["kind"] == "flowchart"
    assert result["edge_labels"] == ["raw text", "attention", "inspect", "accepted"]
    assert result["contains_cjk"] is True
    assert result["visual_type"] == "flowchart"
    assert result["preferred_pptx_export"] == "raster"
    assert result["recommended_width"] >= 760
    assert result["recommended_height"] >= 520
    assert result["primary_path"].endswith(".svg")
    assert result["layout"]["uses_edges"] is True
    assert result["layout"]["edge_routes"]
    assert "rank" in result["layout"]["node_bounds"]["Encode"]
    svg_text = Path(result["svg_path"]).read_text(encoding="utf-8")
    assert 'data-visual-kind="flowchart"' in svg_text
    assert "attention" in svg_text


def test_render_table_asset_accepts_aliases_and_routes_chart_like_calls(tmp_path: Path, monkeypatch) -> None:
    from memslides.tools import asset_services

    captured: dict[str, object] = {}

    def fake_chart_renderer(**kwargs):
        captured.update(kwargs)
        return {
            "kind": "chart",
            "warnings": [],
            "rendered_paths": {"svg": str(tmp_path / "chart.svg")},
            "primary_path": str(tmp_path / "chart.svg"),
        }

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(asset_services, "render_chart_asset_impl", fake_chart_renderer)

    result = asset_services.render_table_asset(
        rows=[{"epoch": "1", "accuracy": "0.91"}],
        chart_type="line",
        x_field="epoch",
        y_fields="accuracy",
        title="Accuracy Trend",
        output_format="both",
    )

    assert result["kind"] == "chart"
    assert result["warnings"] == [
        "render_table_asset received chart-like arguments and routed to render_chart_asset."
    ]
    assert captured["chart_type"] == "line"
    assert captured["x_field"] == "epoch"
    assert captured["y_fields"] == ["accuracy"]
    assert captured["title"] == "Accuracy Trend"
    assert captured["output_format"] == "both"
    assert captured["workspace"] == tmp_path


def test_render_table_asset_title_and_output_format_aliases(tmp_path: Path, monkeypatch) -> None:
    from memslides.tools import asset_services

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sv, "vlc", None)

    result = asset_services.render_table_asset(
        rows=[{"A": "甲", "B": "1"}],
        columns=["A", "B"],
        title="Alias Caption",
        output_format="svg",
    )

    assert result["kind"] == "table"
    assert result["title"] == "Alias Caption"
    assert result["requested_output_mode"] == "svg"
    assert result["svg_path"]
    assert result["contains_cjk"] is True
