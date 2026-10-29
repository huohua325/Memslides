from __future__ import annotations

from pathlib import Path

from memslides.tools import deck_runtime as deck_runtime


def _slide_html(body: str) -> str:
    return f"""<!doctype html>
<html>
<head><meta charset="utf-8"><style>.slide{{position:relative}}</style></head>
<body><div class="slide">{body}</div></body>
</html>"""


def test_generated_anchor_markup_does_not_create_new_html_contracts(tmp_path: Path) -> None:
    html = _slide_html(
        """
        <h1>实验设计与局限</h1>
        <p>Use <strong class="highlight">token</strong> level batching.</p>
        <div aria-hidden="false" class="memslides-anchor-layer" data-anchor-scope="current_deck_with_future_rewrites">
          <div aria-label="data-anchor-role=" class="memslides-anchored-element"
               data-anchor="bottom_right" data-anchor-role="marker"
               data-anchor-text="data-anchor-role=">data-anchor-role=</div>
          <div aria-label="highlight" class="memslides-anchored-element"
               data-anchor="bottom_right" data-anchor-role="marker"
               data-anchor-text="highlight">highlight</div>
          <div aria-label="token" class="memslides-anchored-element"
               data-anchor="bottom_right" data-anchor-role="marker"
               data-anchor-text="token">token</div>
        </div>
        """
    )
    slide_path = tmp_path / "slide_05.html"
    slide_path.write_text(html, encoding="utf-8")

    assert deck_runtime._anchored_contracts_from_html_text(html) == []
    assert (
        deck_runtime._active_anchored_element_contracts_for_path(
            slide_path,
            html,
            include_context=False,
        )
        == []
    )
    assert deck_runtime._validate_anchored_element_contracts(html, [], path=slide_path) == []


def test_temp_pref_marker_still_normalizes_to_one_visible_anchor() -> None:
    html = _slide_html("<p>TEMP-PREF-C1</p>")
    contracts = deck_runtime._anchored_contracts_from_html_text(html)

    assert [contract["text"] for contract in contracts] == ["TEMP-PREF-C1"]

    normalized, report = deck_runtime._normalize_anchored_elements_html(html, contracts=contracts)

    assert report["anchored_element_count"] == 1
    assert normalized.count('data-anchor-text="TEMP-PREF-C1"') == 1
    assert deck_runtime._validate_anchored_element_contracts(normalized, contracts) == []


def test_explicit_user_anchor_contract_remains_valid_and_idempotent() -> None:
    contracts = deck_runtime._anchored_contracts_from_texts('请在右下角放置标签 "DRAFT"')

    assert [contract["text"] for contract in contracts] == ["DRAFT"]

    first, first_report = deck_runtime._normalize_anchored_elements_html(
        _slide_html("<h1>Title</h1><p>Body content</p>"),
        contracts=contracts,
    )
    second, second_report = deck_runtime._normalize_anchored_elements_html(first, contracts=contracts)

    assert first_report["anchored_element_count"] == 1
    assert second_report["anchored_element_count"] == 1
    assert second == first
    assert deck_runtime._validate_anchored_element_contracts(second, contracts) == []
