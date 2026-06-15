from __future__ import annotations

from memslides.pipelines.revision import (
    _build_finalize_pending_inspect_followup,
    _parse_finalize_pending_inspect_slides,
)


def test_parse_finalize_pending_inspect_slides_classifies_next_actions() -> None:
    message = (
        "Cannot finalize modified slides yet. Pending slides: "
        "slide_05.html (never passed `inspect_slide` on the current HTML); "
        "slide_08.html (last inspect failed: overflow); "
        "slide_09.html (visual QA failed: diagram contract missing). "
        "Re-run inspect_slide before finalize."
    )

    parsed = _parse_finalize_pending_inspect_slides(message)

    assert parsed == [
        {
            "slide_name": "slide_05.html",
            "reason": "never passed `inspect_slide` on the current HTML",
            "next_action": "inspect_slide",
        },
        {
            "slide_name": "slide_08.html",
            "reason": "last inspect failed: overflow",
            "next_action": "repair_then_inspect",
        },
        {
            "slide_name": "slide_09.html",
            "reason": "visual QA failed: diagram contract missing",
            "next_action": "repair_then_inspect",
        },
    ]


def test_finalize_pending_followup_blocks_repeat_finalize_and_names_slides() -> None:
    followup = _build_finalize_pending_inspect_followup(
        [
            {
                "slide_name": "slide_05.html",
                "reason": "never passed `inspect_slide` on the current HTML",
                "next_action": "inspect_slide",
            },
            {
                "slide_name": "slide_08.html",
                "reason": "last inspect failed: overflow",
                "next_action": "repair_then_inspect",
            },
        ]
    )

    assert "Do not call `finalize` again yet" in followup
    assert "run `inspect_slide` on `slide_05.html`" in followup
    assert "repair `slide_08.html` if needed, then run `inspect_slide`" in followup
    assert "Only call `finalize` after" in followup
