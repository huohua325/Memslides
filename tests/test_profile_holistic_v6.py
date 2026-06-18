from __future__ import annotations

import json

from memslides.experiment.profile_holistic_v6 import (
    rebuild_profile_holistic_v6_summary,
    validate_profile_holistic_v6_summary,
)
from memslides.experiment.profile_table import collect_profile_results


def test_profile_holistic_v6_rebuild_and_table_parse(tmp_path) -> None:
    eval_dir = tmp_path / "profile_eval_v6_top3_gpt5"
    eval_dir.mkdir()
    dimension_records = [
        _dimension("analyst", "probe_a", "uid-a", "role_decision_fit", 8, 4, "primary_better"),
        _dimension("analyst", "probe_a", "uid-a", "narrative_priority", 7, 5, "primary_better"),
        _dimension("analyst", "probe_a", "uid-a", "visual_manifestation", 6, 3, "primary_better"),
        _dimension("analyst", "probe_a", "uid-a", "action_support", 5, 6, "control_better"),
        _dimension("teacher", "probe_b", "uid-b", "role_decision_fit", 4, 6, "control_better"),
        _dimension("teacher", "probe_b", "uid-b", "narrative_priority", 7, 4, "primary_better"),
        _dimension("teacher", "probe_b", "uid-b", "visual_manifestation", 8, 4, "primary_better"),
        _dimension("teacher", "probe_b", "uid-b", "action_support", 9, 5, "primary_better"),
    ]
    overall_records = [
        _overall("analyst", "probe_a", "uid-a", 8, 5, "primary_better"),
        _overall("teacher", "probe_b", "uid-b", 8, 5, "primary_better"),
    ]
    (eval_dir / "profile_holistic_v6_visual_dimension_records.json").write_text(
        json.dumps({"records": dimension_records}, ensure_ascii=False),
        encoding="utf-8",
    )
    (eval_dir / "profile_holistic_v6_visual_overall_records.json").write_text(
        json.dumps({"records": overall_records}, ensure_ascii=False),
        encoding="utf-8",
    )
    (eval_dir / "profile_table_metadata.json").write_text(
        json.dumps({"framework": "MemSlides (Ours)", "model": "GPT-5"}, ensure_ascii=False),
        encoding="utf-8",
    )

    report = rebuild_profile_holistic_v6_summary(eval_dir)

    assert report["probe_count"] == 2
    assert report["overall"]["strict_positive_persona_rate"] == 1.0
    assert report["overall"]["mean_role_decision_fit_lift"] == 1.0
    validation = validate_profile_holistic_v6_summary(eval_dir)
    assert validation["ok"], validation["issues"]

    results, warnings = collect_profile_results(eval_dir)
    assert not warnings
    assert len(results) == 1
    assert results[0].model == "GPT-5"
    assert results[0].scores["Role Decision"] == 6.0
    assert results[0].scores["Narrative"] == 7.0
    assert results[0].scores["Visual"] == 7.0
    assert results[0].scores["Action Support"] == 7.0


def test_profile_holistic_v6_validate_summary_shape_without_detail_records(tmp_path) -> None:
    eval_dir = tmp_path / "profile_eval_v6"
    eval_dir.mkdir()
    (eval_dir / "profile_holistic_v6_probe_summary.json").write_text(
        json.dumps(
            {
                "dimension_rows": [
                    {"dimension_id": "role_decision_fit", "primary_avg_score": 1},
                    {"dimension_id": "narrative_priority", "primary_avg_score": 1},
                    {"dimension_id": "visual_manifestation", "primary_avg_score": 1},
                    {"dimension_id": "action_support", "primary_avg_score": 1},
                ],
                "overall": {
                    "paired_probe_count": 1,
                    "strict_positive_persona_rate": 1,
                    "mean_overall_lift": 1,
                },
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    validation = validate_profile_holistic_v6_summary(eval_dir)

    assert validation["ok"], validation["issues"]
    assert validation["detail_records_available"] is False


def _dimension(persona: str, probe_id: str, uid: str, dimension_id: str, primary: float, control: float, winner: str) -> dict:
    return {
        "persona": persona,
        "probe_id": probe_id,
        "probe_uid": uid,
        "dimension_id": dimension_id,
        "dimension_label": dimension_id,
        "primary_score": primary,
        "control_score": control,
        "score_lift": primary - control,
        "winner": winner,
        "reason": "synthetic",
    }


def _overall(persona: str, probe_id: str, uid: str, primary: float, control: float, winner: str) -> dict:
    return {
        "persona": persona,
        "probe_id": probe_id,
        "probe_uid": uid,
        "primary_score": primary,
        "control_score": control,
        "score_lift": primary - control,
        "winner": winner,
        "profile_effect": "clear_positive",
        "vote_consistency_rate": 1.0,
        "order_consistent": True,
    }
