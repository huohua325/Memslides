from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class PageAssetBinding:
    page: int
    title: str
    visual_requirement: str = "none"  # required | optional | none
    bound_asset_kind: str = "none"  # figure | table | formula | chart | none
    bound_asset_path: str = ""
    binding_reason: str = ""
    formula_snippet: str = ""
    candidate_assets: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "page": self.page,
            "title": self.title,
            "visual_requirement": self.visual_requirement,
            "bound_asset_kind": self.bound_asset_kind,
            "bound_asset_path": self.bound_asset_path,
            "binding_reason": self.binding_reason,
            "formula_snippet": self.formula_snippet,
            "candidate_assets": self.candidate_assets,
        }


def _load_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _page_formula_snippet(text: str) -> str:
    matches = re.findall(
        r"(softmax\s*\([^\n]{0,160}|QK\^?T[^\n]{0,120}|√d_?k|PE\(pos,[^\n]{0,120}|\$[^\$]{3,160}\$)",
        text,
        flags=re.IGNORECASE,
    )
    if not matches:
        return ""
    return str(matches[0]).strip()


def _caption_mentions_table_number(caption: str, number: int) -> bool:
    text = str(caption or "").lower()
    return f"table {number}" in text or f"table{number}" in text


def _caption_mentions_figure_number(caption: str, number: int) -> bool:
    text = str(caption or "").lower()
    return f"figure {number}" in text or f"figure{number}" in text or f"fig. {number}" in text


def _infer_asset_kind(asset: dict[str, Any]) -> str:
    kind = str(asset.get("kind", "") or asset.get("category", "") or "").strip().lower()
    if kind in {"figure", "table", "chart", "formula"}:
        return kind

    text = " ".join(
        str(asset.get(field, "") or "")
        for field in ("caption", "label", "filename", "path")
    ).lower()
    if "table" in text or re.search(r"\bp\d+_table_", text):
        return "table"
    if "chart" in text:
        return "chart"
    if "figure" in text or "architecture" in text or re.search(r"\bp\d+_figure_", text):
        return "figure"
    return "figure"


class PageAssetPlanner:
    """Bind real workspace assets to each page before layout selection."""

    def __init__(self, workspace: Path | str):
        self.workspace = Path(workspace)

    def build(
        self,
        *,
        manuscript_text: str,
        page_briefs: list[Any],
        asset_manifest: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        asset_manifest = asset_manifest or {}
        figure_manifest = self._load_figure_manifest()
        all_assets = self._collect_assets(asset_manifest, figure_manifest)

        bindings: list[PageAssetBinding] = []
        for brief in page_briefs:
            binding = self._bind_page(
                brief=brief,
                raw_text=str(getattr(brief, "raw_markdown", "") or ""),
                all_assets=all_assets,
            )
            bindings.append(binding)

        return {
            "workspace": str(self.workspace),
            "bindings": [binding.to_dict() for binding in bindings],
        }

    def dump(self, payload: dict[str, Any], output_path: Path | str) -> Path:
        path = Path(output_path)
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        return path

    def _load_figure_manifest(self) -> dict[str, Any]:
        candidates = list(self.workspace.rglob("figure_manifest.json"))
        if not candidates:
            return {}
        # Prefer converted attachment directories within this workspace.
        candidates.sort(key=lambda path: (0 if "converted" in str(path.parent) else 1, len(str(path))))
        return _load_json(candidates[0])

    def _collect_converted_images(self) -> list[dict[str, Any]]:
        assets: list[dict[str, Any]] = []
        seen: set[str] = set()
        for images_dir in sorted(self.workspace.rglob("images")):
            if "converted" not in str(images_dir):
                continue
            for image_path in sorted(images_dir.iterdir()):
                if not image_path.is_file():
                    continue
                if image_path.suffix.lower() not in {".png", ".jpg", ".jpeg", ".webp"}:
                    continue
                resolved = str(image_path.resolve())
                if resolved in seen:
                    continue
                seen.add(resolved)
                assets.append(
                    {
                        "path": resolved,
                        "filename": image_path.name,
                        "caption": "",
                        "kind": _infer_asset_kind({"path": resolved, "filename": image_path.name}),
                        "category": _infer_asset_kind({"path": resolved, "filename": image_path.name}),
                        "exists": True,
                        "within_workspace": True,
                    }
                )
        return assets

    def _collect_assets(
        self,
        asset_manifest: dict[str, Any],
        figure_manifest: dict[str, Any],
    ) -> list[dict[str, Any]]:
        assets: list[dict[str, Any]] = []
        seen_paths: set[str] = set()

        def _append(item: dict[str, Any]) -> None:
            payload = dict(item)
            path = str(payload.get("path", "") or "").strip()
            if path:
                payload["path"] = str(Path(path).expanduser().resolve())
            resolved_path = str(payload.get("path", "") or "")
            if resolved_path and resolved_path in seen_paths:
                return
            payload["caption"] = str(payload.get("caption", "") or "")
            payload["filename"] = str(payload.get("filename", "") or (Path(resolved_path).name if resolved_path else ""))
            inferred_kind = _infer_asset_kind(payload)
            payload["kind"] = inferred_kind
            payload["category"] = inferred_kind
            payload["exists"] = bool(payload.get("exists", True if not resolved_path else Path(resolved_path).exists()))
            payload["within_workspace"] = bool(
                payload.get("within_workspace", bool(resolved_path) and (self.workspace == Path(resolved_path) or self.workspace in Path(resolved_path).parents))
            )
            if resolved_path:
                seen_paths.add(resolved_path)
            assets.append(payload)

        for item in figure_manifest.get("figures", []) or []:
            _append(item)
        for item in asset_manifest.get("assets", []) or []:
            _append(item)
        for item in self._collect_converted_images():
            _append(item)

        assets.sort(
            key=lambda item: (
                0 if item.get("within_workspace") else 1,
                0 if item.get("generated_by_tool") and item.get("kind") in {"chart", "table"} else 1,
                0 if item.get("kind") in {"chart", "table"} else 1 if item.get("kind") == "figure" else 2,
                str(item.get("filename", "")),
            )
        )
        return assets

    def _bind_page(
        self,
        *,
        brief: Any,
        raw_text: str,
        all_assets: list[dict[str, Any]],
    ) -> PageAssetBinding:
        title = str(getattr(brief, "title", "") or "")
        body = str(getattr(brief, "body", "") or "")
        content_shape = str(getattr(brief, "content_shape", "") or "")
        page_purpose = str(getattr(brief, "page_purpose", "content") or "content")
        text = f"{title}\n{body}".lower()
        candidates = self._rank_candidates(text, all_assets)
        formula = _page_formula_snippet(raw_text)

        if page_purpose in {"opening", "ending", "table_of_contents", "section_divider"}:
            return PageAssetBinding(
                page=int(getattr(brief, "page_index", 0) or 0),
                title=title,
                visual_requirement="none",
                bound_asset_kind="none",
                binding_reason=(
                    f"{page_purpose} page should use the template reference rhythm "
                    "without forcing a real visual asset"
                ),
                formula_snippet=formula,
                candidate_assets=[str(item.get("path", "") or item.get("filename", "")) for item in candidates[:3]],
            )

        explicit_asset = self._first_explicit_asset(raw_text, all_assets)
        if explicit_asset:
            return self._binding_from_asset(
                brief,
                explicit_asset,
                "required",
                "manuscript explicitly references this real attachment-derived asset",
                formula,
            )

        referenced_asset = self._resolve_referenced_asset(text, candidates)
        if referenced_asset:
            return self._binding_from_asset(
                brief,
                referenced_asset,
                "required",
                "page text references a specific figure/table from the attachment",
                formula,
            )

        if any(token in text for token in ("架构", "architecture", "mechanism", "attention")):
            figure = self._first_of_kind(candidates, {"figure"})
            if figure:
                return self._binding_from_asset(brief, figure, "required", "page discusses architecture/mechanism and should show a real figure", formula)
        if any(
            token in text
            for token in (
                "结果",
                "bleu",
                "result",
                "results",
                "table",
                "chart",
                "trend",
                "accuracy",
                "指标",
                "消融",
                "训练成本",
            )
        ):
            table = self._first_of_kind(candidates, {"table", "chart"})
            if table:
                return self._binding_from_asset(brief, table, "required", "page discusses evaluation/results and should show a real table/chart", formula)
        if formula and any(token in text for token in ("公式", "attention", "位置编码", "encoding")):
            return PageAssetBinding(
                page=int(getattr(brief, "page_index", 0) or 0),
                title=title,
                visual_requirement="required",
                bound_asset_kind="formula",
                binding_reason="page discusses a core equation/mechanism and should keep a visible formula anchor",
                formula_snippet=formula,
                candidate_assets=[str(item.get("path", "") or item.get("filename", "")) for item in candidates[:3]],
            )
        if getattr(brief, "image_count", 0) or getattr(brief, "table_count", 0):
            asset = candidates[0] if candidates else {}
            if asset:
                return self._binding_from_asset(brief, asset, "optional", "page already contains visual hints", formula)
        return PageAssetBinding(
            page=int(getattr(brief, "page_index", 0) or 0),
            title=title,
            visual_requirement="none",
            bound_asset_kind="formula" if formula else "none",
            binding_reason="text-first page",
            formula_snippet=formula,
            candidate_assets=[str(item.get("path", "") or item.get("filename", "")) for item in candidates[:3]],
        )

    def _binding_from_asset(
        self,
        brief: Any,
        asset: dict[str, Any],
        requirement: str,
        reason: str,
        formula: str,
    ) -> PageAssetBinding:
        return PageAssetBinding(
            page=int(getattr(brief, "page_index", 0) or 0),
            title=str(getattr(brief, "title", "") or ""),
            visual_requirement=requirement,
            bound_asset_kind=str(asset.get("kind", asset.get("category", "figure")) or "figure"),
            bound_asset_path=str(asset.get("path", "") or ""),
            binding_reason=reason,
            formula_snippet=formula,
            candidate_assets=[str(asset.get("path", "") or asset.get("filename", ""))],
        )

    def _rank_candidates(self, text: str, all_assets: list[dict[str, Any]]) -> list[dict[str, Any]]:
        scored: list[tuple[int, dict[str, Any]]] = []
        for asset in all_assets:
            caption = str(asset.get("caption", "") or "").lower()
            label = str(asset.get("label", "") or "").lower()
            kind = _infer_asset_kind(asset)
            score = 0
            if not bool(asset.get("exists", True)):
                score -= 10
            if bool(asset.get("within_workspace")):
                score += 3
            if bool(asset.get("generated_by_tool")):
                score += 3
            if "architecture" in caption and ("架构" in text or "architecture" in text):
                score += 5
            if "transformer - model architecture" in caption or "transformer model architecture" in caption:
                score += 6
            if "attention" in caption and "attention" in text:
                score += 4
            if "bleu" in caption and ("bleu" in text or "结果" in text or "results" in text):
                score += 5
            if "variations" in caption and ("消融" in text or "variation" in text):
                score += 4
            if "training cost" in caption and ("训练成本" in text or "cost" in text):
                score += 4
            if _caption_mentions_table_number(caption, 2) and any(
                token in text for token in ("wmt14", "bleu", "translation results", "machine translation", "training cost", "efficiency")
            ):
                score += 10
            if _caption_mentions_table_number(caption, 3) and any(
                token in text for token in ("ablation", "ablations", "variation", "variations", "multi-head", "dropout")
            ):
                score += 10
            if _caption_mentions_table_number(caption, 4) and any(
                token in text for token in ("transfer", "parsing", "wsj", "f1")
            ):
                score += 10
            if _caption_mentions_figure_number(caption, 2) and any(
                token in text for token in ("attention", "multi-head", "scaled dot-product")
            ):
                score += 8
            if "position" in caption and ("位置编码" in text or "encoding" in text):
                score += 3
            if label and label in text:
                score += 2
            if kind == "table" and any(token in text for token in ("结果", "表", "bleu", "消融", "成本", "results", "training", "cost")):
                score += 2
            if kind == "chart" and any(token in text for token in ("图表", "曲线", "折线图", "柱状图", "趋势", "chart", "trend", "compare", "comparison")):
                score += 4
            if kind == "chart" and any(token in text for token in ("结果", "bleu", "accuracy", "acc", "指标", "results", "performance")):
                score += 4
            if kind == "table" and any(token in text for token in ("wmt14", "ablation", "ablations", "transfer", "parsing", "wsj")):
                score += 4
            if kind == "figure" and any(token in text for token in ("架构", "attention", "机制", "architecture", "mechanism")):
                score += 2
            if kind == "figure" and any(token in text for token in ("architecture", "attention", "mechanism", "encoder", "decoder")):
                score += 2
            if kind == "table" and any(token in text for token in ("architecture", "attention", "mechanism", "encoder", "decoder")):
                score -= 4
            if kind == "chart" and any(token in text for token in ("architecture", "attention", "mechanism", "encoder", "decoder")):
                score -= 3
            if kind == "figure" and any(token in text for token in ("bleu", "results", "training cost", "cost")):
                score -= 1
            scored.append((score, asset))
        scored.sort(key=lambda item: (item[0], int(item[1].get("width", 0) or 0) * int(item[1].get("height", 0) or 0)), reverse=True)
        return [item[1] for item in scored if item[0] > 0] + [item[1] for item in scored if item[0] <= 0]

    @staticmethod
    def _first_of_kind(candidates: list[dict[str, Any]], kinds: set[str]) -> dict[str, Any] | None:
        for candidate in candidates:
            if _infer_asset_kind(candidate) in kinds:
                return candidate
        return candidates[0] if candidates else None

    @staticmethod
    def _first_explicit_asset(raw_text: str, all_assets: list[dict[str, Any]]) -> dict[str, Any] | None:
        if not raw_text:
            return None
        refs = re.findall(r"!\[[^\]]*\]\((?P<path>[^)\s]+)(?:\s+\"[^\"]*\")?\)", raw_text)
        if not refs:
            return None
        ref_names = {Path(ref.strip("<>")).name for ref in refs if ref.strip("<>")}
        for asset in all_assets:
            path = str(asset.get("path", "") or "")
            filename = str(asset.get("filename", "") or "")
            if (path and Path(path).name in ref_names) or (filename and filename in ref_names):
                return asset
        return None

    @staticmethod
    def _resolve_referenced_asset(text: str, candidates: list[dict[str, Any]]) -> dict[str, Any] | None:
        text = str(text or "").lower()
        numbered_targets: list[tuple[str, int]] = []
        if re.search(r"\btable\s*2\b", text):
            numbered_targets.append(("table", 2))
        if re.search(r"\btable\s*3\b", text):
            numbered_targets.append(("table", 3))
        if re.search(r"\btable\s*4\b", text):
            numbered_targets.append(("table", 4))
        if re.search(r"\bfigure\s*1\b|\bfig\.?\s*1\b", text):
            numbered_targets.append(("figure", 1))
        if re.search(r"\bfigure\s*2\b|\bfig\.?\s*2\b", text):
            numbered_targets.append(("figure", 2))

        for kind, number in numbered_targets:
            for asset in candidates:
                asset_kind = _infer_asset_kind(asset)
                caption = str(asset.get("caption", "") or "")
                if asset_kind != kind:
                    continue
                if kind == "table" and _caption_mentions_table_number(caption, number):
                    return asset
                if kind == "figure" and _caption_mentions_figure_number(caption, number):
                    return asset
        return None
