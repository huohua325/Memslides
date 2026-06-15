#!/usr/bin/env python3
"""Local Web Studio smoke for the public MemSlides release.

The script talks to a running local Studio. It uses real model calls and does
not mock generation or revision results.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any


DEFAULT_PROMPT = (
    "Create a 2-slide briefing for city planners about urban heat islands. "
    "Make it practical, visually clear, and suitable for a short local demo."
)
DEFAULT_FEEDBACK = (
    "Make slide 2 more action-oriented with a compact checklist, and keep the deck length unchanged."
)
TERMINAL_STATES = {"succeeded", "failed", "cancelled", "idle"}
RUNNING_MARKERS = {"running", "queued", "cancelling"}


class SmokeError(RuntimeError):
    def __init__(self, stage: str, message: str):
        super().__init__(message)
        self.stage = stage
        self.message = message


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a local MemSlides Web Studio smoke.")
    parser.add_argument("--base-url", default=os.getenv("MEMSLIDES_SMOKE_BASE_URL", "http://127.0.0.1:7860"))
    parser.add_argument("--timeout", type=int, default=int(os.getenv("MEMSLIDES_SMOKE_TIMEOUT", "900")))
    parser.add_argument("--poll-interval", type=float, default=float(os.getenv("MEMSLIDES_SMOKE_POLL_INTERVAL", "5")))
    parser.add_argument("--report-path", default=os.getenv("MEMSLIDES_SMOKE_REPORT_PATH", ""))
    parser.add_argument("--prompt", default=os.getenv("MEMSLIDES_SMOKE_PROMPT", DEFAULT_PROMPT))
    parser.add_argument("--feedback", default=os.getenv("MEMSLIDES_SMOKE_FEEDBACK", DEFAULT_FEEDBACK))
    parser.add_argument("--language", default=os.getenv("MEMSLIDES_SMOKE_LANGUAGE", "en"), choices=["en", "zh"])
    parser.add_argument("--skip-revision", action="store_true", help="Only check generation and initial PPTX export.")
    args = parser.parse_args()

    started = time.time()
    report: dict[str, Any] = {
        "base_url": args.base_url.rstrip("/"),
        "started_at": started,
        "success": False,
        "warnings": [],
    }
    stage = "startup"
    try:
        client = Client(args.base_url)
        stage = "health"
        report["health"] = client.get("/api/health")

        stage = "service_profile"
        profile = ensure_service_profile(client)
        report["service_profile"] = public_profile_summary(profile)
        profile_id = str(profile.get("profile_id") or "")
        if not profile_id:
            raise SmokeError(stage, "No Service Profile is available. Configure one in the Studio or set smoke env vars.")

        stage = "create_session"
        session = client.post_json(
            "/api/sessions",
            {
                "language": args.language,
                "service_profile_id": profile_id,
                "memory_enabled": True,
                "memory_intent": "local_smoke",
            },
        )
        session_id = str(session.get("session_id") or "")
        if not session_id:
            raise SmokeError(stage, f"Session response did not contain session_id: {session}")
        report["session_id"] = session_id

        stage = "generate"
        before_generate = artifact_snapshot(client, session_id)
        generate_payload = {
            "instruction": args.prompt,
            "num_pages": "2",
            "language": args.language,
            "memory_intent": "local_smoke",
            "service_profile_id": profile_id,
            "memory_enabled": "true",
        }
        client.post_form(f"/api/sessions/{session_id}/generate", generate_payload)

        stage = "wait_generation"
        generation_status = wait_for_operation(client, session_id, args.timeout, args.poll_interval)
        report["generation_status"] = compact_status(generation_status)
        if str(generation_status.get("state") or "") != "succeeded":
            raise SmokeError(stage, f"Generation did not succeed: {compact_status(generation_status)}")

        stage = "generation_export_check"
        generation_artifacts = artifact_snapshot(client, session_id)
        initial_pptx = newest_pptx_after(generation_artifacts, before_generate)
        if not initial_pptx:
            raise SmokeError(stage, "Generation succeeded but no fresh PPTX artifact was found.")
        report["initial_pptx"] = artifact_summary(initial_pptx)

        if not args.skip_revision:
            stage = "revise"
            before_revision = artifact_snapshot(client, session_id)
            client.post_json(
                f"/api/sessions/{session_id}/revise",
                {
                    "feedback": args.feedback,
                    "memory_intent": "local_smoke",
                    "service_profile_id": profile_id,
                },
            )

            stage = "wait_revision"
            revision_status = wait_for_operation(client, session_id, args.timeout, args.poll_interval)
            report["revision_status"] = compact_status(revision_status)
            if str(revision_status.get("state") or "") != "succeeded":
                raise SmokeError(stage, f"Revision did not succeed: {compact_status(revision_status)}")
            if str(revision_status.get("export_status") or "") == "partial":
                report["warnings"].append("Revision succeeded with export warnings.")

            stage = "revision_export_check"
            revision_artifacts = artifact_snapshot(client, session_id)
            revision_pptx = newest_pptx_after(revision_artifacts, before_revision, require_modification=True)
            if not revision_pptx:
                raise SmokeError(stage, "Revision succeeded but no fresh modification_N.pptx artifact was found.")
            report["revision_pptx"] = artifact_summary(revision_pptx)

        report["success"] = True
        report["elapsed_seconds"] = round(time.time() - started, 2)
        write_report(args.report_path, report)
        print(json.dumps(report, ensure_ascii=False, indent=2))
    except SmokeError as exc:
        report.update(
            {
                "success": False,
                "failed_stage": exc.stage,
                "error": exc.message,
                "elapsed_seconds": round(time.time() - started, 2),
            }
        )
        write_report(args.report_path, report)
        print(json.dumps(report, ensure_ascii=False, indent=2), file=sys.stderr)
        raise SystemExit(1) from exc
    except Exception as exc:  # noqa: BLE001
        report.update(
            {
                "success": False,
                "failed_stage": stage,
                "error": f"{exc.__class__.__name__}: {exc}",
                "elapsed_seconds": round(time.time() - started, 2),
            }
        )
        write_report(args.report_path, report)
        print(json.dumps(report, ensure_ascii=False, indent=2), file=sys.stderr)
        raise SystemExit(1) from exc


class Client:
    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip("/")

    def get(self, path: str) -> dict[str, Any]:
        return self._request("GET", path)

    def post_json(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        body = json.dumps(payload).encode("utf-8")
        headers = {"Content-Type": "application/json"}
        return self._request("POST", path, data=body, headers=headers)

    def post_form(self, path: str, payload: dict[str, str]) -> dict[str, Any]:
        boundary = f"memslides-smoke-{int(time.time() * 1000)}"
        chunks: list[bytes] = []
        for key, value in payload.items():
            chunks.append(f"--{boundary}\r\n".encode("utf-8"))
            chunks.append(f'Content-Disposition: form-data; name="{key}"\r\n\r\n'.encode("utf-8"))
            chunks.append(str(value).encode("utf-8"))
            chunks.append(b"\r\n")
        chunks.append(f"--{boundary}--\r\n".encode("utf-8"))
        headers = {"Content-Type": f"multipart/form-data; boundary={boundary}"}
        return self._request("POST", path, data=b"".join(chunks), headers=headers)

    def _request(
        self,
        method: str,
        path: str,
        *,
        data: bytes | None = None,
        headers: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        url = f"{self.base_url}{path}"
        request = urllib.request.Request(url, data=data, method=method, headers=headers or {})
        try:
            with urllib.request.urlopen(request, timeout=60) as response:  # noqa: S310
                text = response.read().decode("utf-8")
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"{method} {url} -> HTTP {exc.code}: {detail}") from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"{method} {url} failed: {exc}") from exc
        if not text.strip():
            return {}
        data_obj = json.loads(text)
        if not isinstance(data_obj, dict):
            raise RuntimeError(f"{method} {url} returned non-object JSON: {data_obj!r}")
        return data_obj


def ensure_service_profile(client: Client) -> dict[str, Any]:
    profiles = list_profiles(client)
    ready = [item for item in profiles if item.get("required_ready") or item.get("validation_status") == "valid"]
    if ready:
        return sorted(ready, key=lambda item: (not item.get("is_default"), str(item.get("updated_at") or "")))[0]
    if profiles:
        preferred = next((item for item in profiles if item.get("is_default")), profiles[0])
        profile_id = str(preferred.get("profile_id") or "")
        if not profile_id:
            return {}
        try:
            validated = client.post_json(f"/api/service-profiles/{profile_id}/validate", {})
        except Exception as exc:  # noqa: BLE001
            raise SmokeError("service_profile", f"Service Profile validation failed: {exc}") from exc
        if not validated.get("ok"):
            raise SmokeError("service_profile", f"Service Profile is not ready: {validated.get('message') or validated}")
        profile = validated.get("profile") if isinstance(validated.get("profile"), dict) else {}
        return profile or preferred

    env_profile = smoke_profile_payload_from_env()
    if not env_profile:
        return {}
    saved = client.post_json("/api/service-profiles", env_profile)
    try:
        validated = client.post_json(f"/api/service-profiles/{saved['profile_id']}/validate", {})
        profile = validated.get("profile") if isinstance(validated.get("profile"), dict) else {}
        return profile or saved
    except Exception:
        return saved


def list_profiles(client: Client) -> list[dict[str, Any]]:
    payload = client.get("/api/service-profiles")
    profiles = payload.get("profiles") if isinstance(payload, dict) else []
    return [dict(item) for item in profiles if isinstance(item, dict)]


def smoke_profile_payload_from_env() -> dict[str, Any]:
    llm_key = os.getenv("MEMSLIDES_SMOKE_LLM_API_KEY", "").strip()
    llm_model = os.getenv("MEMSLIDES_SMOKE_LLM_MODEL", "gpt-4.1").strip()
    if not llm_key or not llm_model:
        return {}
    pdf_key = os.getenv("MEMSLIDES_SMOKE_PDF_API_KEY", "").strip()
    pdf_url = os.getenv("MEMSLIDES_SMOKE_PDF_API_URL", "").strip()
    if not pdf_key and not pdf_url:
        return {}
    search_key = os.getenv("MEMSLIDES_SMOKE_TAVILY_API_KEY", "").strip()
    image_key = os.getenv("MEMSLIDES_SMOKE_IMAGE_API_KEY", "").strip()
    image_model = os.getenv("MEMSLIDES_SMOKE_IMAGE_MODEL", "").strip()
    image_base_url = os.getenv("MEMSLIDES_SMOKE_IMAGE_BASE_URL", os.getenv("MEMSLIDES_SMOKE_LLM_BASE_URL", "https://api.openai.com/v1")).strip()
    return {
        "profile_id": os.getenv("MEMSLIDES_SMOKE_PROFILE_ID", "local-smoke"),
        "display_name": "Local smoke profile",
        "max_concurrent": 1,
        "enabled": True,
        "is_default": True,
        "llm": {
            "provider": "openai_compatible",
            "base_url": os.getenv("MEMSLIDES_SMOKE_LLM_BASE_URL", "https://api.openai.com/v1").strip(),
            "model": llm_model,
            "api_key": llm_key,
            "enabled": True,
            "vision_capable": True,
        },
        "pdf": {
            "provider": "pdf_parser_compatible" if pdf_url else "pdf_parser_official",
            "api_key": pdf_key,
            "api_url": pdf_url,
            "enabled": True,
        },
        "embedding": {
            "provider": "openai_compatible",
            "base_url": os.getenv("MEMSLIDES_SMOKE_EMBEDDING_BASE_URL", "https://api.openai.com/v1").strip(),
            "model": os.getenv("MEMSLIDES_SMOKE_EMBEDDING_MODEL", "text-embedding-3-small").strip(),
            "api_key": os.getenv("MEMSLIDES_SMOKE_EMBEDDING_API_KEY", "").strip(),
            "enabled": bool(os.getenv("MEMSLIDES_SMOKE_EMBEDDING_API_KEY", "").strip()),
        },
        "search": {
            "provider": "tavily",
            "api_key": search_key,
            "enabled": bool(search_key),
        },
        "image_generation": {
            "provider": "openai_compatible",
            "base_url": image_base_url,
            "model": image_model,
            "api_key": image_key,
            "enabled": bool(image_key and image_model),
        },
    }


def wait_for_operation(client: Client, session_id: str, timeout: int, poll_interval: float) -> dict[str, Any]:
    deadline = time.time() + timeout
    last_status: dict[str, Any] = {}
    while time.time() < deadline:
        status = client.get(f"/api/sessions/{session_id}/status")
        last_status = status
        state = str(status.get("state") or "").lower()
        phase = str(status.get("phase") or "").lower()
        queued = str(status.get("queued_operation") or "").lower()
        if state in TERMINAL_STATES and not any(marker in phase for marker in RUNNING_MARKERS) and not queued:
            if state == "idle" and phase in {"", "ready"}:
                time.sleep(poll_interval)
                continue
            return status
        time.sleep(poll_interval)
    raise SmokeError("wait_timeout", f"Operation did not finish within {timeout}s. Last status: {compact_status(last_status)}")


def artifact_snapshot(client: Client, session_id: str) -> list[dict[str, Any]]:
    payload = client.get(f"/api/sessions/{session_id}/artifacts")
    files = payload.get("files") if isinstance(payload, dict) else []
    current = ((payload.get("current_deck") or {}).get("files") or []) if isinstance(payload, dict) else []
    history_groups = payload.get("version_history") if isinstance(payload, dict) else []
    history: list[dict[str, Any]] = []
    if isinstance(history_groups, list):
        for group in history_groups:
            if isinstance(group, dict):
                history.extend(item for item in group.get("files", []) if isinstance(item, dict))
    combined = [*(files if isinstance(files, list) else []), *current, *history]
    dedup: dict[str, dict[str, Any]] = {}
    for item in combined:
        if not isinstance(item, dict):
            continue
        if str(item.get("kind") or "").lower() != "pptx":
            continue
        path = str(item.get("path") or item.get("name") or "")
        if not path:
            continue
        dedup[path] = dict(item)
    return list(dedup.values())


def newest_pptx_after(
    artifacts: list[dict[str, Any]],
    before: list[dict[str, Any]],
    *,
    require_modification: bool = False,
) -> dict[str, Any] | None:
    before_mtime = {str(item.get("path") or ""): float(item.get("updated_at") or 0.0) for item in before}
    candidates: list[dict[str, Any]] = []
    for item in artifacts:
        path = str(item.get("path") or "")
        name = str(item.get("name") or Path(path).name)
        if require_modification and not name.lower().startswith("modification_"):
            continue
        if float(item.get("updated_at") or 0.0) > before_mtime.get(path, 0.0):
            candidates.append(item)
    if not candidates:
        return None
    return sorted(candidates, key=lambda item: float(item.get("updated_at") or 0.0), reverse=True)[0]


def public_profile_summary(profile: dict[str, Any]) -> dict[str, Any]:
    return {
        "profile_id": profile.get("profile_id"),
        "display_name": profile.get("display_name"),
        "model": profile.get("model"),
        "required_ready": profile.get("required_ready"),
        "validation_status": profile.get("validation_status"),
    }


def compact_status(status: dict[str, Any]) -> dict[str, Any]:
    return {
        "state": status.get("state"),
        "phase": status.get("phase"),
        "message": status.get("message"),
        "slide_count": status.get("slide_count"),
        "export_status": status.get("export_status"),
        "export_warnings": status.get("export_warnings"),
        "latest_export_fresh": status.get("latest_export_fresh"),
    }


def artifact_summary(item: dict[str, Any]) -> dict[str, Any]:
    return {
        "name": item.get("name"),
        "path": item.get("path"),
        "size": item.get("size"),
        "updated_at": item.get("updated_at"),
        "download_url": item.get("download_url"),
    }


def write_report(path: str, report: dict[str, Any]) -> None:
    if not path:
        return
    target = Path(path).expanduser()
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
