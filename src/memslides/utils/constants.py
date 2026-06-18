import logging
import os
from pathlib import Path


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or raw == "":
        return default
    try:
        return int(raw)
    except ValueError:
        return default


PACKAGE_DIR = Path(__file__).resolve().parent.parent

LOGGING_LEVEL = _env_int("MEMSLIDES_LOG_LEVEL", logging.WARNING)
MAX_LOGGING_LENGTH = _env_int("MEMSLIDES_MAX_LOGGING_LENGTH", 1024)

RETRY_TIMES = _env_int("MEMSLIDES_RETRY_TIMES", 10)
MAX_RETRY_INTERVAL = _env_int("MEMSLIDES_MAX_RETRY_INTERVAL", 60)
MAX_TOOLCALL_PER_TURN = _env_int("MEMSLIDES_MAX_TOOLCALL_PER_TURN", 15)
TOOL_CUTOFF_LEN = _env_int("MEMSLIDES_TOOL_CUTOFF_LEN", 8000)
CONTEXT_LENGTH_LIMIT = _env_int("MEMSLIDES_CONTEXT_LENGTH_LIMIT", 90_000)
MAX_TOOL_RESULT_CHARS = _env_int("MEMSLIDES_MAX_TOOL_RESULT_CHARS", 16_000)
MAX_AGENT_ITERATIONS = _env_int("MEMSLIDES_MAX_AGENT_ITERATIONS", 50)
MAX_MODIFY_ITERATIONS = _env_int("MEMSLIDES_MAX_MODIFY_ITERATIONS", 25)
STUCK_DETECTION_THRESHOLD = _env_int("MEMSLIDES_STUCK_DETECTION_THRESHOLD", 2)

PIXEL_MULTIPLE = _env_int("MEMSLIDES_PIXEL_MULTIPLE", 16)
MAX_RAW_CHAINS_PER_KEY = _env_int("MEMSLIDES_MAX_RAW_CHAINS_PER_KEY", 20)
MAX_LTM_EXPERIENCES_PER_TOOL = _env_int("MEMSLIDES_MAX_LTM_EXPERIENCES_PER_TOOL", 10)
MCP_CONNECT_TIMEOUT = _env_int("MEMSLIDES_MCP_CONNECT_TIMEOUT", 120)
MCP_CALL_TIMEOUT = _env_int("MEMSLIDES_MCP_CALL_TIMEOUT", 1800)

READ_ONLY_TOOLS = frozenset(
    {
        "browse_web",
        "get_file_info",
        "list_directory",
        "list_files",
        "read_file",
        "read_slide_snapshot",
        "search_web",
    }
)

OVERFLOW_MARKER = (
    "[OUTPUT_TRUNCATED: showing {max_chars} of {actual_len} characters]"
)
CUTOFF_WARNING = (
    "\n\n[MemSlides note: output truncated after {line} lines. "
    "Continue reading from {resource_id} with an offset or narrower query.]"
)

DEFAULT_CACHE_BASE = Path(
    os.getenv("MEMSLIDES_DEFAULT_CACHE_ROOT", str(Path.home() / ".cache" / "memslides"))
).expanduser()

_workspace_root = os.getenv("MEMSLIDES_WORKSPACE_BASE", "").strip()
WORKSPACE_BASE = (
    Path(_workspace_root).expanduser()
    if _workspace_root and _workspace_root not in {"workspace", "./workspace"}
    else DEFAULT_CACHE_BASE
)

DEFAULT_LOG_DIR = WORKSPACE_BASE / "logs"
DEFAULT_GLOBAL_MEMORY_DIR = WORKSPACE_BASE / ".memory"
DEFAULT_GLOBAL_MEMORY_DB = DEFAULT_GLOBAL_MEMORY_DIR / "global_memory.db"
DEFAULT_GLOBAL_MEMORY_V2_DB = DEFAULT_GLOBAL_MEMORY_DIR / "global_memory_v2.db"
DEFAULT_TEMPLATES_DIR = WORKSPACE_BASE / "templates"
TOOL_CACHE = PACKAGE_DIR / ".tools.json"

GLOBAL_ENV_LIST = [
    "ALL_PROXY",
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "NO_PROXY",
    "all_proxy",
    "http_proxy",
    "https_proxy",
    "no_proxy",
]

PDF_OPTIONS = {
    "display_header_footer": False,
    "landscape": False,
    "margin": {
        "bottom": "0mm",
        "left": "0mm",
        "right": "0mm",
        "top": "0mm",
    },
    "page_ranges": "1",
    "prefer_css_page_size": False,
    "print_background": True,
    "scale": 1,
}

AGENT_PROMPT = """
<MemSlides Runtime>
Date: {time}
Workspace: {workspace}
Execution environment: Debian-based container

Available local utilities include Python, Node.js, ImageMagick, Mermaid CLI,
curl, wget, python-pptx, matplotlib, plotly, and common Unix tools. Install
small missing packages only when they are necessary for the current slide task.
</MemSlides Runtime>

<Tool Use Rules>
- Explore normally until runtime warnings say the remaining budget is low.
- Very long tool outputs are shortened after {cutoff_len} characters and saved
  to a local resource path when possible.
- When using tools, write a concise reason in the assistant message before the
  tool calls.
- Use no more than {max_toolcall_per_turn} tool calls in one assistant turn.
</Tool Use Rules>
"""

TOOL_REASON_PROMPT = """
<Tool Reason>
For every tool batch, include one short `Reason:` sentence that names the next
concrete step and why the tool result is needed. Keep it brief and operational.
</Tool Reason>
"""

OFFLINE_PROMPT = """
<Offline Mode>
Network-dependent tools are unavailable. Use local files, cached artifacts, and
installed utilities to complete the slide work.
</Offline Mode>
"""

CONTEXT_MODE_PROMPT = """
<Context Management>
When context is compacted, immediately preserve important generated files and
refer back to the saved summary before continuing. Do not rely on unsaved
conversation state for slide edits or verification.
</Context Management>
"""

HALF_BUDGET_NOTICE_MSG = {
    "type": "text",
    "text": (
        "<NOTICE>About half of the working budget has been used. Focus on the "
        "main deliverable and avoid optional exploration.</NOTICE>"
    ),
}

URGENT_BUDGET_NOTICE_MSG = {
    "type": "text",
    "text": (
        "<URGENT>The working budget is nearly exhausted. Complete the essential "
        "slide work and call `finalize` now.</URGENT>"
    ),
}

HIST_LOST_MSG = {
    "type": "text",
    "text": "<NOTICE>Earlier history has been compacted into a local summary.</NOTICE>",
}

CONTINUE_MSG = {
    "type": "text",
    "text": "<NOTICE>Resume from the saved summary and continue the current task.</NOTICE>",
}

LAST_ITER_MSG = {
    "type": "text",
    "text": (
        "<URGENT>This is the final available iteration. Finish the core task and "
        "call `finalize`.</URGENT>"
    ),
}

STUCK_NUDGE_MSG = {
    "type": "text",
    "text": (
        "<NOTICE>The recent steps look repetitive. Change strategy, inspect a "
        "different artifact, or finalize if the task is complete.</NOTICE>"
    ),
}

NONPRODUCTIVE_NUDGE_MSG = {
    "type": "text",
    "text": (
        "<NOTICE>Planning messages do not modify slides. Make a concrete file "
        "change with the slide tools, choose reasonable defaults for unspecified "
        "details, or finalize if no more edits are needed.</NOTICE>"
    ),
}

FORCE_FINALIZE_MSG = {
    "type": "text",
    "text": (
        "<URGENT>The maximum iteration count has been reached. Call `finalize` "
        "now with the current result.</URGENT>"
    ),
}

MEMORY_COMPACT_MSG = """
Create a compact continuation summary for the current MemSlides work session and
save it in the workspace before doing anything else.

<summary_requirements>
1. Modification checklist
   - List every file that was modified or still needs modification.
   - For each file, include the exact status and the next required action.

2. Evidence and artifacts
   - Record important paths, generated files, observations, and test outputs.

3. Open risks
   - Name unresolved assumptions, missing verification, or fragile decisions.

4. Next steps
   - Put unfinished required work first.
   - Include enough detail for another agent to continue without old history.
</summary_requirements>

<important>
- Write the summary primarily in {language}.
- Do not use vague references to previous messages.
- Save the summary immediately; unsaved conversation history may be removed.
</important>
"""
