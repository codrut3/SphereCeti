# SPDX-License-Identifier: Apache-2.0
"""Standalone, stdlib-only read-only Google Gemini API review agent.

Invoked as an isolated subprocess (`python -I gemini_agent.py --model MODEL`) with
`cwd` set to the read-only review workspace and a minimal environment containing only
`GEMINI_API_KEY`. Reads the full rubric prompt from stdin, runs a bounded read-only
function-calling loop against Google's Gemini REST API (`generateContent`), and writes
one structured JSON result (final visible text, token usage, safe workspace-only tool
trace, and structured error fields) to stdout.

No shell, file write, file edit, or arbitrary network tool is exposed. Every filesystem
access is resolved via `realpath` and jailed inside the workspace root.
"""
from __future__ import annotations

import argparse
import fnmatch
import json
import os
import re
import ssl
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

DEFAULT_API_BASE = "https://generativelanguage.googleapis.com/v1beta"
LOOPBACK_API_BASE = re.compile(r"^http://127\.0\.0\.1:[1-9][0-9]*/[A-Za-z0-9._/-]*\Z")
DEFAULT_MAX_TURNS = 30
REQUEST_TIMEOUT_S = 180
MAX_TOOL_TRACE = 40
MAX_TOOL_BYTES = 64 * 1024
MAX_GREP_FILE_BYTES = 4 * 1024 * 1024
TRANSIENT_HTTP = frozenset({500, 502, 503, 504})
OUTSIDE = "<outside-workspace>"
MISSING = "<not-found>"

SYSTEM_INSTRUCTION = (
    "You are running inside a read-only code review sandbox rooted at the current workspace. "
    "Use the provided read-only tools (`read_file`, `grep`, `list_dir`, `find_files`) whenever "
    "the review instructions ask you to read, grep, or glob files under `./code`, `./approved`, "
    "or `./dependencies`. Never fabricate file contents or grep hits. Follow the user prompt's "
    "output format and verdict marker instructions exactly."
)

TOOLS = [{
    "functionDeclarations": [
        {
            "name": "read_file",
            "description": (
                "Read a UTF-8 text file inside the review workspace (read-only). "
                "Returns 1-based numbered lines. Use offset and limit for large files."
            ),
            "parameters": {
                "type": "OBJECT",
                "properties": {
                    "path": {
                        "type": "STRING",
                        "description": (
                            "Path relative to the workspace root (e.g. 'code/SphereCeti/Foo.lean', "
                            "'approved/README.md', 'dependencies/TauCeti/...')."
                        ),
                    },
                    "offset": {
                        "type": "INTEGER",
                        "description": "1-based starting line number (default 1).",
                    },
                    "limit": {
                        "type": "INTEGER",
                        "description": "Maximum number of lines to return (default 250, max 500).",
                    },
                },
                "required": ["path"],
            },
        },
        {
            "name": "grep",
            "description": (
                "Search text files under a workspace file or directory with a Python regular "
                "expression (read-only). Returns 'path:line:text' matches."
            ),
            "parameters": {
                "type": "OBJECT",
                "properties": {
                    "pattern": {
                        "type": "STRING",
                        "description": "Regular expression pattern to search for.",
                    },
                    "path": {
                        "type": "STRING",
                        "description": "Workspace-relative file or directory to search (default '.').",
                    },
                    "glob": {
                        "type": "STRING",
                        "description": "Optional filename glob filter, e.g. '*.lean' or '*.md'.",
                    },
                    "case_insensitive": {
                        "type": "BOOLEAN",
                        "description": "If true, match case-insensitively (default false).",
                    },
                    "max_matches": {
                        "type": "INTEGER",
                        "description": "Maximum matching lines to return (default 100, max 200).",
                    },
                },
                "required": ["pattern"],
            },
        },
        {
            "name": "list_dir",
            "description": "List files and subdirectories in a workspace directory (read-only).",
            "parameters": {
                "type": "OBJECT",
                "properties": {
                    "path": {
                        "type": "STRING",
                        "description": "Workspace-relative directory path (default '.').",
                    },
                    "max_entries": {
                        "type": "INTEGER",
                        "description": "Maximum entries to return (default 200, max 500).",
                    },
                },
            },
        },
        {
            "name": "find_files",
            "description": "Find files under a workspace directory matching a glob pattern (read-only).",
            "parameters": {
                "type": "OBJECT",
                "properties": {
                    "pattern": {
                        "type": "STRING",
                        "description": "Glob pattern matched against filename or relative path, e.g. '*.lean'.",
                    },
                    "path": {
                        "type": "STRING",
                        "description": "Workspace-relative root directory to search under (default '.').",
                    },
                    "max_results": {
                        "type": "INTEGER",
                        "description": "Maximum file paths to return (default 200, max 500).",
                    },
                },
                "required": ["pattern"],
            },
        },
    ]
}]


def clamp_int(value, default: int, low: int, high: int) -> int:
    if isinstance(value, bool):
        return default
    if isinstance(value, (int, float)):
        return max(low, min(high, int(value)))
    return default


def resolve_in_workspace(root: str, raw: str | None) -> tuple[str | None, str, str | None]:
    """Resolve `raw` under workspace `root`.

    Returns `(resolved_path, trace_target, error_message)`. Any escape outside `root`
    (including via symlinks) fails closed and traces as `<outside-workspace>`.
    """
    rel = "." if raw is None or raw == "" else raw
    if not isinstance(rel, str):
        return None, OUTSIDE, "path must be a string"
    if os.path.isabs(rel):
        return None, OUTSIDE, "absolute paths are not permitted; use a workspace-relative path"
    try:
        resolved = os.path.realpath(os.path.join(root, rel))
        if os.path.commonpath([resolved, root]) != root:
            return None, OUTSIDE, "path escapes the review workspace"
    except (OSError, ValueError):
        return None, OUTSIDE, "invalid workspace path"
    if not os.path.exists(resolved):
        return None, MISSING, f"path does not exist: {rel}"
    trace = os.path.relpath(resolved, root)
    return resolved, trace, None


def truncate_text(text: str, limit: int = MAX_TOOL_BYTES) -> str:
    raw = text.encode("utf-8", errors="replace")
    if len(raw) <= limit:
        return text
    return raw[:limit].decode("utf-8", errors="ignore") + "\n...[truncated]"


def tool_read_file(root: str, args: dict) -> tuple[str, str, bool]:
    resolved, trace, err = resolve_in_workspace(root, args.get("path"))
    if err:
        return err, trace, False
    if not os.path.isfile(resolved):
        return f"not an ordinary file: {trace}", trace, False
    offset = clamp_int(args.get("offset"), 1, 1, 10_000_000)
    limit = clamp_int(args.get("limit"), 250, 1, 500)
    try:
        with open(resolved, "rb") as fh:
            data = fh.read()
    except OSError as exc:
        return f"could not read file: {exc}", trace, False
    if b"\x00" in data[:4096]:
        return f"binary file ({len(data)} bytes); text reading skipped", trace, False
    lines = data.decode("utf-8", errors="replace").splitlines()
    total = len(lines)
    if total == 0:
        return f"{trace} (empty file, 0 lines)", trace, True
    start = min(offset, total + 1) - 1
    end = min(start + limit, total)
    out = [f"{i}: {lines[i - 1]}" for i in range(start + 1, end + 1)]
    header = f"File: {trace} (lines {start + 1}-{end} of {total})"
    if end < total:
        header += f" — call read_file with offset={end + 1} for more"
    return truncate_text(header + "\n" + "\n".join(out)), trace, True


def _iter_workspace_files(root: str, start: str):
    """Yield `(abs_path, rel_path)` for readable files inside `root` without escaping symlinks."""
    if os.path.isfile(start):
        yield start, os.path.relpath(start, root)
        return
    for dirpath, dirnames, filenames in os.walk(start, followlinks=False):
        dirnames.sort()
        filenames.sort()
        for name in filenames:
            full = os.path.join(dirpath, name)
            try:
                real = os.path.realpath(full)
                if os.path.commonpath([real, root]) != root or not os.path.isfile(real):
                    continue
            except (OSError, ValueError):
                continue
            yield real, os.path.relpath(full, root)


def tool_grep(root: str, args: dict) -> tuple[str, str, bool]:
    pattern = args.get("pattern")
    if not isinstance(pattern, str) or not pattern:
        return "pattern must be a non-empty string", "", False
    flags = re.IGNORECASE if args.get("case_insensitive") else 0
    try:
        rx = re.compile(pattern, flags)
    except re.error as exc:
        return f"invalid regular expression: {exc}", "", False
    resolved, trace, err = resolve_in_workspace(root, args.get("path", "."))
    if err:
        return err, trace, False
    glob_pat = args.get("glob")
    if glob_pat is not None and not isinstance(glob_pat, str):
        return "glob must be a string", trace, False
    max_matches = clamp_int(args.get("max_matches"), 100, 1, 200)
    matches = []
    truncated = False
    for real, rel in _iter_workspace_files(root, resolved):
        if glob_pat and not (fnmatch.fnmatch(os.path.basename(rel), glob_pat) or fnmatch.fnmatch(rel, glob_pat)):
            continue
        try:
            if os.path.getsize(real) > MAX_GREP_FILE_BYTES:
                continue
            with open(real, "rb") as fh:
                raw = fh.read()
        except OSError:
            continue
        if b"\x00" in raw[:4096]:
            continue
        text = raw.decode("utf-8", errors="replace")
        if not rx.search(text):
            continue
        for lineno, line in enumerate(text.splitlines(), 1):
            if rx.search(line):
                matches.append(f"{rel}:{lineno}:{line[:300]}")
                if len(matches) >= max_matches:
                    truncated = True
                    break
        if truncated:
            break
    if not matches:
        return f"No matches for {pattern!r} in {trace}.", trace, True
    footer = f"\n...[stopped at {max_matches} matches]" if truncated else ""
    return truncate_text("\n".join(matches) + footer), trace, True


def tool_list_dir(root: str, args: dict) -> tuple[str, str, bool]:
    resolved, trace, err = resolve_in_workspace(root, args.get("path", "."))
    if err:
        return err, trace, False
    if not os.path.isdir(resolved):
        return f"not a directory: {trace}", trace, False
    max_entries = clamp_int(args.get("max_entries"), 200, 1, 500)
    try:
        names = sorted(os.listdir(resolved))
    except OSError as exc:
        return f"could not list directory: {exc}", trace, False
    entries = []
    for name in names[:max_entries]:
        full = os.path.join(resolved, name)
        entries.append(name + "/" if os.path.isdir(full) else name)
    header = f"Directory: {trace} ({len(names)} entries)"
    if len(names) > max_entries:
        header += f" [showing first {max_entries}]"
    return truncate_text(header + "\n" + "\n".join(entries)), trace, True


def tool_find_files(root: str, args: dict) -> tuple[str, str, bool]:
    pattern = args.get("pattern")
    if not isinstance(pattern, str) or not pattern:
        return "pattern must be a non-empty string", "", False
    resolved, trace, err = resolve_in_workspace(root, args.get("path", "."))
    if err:
        return err, trace, False
    max_results = clamp_int(args.get("max_results"), 200, 1, 500)
    found = []
    truncated = False
    for _, rel in _iter_workspace_files(root, resolved):
        if fnmatch.fnmatch(os.path.basename(rel), pattern) or fnmatch.fnmatch(rel, pattern):
            found.append(rel)
            if len(found) >= max_results:
                truncated = True
                break
    if not found:
        return f"No files matching {pattern!r} under {trace}.", trace, True
    footer = f"\n...[stopped at {max_results} files]" if truncated else ""
    return truncate_text("\n".join(found) + footer), trace, True


TOOL_HANDLERS = {
    "read_file": tool_read_file,
    "grep": tool_grep,
    "list_dir": tool_list_dir,
    "find_files": tool_find_files,
}


def dispatch_tool(root: str, name: str, args) -> tuple[str, str, bool]:
    handler = TOOL_HANDLERS.get(name)
    if handler is None:
        return f"unknown or disallowed tool: {name!r}", "", False
    if not isinstance(args, dict):
        args = {}
    return handler(root, args)


def resolve_api_base(explicit: str | None) -> str:
    candidate = explicit or os.environ.get("SPHERECETI_GEMINI_API_BASE") or ""
    if not candidate:
        return DEFAULT_API_BASE
    if not LOOPBACK_API_BASE.fullmatch(candidate):
        raise ValueError("custom Gemini API base must be a loopback http://127.0.0.1:<port>/ URL")
    return candidate.rstrip("/")


def parse_http_error(code: int, raw: bytes) -> tuple[int, str]:
    text = raw.decode("utf-8", errors="replace").strip()
    try:
        payload = json.loads(text)
    except ValueError:
        return code, text[:500] or f"HTTP {code}"
    err = payload.get("error") if isinstance(payload, dict) else None
    if isinstance(err, dict):
        status = err.get("status")
        msg = err.get("message")
        pieces = [p for p in (status, msg) if isinstance(p, str) and p]
        if pieces:
            return code, f"HTTP {code} " + ": ".join(pieces)
    return code, text[:500] or f"HTTP {code}"


def post_generate_content(api_base: str, model: str, api_key: str, body: dict,
                          timeout: float = REQUEST_TIMEOUT_S) -> tuple[dict | None, int | None, str | None]:
    url = f"{api_base}/models/{urllib.parse.quote(model, safe='')}:generateContent"
    encoded = json.dumps(body).encode("utf-8")
    context = ssl.create_default_context() if url.startswith("https://") else None
    delays = (1.0, 3.0)
    for attempt in range(len(delays) + 1):
        req = urllib.request.Request(
            url,
            data=encoded,
            method="POST",
            headers={
                "Content-Type": "application/json",
                "x-goog-api-key": api_key,
            },
        )
        try:
            with urllib.request.urlopen(req, timeout=timeout, context=context) as resp:
                raw = resp.read()
            data = json.loads(raw.decode("utf-8", errors="replace"))
            if not isinstance(data, dict):
                return None, None, "Gemini API returned a non-object JSON payload"
            return data, None, None
        except urllib.error.HTTPError as exc:
            try:
                err_body = exc.read()
            except OSError:
                err_body = b""
            code, msg = parse_http_error(exc.code, err_body)
            if code in TRANSIENT_HTTP and attempt < len(delays):
                time.sleep(delays[attempt])
                continue
            return None, code, msg
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            if attempt < len(delays):
                time.sleep(delays[attempt])
                continue
            return None, None, f"connection error: {exc}"
        except ValueError as exc:
            return None, None, f"invalid JSON response from Gemini API: {exc}"
    return None, None, "Gemini API request failed"


def visible_text(parts: list) -> str:
    """Collect visible assistant text, excluding `thought: true` reasoning summaries."""
    chunks = []
    for part in parts:
        if isinstance(part, dict) and not part.get("thought") and isinstance(part.get("text"), str):
            chunks.append(part["text"])
    return "\n".join(chunks)


def run_agent(prompt: str, cwd: str, model: str, api_key: str, *,
              api_base: str = DEFAULT_API_BASE, max_turns: int = DEFAULT_MAX_TURNS,
              timeout: float = REQUEST_TIMEOUT_S) -> dict:
    root = os.path.realpath(cwd)
    if not api_key or not api_key.strip():
        return {
            "text": "",
            "usage": {"input_tokens": 0, "cached_input_tokens": 0, "output_tokens": 0},
            "tool_trace": [],
            "tool_trace_meta": {"total_calls": 0, "trace_truncated": False},
            "is_error": True,
            "error_status": 401,
            "error_message": "GEMINI_API_KEY is not set",
        }
    contents: list[dict] = [{"role": "user", "parts": [{"text": prompt}]}]
    fresh_in = cached_in = out_tok = reas_tok = 0
    trace: list[dict] = []
    total_calls = 0
    final_text = ""

    def usage_dict():
        u = {"input_tokens": fresh_in, "cached_input_tokens": cached_in, "output_tokens": out_tok}
        if reas_tok:
            u["reasoning_output_tokens"] = reas_tok
        return u

    def result(*, text: str = "", is_error: bool = False, error_status: int | None = None,
               error_message: str | None = None) -> dict:
        return {
            "text": text,
            "usage": usage_dict(),
            "tool_trace": trace,
            "tool_trace_meta": {"total_calls": total_calls, "trace_truncated": total_calls > len(trace)},
            "is_error": is_error,
            "error_status": error_status,
            "error_message": error_message,
        }

    for turn in range(max_turns + 1):
        finalizing = turn == max_turns
        body = {
            "systemInstruction": {"parts": [{"text": SYSTEM_INSTRUCTION}]},
            "contents": contents,
        }
        if not finalizing:
            body["tools"] = TOOLS
        data, err_status, err_msg = post_generate_content(api_base, model, api_key.strip(), body, timeout=timeout)
        if data is None:
            return result(text=final_text, is_error=True, error_status=err_status, error_message=err_msg)
        um = data.get("usageMetadata")
        if isinstance(um, dict):
            prompt_t = um.get("promptTokenCount") or 0
            cached_t = um.get("cachedContentTokenCount") or 0
            cand_t = um.get("candidatesTokenCount") or 0
            thought_t = um.get("thoughtsTokenCount") or 0
            fresh_in += max(0, int(prompt_t) - int(cached_t))
            cached_in += max(0, int(cached_t))
            out_tok += max(0, int(cand_t) + int(thought_t))
            reas_tok += max(0, int(thought_t))
        candidates = data.get("candidates")
        if not isinstance(candidates, list) or not candidates:
            feedback = data.get("promptFeedback")
            reason = feedback.get("blockReason") if isinstance(feedback, dict) else "empty"
            return result(text=final_text, is_error=True,
                          error_message=f"Gemini returned no candidates ({reason})")
        cand = candidates[0] if isinstance(candidates[0], dict) else {}
        content = cand.get("content") if isinstance(cand.get("content"), dict) else {}
        parts = content.get("parts") if isinstance(content.get("parts"), list) else []
        text = visible_text(parts)
        if text:
            final_text = text
        calls = [p["functionCall"] for p in parts
                 if isinstance(p, dict) and isinstance(p.get("functionCall"), dict)]
        if not calls:
            return result(text=final_text)
        # Preserve the model's exact parts (including thoughtSignature and thought parts) verbatim.
        contents.append({"role": "model", "parts": parts})
        response_parts = []
        for call in calls:
            name = str(call.get("name") or "")
            args = call.get("args")
            output, target, ok = dispatch_tool(root, name, args)
            total_calls += 1
            if len(trace) < MAX_TOOL_TRACE:
                entry = {"tool": name[:40], "ok": ok}
                if target:
                    entry["target"] = target
                trace.append(entry)
            fr = {"name": name, "response": {"output": output} if ok else {"error": output}}
            if call.get("id"):
                fr["id"] = call["id"]
            response_parts.append({"functionResponse": fr})
        if turn + 1 == max_turns:
            response_parts.append({
                "text": "Tool budget reached. Produce your final review and verdict JSON now without further tool calls."
            })
        contents.append({"role": "user", "parts": response_parts})
    return result(text=final_text)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Read-only Google Gemini API reviewer agent")
    ap.add_argument("--model", required=True)
    ap.add_argument("--max-turns", type=int, default=DEFAULT_MAX_TURNS)
    ap.add_argument("--api-base", default=None)
    args = ap.parse_args(argv)
    try:
        api_base = resolve_api_base(args.api_base)
    except ValueError as exc:
        print(json.dumps({
            "text": "",
            "usage": {"input_tokens": 0, "cached_input_tokens": 0, "output_tokens": 0},
            "tool_trace": [],
            "tool_trace_meta": {"total_calls": 0, "trace_truncated": False},
            "is_error": True,
            "error_status": None,
            "error_message": str(exc),
        }))
        return 2
    prompt = sys.stdin.read()
    out = run_agent(
        prompt,
        os.getcwd(),
        args.model,
        os.environ.get("GEMINI_API_KEY", ""),
        api_base=api_base,
        max_turns=max(1, args.max_turns),
    )
    print(json.dumps(out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
