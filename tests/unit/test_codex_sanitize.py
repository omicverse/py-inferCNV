"""Tests for scripts/codex_review_sanitize.py — output strip rules."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "codex_review_sanitize.py"


def run_sanitizer(input_text: str, tmp_path: Path) -> str:
    src = tmp_path / "in.md"
    dst = tmp_path / "out.md"
    src.write_text(input_text, encoding="utf-8")
    subprocess.run(
        [sys.executable, str(SCRIPT), str(src), "-o", str(dst)],
        check=True,
        capture_output=True,
    )
    return dst.read_text(encoding="utf-8")


def test_removes_suspicious_urls(tmp_path):
    text = "Check out www.k-dense.ai for benchmarks. Also https://github.com/broadinstitute/inferCNV is useful."
    out = run_sanitizer(text, tmp_path)
    assert "k-dense.ai" not in out
    assert "SUSPICIOUS_URL_REMOVED" in out
    assert "github.com" in out


def test_removes_skill_meta_lines(tmp_path):
    text = 'Using `python-performance-optimization`; I will skim those guides.\nActual content here.\n'
    out = run_sanitizer(text, tmp_path)
    assert "I will skim" not in out
    assert "Using `python-performance" not in out
    assert "Actual content" in out


def test_removes_planning_bullets(tmp_path):
    text = "• Read skill guidance\n→ Assess architecture decisions\n✓ Draft concise review\n**Q1** Real content."
    out = run_sanitizer(text, tmp_path)
    assert "Read skill guidance" not in out
    assert "Assess architecture" not in out
    assert "Draft concise" not in out
    assert "**Q1**" in out


def test_removes_sandbox_errors(tmp_path):
    text = "bwrap: loopback: Failed RTM_NEWADDR: Operation not permitted\nSandbox(Denied { output:\nnetwork_policy_decision: None\nKept content."
    out = run_sanitizer(text, tmp_path)
    assert "bwrap:" not in out
    assert "Sandbox(Denied" not in out
    assert "network_policy_decision" not in out
    assert "Kept content" in out


def test_removes_meta_lines(tmp_path):
    text = "session id: 019dae0a-b6d5\ntokens used: 47809\nreasoning summaries: none\nprovider: openai\napproval: never\nsandbox: read-only\n**Verdict:** fine."
    out = run_sanitizer(text, tmp_path)
    for pat in ["session id:", "tokens used", "reasoning summaries:", "provider:", "approval:", "sandbox:"]:
        assert pat not in out, f"leaked: {pat}"
    assert "**Verdict:**" in out


def test_removes_deprecation_warnings(tmp_path):
    text = "warning: `OPENAI_BASE_URL` is deprecated. Set `openai_base_url` in config.toml instead.\nReal text."
    out = run_sanitizer(text, tmp_path)
    assert "OPENAI_BASE_URL" not in out
    assert "Real text" in out


def test_preserves_qa_content(tmp_path):
    text = "**Q1**\n- **Verdict:** Adjustment recommended.\n- **Modification:** Use scipy.ndimage.\n- **Reason:** Idiomatic."
    out = run_sanitizer(text, tmp_path)
    assert "Q1" in out
    assert "Verdict" in out
    assert "Adjustment recommended" in out
    assert "scipy.ndimage" in out
