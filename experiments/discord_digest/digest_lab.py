"""Discord digest lab: two-pass story pick + multi-model comparison + Opus judge.

  uv run python experiments/discord_digest/digest_lab.py stories <transcript.md> [--model claude:haiku]
  uv run python experiments/discord_digest/digest_lab.py digest <run_dir> --pick 1,3,4 [--preset ...] [--models ...]
  uv run python experiments/discord_digest/digest_lab.py judge <digest_dir>

Model specs: groq:<model-id> or claude:<alias> (haiku | sonnet | opus).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path

import yaml
from dotenv import load_dotenv

from pidcast.discord_recap import STORIES_SYSTEM, parse_stories, segment
from pidcast.providers.claude_provider import run_claude

LAB_DIR = Path(__file__).resolve().parent
REPO_ROOT = LAB_DIR.parents[1]
RUNS_DIR = LAB_DIR / "runs"
DEFAULT_PRESET = LAB_DIR / "presets" / "discord-digest-uk.yaml"
DEFAULT_MODELS = ["groq:openai/gpt-oss-120b", "claude:haiku", "claude:sonnet", "claude:opus"]
JUDGE_MODEL = "claude:opus"

load_dotenv(REPO_ROOT / ".env")


@dataclass
class LLMResult:
    model: str
    text: str
    latency_s: float
    input_tokens: int | None = None
    output_tokens: int | None = None
    cost_usd: float | None = None
    rate_limits: dict[str, str] = field(default_factory=dict)
    error: str | None = None


# ---------- LLM backends ----------


def call_groq(
    model: str,
    system: str,
    user: str,
    *,
    temperature: float,
    seed: int | None,
    max_tokens: int,
    reasoning_effort: str | None,
    json_mode: bool,
) -> LLMResult:
    from groq import Groq, RateLimitError

    client = Groq(api_key=os.environ["GROQ_API_KEY"])
    kwargs = {
        "model": model,
        "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
        "temperature": temperature,
        "max_completion_tokens": max_tokens,
    }
    if seed is not None:
        kwargs["seed"] = seed
    if reasoning_effort:
        kwargs["reasoning_effort"] = reasoning_effort
        if "gpt-oss" not in model:
            # Non-gpt-oss reasoning models default to "raw", which inlines <think> into content
            kwargs["reasoning_format"] = "hidden"
    if json_mode:
        kwargs["response_format"] = {"type": "json_object"}

    start = time.monotonic()
    try:
        raw = client.chat.completions.with_raw_response.create(**kwargs)
    except RateLimitError as e:
        # Free tier is 8k TPM; back-to-back passes on one transcript trip it, a single retry clears it
        wait = float(e.response.headers.get("retry-after", 60))
        print(f"   Groq 429, retrying in {wait:.0f}s", flush=True)
        time.sleep(wait)
        raw = client.chat.completions.with_raw_response.create(**kwargs)
    resp = raw.parse()
    return LLMResult(
        model=f"groq:{model}",
        text=(resp.choices[0].message.content or "").strip(),
        latency_s=round(time.monotonic() - start, 1),
        input_tokens=resp.usage.prompt_tokens,
        output_tokens=resp.usage.completion_tokens,
        rate_limits={k: v for k, v in raw.headers.items() if k.startswith("x-ratelimit")},
    )


def call_claude(alias: str, system: str, user: str, *, timeout: int = 600) -> LLMResult:
    # Same isolated runner the product uses, so lab results transfer to `pidcast discord`
    start = time.monotonic()
    run = run_claude(user, alias, system_prompt=system, timeout=timeout)
    return LLMResult(
        model=f"claude:{run.model}",
        text=run.text,
        latency_s=round(time.monotonic() - start, 1),
        input_tokens=run.input_tokens,
        output_tokens=run.output_tokens,
        cost_usd=run.cost_usd,
    )


def call_model(
    spec: str,
    system: str,
    user: str,
    *,
    temperature: float = 0.3,
    seed: int | None = 42,
    max_tokens: int = 3000,
    reasoning_effort: str | None = None,
    json_mode: bool = False,
) -> LLMResult:
    backend, _, model = spec.partition(":")
    # "groq:<model>@<effort>" overrides the preset's reasoning_effort for that model
    model, _, effort_override = model.partition("@")
    reasoning_effort = effort_override or reasoning_effort
    try:
        if backend == "groq":
            result = call_groq(
                model,
                system,
                user,
                temperature=temperature,
                seed=seed,
                max_tokens=max_tokens,
                reasoning_effort=reasoning_effort,
                json_mode=json_mode,
            )
            if reasoning_effort:
                result.model = f"{result.model}@{reasoning_effort}"
            return result
        if backend == "claude":
            return call_claude(model, system, user)
        raise ValueError(f"Unknown backend in model spec: {spec}")
    except Exception as e:  # noqa: BLE001 - one model failing must not sink the comparison
        return LLMResult(model=spec, text="", latency_s=0.0, error=f"{type(e).__name__}: {e}")


def parse_json(text: str) -> dict:
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip())
    match = re.search(r"\{.*\}", text, re.DOTALL)
    return json.loads(match.group(0) if match else text)


# ---------- Transcript handling ----------


def load_transcript(path: Path) -> tuple[dict, str]:
    raw = path.read_text(encoding="utf-8")
    if raw.startswith("---"):
        _, fm, body = raw.split("---", 2)
        return yaml.safe_load(fm) or {}, body.strip()
    return {}, raw.strip()


def cmd_stories(args: argparse.Namespace) -> None:
    transcript = Path(args.transcript).expanduser()
    meta, body = load_transcript(transcript)
    segments = segment(body)
    numbered = "\n".join(f"[{i}] {s}" for i, s in enumerate(segments))
    user = f"Title: {meta.get('title', transcript.stem)}\n\n{numbered}"

    result = call_model(
        args.model, STORIES_SYSTEM, user, temperature=0.0, json_mode=True, reasoning_effort="low"
    )
    if result.error:
        sys.exit(f"Story pass failed: {result.error}")
    stories = [asdict(s) for s in parse_stories(result.text, len(segments))]

    run_dir = RUNS_DIR / transcript.stem
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "segments.json").write_text(json.dumps(segments, ensure_ascii=False, indent=1))
    (run_dir / "stories.json").write_text(
        json.dumps(
            {
                "transcript": str(transcript),
                "meta": meta,
                "model": result.model,
                "usage": {"in": result.input_tokens, "out": result.output_tokens},
                "rate_limits": result.rate_limits,
                "stories": stories,
            },
            ensure_ascii=False,
            indent=2,
            default=str,
        )
    )

    print(
        f"{len(segments)} segments, story pass by {result.model} "
        f"({result.input_tokens} in / {result.output_tokens} out, {result.latency_s}s)"
    )
    if result.rate_limits:
        print("rate limits:", result.rate_limits)
    for s in stories:
        tag = "" if s["kind"] == "story" else f"  [{s['kind']}]"
        print(
            f"{s['id']:>3}. {s['headline']}{tag}\n     {s['gist']}  (lines {s['start']}-{s['end']})"
        )
    print(f"\nRun dir: {run_dir}")


# ---------- Digest pass ----------


def build_digest_prompt(run_dir: Path, picks: list[int], preset: dict) -> tuple[str, str]:
    data = json.loads((run_dir / "stories.json").read_text())
    segments = json.loads((run_dir / "segments.json").read_text())
    by_id = {s["id"]: s for s in data["stories"]}
    missing = [p for p in picks if p not in by_id]
    if missing:
        sys.exit(f"Unknown story ids: {missing}")

    meta = data["meta"]
    blocks = []
    for n, pid in enumerate(picks, 1):
        s = by_id[pid]
        text = " ".join(segments[s["start"] : s["end"] + 1])
        blocks.append(f"### Story {n}: {s['headline']}\n{text}")
    source_line = " | ".join(str(x) for x in (meta.get("channel"), meta.get("date")) if x)
    user = preset["user_prompt"].format(
        source_line=source_line or "Podcast",
        title=meta.get("title", ""),
        n_stories=len(picks),
        stories_block="\n\n".join(blocks),
    )
    return preset["system_prompt"], user


def cmd_digest(args: argparse.Namespace) -> None:
    run_dir = Path(args.run_dir)
    preset_path = Path(args.preset)
    preset = yaml.safe_load(preset_path.read_text())
    picks = [int(x) for x in args.pick.split(",")]
    system, user = build_digest_prompt(run_dir, picks, preset)

    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    out_dir = run_dir / f"{preset['name']}-{preset['version']}_{stamp}"
    out_dir.mkdir(parents=True)
    (out_dir / "prompt.txt").write_text(f"=== SYSTEM ===\n{system}\n=== USER ===\n{user}")

    results = []
    for spec in args.models:
        print(f"-> {spec} ...", flush=True)
        r = call_model(
            spec,
            system,
            user,
            temperature=preset["temperature"],
            seed=preset.get("seed"),
            max_tokens=preset["max_output_tokens"],
            reasoning_effort=preset.get("reasoning_effort"),
        )
        status = r.error or f"{len(r.text)} chars, {r.latency_s}s"
        print(f"   {status}")
        results.append(asdict(r))

    (out_dir / "results.json").write_text(
        json.dumps(
            {
                "preset": str(preset_path),
                "preset_name": preset["name"],
                "preset_version": preset["version"],
                "preset_sha": hashlib.sha256(preset_path.read_bytes()).hexdigest()[:12],
                "picks": picks,
                "budget_chars": preset["budget_chars"],
                "min_chars": preset.get("min_chars", 0),
                "results": results,
            },
            ensure_ascii=False,
            indent=2,
        )
    )

    if not args.no_judge:
        run_judge(out_dir, user)
    write_report(out_dir)
    print(f"\nReport: {out_dir / 'report.md'}")


# ---------- Judge ----------

JUDGE_SYSTEM = """You are a strict editor evaluating Ukrainian Discord recaps of tech news stories.
You get the source excerpts, the style brief, and several anonymized candidates (A, B, ...).
Score each candidate independently, then rank them. Return ONLY JSON:
{"candidates": {"A": {
   "fidelity": 1-5, "factual_errors": ["claim not supported by the source, quoted"],
   "coverage": 1-5, "voice": 1-5, "ukrainian": 1-5, "russianisms_or_calques": ["..."],
   "ai_tells": ["generic/throat-clearing phrases, quoted"], "swear_count": 0,
   "format_ok": true, "overall": 1-10, "verdict": "one sentence"}},
 "ranking": ["A", "B"], "notes": "one or two sentences on what separates the best from the rest"}
Scoring guide:
- fidelity: every number, name, and claim traceable to the source. One invented figure caps it at 2.
- coverage: every selected story present with its key fact, nothing extra.
- voice: Les Podereviansky register - short, flat, dark irony, swearing that lands a line rather than decorating it. Generic AI recap prose scores 1-2.
- ukrainian: natural modern conversational Ukrainian, no Russianisms or English calques.
- format_ok: follows the STRUCTURE/FORM section of the style brief exactly and fits its length cap.
Do not reward length for its own sake. Be harsh; 5 means you would paste it unedited."""


def run_judge(out_dir: Path, digest_user_prompt: str | None = None) -> None:
    data = json.loads((out_dir / "results.json").read_text())
    if digest_user_prompt is None:
        digest_user_prompt = (out_dir / "prompt.txt").read_text().split("=== USER ===\n", 1)[1]
    ok = [r for r in data["results"] if not r["error"] and r["text"]]
    rng = random.Random(42)
    shuffled = ok[:]
    rng.shuffle(shuffled)
    labels = {chr(65 + i): r["model"] for i, r in enumerate(shuffled)}
    cands = "\n\n".join(
        f"=== Candidate {lbl} ({len(r['text'])} chars) ===\n{r['text']}"
        for lbl, r in zip(labels, shuffled, strict=True)
    )
    brief = yaml.safe_load(Path(data["preset"]).read_text())["system_prompt"]
    user = f"# Style brief\n{brief}\n\n# Source\n{digest_user_prompt}\n\n# Candidates\n{cands}"

    if not labels:
        print("   no successful candidates, skipping judge")
        return
    print(f"-> judging {len(labels)} candidates with {JUDGE_MODEL} ...", flush=True)
    r = call_model(JUDGE_MODEL, JUDGE_SYSTEM, user)
    if r.error:
        print(f"   judge failed: {r.error}")
        return
    verdict = parse_json(r.text)
    verdict["labels"] = labels
    verdict["judge_cost_usd"] = r.cost_usd
    (out_dir / "judge.json").write_text(json.dumps(verdict, ensure_ascii=False, indent=2))


def cmd_judge(args: argparse.Namespace) -> None:
    out_dir = Path(args.digest_dir)
    run_judge(out_dir)
    write_report(out_dir)
    print(f"Report: {out_dir / 'report.md'}")


# ---------- Report ----------


def write_report(out_dir: Path) -> None:
    data = json.loads((out_dir / "results.json").read_text())
    judge_path = out_dir / "judge.json"
    judge = json.loads(judge_path.read_text()) if judge_path.exists() else None
    scores = {}
    if judge:
        scores = {model: judge["candidates"].get(lbl, {}) for lbl, model in judge["labels"].items()}

    budget, floor = data["budget_chars"], data["min_chars"]
    lines = [
        f"# {data['preset_name']} {data['preset_version']} (sha {data['preset_sha']})",
        f"Picks: {data['picks']}",
        "",
        "| Model | Chars | Fits | Latency | Tokens in/out | Cost | Overall | Fid | Cov | Voice | UA |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in data["results"]:
        if r["error"]:
            lines.append(f"| {r['model']} | - | ERROR | | | | | | | | |")
            continue
        n = len(r["text"])
        fits = "yes" if floor <= n <= budget else ("over" if n > budget else "short")
        s = scores.get(r["model"], {})
        cost = f"${r['cost_usd']:.3f}" if r["cost_usd"] is not None else "free tier"
        lines.append(
            f"| {r['model']} | {n} | {fits} | {r['latency_s']}s | {r['input_tokens']}/{r['output_tokens']} "
            f"| {cost} | {s.get('overall', '')} | {s.get('fidelity', '')} | {s.get('coverage', '')} "
            f"| {s.get('voice', '')} | {s.get('ukrainian', '')} |"
        )
    if judge:
        ranking = [judge["labels"].get(lbl, lbl) for lbl in judge.get("ranking", [])]
        lines += [
            "",
            f"**Judge ranking:** {' > '.join(ranking)}",
            "",
            judge.get("notes", ""),
            "",
            f"Judge cost: ${judge.get('judge_cost_usd') or 0:.3f}",
        ]

    for r in data["results"]:
        lines += ["", f"## {r['model']}"]
        if r["error"]:
            lines += ["", f"Error: `{r['error']}`"]
            continue
        s = scores.get(r["model"])
        if s:
            lines += ["", f"_Judge:_ {s.get('verdict', '')}"]
            for key in ("factual_errors", "russianisms_or_calques", "ai_tells"):
                if s.get(key):
                    lines.append(f"- {key}: {'; '.join(s[key])}")
            lines.append(f"- swears: {s.get('swear_count')}, format_ok: {s.get('format_ok')}")
        if r.get("rate_limits"):
            lines.append(f"- rate limits: {r['rate_limits']}")
        lines += ["", "```", r["text"], "```"]

    (out_dir / "report.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = parser.add_subparsers(required=True)

    p = sub.add_parser("stories", help="Pass 1: list stories in a transcript")
    p.add_argument("transcript")
    p.add_argument("--model", default="groq:openai/gpt-oss-120b")
    p.set_defaults(func=cmd_stories)

    p = sub.add_parser("digest", help="Pass 2: write the digest for picked stories across models")
    p.add_argument("run_dir")
    p.add_argument("--pick", required=True, help="Comma-separated story ids, in output order")
    p.add_argument("--preset", default=str(DEFAULT_PRESET))
    p.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    p.add_argument("--no-judge", action="store_true")
    p.set_defaults(func=cmd_digest)

    p = sub.add_parser("judge", help="(Re)run the Opus judge on a digest dir")
    p.add_argument("digest_dir")
    p.set_defaults(func=cmd_judge)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
