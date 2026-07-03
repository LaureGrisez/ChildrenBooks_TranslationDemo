#!/usr/bin/env python3
"""Recompute selection/synthesis-bias statistics from translation artifacts.

The program is deliberately dependency-free.  It discovers completed runs below
``translation`` and ``client_presentation`` and prints a Markdown report.  Panel
JSON is preferred when present; older runs are still included in the whole-book
tables.  Malformed/incomplete artifacts are skipped with an explicit warning.
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Any, Iterable

WORD_RE = re.compile(r"[^\W_]+(?:['’][^\W_]+)*", re.UNICODE)
STAMP_RE = re.compile(r"^\d{8}_\d{6}(?:_\d+)?$")
CRITERIA = ("faithfulness", "naturalness", "child_friendliness", "read_aloud",
            "continuity", "glossary_compliance", "structure")


def aggregate_ranking(judgments: list[dict[str, Any]]) -> list[str]:
    """Reproduce production aggregation for leave-one-judge-out analysis."""
    options=list(judgments[0]["overall_ranking"]); totals={o:0.0 for o in options}
    pair={o:0.0 for o in options}; rank={o:0.0 for o in options}; scores={o:0.0 for o in options}
    errors: dict[str,list[str]]=defaultdict(list)
    for result in judgments:
        confidence=float(result.get("confidence",1.0)); order=result["overall_ranking"]
        positions={o:i for i,o in enumerate(order)}; possible=max(1,len(order)-1)
        pair_points={o:0.0 for o in options}
        for left,right in combinations(order,2): pair_points[left if positions[left]<positions[right] else right] += 1/possible
        for option in options:
            pair[option] += pair_points[option]*confidence
            rank[option] += ((len(options)-1-positions[option])/possible)*confidence
            normalized=[]
            for criterion in CRITERIA:
                values={o:float(result["option_scores"][o][criterion]) for o in options}
                low,high=min(values.values()),max(values.values())
                normalized.append(.5 if high==low else (values[option]-low)/(high-low))
            scores[option] += statistics.mean(normalized)*confidence
            errors[option].extend(str(e).strip().casefold() for e in result["option_scores"][option].get("critical_errors",[]) if str(e).strip())
    count=len(judgments); confirmed={o:[e for e,n in Counter(errors[o]).items() if n>=2] for o in options}
    for option in options:
        totals[option]=.45*pair[option]/count+.35*rank[option]/count+.20*scores[option]/count-(1.0 if confirmed[option] else 0.0)
    return sorted(options,key=lambda o:(bool(confirmed[o]),-totals[o],o))


def tokens(text: str) -> list[str]:
    return WORD_RE.findall(text.casefold())


def rouge_l(left: str, right: str) -> float:
    """ROUGE-L F1 over case-folded Unicode word tokens."""
    a, b = tokens(left), tokens(right)
    if not a or not b:
        return float(a == b)
    # One-row LCS keeps memory bounded for whole books.
    row = [0] * (len(b) + 1)
    for x in a:
        previous = 0
        for j, y in enumerate(b, 1):
            old = row[j]
            row[j] = previous + 1 if x == y else max(row[j], row[j - 1])
            previous = old
    lcs = row[-1]
    precision, recall = lcs / len(b), lcs / len(a)
    return 2 * precision * recall / (precision + recall) if lcs else 0.0


def paragraphs(text: str) -> list[str]:
    return [re.sub(r"\s+", " ", p).strip() for p in re.split(r"\n\s*\n", text.strip()) if p.strip()]


def model_id(path: Path) -> str:
    name = path.stem.casefold()
    rules = (("gpt5_5", "GPT-5.5"), ("gpt-5.5", "GPT-5.5"),
             ("gpt4o", "GPT-4o"), ("gpt-4o", "GPT-4o"),
             ("claude", "Claude Sonnet 4.6"), ("gemini", "Gemini"),
             ("google", "Google Translate"))
    for needle, label in rules:
        if needle in name:
            return label
    return path.stem


def label(candidate: str) -> str:
    c = candidate.casefold()
    if "gpt5" in c or "gpt-5.5" in c: return "GPT-5.5"
    if "gpt4" in c or "gpt-4o" in c: return "GPT-4o"
    if "claude" in c: return "Claude Sonnet 4.6"
    if "gemini" in c: return "Gemini"
    if "google" in c: return "Google Translate"
    return candidate


def judge_label(model: str) -> str:
    m = model.casefold()
    if "claude" in m: return "Claude Sonnet 4.6"
    if "gpt-5.5" in m or "gpt5" in m: return "GPT-5.5"
    if "gemini-3.5" in m: return "Gemini 3.5 Flash"
    if "gemini-3.1" in m: return "Gemini 3.1 Pro Preview"
    if "gemini-2.5" in m: return "Gemini 2.5 Flash"
    return model


def same_family(judge: str, candidate: str) -> bool:
    return ((judge.startswith("Gemini") and candidate == "Gemini") or
            (judge.startswith("Claude") and candidate == "Claude Sonnet 4.6") or
            (judge == "GPT-5.5" and candidate in {"GPT-5.5", "GPT-4o"}))


@dataclass
class Run:
    path: Path
    language: str
    final: str
    candidates: dict[str, str]
    panel: dict[str, Any] | None

    @property
    def workflow(self) -> str:
        parts = self.path.parts
        named = {
            "version_1_text_image_summaries": "Text + image summaries",
            "version_2_multimodal_two_candidates": "Multimodal, two candidates",
            "version_3_full_stack": "Full multimodal stack",
            "version_4_creative_full_stack": "Creative full stack",
        }
        for part, title in named.items():
            if part in parts:
                return title
        if "client_presentation" in parts:
            return "Preprocessed panel"
        if self.panel:
            return ("Preprocessed five-candidate panel" if len(self.panel.get("finals", {})) == 31
                    else "Early five-candidate panel")
        return "Historical workflow"

    @property
    def final_paragraphs(self) -> list[str]:
        if self.panel and self.panel.get("finals"):
            return list(self.panel["finals"].values())
        return paragraphs(self.final)


def load_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def discover(roots: Iterable[Path]) -> tuple[list[Run], list[str]]:
    runs, warnings = [], []
    seen: set[Path] = set()
    for root in roots:
        if not root.exists():
            warnings.append(f"missing root: {root}")
            continue
        for cdir in root.rglob("candidates"):
            base = cdir.parent.resolve()
            if base in seen: continue
            seen.add(base)
            candidate_files = sorted(cdir.glob("*.txt"))
            final_files = [p for p in base.glob("*.txt") if p.is_file()]
            if not candidate_files or not final_files:
                warnings.append(f"incomplete run skipped: language={base.parent.name}, source={root.name}")
                continue
            # There should be one; newest mtime is safest for format migrations.
            final_path = max(final_files, key=lambda p: p.stat().st_mtime)
            try:
                candidates = {model_id(p): p.read_text(encoding="utf-8") for p in candidate_files}
                final = final_path.read_text(encoding="utf-8")
                panel_dir = base / "panel"
                panel = None
                required = ["aggregates.json", "judge_results.json", "final_paragraphs.json"]
                if all((panel_dir / p).exists() for p in required):
                    panel = {"aggregates": load_json(panel_dir / "aggregates.json"),
                             "judges": load_json(panel_dir / "judge_results.json"),
                             "finals": load_json(panel_dir / "final_paragraphs.json")}
                    alignment = panel_dir / "alignment.json"
                    if alignment.exists():
                        panel["alignment"] = load_json(alignment)
                # Current layouts put the ISO language directory immediately
                # above the timestamp, regardless of any workflow prefix.
                language = base.parent.name
                runs.append(Run(base, language, final, candidates, panel))
            except (OSError, UnicodeError, json.JSONDecodeError) as exc:
                warnings.append(f"unreadable run skipped: language={base.parent.name}, source={root.name}: {exc}")
    return sorted(runs, key=lambda r: str(r.path)), warnings


def candidate_paragraphs(run: Run) -> dict[str, list[str]]:
    return {name: paragraphs(text) for name, text in run.candidates.items()}


def nearest(text: str, choices: dict[str, str]) -> tuple[str, float]:
    scored = [(rouge_l(text, value), key) for key, value in choices.items()]
    # max() is stable: an exact tie is attributed to the first artifact seen.
    # Duplicate-adjusted selection statistics below remove this ordering effect.
    score, key = max(scored, key=lambda x: x[0])
    return key, score


def mean_pairs(texts: list[str]) -> tuple[float, float, float] | None:
    scores = [rouge_l(a, b) for a, b in combinations(texts, 2)]
    return (statistics.mean(scores), min(scores), max(scores)) if scores else None


def valid_judgments(run: Run, pid: str) -> list[tuple[str, dict[str, Any]]]:
    assert run.panel
    output = []
    for record in run.panel["judges"].get(pid, {}).values():
        result = record.get("result")
        if record.get("status") == "ok" and isinstance(result, dict) and result.get("overall_ranking"):
            output.append((judge_label(str(record.get("model", "unknown"))), result))
    return output


def split_credit(winner: str, texts: dict[str, str]) -> dict[str, float]:
    same = [c for c, text in texts.items() if text.strip() == texts[winner].strip()]
    return {c: 1 / len(same) for c in same}


def render(runs: list[Run], warnings: list[str]) -> str:
    out = ["# Automatically computed translation-bias statistics", ""]
    out += [f"Discovered **{len(runs)} completed runs**: " + ", ".join(f"{k}={v}" for k,v in sorted(Counter(r.language for r in runs).items())) + ".", ""]

    out += ["## Final-to-final similarity", "", "| Language | Paragraphs | Runs | Mean ROUGE-L | Range |", "|---|---:|---:|---:|---:|"]
    cohorts: dict[tuple[str,int], list[Run]] = defaultdict(list)
    for run in runs: cohorts[(run.language, len(run.final_paragraphs))].append(run)
    for (lang, count), group in sorted(cohorts.items()):
        value = mean_pairs([r.final for r in group])
        if value: out.append(f"| {lang} | {count} | {len(group)} | {value[0]:.3f} | {value[1]:.3f}–{value[2]:.3f} |")
        else: out.append(f"| {lang} | {count} | {len(group)} | not measurable | — |")

    out += ["", "## Candidate and synthesis similarity by language/workflow", "", "| Language | Workflow | Runs | Paragraphs | Candidate-pool mean | Closest final candidate(s) | Final-to-candidate ROUGE-L |", "|---|---|---:|---:|---:|---|---:|"]
    workflow_groups: dict[tuple[str,str,int], list[Run]] = defaultdict(list)
    for run in runs: workflow_groups[(run.language, run.workflow, len(run.final_paragraphs))].append(run)
    for (lang, workflow, pcount), group in sorted(workflow_groups.items()):
        pools = [mean_pairs(list(r.candidates.values()))[0] for r in group]
        nearest_values = [nearest(r.final, r.candidates) for r in group]
        nearest_counts = Counter(name for name, _ in nearest_values)
        out.append(f"| {lang} | {workflow} | {len(group)} | {pcount} | {statistics.mean(pools):.3f} | "
                   f"{', '.join(f'{n} ({c})' for n,c in nearest_counts.most_common())} | {statistics.mean(s for _,s in nearest_values):.3f} |")

    out += ["", "### Final similarity to every candidate", "", "| Language | Workflow | Candidate | Runs present | Mean final ROUGE-L |", "|---|---|---|---:|---:|"]
    candidate_group_scores: dict[tuple[str,str,str], list[float]] = defaultdict(list)
    for run in runs:
        for candidate, text in run.candidates.items():
            candidate_group_scores[(run.language, run.workflow, candidate)].append(rouge_l(run.final, text))
    for (lang, workflow, candidate), scores in sorted(candidate_group_scores.items()):
        out.append(f"| {lang} | {workflow} | {candidate} | {len(scores)} | {statistics.mean(scores):.3f} |")

    panel_runs = [r for r in runs if r.panel]
    agreement = total_aligned = toward = away = gpt_first = non_gpt_first = duplicates = eligible_paras = 0
    wins, adjusted_wins, eligible = Counter(), Counter(), Counter()
    judge_n, judge_agree, judge_conf = Counter(), Counter(), defaultdict(list)
    choice_credit, choice_exposure = Counter(), Counter()
    pivotal_removals, pivotal_model, pivotal_semantic = Counter(), Counter(), Counter()
    pivotal_by_size: dict[tuple[int,str], list[int]] = defaultdict(lambda:[0,0,0])
    family_raw, family_eligible = Counter(), Counter()
    per_run_rows = []
    for run in panel_runs:
        cps = candidate_paragraphs(run)
        # A failed-provider file can contain a one-line error while panel JSON
        # still has usable decisions.  Such a run belongs in selection stats,
        # but cannot support final-to-candidate paragraph attribution.
        if any(len(ps) != len(run.panel["finals"]) for ps in cps.values()):
            continue
        aligned_options = {item["paragraph_id"]: item.get("options", {})
                           for item in run.panel.get("alignment", []) if isinstance(item, dict)}
        firsts, nears = Counter(), Counter()
        exact_gpt = 0
        for pid, agg in run.panel["aggregates"].items():
            idx = int(re.sub(r"\D", "", pid)) - 1
            if pid not in run.panel["finals"] or any(idx >= len(v) for v in cps.values()): continue
            raw_options = aligned_options.get(pid)
            texts = ({label(name): str(text) for name, text in raw_options.items()}
                     if raw_options else {name: ps[idx] for name, ps in cps.items()})
            winner = label(agg["ranking"][0]); final = run.panel["finals"][pid]
            nearest_name, _ = nearest(final, texts); nearest_name = label(nearest_name)
            total_aligned += 1; firsts[winner] += 1; nears[nearest_name] += 1
            agreement += winner == nearest_name
            if winner == "GPT-5.5":
                gpt_first += 1; away += nearest_name != "GPT-5.5"
            else:
                non_gpt_first += 1; toward += nearest_name == "GPT-5.5"
            # Use the normalized paragraphs from the persisted candidate book;
            # alignment snapshots preserve line wrapping that final JSON omits.
            if "GPT-5.5" in cps and final.strip() == cps["GPT-5.5"][idx].strip(): exact_gpt += 1
            eligible_paras += 1
            for name in texts: eligible[label(name)] += 1
            wins[winner] += 1
            if len({v.strip() for v in texts.values()}) < len(texts): duplicates += 1
            for name, credit in split_credit(winner, texts).items(): adjusted_wins[label(name)] += credit
            for jname, result in valid_judgments(run, pid):
                judge_n[jname] += 1; judge_conf[jname].append(float(result.get("confidence",1)))
                judge_agree[jname] += label(result["overall_ranking"][0]) == winner
                for candidate in texts: choice_exposure[(jname,label(candidate))] += 1
                jwinner = label(result["overall_ranking"][0])
                for name, credit in split_credit(jwinner, texts).items(): choice_credit[(jname,label(name))] += credit
        per_run_rows.append((run, firsts, nears, exact_gpt))

    # Selection denominators come from each stored panel decision, including
    # the historical run whose failed Gemini file prevents book alignment.
    # alignment.json is the authoritative snapshot of candidates actually seen.
    wins, adjusted_wins, eligible = Counter(), Counter(), Counter()
    judge_n, judge_agree, judge_conf = Counter(), Counter(), defaultdict(list)
    choice_credit, choice_exposure = Counter(), Counter()
    family_raw, family_eligible = Counter(), Counter()
    duplicates = eligible_paras = 0
    for run in panel_runs:
        aligned = {item["paragraph_id"]: item.get("options", {})
                   for item in run.panel.get("alignment", []) if isinstance(item, dict)}
        for pid, agg in run.panel["aggregates"].items():
            texts = {label(k): str(v) for k, v in aligned.get(pid, {}).items()}
            candidates_here = [label(c) for c in agg.get("ranking", [])]
            if not candidates_here:
                continue
            winner = candidates_here[0]
            eligible_paras += 1; wins[winner] += 1
            for name in candidates_here: eligible[name] += 1
            if texts and len({v.strip() for v in texts.values()}) < len(texts): duplicates += 1
            credits = split_credit(winner, texts) if winner in texts else {winner: 1.0}
            for name, credit in credits.items(): adjusted_wins[name] += credit
            for jname, result in valid_judgments(run, pid):
                judge_n[jname] += 1; judge_conf[jname].append(float(result.get("confidence",1)))
                judge_agree[jname] += label(result["overall_ranking"][0]) == winner
                for candidate in candidates_here: choice_exposure[(jname,candidate)] += 1
                jwinner = label(result["overall_ranking"][0])
                if any(same_family(jname, candidate) for candidate in candidates_here):
                    family_eligible[jname] += 1
                    family_raw[jname] += same_family(jname, jwinner)
                jcredits = split_credit(jwinner, texts) if jwinner in texts else {jwinner: 1.0}
                for name, credit in jcredits.items(): choice_credit[(jname,name)] += credit
            judgments = valid_judgments(run, pid)
            raw_results = [result for _, result in judgments]
            for removed, (jname, _) in enumerate(judgments):
                remaining = [result for index, result in enumerate(raw_results) if index != removed]
                if not remaining:
                    continue
                recomputed = aggregate_ranking(remaining)[0]
                stored = agg["ranking"][0]
                pivotal_removals[jname] += 1
                sized=pivotal_by_size[(len(judgments),jname)]; sized[0] += 1
                if recomputed != stored:
                    pivotal_model[jname] += 1
                    sized[1] += 1
                    raw_texts = aligned.get(pid, {})
                    if not raw_texts or raw_texts.get(recomputed, "").strip() != raw_texts.get(stored, "").strip():
                        pivotal_semantic[jname] += 1
                        sized[2] += 1

    out += ["", "## Panel selection and synthesis by language/workflow", "", "| Language | Workflow | Runs | First-ranked options | Nearest final paragraphs | Exact GPT-5.5 copies |", "|---|---|---:|---|---|---:|"]
    fmt = lambda c: ", ".join(f"{k} {v}" for k,v in c.most_common())
    panel_groups: dict[tuple[str,str], tuple[int,Counter,Counter,int]] = {}
    for run, firsts, nears, exact in per_run_rows:
        key=(run.language,run.workflow); n, f, nr, ex=panel_groups.get(key,(0,Counter(),Counter(),0))
        panel_groups[key]=(n+1,f+firsts,nr+nears,ex+exact)
    for (lang, workflow), (count, firsts, nears, exact) in sorted(panel_groups.items()):
        out.append(f"| {lang} | {workflow} | {count} | {fmt(firsts)} | {fmt(nears)} | {exact} |")
    if total_aligned:
        out += ["", f"Panel first rank equals nearest final: **{agreement}/{total_aligned} ({agreement/total_aligned:.1%})**.",
                f"GPT-5.5 first-ranked: **{gpt_first}/{total_aligned} ({gpt_first/total_aligned:.1%})**.",
                f"When GPT-5.5 was not first, synthesis moved toward it: **{toward}/{non_gpt_first} ({toward/non_gpt_first:.1%})**.",
                f"When GPT-5.5 was first, synthesis moved away: **{away}/{gpt_first} ({away/gpt_first:.1%})**."]

    out += ["", "## Exposure-normalized selection", "", "| Candidate | Eligible paragraphs | Raw wins | Duplicate-adjusted wins | Adjusted win rate | Adjusted first-place votes / exposures |", "|---|---:|---:|---:|---:|---:|"]
    for name in sorted(eligible, key=lambda n: (-eligible[n], n)):
        votes=sum(v for (j,c),v in choice_credit.items() if c==name); exposures=sum(v for (j,c),v in choice_exposure.items() if c==name)
        out.append(f"| {name} | {eligible[name]} | {wins[name]} | {adjusted_wins[name]:.2f} | {adjusted_wins[name]/eligible[name]:.1%} | {votes:.2f}/{exposures} ({votes/exposures:.1%}) |")
    if eligible_paras: out += ["", f"Strictly identical candidates occur in **{duplicates}/{eligible_paras} paragraphs ({duplicates/eligible_paras:.1%})**."]

    out += ["", "## Judge behavior", "", "| Judge | Valid judgments | Aggregate agreement | Same-family candidate first / eligible | Mean confidence |", "|---|---:|---:|---:|---:|"]
    for name in sorted(judge_n):
        den=family_eligible[name]; num=family_raw[name]
        family_cell=f"{num}/{den} ({num/den:.1%})" if den else "not available"
        out.append(f"| {name} | {judge_n[name]} | {judge_agree[name]/judge_n[name]:.1%} | {family_cell} | {statistics.mean(judge_conf[name]):.3f} |")
    candidates = sorted(eligible)
    out += ["", "### Conditional duplicate-adjusted first choices", "", "| Judge | " + " | ".join(candidates) + " |", "|---|" + "---:|"*len(candidates)]
    for judge in sorted(judge_n):
        cells=[]
        for candidate in candidates:
            den=choice_exposure[(judge,candidate)]; num=choice_credit[(judge,candidate)]
            cells.append(f"{num/den:.1%} ({num:.2f}/{den})" if den else "not present")
        out.append("| " + judge + " | " + " | ".join(cells) + " |")

    out += ["", "### Leave-one-judge-out influence", "", "| Judge | Removals | Model-winner changes | Semantic winner changes | Semantic pivotal rate | Mean confidence |", "|---|---:|---:|---:|---:|---:|"]
    for judge in sorted(pivotal_removals):
        n=pivotal_removals[judge]
        out.append(f"| {judge} | {n} | {pivotal_model[judge]} | {pivotal_semantic[judge]} | {pivotal_semantic[judge]/n:.1%} | {statistics.mean(judge_conf[judge]):.3f} |")

    out += ["", "### Influence by panel size", "", "| Judges in panel | Judge removed | Removals | Model changes | Semantic changes |", "|---:|---|---:|---:|---:|"]
    for (size,judge),(n,model_changes,semantic_changes) in sorted(pivotal_by_size.items()):
        out.append(f"| {size} | {judge} | {n} | {model_changes} ({model_changes/n:.1%}) | {semantic_changes} ({semantic_changes/n:.1%}) |")

    if warnings:
        out += ["", "## Warnings", ""] + [f"- {w}" + (f" ({count} runs)" if count > 1 else "") for w,count in Counter(warnings).items()]
    out += ["", "ROUGE-L uses case-folded Unicode word tokens. Exact-duplicate correction strips outer whitespace only. Runs are grouped by language and observed paragraph count; no language list is hard-coded for reporting.", ""]
    return "\n".join(out)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", action="append", type=Path, help="artifact root (repeatable)")
    parser.add_argument("--output", type=Path, help="write Markdown here instead of stdout")
    args = parser.parse_args()
    roots = args.root or [Path("translation"), Path("client_presentation")]
    runs, warnings = discover(roots)
    if not runs:
        print("error: no completed runs found", file=sys.stderr); return 2
    report = render(runs, warnings)
    if args.output: args.output.write_text(report, encoding="utf-8")
    else: print(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
