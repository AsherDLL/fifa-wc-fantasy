"""CLI entry point.

    python -m fifa_fantasy.live --recommendation results/<host>_recommendation_GROUP_MD1_<date>.json

Reads the squad + lineup from the recommendation JSON, joins with the
latest live data, and produces a markdown report under results/.
"""

from __future__ import annotations

import argparse
import json
import re
import socket
from datetime import datetime, timezone
from pathlib import Path

from .captain import build_playbook, live_recommendation
from .report import render_captain, render_round_summary, render_subs
from .state import load_live_state
from .subs import recommend_subs

DEFAULT_RESULTS = Path("results")


def _hostname() -> str:
    raw = socket.gethostname() or "unknown"
    return re.sub(r"[^A-Za-z0-9_.-]", "_", raw)


def main() -> None:
    parser = argparse.ArgumentParser(prog="fifa_fantasy.live")
    parser.add_argument("--recommendation", type=Path, required=True,
                        help="recommendation JSON saved by python -m fifa_fantasy.optimizer")
    parser.add_argument("--round", type=int, default=None,
                        help="override the auto-detected current round")
    parser.add_argument("--captain-only", action="store_true")
    parser.add_argument("--subs-only", action="store_true")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_RESULTS)
    args = parser.parse_args()

    rec = json.loads(args.recommendation.read_text())
    squad_ids = list(rec["squad_player_ids"])
    starter_ids = list(rec["lineup"]["starter_ids"])
    bench_order = list(rec["lineup"]["bench_ids_priority_order"])
    captain_id = int(rec["lineup"]["captain_id"])
    stage = rec["stage"]

    state = load_live_state(
        squad_player_ids=squad_ids,
        target_round=args.round,
    )
    any_completed = (state.players["match_status"] == "completed").any()

    playbook = build_playbook(state, starter_ids)
    live_rec = live_recommendation(state, starter_ids, captain_id) if any_completed else None
    subs_advice = recommend_subs(state, starter_ids, bench_order)

    sections = [render_round_summary(state)]
    if not args.subs_only:
        sections.append(render_captain(playbook, live_rec))
    if not args.captain_only:
        sections.append(render_subs(subs_advice))

    mode = "live" if any_completed else "playbook"
    title = (f"# Live decisions ({mode}) - {stage} round {state.round_id}\n\n"
             f"Source recommendation: `{args.recommendation.name}`\n")
    body = title + "\n" + "\n".join(sections)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%SZ")
    out_path = args.out_dir / f"{_hostname()}_live_{stage}_R{state.round_id}_{ts}.md"
    out_path.write_text(body)
    print(f"live report written -> {out_path}")
    print()
    print(body)


if __name__ == "__main__":
    main()
