"""Run all evaluation suites and produce a combined summary.

Usage:
    python -m evals.run_all                  # both suites
    python -m evals.run_all --suite chat     # chat only
    python -m evals.run_all --suite byod     # byod only
"""

import argparse
import importlib
import sys
import traceback


SUITES = {
    "chat": "evals.eval_chat",
    "byod": "evals.eval_byod",
}


def main():
    parser = argparse.ArgumentParser(description="Run evaluation suites")
    parser.add_argument(
        "--suite",
        choices=["chat", "byod", "all"],
        default="all",
        help="Which suite to run (default: all)",
    )
    args = parser.parse_args()

    suites = list(SUITES.keys()) if args.suite == "all" else [args.suite]
    failed = []

    for name in suites:
        print(f"\n{'#' * 60}")
        print(f"# Running: {name}")
        print(f"{'#' * 60}\n")
        try:
            mod = importlib.import_module(SUITES[name])
            mod.main()
        except Exception as exc:
            print(f"\n[FAILED] {name}: {exc}")
            traceback.print_exc()
            failed.append(name)

    print(f"\n{'=' * 60}")
    if failed:
        print(f"DONE — {len(failed)} suite(s) failed: {', '.join(failed)}")
        sys.exit(1)
    else:
        print(f"DONE — all {len(suites)} suite(s) passed.")


if __name__ == "__main__":
    main()
