"""
run_demo.py
───────────
Course Concept: Demonstration of Agentic Architecture Scenarios (Units 3 & 4).

Group 4: Aryan Maradwar, Asta Joseph, Dhiraj Sarawale, Ridhima Maliya

Three scripted scenarios for professor evaluation and viva demonstration:
  Scenario 1: Clean Base Run
              - Original 9-node LangGraph pipeline
              - Deterministic calculation + ReportLab PDF generation
  Scenario 2: Missing-Data Feedback Loop
              - Unit 3 Act-Observe-Reflect-Adjust cycle
              - Resolves missing amounts from data/answers.json
              - Recalculates category and grand totals
  Scenario 3: Long-Term Memory & User Correction
              - Adds a user correction to ChromaDB vector store
              - Re-runs pipeline showing memory lookup hit and LLM skip
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path

# Ensure UTF-8 output on Windows
if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf8"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def banner(title: str):
    print("\n" + "=" * 75)
    print(f"   {title}")
    print("=" * 75 + "\n")


def run_command(cmd: list[str], description: str):
    banner(f"DEMO STEP: {description}")
    print(f"Command: {' '.join(cmd)}\n")
    start = time.time()
    env = dict(sys.modules["os"].environ)
    env["PYTHONIOENCODING"] = "utf-8"
    result = subprocess.run(cmd, env=env)
    elapsed = time.time() - start
    print(f"\n[Completed in {elapsed:.2f}s with exit code {result.returncode}]\n")
    if result.returncode != 0:
        print(f"WARNING: Command returned non-zero exit code {result.returncode}")
    return result.returncode


def scenario_1():
    banner("SCENARIO 1: Clean Base Run (Phase 0 Base Pipeline)")
    print("Concept: 9-node LangGraph DAG with deterministic calculation & PDF publishing.")
    print("Expected: 20 rows ingested, Flights=₹2,075, Grand Total=₹3,530.00, 2 exceptions audited.\n")
    cmd = [sys.executable, "main.py", "--mode", "base"]
    run_command(cmd, "Running Base LangGraph Pipeline")


def scenario_2():
    banner("SCENARIO 2: Missing-Data Feedback Loop (Unit 3)")
    print("Concept: Act-Observe-Reflect-Adjust cycle with retry budget.")
    print("Expected: Detects missing amounts for Lyft & Parking Authority, resolves them via answers.json,")
    print("          and dynamically recalculates Grand Total to ₹3,653.25.\n")
    cmd = [
        sys.executable,
        "main.py",
        "--mode",
        "extended",
        "--auto-approve",
        "--answers",
        "data/answers.json",
    ]
    run_command(cmd, "Running Extended Pipeline with Feedback Loop")


def scenario_3():
    banner("SCENARIO 3: Long-Term Memory & User Correction (Unit 3 & 4)")
    print("Concept: ChromaDB vector store with BAAI/bge-small-en-v1.5 embeddings.")
    print("Part A: Save human correction for 'Olive Garden=Meals' into episodic memory.")
    print("Part B: Re-run pipeline showing memory match with high similarity and LLM skip.\n")

    # Step A: Save correction
    cmd_a = [sys.executable, "main.py", "--correct", "Olive Garden=Meals"]
    run_command(cmd_a, "Part A: Registering User Correction in Long-Term Memory")

    time.sleep(1)

    # Step B: Re-run pipeline
    cmd_b = [
        sys.executable,
        "main.py",
        "--mode",
        "extended",
        "--auto-approve",
        "--answers",
        "data/answers.json",
    ]
    run_command(cmd_b, "Part B: Re-running Extended Pipeline with Vector Memory Active")


def main():
    parser = argparse.ArgumentParser(description="Expense Report Agent Demo Runner")
    parser.add_argument(
        "--scenario",
        type=int,
        choices=[1, 2, 3],
        default=None,
        help="Run a specific scenario (1, 2, or 3). If omitted, runs all 3 sequentially.",
    )
    args = parser.parse_args()

    banner("EXPENSE REPORT AGENT — AIA_7 GROUP 4 VIVA DEMONSTRATION")
    print("Authors: Aryan Maradwar · Asta Joseph · Dhiraj Sarawale · Ridhima Maliya")
    print("Platform: Fully Local, Open-Source (LangGraph, ChromaDB, Sentence-Transformers, Ollama)\n")

    if args.scenario == 1:
        scenario_1()
    elif args.scenario == 2:
        scenario_2()
    elif args.scenario == 3:
        scenario_3()
    else:
        print("Running all 3 viva demo scenarios in sequence...\n")
        scenario_1()
        input("\n>>> Press [ENTER] to proceed to Scenario 2 (Missing-Data Feedback Loop)... ")
        scenario_2()
        input("\n>>> Press [ENTER] to proceed to Scenario 3 (Long-Term Memory & Correction)... ")
        scenario_3()
        banner("ALL DEMO SCENARIOS COMPLETE")


if __name__ == "__main__":
    main()
