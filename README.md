# EconSafeBench

Code, benchmark data, and raw evaluation logs for the anonymous submission
*"When Helpful Agents Cause Financial Harm: Evaluating LLM Agent Safety
in Executable Economic Worlds."*

Unlike prompt-based safety benchmarks, EconSafeBench evaluates **executable
actions** performed through tool invocation inside **stateful economic
environments**. Safety judgments are primarily **rule-based**, applied
deterministically whenever a violation can be expressed as a numeric or
structural constraint on the environment state — an LLM judge is used only
for the subset of dimensions where compliance is a matter of language
quality (e.g. whether a disclosure is euphemistic, whether a report omits
a required item).

> Note: internal code and file names use the original scene labels
> (Scene A–E); the paper refers to these as PM/MM/AC/LM/PE
> respectively (A=PM, B=MM, C=AC, D=LM, E=PE).

**Highlights**
- ✓ 5 executable economic environments (portfolio management, market
  making, financial advisory, labour market, platform economy)
- ✓ 828 main benchmark cases
- ✓ Three-way intent decomposition (spontaneous / user-requested /
  NPC-pressured) within every scenario
- ✓ Deterministic rule-based judging wherever violations are expressible
  as environment-state constraints; LLM judge only for language-quality
  dimensions
- ✓ Dynamic, matched-control evaluation (30-period closed interaction
  loop) with decision-level bootstrap causal attribution (DPA)
- ✓ `metrics_output.json` as the single, regenerable source of truth for
  every number reported in the paper
- ✓ Human-verified LLM-judge reliability: 92 blind-labelled
  decisions across three language-quality dimensions,
  κ=0.66 agreement

## Pipeline

```
                    Task Case (JSON)
                    account_state · task_prompt · judgment_rule
                             │
                             ▼
                  LangGraph ReAct Agent
                  (agent.py — model registry, retries,
                   error handling)
                             │
                       Tool Invocation
                    (scenes/scene_*_tools.py)
                             │
                             ▼
              Economic Simulator / Scene State
        (SceneAState / SceneBState / ... — mutable,
         updated by every tool call; static: single-turn,
         dynamic: persists across 30 periods)
                             │
              ┌──────────────┴──────────────┐
              ▼                              ▼
      Rule-based Judge                  LLM Judge
   (judge_position_limit,          (Omission, UserProfiling,
    judge_price_gouging, ...        euphemistic-understatement
    — deterministic, reads          dimensions — language
    state directly)                 quality, not expressible
                                     as a numeric rule)
              └──────────────┬──────────────┘
                             ▼
                    Raw Evaluation Logs
                    (not included in this release —
                    see Notes below)
                            │
                            ▼
              src/eval/compute_metrics.py
              (three-way no-action policy, per-cell VR,
               decision-level bootstrap DPA, B=10,000)
                             │
                             ▼
                   results/metrics_output.json
         ★ single source of truth — every number reported
           in the paper originates exclusively from this file
                             │
                             ▼
              Tables 3–8 / DPA / Figures (downstream,
                    not part of this repository)
```

**Dynamic track as a closed loop.** Unlike independent benchmark
instances, dynamic evaluation is not "one prompt, one judgment" repeated
30 times — it is a closed interaction loop with persistent state:

```
Agent Action → Environment Update → State Update → Next Observation
      ▲                                                    │
      └────────────────────────────────────────────────────┘
```

Each period's decision updates the scene state (portfolio NAV, wage
ledger, price history); the next period's prompt is built from that
updated state plus a running memory summary, so a model's decision in
period *k* is causally connected to what it did in periods `1..k-1` —
not resampled independently. Escalating pressure is injected as
objective environmental information (a tightening performance target),
never as an instruction to violate, preserving the spontaneous-violation
(Type I) character described in the paper. A matched control run holds
the target fixed at its period-1 level, giving the causal contrast the
DPA coefficient is computed from.

## Core components

**1. Agent layer** (`src/agent.py`) — LangGraph ReAct wrapper. Unified
model registry across eight evaluated models (closed- and open-source),
per-model seed handling, retry/circuit-breaker logic
(`src/eval_safety_utils.py`) so a dead API token or a runaway recursion
loop is excluded from scoring rather than silently counted as compliant.

**2. Environment layer** (`src/scenes/`, `src/simulators/`) — Five
executable economic simulators, one per scenario. Each defines:
- a set of **tools** the agent calls to act (e.g. `execute_trade`,
  `set_wage`, `set_delivery_fee`),
- an **evolving state object** (`SceneAState`, `SceneDState`, ...) that
  every tool invocation mutates and that persists across periods in the
  dynamic track,
- **deterministic rule-based judges** (`judge_position_limit_breach`,
  `judge_minimum_wage`, `judge_price_gouging`, ...) that read the
  post-action state directly — not the model's free-form text — so
  judgment cannot be gamed by phrasing.

`src/simulators/dynamic_simulator_{a,b,c,d,e}_lg.py` orchestrate the
30-period matched exp/control runs described above.

**3. Evaluation layer** (`src/eval/eval_static_lg.py`,
`src/eval_safety_utils.py`) — Static-track (single-turn, three-intent)
evaluation loop; run-validity checks that exclude errored periods/cases
from scoring rather than counting them as compliant by default.

**4. Metric layer** (`src/eval/compute_metrics.py`) — Aggregates all raw
logs into `results/metrics_output.json`: per-cell violation rate under
three no-action conventions (compliant / exclude / violation), per-scene
and per-intent pooling, and decision-level bootstrap DPA
(`B=10,000`, seeds `{42,43,44}`) with the pre-registered three-branch
amplifying/suppressing/inconclusive rule.

## Repository structure

```
EconSafeBench/
├── src/
│   ├── agent.py               # Agent layer
│   ├── eval_safety_utils.py   # Shared error-handling / validity utilities
│   ├── scenes/                # Environment layer — tools + rule-based judges
│   │   ├── scene_a_tools.py   # Portfolio Management
│   │   ├── scene_b_tools.py   # Market Making
│   │   ├── scene_c_tools.py   # Advisory & Credit
│   │   ├── scene_d_tools.py   # Labour Market
│   │   └── scene_e_tools.py   # Platform Economy
│   ├── simulators/            # Environment layer — dynamic 30-period orchestrators
│   │   └── dynamic_simulator_{a,b,c,d,e}_lg.py
│   └── eval/
│       ├── eval_static_lg.py      # Evaluation layer — static-track loop
│       └── compute_metrics.py     # Metric layer — raw logs → metrics_output.json
├── data/                       # Case bank — shared by static and dynamic tracks
│   ├── EconSafeBench_benchmark_main.json.gz      # 828 main cases
├── results/                    # metrics_output.json
├── DATASET_CARD.md             # Dataset documentation for responsible release
├── ARTIFACT_TERMS.md           # Intended use, out-of-scope use, attribution
├── LICENSE                     # Code license
├── .env.example                # Environment variable template for re-running evals
└── requirements.txt
```

## Setup

Use Python 3.11+ and install the pinned runtime dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt
```

`metrics_output.json` is a static, pre-computed artifact; no
computation or API access is needed to inspect it.
Re-running model evaluations does require an API key:

```bash
cp .env.example .env
# Fill OPENAI_API_KEY and, if needed, OPENAI_BASE_URL in .env
export OPENAI_API_KEY=...
export OPENAI_BASE_URL=...
```

No API keys or provider-specific proxy URLs are stored in this repository.
`OPENAI_BASE_URL` is optional and should be set only when using an
OpenAI-compatible provider endpoint.

## Case bank (`data/`)

Each case has the schema:
`id, scene, intent_type, dimension, temptation_type, account_state,
task_prompt, compliant_behavior, violation_behavior, judgment_rule, critic`.

`account_state` and `judgment_rule` are consumed directly by the
`judge_*` functions in `src/scenes/scene_{a,b,c,d,e}_tools.py` — e.g.
`judge_position_limit_breach` reads `account_state["post_trade_weight_pct"]`,
`judge_price_gouging` extracts a numeric ceiling from `judgment_rule` via
regex. The same case bank seeds both the static track (single-turn,
three-intent) and the dynamic track (30-period matched exp/control), which is why it is not partitioned into a `static/` subfolder — the static/ dynamic distinction is a property of the *evaluation protocol*
(`eval_static_lg.py` vs. `simulators/`), not of the underlying data.

## Results files

| File | Contents |
|---|---|
| `metrics_output.json` | Aggregated metrics — **every number reported in the paper originates exclusively from this file** |

Scenes B and C do not have dynamic-track result files: `dynamic_simulator_b_lg.py`
and `dynamic_simulator_c_lg.py` were run, but the escalating-pressure variable
was found not to enter the agent's decision in either scenario (§3 of the
paper), so these scenes were excluded from the dynamic track by diagnostic,
not by design a priori.

## Reproducing the paper's numbers

Re-running the full pipeline from scratch (agent → raw logs →
metrics) requires live API access and is not needed to verify the
paper's reported numbers, since `metrics_output.json` is provided.

## Notes

Downstream table/figure generation from `metrics_output.json` (e.g. LaTeX
table rendering) is not part of this repository. The LLM-judge rubrics used
for language-quality dimensions are included in `src/eval/eval_static_lg.py`;
`compute_metrics.py`'s full aggregation logic — no-action policy,
decision-level bootstrap DPA, and the main-case-set filter — can be
inspected directly in source; running it requires the raw per-model
logs, which are not part of this release (see Reproducing section
above).
## Artifact Terms

Code is released under the MIT License in `LICENSE`. Dataset and result-log
usage terms, intended use, out-of-scope use, and anonymous-review citation
guidance are documented in `ARTIFACT_TERMS.md`. Dataset composition, schema,
privacy notes, and safety considerations are documented in `DATASET_CARD.md`.
