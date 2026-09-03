# Dataset Card: EconSafeBench

## Summary

EconSafeBench is a synthetic benchmark for evaluating whether LLM agents cause
financial or economic harm through executable tool calls in stateful simulated
environments. It covers portfolio management, market making, financial
advisory, labour market, and platform economy scenarios.

## Files

- `data/EconSafeBench_benchmark_main.json.gz`: 828 main benchmark cases across 53 evaluated cells.
- `results/metrics_output.json`: aggregated evaluation metrics
  (violation rates, DPA, bootstrap CIs) computed from raw model
  evaluation logs via `src/eval/compute_metrics.py`. Raw per-model
  logs are not included in this release; see the paper's
  Reproducibility notes for the aggregation methodology.

## Schema

Each case contains:

- `id`: case identifier.
- `scene`: one of `A`, `B`, `C`, `D`, `E`.
- `intent_type`: `TypeI`, `TypeII`, or `TypeIII`.
- `dimension`: safety dimension such as rule violation, deception, privacy, or
  discrimination.
- `temptation_type`: pressure or incentive category.
- `account_state`: structured simulator state consumed by tool and judge code.
- `task_prompt`: prompt shown to the agent.
- `compliant_behavior`: expected safe behavior.
- `violation_behavior`: unsafe behavior being tested.
- `judgment_rule`: deterministic or rubric-based judgment criterion.
- `critic`: construction-time quality score.

## Construction

Cases are synthetic and were designed to instantiate concrete economic safety constraints. The main split is capped at 16 cases per evaluated cell. 


## Personal Data

The benchmark does not contain real personal data. Names, clients, workers, accounts, portfolios, prices, wages, and platform events are synthetic. Some scenarios intentionally mention protected or sensitive attributes to test whether agents avoid using them in prohibited ways.

## Safety and Ethical Considerations

The benchmark includes adversarial prompts involving financial misconduct, privacy misuse, wage violations, deceptive disclosures, and discriminatory pricing. These examples are intended only for safety evaluation. They should not be used as operational guidance.

## Recommended Reporting

When using this dataset, report:

- Dataset version from `meta.version`.
- Number of cases and cells evaluated.
- Models, seeds, provider endpoints, and tool-calling settings.
- No-action policy used for violation-rate denominators.
- Error and exclusion criteria.
- Whether LLM-judge dimensions were re-run or only existing raw logs were used.
