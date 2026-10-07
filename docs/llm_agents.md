# LLM agents

Three agents answer questions about pairs. They run on a model hosted
locally, and they get every number from tools on an MCP server.

**Design rule:** the LLM never makes a trading decision. It calls tools and
explains what they returned. Decisions come from statistics and the ML model.

```
question ──► agent API (8003) ──► coordinator ──► features analyst ─┐
                                        │                           ├─► MCP server (8002)
                                        └──────► regime analyst ────┘        │
                                                                             ├─► signal_api (8000) ─► Feast online store, model
llama.cpp server (8085) ◄── every agent's model calls                        └─► regime_api (8001) ─► drift detection
```

## 1. Inference platform and custom model

- **Server:** llama.cpp (`ghcr.io/ggml-org/llama.cpp:server`), Compose service
  `llama-cpp`, profile `llm`, OpenAI-compatible API on port 8085.
- **Custom model:** Qwen2.5-3B-Instruct, GGUF, 4-bit quantised (`Q4_K_M`,
  2.1 GB), downloaded by `scripts/download_model.sh` into the git-ignored
  `models/` folder and mounted into the container.
- **Flags:** `--jinja` (tool calling), `--metrics` (Prometheus), `-np` parallel
  request slots, `-c` context size, `-t` threads per slot.
- **Memory:** about 4 GB. Docker Desktop needs more than its default; this
  stack runs with 10.7 GB.

Start it:

```
bash scripts/download_model.sh
docker compose -f platform/compose.yml --profile core --profile llm up -d llama-cpp
```

### Benchmark and optimisation

`services/agent/bench_llm.py` streams 12 completions, two at a time, and
measures time to first token (TTFT), latency and tokens per second. Docker has
10 CPUs. One run per setting, so differences of a few percent are noise.

| Setting | TTFT median | Latency median | Latency p95 | Tokens/s per stream | Throughput |
|---|---|---|---|---|---|
| Baseline: `-t 4 -np 2` | 0.09 s | 3.92 s | 5.12 s | 17.7 | 31.2 tok/s |
| `-t 8 -np 2` | 0.21 s | 5.95 s | 7.00 s | 12.7 | 23.7 tok/s |
| `-t 8 -np 1` | 3.20 s | 6.33 s | 9.23 s | 22.5 | 20.7 tok/s |
| **`-t 5 -np 2`** | 0.08 s | 3.77 s | 5.50 s | 20.5 | **36.0 tok/s** |

What the runs show:

- **More threads is not faster.** Two slots with 8 threads each ask for 16
  threads on 10 CPUs; they fight each other and throughput drops 24%.
- **One slot makes each answer faster but makes callers wait.** With `-np 1`
  the second request queues, so its first token arrives after 3.2 s.
- **Best: threads × slots = CPUs.** `-t 5 -np 2` uses exactly 10 threads and
  gives 15% more throughput than the baseline.

Raw results: `pngs/llm_bench_*.json`. Re-run:

```
uv run python -m services.agent.bench_llm --requests 12 --concurrency 2 --label my_run
```

## 2. MCP server

`services/mcp_server/` (FastMCP, streamable HTTP, port 8002).

| Tool | Calls | Returns |
|---|---|---|
| `get_pair_features(pair_id)` | `signal_api` `POST /signal` | Features from the Feast online store, model probability, take/skip, model version |
| `check_regime(pair_id)` | `regime_api` `POST /regime` | Regime, cointegration p-value, PSI |

- A malformed pair ID is rejected before any API call.
- API errors and unreachable services come back as `{"error": ..., "status": ...}`,
  so the model can read and report the failure.
- `GET /healthz` for liveness.

```
uv run python -m services.mcp_server.server
```

## 3. Agents

`services/agent/`.

| Agent | Tools | Job |
|---|---|---|
| `features_analyst` | `get_pair_features` | Report z-score, model probability and the model's decision |
| `regime_analyst` | `check_regime` | Report the regime and the statistics behind it |
| `coordinator` | the two analysts, as tools | Combine both reports into one answer |

- **Loop** (`agent.py`): ask the model → run the tools it requests → feed the
  results back → repeat until it answers in text, up to a step limit.
- **Deterministic:** temperature 0 and a fixed seed.
- **Tools come from MCP** (`mcp_tools.py`): the agent lists the server's tools
  and calls them over the MCP connection.
- **Failures are data:** an unknown tool, bad arguments or a crashing tool
  becomes an error result the model reads; the conversation continues.
- **PII guard** (`safety.py`): a prompt containing an email address, phone
  number or card number is refused before the model sees it.

Chat API (port 8003):

```
uv run uvicorn services.agent.app:build_default_app --factory --port 8003
POST /ask  {"question": "Is KO__PEP still tradeable?", "agent": "coordinator"}
```

### A/B test

The agent API runs two variants side by side and splits sessions between
them.

| Variant | Model | Prompt set |
|---|---|---|
| `champion` | Qwen2.5-3B | `default`: the prompts above |
| `challenger` | Qwen2.5-3B | `brief`: at most two sentences; the coordinator starts with `Tradeable:` or `Not tradeable:` and the two numbers that decide it |

- **Split:** a stable hash of `session_id` (or of the question when there is
  none), the same function `signal_api` uses (`services/ab.py`). A session
  always talks to the same variant. `CHALLENGER_SHARE` (default 0.5).
- **Forced variant:** `"variant": "champion" | "challenger"` in the request
  skips the hash. The evaluation uses it to ask both variants the same question.
- **Response:** carries `variant`.
- **Metrics:** `agent_ab_requests_total{variant, agent, outcome}` (outcome:
  `answered`, `tool_failed`, `unfinished`, `blocked`) and the
  `agent_ab_seconds{variant, agent}` histogram. Both are in the Grafana
  dashboard (LLM row).
- **Comparing two models instead:** point the challenger at a second
  llama.cpp server with `CHALLENGER_LLM_BASE_URL` and `CHALLENGER_LLM_MODEL`.
  Prompt sets: `CHAMPION_PROMPTS`, `CHALLENGER_PROMPTS`.

**Evaluation** (`services/agent/ab_eval.py`): 6 questions (3 pairs × 2
wordings), each asked once per variant, one at a time. Every answer is
scored with deterministic checks:

| Check | Pass when |
|---|---|
| Finished | Not refused, did not hit the step limit |
| Called all tools | Both analysts were asked |
| Right pair | Every tool call used the pair in the question |
| No failed tools | No tool returned an error |
| Grounded | Every number in the answer matches a value `signal_api` or `regime_api` returns for that pair, at the precision it is written (0.688 may appear as 0.69 or 68.8%) |
| Verdict format | Answer starts with `Tradeable:` or `Not tradeable:` (reported, not part of pass) |

Result on Qwen2.5-3B (`pngs/phase4_llm_ab.json`):

| Variant | Pass | Grounded | Verdict format | Latency median | Output tokens median |
|---|---|---|---|---|---|
| champion | 6/6 | 6/6 | 0/6 | 38.9 s | 122.5 |
| challenger | 5/6 | 5/6 | 6/6 | 20.9 s | 72 |

- The brief prompt is about twice as fast and uses 41% fewer output tokens.
- It invented a number once: "Tradeable: 79.92, 100" for XOM__CVX. No tool
  returns 100. The tool-call checks alone would have passed it.
- Its numbers come without labels ("Tradeable: 2.75, 0.0012" does not say
  that 2.75 is the z-score). The champion names every number.
- **Decision:** keep the champion. Speed does not make up for an invented
  number. A next challenger would keep the format and require a label per
  number.
- Latency varies by a few seconds between runs (an earlier identical run
  measured 36.0 s and 24.8 s); the answers were identical both times.

Per-variant counters after the runs: `pngs/phase4_llm_ab_metrics.txt`.

```
uv run python -m services.agent.ab_eval --base-url http://localhost:8003 \
    --signal-url http://localhost:8000 --regime-url http://localhost:8001 \
    --out docs/pngs/phase4_llm_ab.json
```

This is an offline comparison on fixed questions, like the model A/B in
`signal_api`: no trading follows from either variant's answer.

## 4. Notebook

[`notebooks/agent_demo.ipynb`](../notebooks/agent_demo.ipynb), saved with
outputs from the real model. It shows the agents pulling features from the
feature store and running drift detection through the MCP server, the
coordinator combining both, an unknown pair, a refused prompt, and the
telemetry collected.

Example from the notebook:

> **Q:** Is KO__PEP still tradeable?
> **Coordinator** (15.9 s, 904 + 119 tokens; called both analysts): The pair
> KO__PEP is still tradeable as the features analyst indicates a model
> probability of 0.688, suggesting a favorable trade. The regime analyst
> confirms the regime is stable with a cointegration p-value of 0.0095 and PSI
> of 0.12.

## 5. Telemetry

Exposed at the agent API's `/metrics`.

| Rubric item | Metric |
|---|---|
| Input, output tokens | `llm_tokens_total{model, kind}` |
| Round-trip time of a generation | `llm_round_trip_seconds` (histogram) |
| TTFT | `llm_time_to_first_token_seconds` (histogram) |
| Prompts caught for PII | `agent_pii_blocked_total{kind}` |
| Calls per agent | `agent_calls_total{agent}` |
| Calls per MCP tool | `agent_tool_calls_total{tool}` |
| Failures per tool | `agent_tool_failures_total{tool}` |

In the agent loop TTFT is the prompt-processing time llama.cpp reports; the
benchmark measures it directly from the stream.

## 6. Tests

```
uv run pytest tests/services/test_agent.py tests/services/test_mcp_server.py \
    tests/services/test_agent_api.py tests/services/test_bench_llm.py \
    tests/services/test_ab_eval.py
```

124 tests. The model is replaced by a scripted fake; the MCP connection is real
(in-memory). Partitions and boundaries are documented above the parametrized
tests: malformed tool requests, step limit, request validation, PII kinds, A/B
share 0 and 1, run outcomes, grounding precision. A Hypothesis test checks that
the same run gives the same answer.

## Not done

Kubernetes items of the rubric (Helm deployment of MCP tools, multi-replica
and auto-scaled agents, sandbox, agent registry, warm-up mode), the RAG
pipeline, and Compose services for the MCP server and agent API.
