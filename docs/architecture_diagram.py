"""Generate pairlab system deployment architecture diagram."""

import os
os.chdir("/Users/hoapham/pairlab/docs")

from diagrams import Diagram, Cluster, Edge
from diagrams.aws.storage import S3
from diagrams.onprem.queue import Kafka
from diagrams.onprem.workflow import Airflow
from diagrams.onprem.database import PostgreSQL
from diagrams.onprem.inmemory import Redis
from diagrams.onprem.monitoring import Prometheus, Grafana
from diagrams.onprem.container import Docker
from diagrams.onprem.vcs import Github
from diagrams.onprem.ci import GithubActions
from diagrams.onprem.network import Nginx
from diagrams.onprem.compute import Server
from diagrams.programming.language import Python
from diagrams.programming.framework import FastAPI
from diagrams.generic.storage import Storage
from diagrams.generic.compute import Rack

graph_attr = {
    "fontsize":      "13",
    "bgcolor":       "white",
    "pad":           "0.8",
    "splines":       "curved",
    "nodesep":       "0.5",
    "ranksep":       "0.9",
    "fontname":      "Helvetica",
    "label":         "pairlab — Full System Deployment Architecture\n"
                     "Phase 1: Engine + Generator  |  Phase 2: Data Platform  |  Phase 3: ML + Serving + LLM Agent",
    "labelloc":      "t",
    "labelfontsize": "18",
    "fontcolor":     "#1e293b",
    "size":          "28,18",
    "dpi":           "150",
}

node_attr = {
    "fontsize":  "11",
    "fontname":  "Helvetica",
    "width":     "1.6",
    "height":    "1.0",
    "fixedsize": "false",
}

with Diagram(
    "",
    filename="pngs/architecture",
    outformat="png",
    graph_attr=graph_attr,
    node_attr=node_attr,
    direction="TB",
    show=False,
):
    # ── Actors (top-level) ──────────────────────────────────────
    developer = Python("Developer")
    analyst   = Python("Analyst /\nResearcher")

    # ══════════════════════════════════════════════════════════════
    # ROW 1 — CI / CD
    # ══════════════════════════════════════════════════════════════
    with Cluster("ns: ci  |  GitHub Actions"):
        gh       = Github("GitHub\nRepo")
        ci_run   = GithubActions("GitHub Actions\nlint → test → docker")
        registry = Docker("GHCR\nDocker Registry")
        gh >> Edge(label="trigger CI") >> ci_run >> Edge(label="push image") >> registry

    developer >> Edge(label="① git push", color="#3b82f6", style="bold") >> gh

    # ══════════════════════════════════════════════════════════════
    # ROW 2 — Data Sources
    # ══════════════════════════════════════════════════════════════
    with Cluster("ns: data-generation  |  Synthetic + Live"):
        generator  = Python("Data Generator\n(synthetic OHLCV)\nskew / dup / schema-v")
        minio_raw  = S3("MinIO\nvendor-raw/\n(Parquet)")
        binance_ws = Server("Binance WS\nTick Producer")
        generator >> Edge(label="② Parquet", color="#16a34a") >> minio_raw

    # ══════════════════════════════════════════════════════════════
    # ROW 3 — Streaming
    # ══════════════════════════════════════════════════════════════
    with Cluster("ns: streaming  |  Redpanda + Flink"):
        redpanda   = Kafka("Redpanda\nticks.raw\n(burst/late/dup)")
        flink      = Server("Flink SQL\ntumbling 1-min OHLCV\nwatermark + dedup")
        bars_topic = Kafka("Redpanda\nbars.1m")

        redpanda   >> Edge(label="④ stream ticks", color="#ea580c") >> flink
        flink      >> Edge(label="⑤ 1-min bars",   color="#ea580c") >> bars_topic

    binance_ws >> Edge(label="③ live ticks\n(burst / late arrivals)",
                       color="#ea580c", style="bold") >> redpanda

    # ══════════════════════════════════════════════════════════════
    # ROW 4 — Batch Processing
    # ══════════════════════════════════════════════════════════════
    with Cluster("ns: data-platform  |  Airflow + Spark"):
        airflow = Airflow("Airflow\nDP1 / DP2 / DP3 DAGs\n(connections in Airflow UI)")
        spark   = Rack("Spark\nskew: salting + AQE\ncardinality: bucket join\nschema: mergeSchema")
        airflow >> Edge(label="⑦ trigger job", color="#7c3aed") >> spark

    minio_raw >> Edge(label="DP1 bronze ingest", color="#7c3aed", style="dashed") >> airflow

    # ══════════════════════════════════════════════════════════════
    # ROW 5 — Storage Layer
    # ══════════════════════════════════════════════════════════════
    with Cluster("ns: warehouse  |  Delta Lake + PostgreSQL DWH"):
        delta  = S3("MinIO  Delta Lake\nBronze → Silver → Gold\nOPTIMIZE + ZORDER(symbol)")
        pg_dwh = PostgreSQL("PostgreSQL  Gold\ndim_symbol (SCD2)\nfact_daily_bar\nfeat_pair_daily\nobt_pair_backtest_input")

        spark >> Edge(label="⑧ write Delta\n(partitioned by dt)",  color="#7c3aed") >> delta
        spark >> Edge(label="⑨ copy Gold\n(indexed symbol+ts)",    color="#7c3aed") >> pg_dwh

    bars_topic >> Edge(label="Flink sink → bars.1m", color="#ea580c", style="dashed") >> pg_dwh

    with Cluster("ns: feature-store  |  Feast"):
        feast_offline = PostgreSQL("Feast  Offline Store\nfeat_symbol_daily (TTL 3d)\nfeat_pair_daily   (TTL 3d)\nfeat_bar_1m       (TTL 10m)")
        feast_online  = Redis("Feast  Online Store\n(Redis)\nTTL: 3d / 10 min")
        feast_offline >> Edge(label="⑩ incremental\nmaterialize", color="#0d9488") >> feast_online

    pg_dwh >> Edge(label="DP3 feature compute", color="#7c3aed", style="dashed") >> feast_offline

    # ══════════════════════════════════════════════════════════════
    # ROW 6 — ML + Engine
    # ══════════════════════════════════════════════════════════════
    with Cluster("ns: ml  |  LightGBM + MLflow"):
        mlflow   = Server("MLflow\nModel Registry\n(PG backend + MinIO)")
        training = Airflow("Training DAG\ntime-split + purge\n+ embargo\n(data versioned by\nDelta time travel)")
        lgbm     = Python("LightGBM\nMeta-label classifier\nAUC + Sharpe↑ vs baseline")

        training >> Edge(label="train",              color="#4338ca") >> lgbm
        lgbm     >> Edge(label="⑪ register model",  color="#4338ca") >> mlflow

    feast_offline >> Edge(label="read features + labels", color="#4338ca", style="dashed") >> training

    with Cluster("ns: engine  |  pairlab  (pure Python, no infra imports)"):
        engine    = Python("pairlab Engine\nevent-driven backtester\n(no-lookahead guaranteed\nby design + test)")
        cli       = Python("CLI\npairlab backtest\n--config demo.yaml")
        tearsheet = Storage("Tear Sheet\nHTML + PNGs\nresults.json\n(Sharpe/MDD/PF/Calmar)")
        cli >> engine >> Edge(label="⑰ output") >> tearsheet

    pg_dwh >> Edge(label="obt_pair_backtest_input\n(point-in-time filtered)",
                   color="#4b5563", style="dashed") >> engine

    # ══════════════════════════════════════════════════════════════
    # ROW 7 — Serving + LLM
    # ══════════════════════════════════════════════════════════════
    with Cluster("ns: api-serving  |  FastAPI + NGINX"):
        gateway    = Nginx("NGINX Gateway\nBasic Auth\nrate limit\nHTTPS")
        signal_api = FastAPI("signal_api\nFeast online\n→ model predict\n→ {prob, take_trade\n   model_version}")
        regime_api = FastAPI("regime_api\nrolling coint p-value\n+ PSI drift score\n→ {stable|shifting|broken}")
        gateway >> signal_api
        gateway >> regime_api

    feast_online >> Edge(label="⑫ online features", color="#0284c7")               >> signal_api
    mlflow       >> Edge(label="champion alias\npull model",
                         color="#4338ca", style="dashed")                           >> signal_api
    pg_dwh       >> Edge(label="spread history",    color="#0284c7", style="dashed") >> regime_api
    registry     >> Edge(label="deploy\n(CI/CD)",   color="#3b82f6", style="dashed") >> gateway

    with Cluster("ns: llm-inference  |  Ollama + FastMCP"):
        ollama = Server("Ollama\nQwen 7B  (local)\nno external API")
        mcp    = FastAPI("MCP Server\n(FastMCP)\ntools:\nget_pair_features\ncheck_regime")
        agent  = Python("Analyst Agent\ntool-calling loop\n\"Is KO/PEP tradeable?\"\n→ features + regime\n→ plain-English answer")

        mcp >> Edge(label="⑭ get_pair_features",    color="#db2777") >> signal_api
        mcp >> Edge(label="⑮ check_regime",         color="#db2777") >> regime_api
        ollama >> Edge(label="LLM inference",        color="#db2777") >> agent
        mcp    >> Edge(label="⑬ tool results",       color="#db2777") >> agent

    analyst >> Edge(label="\"Is KO/PEP\nstill tradeable?\"",
                    color="#db2777", style="bold") >> agent
    agent   >> Edge(label="plain-English\nanswer",
                    color="#db2777", style="bold") >> analyst

    # ══════════════════════════════════════════════════════════════
    # ROW 8 — Observability
    # ══════════════════════════════════════════════════════════════
    with Cluster("ns: observability  |  Prometheus + Grafana"):
        prom    = Prometheus("Prometheus\nscrape /metrics\nall services")
        grafana = Grafana("Grafana\nAPI req/s · failures\ndrift PSI · Sharpe\ncontainer CPU/RAM")
        prom >> Edge(label="visualize") >> grafana

    signal_api >> Edge(label="⑯ metrics",  color="#dc2626", style="dashed") >> prom
    regime_api >> Edge(label="drift PSI",  color="#dc2626", style="dashed") >> prom
