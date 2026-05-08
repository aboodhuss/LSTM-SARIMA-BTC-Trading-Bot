"""
main.py – FastAPI server: the central API + WebSocket hub for the trading bot.

This module spins up the backend server that:
  1. Connects to the Binance BTC/USDT WebSocket for live 1-minute candle data.
  2. Runs the LSTM and LSTM + SARIMA inference pipeline.
  3. Broadcasts real-time predictions, portfolio state, and telemetry to the
     React dashboard via a WebSocket on /ws.
  4. Exposes REST endpoints for configuration, simulation control, network
     testing, and model history.

Network testing
---------------
The /network-test/* endpoints drive the multi-model comparison table shown
in the dashboard.  They iterate through simulated network conditions
(latency, jitter, packet loss) and run each model family (LSTM and
LSTM+SARIMA) side-by-side.  See model_comparison.py for the replay engine.

Data sources
------------
  • Live market data  : Binance WebSocket (wss://stream.binance.com)
  • Historical data   : Binance REST API (BTCUSDT 1m klines)
  • SARIMA profile    : Calibrated on Kaggle BTC minute data
      https://www.kaggle.com/datasets/swaptr/bitcoin-historical-data

Usage
-----
    cd backend
    uvicorn main:app --host 0.0.0.0 --port 8000
"""

import asyncio
import logging
import math
import os
import re
import statistics
from datetime import datetime, timezone
from decimal import Decimal
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
from contextlib import asynccontextmanager
from typing import Dict, List, Optional

from data_ingestion import (
    bootstrap_historical_data,
    ingest_binance_data,
    get_latest_state,
    get_model_history,
    get_network_config,
    get_simulation_config,
    live_training_loop,
    start_live_simulation,
    set_network_config,
    set_simulation_config,
)
from model_comparison import DEFAULT_FETCH_LOOPS, DEFAULT_LIVE_PHASE_CANDLES, generate_model_comparison_report

logger = logging.getLogger("API_Server")
logging.basicConfig(level=logging.INFO)

# --- State ---
active_connections: list[WebSocket] = []
ingestion_task = None
live_training_task = None

NETWORK_SWEEP_PHASES = [
    {"name": "baseline", "enabled": False, "latencyMs": 0.0, "jitterMs": 0.0, "packetLossPct": 0.0},
    {"name": "latency_300", "enabled": True, "latencyMs": 300.0, "jitterMs": 0.0, "packetLossPct": 0.0},
    {"name": "latency_800_jitter_200", "enabled": True, "latencyMs": 800.0, "jitterMs": 200.0, "packetLossPct": 0.0},
    {"name": "loss_3pct", "enabled": True, "latencyMs": 0.0, "jitterMs": 0.0, "packetLossPct": 3.0},
    {"name": "combined", "enabled": True, "latencyMs": 800.0, "jitterMs": 200.0, "packetLossPct": 5.0},
    {"name": "sarima_stress_loss_15pct", "enabled": True, "latencyMs": 0.0, "jitterMs": 0.0, "packetLossPct": 15.0},
]
NETWORK_SWEEP_SIM_OVERRIDES = {
    "minConfidencePct": 33.0,
    "minTradeNotional": 200.0,
    "minAllocationPct": 5.0,
    "maxAllocationPct": 25.0,
    "ignoreFees": True,
    "dynamicThresholds": False,
}

network_sweep_state = {
    "running": False,
    "stop_requested": False,
    "task": None,
    "started_at": None,
    "ended_at": None,
    "current_phase": None,
    "current_model": None,
    "fetch_loops": DEFAULT_FETCH_LOOPS,
    "poll_seconds": 2.0,
    "samples": [],
    "completed_runs": 0,
    "total_runs": len(NETWORK_SWEEP_PHASES) * 2,
    "current_samples": 0,
    "estimated_samples": 0,
    "latest_report": None,
    "original_simulation": None,
}


def json_safe(value):
    """Convert live model state into values FastAPI can safely serialize."""
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [json_safe(item) for item in value]
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, Decimal):
        as_float = float(value)
        return as_float if math.isfinite(as_float) else None
    if isinstance(value, datetime):
        return value.isoformat()
    if hasattr(value, "item"):
        try:
            return json_safe(value.item())
        except Exception:
            return str(value)
    if hasattr(value, "isoformat"):
        try:
            return value.isoformat()
        except Exception:
            return str(value)
    return value


def _utc_now_iso():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


async def ensure_market_state():
    latest = get_latest_state()
    if latest.get("history") or latest.get("candle"):
        return latest

    await asyncio.to_thread(bootstrap_historical_data, "BTCUSDT", 200)
    return get_latest_state()


def _collect_network_sample(phase_name: str) -> Dict[str, object]:
    latest = get_latest_state()
    telemetry = latest.get("telemetry", {})
    prediction = latest.get("prediction", {})
    portfolio = latest.get("portfolio", {})
    return {
        "timestamp_utc": _utc_now_iso(),
        "phase": phase_name,
        "action": str(prediction.get("action", "")),
        "confidence": float(prediction.get("confidence") or 0.0),
        "delta_ms": float(telemetry.get("deltaMs") or 0.0),
        "simulated_delay_ms": float(telemetry.get("simulatedDelayMs") or 0.0),
        "packet_loss_pct": float(telemetry.get("packetLossPct") or 0.0),
        "processed_candles": int(telemetry.get("processedCandles") or 0),
        "dropped_candles": int(telemetry.get("droppedCandles") or 0),
        "imputed_rate_pct": float(telemetry.get("imputedRatePct") or 0.0),
        "data_quality": float(telemetry.get("dataQuality") or 1.0),
        "trade_count": int(portfolio.get("tradeCount") or 0),
        "total_pnl": float(portfolio.get("totalPnl") or 0.0),
    }


def _summarize_network_samples(samples: List[Dict[str, object]]) -> List[Dict[str, object]]:
    grouped: Dict[str, List[Dict[str, object]]] = {}
    for row in samples:
        grouped.setdefault(str(row["phase"]), []).append(row)

    summary_rows = []
    for phase in [phase["name"] for phase in NETWORK_SWEEP_PHASES]:
        rows = grouped.get(phase, [])
        if not rows:
            continue

        conf = [float(x["confidence"]) for x in rows]
        delays = [float(x["delta_ms"]) for x in rows]
        sim_delays = [float(x["simulated_delay_ms"]) for x in rows]
        imputed_rates = [float(x.get("imputed_rate_pct") or 0.0) for x in rows]
        data_quality = [float(x.get("data_quality") or 1.0) for x in rows]
        actions = [str(x["action"]) for x in rows]
        non_hold = len([x for x in rows if x["action"] in {"BUY", "SELL"}])
        low_conf = len([x for x in rows if float(x["confidence"]) < 0.55])
        high_delay = len([d for d in delays if d >= 500.0])
        action_flips = 0
        for prev, cur in zip(actions, actions[1:]):
            if prev != cur:
                action_flips += 1

        sorted_delays = sorted(delays)
        p95_index = max(0, math.ceil(0.95 * len(sorted_delays)) - 1)

        processed_delta = int(rows[-1]["processed_candles"]) - int(rows[0]["processed_candles"])
        dropped_delta = int(rows[-1]["dropped_candles"]) - int(rows[0]["dropped_candles"])
        trade_delta = int(rows[-1]["trade_count"]) - int(rows[0]["trade_count"])
        pnl_start = float(rows[0]["total_pnl"])
        pnl_end = float(rows[-1]["total_pnl"])
        pnl_delta = pnl_end - pnl_start

        summary_rows.append(
            {
                "phase": phase,
                "samples": len(rows),
                "mean_confidence": statistics.fmean(conf) if conf else 0.0,
                "mean_decision_delay_ms": statistics.fmean(delays) if delays else 0.0,
                "p95_decision_delay_ms": sorted_delays[p95_index] if sorted_delays else 0.0,
                "mean_simulated_delay_ms": statistics.fmean(sim_delays) if sim_delays else 0.0,
                "low_conf_rate": low_conf / max(1, len(rows)),
                "non_hold_rate": non_hold / max(1, len(rows)),
                "high_delay_rate": high_delay / max(1, len(rows)),
                "action_flip_rate": action_flips / max(1, len(rows) - 1),
                "processed_candles_delta": processed_delta,
                "dropped_candles_delta": dropped_delta,
                "drop_rate_pct_from_counter": (dropped_delta / max(1, processed_delta + dropped_delta)) * 100.0,
                "mean_imputed_rate_pct": statistics.fmean(imputed_rates) if imputed_rates else 0.0,
                "mean_data_quality": statistics.fmean(data_quality) if data_quality else 1.0,
                "trade_count_delta": trade_delta,
                "pnl_start": pnl_start,
                "pnl_end": pnl_end,
                "pnl_delta": pnl_delta,
            }
        )

    baseline_row = next((row for row in summary_rows if row["phase"] == "baseline"), None)
    baseline_delta = baseline_row["pnl_delta"] if baseline_row else 0.0
    for row in summary_rows:
        row["pnl_impact_vs_baseline"] = row["pnl_delta"] - baseline_delta
    return summary_rows


def _build_report_paragraph(summary_rows: List[Dict[str, object]]) -> str:
    if not summary_rows:
        return "No samples were captured. Please run the test again with a longer duration."

    baseline = next((row for row in summary_rows if row["phase"] == "baseline"), summary_rows[0])
    delay_worst = max(summary_rows, key=lambda r: r["mean_decision_delay_ms"])
    loss_worst = max(summary_rows, key=lambda r: r["drop_rate_pct_from_counter"])
    confidence_lowest = min(summary_rows, key=lambda r: r["mean_confidence"])
    quality_lowest = min(summary_rows, key=lambda r: r.get("mean_data_quality", 1.0))
    pnl_best = max(summary_rows, key=lambda r: r["pnl_delta"])
    pnl_worst = min(summary_rows, key=lambda r: r["pnl_delta"])
    degraded = [r for r in summary_rows if r["phase"] != "baseline"]
    positive_under_degradation = [r for r in degraded if r["pnl_delta"] > 0]

    stochastic_note = ""
    if positive_under_degradation:
        names = ", ".join([f"'{row['phase']}'" for row in positive_under_degradation])
        stochastic_note = (
            f" Positive PnL in degraded phases ({names}) is likely stochastic variation, "
            "not evidence that network degradation improves model quality."
        )

    return (
        "Across the run, network impairment changed both responsiveness and behaviour in measurable ways. "
        f"The highest mean delay was in '{delay_worst['phase']}' at {delay_worst['mean_decision_delay_ms']:.1f} ms, "
        f"compared with {baseline['mean_decision_delay_ms']:.1f} ms in baseline. "
        f"The largest counter-based drop rate occurred in '{loss_worst['phase']}' at {loss_worst['drop_rate_pct_from_counter']:.2f}%. "
        f"Model confidence was lowest in '{confidence_lowest['phase']}' at {confidence_lowest['mean_confidence']:.3f}. "
        f"Input quality was lowest in '{quality_lowest['phase']}' at {quality_lowest['mean_data_quality']:.3f}. "
        f"PnL moved most positively in '{pnl_best['phase']}' ({pnl_best['pnl_delta']:+.2f}) and most negatively in "
        f"'{pnl_worst['phase']}' ({pnl_worst['pnl_delta']:+.2f}). "
        "This indicates degradation affects not only timing but also signal quality and decision consistency, with combined "
        "or loss-heavy phases typically producing the strongest operational risk."
        + stochastic_note
    )


async def _run_network_sweep(fetch_loops: int, poll_seconds: float):
    # Reuse the existing network-test UI flow, but now generate a single
    # side-by-side comparison report across all supported model families.
    # This now follows the live-market requirement: each phase is evaluated
    # on future Binance candles rather than a fixed historical replay.
    network_sweep_state["started_at"] = _utc_now_iso()
    network_sweep_state["ended_at"] = None
    network_sweep_state["current_phase"] = "replay-prep"
    network_sweep_state["current_model"] = None
    network_sweep_state["samples"] = []
    network_sweep_state["completed_runs"] = 0
    network_sweep_state["current_samples"] = 0
    network_sweep_state["estimated_samples"] = 0
    network_sweep_state["latest_report"] = None

    try:
        await asyncio.sleep(0)
        network_sweep_state["current_phase"] = "multi-model-comparison"

        def should_continue():
            return not network_sweep_state["stop_requested"]

        def update_progress(payload: Dict[str, object]):
            network_sweep_state["current_phase"] = payload.get("currentPhase") or network_sweep_state["current_phase"]
            network_sweep_state["current_model"] = payload.get("currentModel")
            network_sweep_state["completed_runs"] = int(payload.get("completedRuns") or 0)
            network_sweep_state["total_runs"] = int(payload.get("totalRuns") or network_sweep_state["total_runs"])
            network_sweep_state["current_samples"] = int(payload.get("currentSamples") or 0)
            network_sweep_state["estimated_samples"] = int(payload.get("estimatedSamples") or 0)
            rows = payload.get("rows")
            if isinstance(rows, list):
                network_sweep_state["samples"] = rows
            report_payload = payload.get("report")
            if isinstance(report_payload, dict):
                network_sweep_state["latest_report"] = report_payload

        report = await asyncio.to_thread(
            generate_model_comparison_report,
            fetch_loops=fetch_loops,
            progress_callback=update_progress,
            should_continue=should_continue,
        )
        network_sweep_state["samples"] = report.get("summary_rows", [])
        network_sweep_state["completed_runs"] = int(report.get("completed_runs") or len(network_sweep_state["samples"]))
        network_sweep_state["total_runs"] = int(report.get("total_runs") or network_sweep_state["total_runs"])
        network_sweep_state["current_samples"] = 0
        network_sweep_state["estimated_samples"] = int(report.get("estimated_samples_per_run") or 0)
        network_sweep_state["latest_report"] = report
    finally:
        network_sweep_state["running"] = False
        network_sweep_state["stop_requested"] = False
        network_sweep_state["current_phase"] = None
        network_sweep_state["current_model"] = None
        network_sweep_state["current_samples"] = 0
        network_sweep_state["ended_at"] = _utc_now_iso()
        network_sweep_state["task"] = None
        network_sweep_state["original_simulation"] = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Start the background Binance ingestion loop
    logger.info("Starting Binance WebSocket Ingestion...")
    global ingestion_task, live_training_task
    ingestion_task = asyncio.create_task(ingest_binance_data())
    live_training_task = asyncio.create_task(live_training_loop())
    yield
    # Shutdown
    logger.info("Shutting down Binance ingestion...")
    if ingestion_task:
        ingestion_task.cancel()
    if live_training_task:
        live_training_task.cancel()
    if network_sweep_state.get("task"):
        network_sweep_state["stop_requested"] = True
        network_sweep_state["task"].cancel()

app = FastAPI(lifespan=lifespan)

DEFAULT_FRONTEND_ORIGINS = (
    "http://localhost:5173,"
    "http://127.0.0.1:5173,"
    "http://localhost:3000,"
    "http://127.0.0.1:3000"
)


def _normalize_origin(origin: str) -> str:
    return origin.strip().rstrip("/")


def _configured_frontend_origins() -> List[str]:
    raw_origins = os.getenv("FRONTEND_ORIGINS", DEFAULT_FRONTEND_ORIGINS)
    origins = [_normalize_origin(origin) for origin in raw_origins.split(",") if origin.strip()]
    vercel_frontend_url = os.getenv("VERCEL_FRONTEND_URL")
    if vercel_frontend_url:
        origins.append(_normalize_origin(vercel_frontend_url))
    return list(dict.fromkeys(origins))


frontend_origins = _configured_frontend_origins()
allow_all_origins = "*" in frontend_origins
frontend_origin_regex = os.getenv("FRONTEND_ORIGIN_REGEX", r"https://.*\.vercel\.app")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"] if allow_all_origins else frontend_origins,
    allow_origin_regex=None if allow_all_origins else frontend_origin_regex,
    allow_credentials=not allow_all_origins,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _origin_allowed(origin: Optional[str]) -> bool:
    if not origin or allow_all_origins:
        return True
    normalized_origin = _normalize_origin(origin)
    if normalized_origin in frontend_origins:
        return True
    return bool(frontend_origin_regex and re.fullmatch(frontend_origin_regex, normalized_origin))


class SimulationConfigUpdate(BaseModel):
    initialEquity: Optional[float] = None
    minAllocationPct: Optional[float] = None
    maxAllocationPct: Optional[float] = None
    minConfidencePct: Optional[float] = None
    feeRatePct: Optional[float] = None
    ignoreFees: Optional[bool] = None
    minTradeNotional: Optional[float] = None
    allowLong: Optional[bool] = None
    allowShort: Optional[bool] = None


class SimulationStartRequest(BaseModel):
    budget: float
    ignoreFees: Optional[bool] = None


class NetworkConfigUpdate(BaseModel):
    enabled: Optional[bool] = None
    latencyMs: Optional[float] = None
    jitterMs: Optional[float] = None
    packetLossPct: Optional[float] = None


class NetworkTestStartRequest(BaseModel):
    fetchLoops: Optional[int] = DEFAULT_FETCH_LOOPS
    pollSeconds: Optional[float] = 2.0


@app.get("/network-test/status")
async def network_test_status():
    return {
        "status": "ok",
        "running": bool(network_sweep_state["running"]),
        "currentPhase": network_sweep_state["current_phase"],
        "currentModel": network_sweep_state["current_model"],
        "startedAt": network_sweep_state["started_at"],
        "endedAt": network_sweep_state["ended_at"],
        "fetchLoops": network_sweep_state["fetch_loops"],
        "pollSeconds": network_sweep_state["poll_seconds"],
        "sampleCount": len(network_sweep_state["samples"]),
        "completedRuns": network_sweep_state["completed_runs"],
        "totalRuns": network_sweep_state["total_runs"],
        "currentSamples": network_sweep_state["current_samples"],
        "estimatedSamples": network_sweep_state["estimated_samples"],
        "phases": [phase["name"] for phase in NETWORK_SWEEP_PHASES],
    }


@app.get("/network-test/report")
async def network_test_report():
    return {
        "status": "ok",
        "running": bool(network_sweep_state["running"]),
        "report": network_sweep_state["latest_report"],
    }


@app.post("/network-test/start")
async def start_network_test(payload: NetworkTestStartRequest):
    if network_sweep_state["running"]:
        return {
            "status": "busy",
            "message": "Network test is already running.",
            "running": True,
            "currentPhase": network_sweep_state["current_phase"],
        }

    fetch_loops = max(1, int(payload.fetchLoops or DEFAULT_FETCH_LOOPS))
    poll_seconds = max(0.5, float(payload.pollSeconds or 2.0))

    network_sweep_state["running"] = True
    network_sweep_state["stop_requested"] = False
    network_sweep_state["fetch_loops"] = fetch_loops
    network_sweep_state["poll_seconds"] = poll_seconds
    network_sweep_state["task"] = asyncio.create_task(_run_network_sweep(fetch_loops, poll_seconds))

    return {
        "status": "ok",
        "message": "Completion-based network test started.",
        "running": True,
        "fetchLoops": fetch_loops,
        "pollSeconds": poll_seconds,
        "completedRuns": 0,
        "totalRuns": len(NETWORK_SWEEP_PHASES) * 2,
        "currentSamples": 0,
        "estimatedSamples": DEFAULT_LIVE_PHASE_CANDLES,
        "phases": [phase["name"] for phase in NETWORK_SWEEP_PHASES],
    }


@app.post("/network-test/stop")
async def stop_network_test():
    if not network_sweep_state["running"]:
        return {
            "status": "ok",
            "message": "No active run. Returning latest report.",
            "running": False,
            "report": network_sweep_state["latest_report"],
        }

    network_sweep_state["stop_requested"] = True
    task = network_sweep_state.get("task")
    if task:
        await task

    return {
        "status": "ok",
        "message": "Network test stopped. Partial report generated.",
        "running": False,
        "report": network_sweep_state["latest_report"],
    }

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    origin = websocket.headers.get("origin")
    if not _origin_allowed(origin):
        logger.warning("Rejected WebSocket connection from unauthorized origin: %s", origin)
        await websocket.close(code=1008)
        return

    await websocket.accept()
    active_connections.append(websocket)
    logger.info(f"Client connected. Total clients: {len(active_connections)}")
    try:
        await ensure_market_state()
        while True:
            # We poll the latest state and broadcast it to the UI every second
            await asyncio.sleep(1)
            state = get_latest_state()
            if state["candle"]:
                await websocket.send_json(json_safe(state))
    except WebSocketDisconnect:
        active_connections.remove(websocket)
        logger.info("Client disconnected.")
    except Exception as e:
         logger.error(f"WebSocket Error: {e}")
         if websocket in active_connections:
             active_connections.remove(websocket)

@app.get("/health")
async def health_check():
    latest = await ensure_market_state()
    return json_safe({"status": "ok", "latest_state": latest})


@app.get("/config")
async def get_config():
    return {"status": "ok", "simulation": get_simulation_config()}


@app.get("/model-history")
async def model_history():
    latest = await ensure_market_state()
    return json_safe({"status": "ok", "models": get_model_history(), "latest_state": latest})


@app.get("/network/profile")
async def get_network_profile():
    latest = await ensure_market_state()
    return json_safe({"status": "ok", "network": get_network_config(), "latest_state": latest})


@app.post("/config")
async def update_config(payload: SimulationConfigUpdate):
    updated = set_simulation_config(payload.model_dump(exclude_none=True))
    latest = await ensure_market_state()
    return json_safe({"status": "ok", "simulation": updated, "latest_state": latest})


@app.post("/simulation/start")
async def start_simulation(payload: SimulationStartRequest):
    await ensure_market_state()
    updates = {"initialEquity": payload.budget}
    if payload.ignoreFees is not None:
        updates["ignoreFees"] = payload.ignoreFees
    updated = start_live_simulation(updates)
    latest = await ensure_market_state()
    return json_safe({"status": "ok", "simulation": updated, "latest_state": latest})


@app.post("/network/profile")
async def update_network_profile(payload: NetworkConfigUpdate):
    updated = set_network_config(payload.model_dump(exclude_none=True))
    latest = await ensure_market_state()
    return json_safe({"status": "ok", "network": updated, "latest_state": latest})

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=int(os.getenv("PORT", "8000")), reload=False)
