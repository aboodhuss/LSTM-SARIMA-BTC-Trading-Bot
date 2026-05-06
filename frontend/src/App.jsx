import { useEffect, useMemo, useRef, useState } from 'react';
import { CandlestickSeries, createChart } from 'lightweight-charts';
import {
  Activity,
  ArrowLeft,
  ArrowDownRight,
  ArrowUpRight,
  BrainCircuit,
  ExternalLink,
  Gauge,
  LineChart,
  Radar,
  PlayCircle,
  Save,
  ShieldCheck,
  SlidersHorizontal,
  Sparkles,
  StopCircle,
  TrendingUp,
  Wallet,
  Wifi,
} from 'lucide-react';

const INITIAL_STATE = {
  history: [],
  candle: null,
  patterns: {
    Bullish_FVG: false,
    Bearish_FVG: false,
    HH: false,
    LL: false,
    HL: false,
    LH: false,
  },
  prediction: {
    action: 'HOLD',
    confidence: 0,
    targetPrice: null,
    projectedMovePct: 0,
  },
  portfolio: {
    equity: 100000,
    realizedPnl: 0,
    unrealizedPnl: 0,
    totalPnl: 0,
    feesPaid: 0,
    position: 'FLAT',
    positionSize: 0,
    deployedCapital: 0,
    entryPrice: null,
    tradeCount: 0,
    winRate: 0,
    signalAccuracy: 0,
  },
  telemetry: {
    tickTime: null,
    receivedTime: null,
    signalTime: null,
    deltaMs: 0,
    updatesPerMinute: 0,
    packetLossPct: 0,
    latencyMs: 0,
    volatilityPct: 0,
    rangePct: 0,
    trendBiasPct: 0,
    confluenceScore: 0,
  },
  simulation: {
    initialEquity: 100000,
    minAllocationPct: 2,
    maxAllocationPct: 12,
    minConfidencePct: 58,
    feeRatePct: 0.1,
    ignoreFees: false,
    minTradeNotional: 1000,
    allowLong: true,
    allowShort: true,
  },
  networkTest: {
    enabled: false,
    latencyMs: 0,
    jitterMs: 0,
    packetLossPct: 0,
  },
  networkAutomation: {
    running: false,
    currentPhase: null,
    currentModel: null,
    startedAt: null,
    endedAt: null,
    fetchLoops: 1,
    pollSeconds: 2,
    sampleCount: 0,
    completedRuns: 0,
    totalRuns: 12,
    currentSamples: 0,
    estimatedSamples: 0,
    phases: [],
    report: null,
  },
  simulationSummary: {
    budget: 100000,
    aiPolicy: {
      allocationRangePct: [2, 12],
      minConfidencePct: 58,
      feeRatePct: 0.1,
      configuredFeeRatePct: 0.1,
      ignoreFees: false,
      roundTripCostPct: 0.2,
      feeAwareTradeGatePct: 0.28,
      minTradeNotional: 1000,
      allowLong: true,
      allowShort: true,
    },
  },
  training: {},
  liveTraining: {
    enabled: true,
    running: false,
    intervalSec: 900,
    warmupSec: 90,
    lastStartedAt: null,
    lastCompletedAt: null,
    lastPromotedAt: null,
    nextRunAt: null,
    activeModelVersion: null,
    championModelLabel: null,
    championModelPath: null,
    championScore: null,
    championMetrics: null,
    promotionCount: 0,
    tradeFeedbackCount: 0,
    lastPromotionDecision: 'pending',
    lastError: null,
    lastCandidate: null,
    shadowEvaluation: {
      status: 'idle',
      decision: 'no shadow evaluation running',
      candlesObserved: 0,
      minCandles: 120,
      minTrades: 6,
      candidateReplay: null,
      championReplay: null,
      candidateShadow: null,
      championShadow: null,
    },
  },
  modelHistory: [],
  modelTraces: [],
  activity: [],
  blotter: [],
  logs: [],
};

const numberFormat = new Intl.NumberFormat('en-US', {
  maximumFractionDigits: 2,
  minimumFractionDigits: 2,
});

const percentFormat = new Intl.NumberFormat('en-US', {
  maximumFractionDigits: 2,
  minimumFractionDigits: 2,
});

const EMPTY_VALUE = 'Not available';

function formatCurrency(value) {
  if (value === null || value === undefined || Number.isNaN(value)) {
    return EMPTY_VALUE;
  }

  const formatted = `$${numberFormat.format(Math.abs(value))}`;
  return value < 0 ? `-${formatted}` : formatted;
}

function formatPercent(value) {
  if (value === null || value === undefined || Number.isNaN(value)) {
    return EMPTY_VALUE;
  }

  const formatted = `${percentFormat.format(Math.abs(value))}%`;
  return value < 0 ? `(${formatted})` : `+${formatted}`;
}

function formatTimestamp(ms) {
  if (!ms) {
    return EMPTY_VALUE;
  }

  return new Date(ms).toLocaleTimeString();
}

function formatDateTime(value) {
  if (!value) {
    return EMPTY_VALUE;
  }

  const date = new Date(value);
  if (Number.isNaN(date.getTime())) {
    return value;
  }

  return date.toLocaleString();
}

function formatPlainNumber(value, digits = 2) {
  const numeric = Number(value);
  if (!Number.isFinite(numeric)) {
    return EMPTY_VALUE;
  }
  const formatted = Math.abs(numeric).toFixed(digits);
  return numeric < 0 ? `-${formatted}` : formatted;
}

function isFinitePrice(value) {
  return Number.isFinite(value) && value > 0 && value < 1_000_000;
}

function sanitizeHistory(history, fallback = []) {
  if (!Array.isArray(history)) {
    return fallback;
  }

  return history.filter((candle) => (
    candle
    && Number.isFinite(candle.time)
    && isFinitePrice(candle.open)
    && isFinitePrice(candle.high)
    && isFinitePrice(candle.low)
    && isFinitePrice(candle.close)
    && candle.high >= candle.low
  ));
}

function sanitizeCandle(candle, fallback = null) {
  if (
    !candle
    || !Number.isFinite(candle.time)
    || !isFinitePrice(candle.open)
    || !isFinitePrice(candle.high)
    || !isFinitePrice(candle.low)
    || !isFinitePrice(candle.close)
    || candle.high < candle.low
  ) {
    return fallback;
  }

  return candle;
}

function modelScoreForItem(item) {
  if (!item) {
    return null;
  }

  return item.candidate?.shadowEvaluation?.score ?? item.candidate?.evaluation?.score ?? null;
}

function ModelPnLCard({ trace }) {
  const points = Array.isArray(trace?.points) ? trace.points : [];
  const width = 320;
  const height = 160;
  const padding = 18;

  if (!trace) {
    return null;
  }

  const values = points.length ? points.map((point) => point.pnl ?? 0) : [0];
  const minValue = Math.min(...values);
  const maxValue = Math.max(...values);
  const span = Math.max(1, maxValue - minValue);
  const stepX = points.length > 1 ? (width - padding * 2) / (points.length - 1) : 0;
  const mapped = (points.length ? points : [{ time: Date.now(), pnl: 0 }]).map((point, index) => {
    const x = padding + stepX * index;
    const y = height - padding - (((point.pnl ?? 0) - minValue) / span) * (height - padding * 2);
    return { ...point, x, y };
  });
  const path = mapped.map((point, index) => `${index === 0 ? 'M' : 'L'} ${point.x} ${point.y}`).join(' ');
  const latestPnl = trace.latestPnl ?? points.at(-1)?.pnl ?? 0;

  return (
    <div className="model-pnl-card">
      <div className="model-pnl-head">
        <div>
          <div className="panel-eyebrow">{trace.role ?? 'historical'}</div>
          <h3>{trace.modelLabel}</h3>
        </div>
        <div className={`model-pnl-badge ${latestPnl >= 0 ? 'model-pnl-badge-positive' : 'model-pnl-badge-negative'}`}>
          {trace.status ?? 'archived'}
        </div>
      </div>
      <div className="model-pnl-meta">
        <span>{formatCurrency(latestPnl)}</span>
        <span>{`${trace.tradeCount ?? 0} trades`}</span>
      </div>
      <svg viewBox={`0 0 ${width} ${height}`} className="model-pnl-svg" role="img" aria-label={`${trace.modelLabel} live profit and loss`}>
        <line x1={padding} y1={padding} x2={padding} y2={height - padding} className="model-axis" />
        <line x1={padding} y1={height - padding} x2={width - padding} y2={height - padding} className="model-axis" />
        <path d={path} className={latestPnl >= 0 ? 'model-line model-line-positive' : 'model-line model-line-negative'} />
        {mapped.length ? (
          <circle
            cx={mapped.at(-1).x}
            cy={mapped.at(-1).y}
            r={5}
            className={latestPnl >= 0 ? 'model-point model-point-applied' : 'model-point'}
          />
        ) : null}
      </svg>
      <div className="model-pnl-foot">
        <span>{points.length ? 'Live trace' : 'Snapshot only'}</span>
        <span>{trace.lastUpdated ? formatDateTime(trace.lastUpdated) : EMPTY_VALUE}</span>
      </div>
    </div>
  );
}

function getHttpBaseUrl() {
  const configuredUrl = import.meta.env.VITE_API_BASE_URL;
  if (configuredUrl) {
    return configuredUrl.replace(/\/$/, '');
  }

  const apiHost = window.location.hostname || 'localhost';
  return `${window.location.protocol}//${apiHost}:8000`;
}

function getWsUrl() {
  const configuredUrl = import.meta.env.VITE_WS_URL;
  if (configuredUrl) {
    return configuredUrl.replace(/\/$/, '');
  }

  return `${getHttpBaseUrl().replace(/^https:/, 'wss:').replace(/^http:/, 'ws:')}/ws`;
}

function StatCard({ icon: Icon, label, value, hint, tone = 'neutral' }) {
  return (
    <div className={`stat-card stat-card-${tone}`}>
      <div className="stat-card-header">
        <span>{label}</span>
        <Icon size={16} />
      </div>
      <div className="stat-card-value">{value}</div>
      <div className="stat-card-hint">{hint}</div>
    </div>
  );
}

function Panel({ title, eyebrow, action, children }) {
  return (
    <section className="panel">
      <div className="panel-header">
        <div>
          {eyebrow ? <div className="panel-eyebrow">{eyebrow}</div> : null}
          <h2>{title}</h2>
        </div>
        {action}
      </div>
      {children}
    </section>
  );
}

function ModelTimelineChart({ models, selectedSequence, onSelect }) {
  const points = models.filter((item) => modelScoreForItem(item) !== null);
  if (!points.length) {
    return <div className="empty-state">No model history has been recorded yet.</div>;
  }

  const width = 760;
  const height = 280;
  const padding = 32;
  const scores = points.map((item) => modelScoreForItem(item) ?? 0);
  const minScore = Math.min(...scores);
  const maxScore = Math.max(...scores);
  const span = Math.max(1, maxScore - minScore);
  const stepX = points.length > 1 ? (width - padding * 2) / (points.length - 1) : 0;
  const mapped = points.map((item, index) => {
    const x = padding + stepX * index;
    const y = height - padding - (((modelScoreForItem(item) ?? 0) - minScore) / span) * (height - padding * 2);
    return { ...item, x, y };
  });
  const path = mapped.map((point, index) => `${index === 0 ? 'M' : 'L'} ${point.x} ${point.y}`).join(' ');

  return (
    <div className="model-chart">
      <div className="model-chart-copy">
        <div>
          <div className="panel-eyebrow">Fair model comparison score</div>
          <h2>Model improvement timeline</h2>
        </div>
        <div className="model-chart-legend">
          <span>Y axis: shadow score when available, otherwise replay score.</span>
        </div>
      </div>
      <svg viewBox={`0 0 ${width} ${height}`} className="model-chart-svg" role="img" aria-label="Model performance timeline">
        <line x1={padding} y1={padding} x2={padding} y2={height - padding} className="model-axis" />
        <line x1={padding} y1={height - padding} x2={width - padding} y2={height - padding} className="model-axis" />
        <path d={path} className="model-line" />
        {mapped.map((point) => (
          <g key={point.sequence}>
            <circle
              cx={point.x}
              cy={point.y}
              r={selectedSequence === point.sequence ? 8 : 6}
              className={point.promoted ? 'model-point model-point-applied' : 'model-point'}
              onClick={() => onSelect(point.sequence)}
            />
            <text x={point.x} y={height - 10} textAnchor="middle" className="model-label">
              {point.modelLabel}
            </text>
          </g>
        ))}
      </svg>
    </div>
  );
}

function ModelLab() {
  const httpBaseUrl = getHttpBaseUrl();
  const [models, setModels] = useState([]);
  const [selectedSequence, setSelectedSequence] = useState(null);
  const [liveTraining, setLiveTraining] = useState(INITIAL_STATE.liveTraining);
  const [portfolio, setPortfolio] = useState(INITIAL_STATE.portfolio);
  const [modelTraces, setModelTraces] = useState(INITIAL_STATE.modelTraces);

  useEffect(() => {
    let active = true;

    const fetchModelHistory = async () => {
      const response = await fetch(`${httpBaseUrl}/model-history`);
      if (!response.ok) {
        return;
      }

      const payload = await response.json();
      if (!active) {
        return;
      }

      const nextModels = Array.isArray(payload.models) ? payload.models : [];
      setModels(nextModels);
      setSelectedSequence((previous) => previous ?? nextModels.at(-1)?.sequence ?? null);
      setLiveTraining(payload.latest_state?.liveTraining ?? INITIAL_STATE.liveTraining);
      setPortfolio(payload.latest_state?.portfolio ?? INITIAL_STATE.portfolio);
      setModelTraces(Array.isArray(payload.latest_state?.modelTraces) ? payload.latest_state.modelTraces : []);
    };

    fetchModelHistory().catch(() => {});
    const interval = window.setInterval(() => {
      fetchModelHistory().catch(() => {});
    }, 10000);

    return () => {
      active = false;
      window.clearInterval(interval);
    };
  }, [httpBaseUrl]);

  const selectedModel = models.find((item) => item.sequence === selectedSequence) ?? models.at(-1) ?? null;
  const orderedTraces = [...modelTraces].sort((a, b) => {
    const left = a.lastUpdated ?? 0;
    const right = b.lastUpdated ?? 0;
    return right - left;
  });
  return (
    <div className="shell model-shell">
      <div className="shell-backdrop" />
      <header className="hero model-hero">
        <div className="hero-copy">
          <div className="hero-kicker">Model lab</div>
          <h1>Version by version evidence for live model updates.</h1>
          <p>
            Each entry records when a new model version was evaluated and the evidence behind that decision. The
            portfolio does not reset when the model changes. This page shows replay evidence and the account snapshot
            that existed when each decision was made.
          </p>
        </div>
        <div className="hero-side">
          <a className="nav-link" href="/">
            <ArrowLeft size={16} />
            <span>Back to dashboard</span>
          </a>
          <div className="price-block">
            <span>Live model status</span>
            <strong>{liveTraining.running ? 'Retraining' : 'Monitoring'}</strong>
            <small>{liveTraining.lastPromotionDecision ?? 'Waiting for first model update'}</small>
          </div>
        </div>
      </header>

      <section className="stats-grid">
        <StatCard icon={LineChart} label="Current total PnL" value={formatCurrency(portfolio.totalPnl)} hint={`${portfolio.tradeCount} closed trades so far`} tone={portfolio.totalPnl > 0 ? 'positive' : 'neutral'} />
        <StatCard icon={Radar} label="Models recorded" value={String(models.length)} hint={`${liveTraining.promotionCount ?? 0} model updates applied`} tone="neutral" />
        <StatCard icon={BrainCircuit} label="Feedback samples" value={String(liveTraining.tradeFeedbackCount ?? 0)} hint="Closed trades being fed back into retraining" tone="neutral" />
        <StatCard icon={TrendingUp} label="Last update" value={liveTraining.lastPromotedAt ?? EMPTY_VALUE} hint={`Active version ${liveTraining.activeModelVersion ?? EMPTY_VALUE}`} tone="neutral" />
      </section>

      <main className="dashboard-grid model-grid">
        <div className="primary-column">
          <Panel eyebrow="Model timeline" title="Evaluation score at each model decision">
            <ModelTimelineChart models={models} selectedSequence={selectedSequence} onSelect={setSelectedSequence} />
          </Panel>

          <Panel eyebrow="Model ledger" title="Recorded model results">
            <div className="model-history-table">
              <div className="model-history-head">
                <span>Model</span>
                <span>When</span>
                <span>Status</span>
                <span>Score</span>
                <span>Accuracy</span>
              </div>
              {models.length ? models.slice().reverse().map((item) => (
                <button
                  type="button"
                  key={item.sequence}
                  className={`model-history-row ${selectedSequence === item.sequence ? 'model-history-row-active' : ''}`}
                  onClick={() => setSelectedSequence(item.sequence)}
                >
                  <span>{item.modelLabel}</span>
                  <span>{formatDateTime(item.timestamp)}</span>
                  <span>{item.promoted ? 'Applied' : 'Rejected'}</span>
                  <span className={(modelScoreForItem(item) ?? 0) >= 0 ? 'pnl-positive' : 'pnl-negative'}>
                    {modelScoreForItem(item) ?? EMPTY_VALUE}
                  </span>
                  <span>{`${Math.round((item.candidate?.valAccuracy ?? 0) * 100)}%`}</span>
                </button>
              )) : (
                <div className="empty-state">No model candidates have been logged yet.</div>
              )}
            </div>
          </Panel>
        </div>

        <div className="secondary-column">
          <Panel
            eyebrow="Selected model"
            title={selectedModel ? `${selectedModel.modelLabel} details` : 'Model details'}
            action={<ExternalLink size={18} className="panel-icon" />}
          >
            {selectedModel ? (
              <div className="list-table">
                <div className="list-row">
                  <span>Decision time</span>
                  <strong>{formatDateTime(selectedModel.timestamp)}</strong>
                </div>
                <div className="list-row">
                  <span>Update result</span>
                  <strong>{selectedModel.promoted ? 'Applied to live model' : 'Rejected'}</strong>
                </div>
                <div className="list-row">
                  <span>Decision reason</span>
                  <strong>{selectedModel.decision}</strong>
                </div>
                <div className="list-row">
                  <span>PnL at implementation</span>
                  <strong>{formatCurrency(selectedModel.portfolioAtDecision?.totalPnl)}</strong>
                </div>
                <div className="list-row">
                  <span>Realized PnL at implementation</span>
                  <strong>{formatCurrency(selectedModel.portfolioAtDecision?.realizedPnl)}</strong>
                </div>
                <div className="list-row">
                  <span>Trade count at implementation</span>
                  <strong>{selectedModel.portfolioAtDecision?.tradeCount ?? EMPTY_VALUE}</strong>
                </div>
                <div className="list-row">
                  <span>New version validation accuracy</span>
                  <strong>{`${Math.round((selectedModel.candidate?.valAccuracy ?? 0) * 100)}%`}</strong>
                </div>
                <div className="list-row">
                  <span>New version replay score</span>
                  <strong>{selectedModel.candidate?.evaluation?.score ?? EMPTY_VALUE}</strong>
                </div>
                <div className="list-row">
                  <span>New version live evaluation score</span>
                  <strong>{selectedModel.candidate?.shadowEvaluation?.score ?? EMPTY_VALUE}</strong>
                </div>
                <div className="list-row">
                  <span>New version replay net PnL</span>
                  <strong>{formatCurrency(selectedModel.candidate?.evaluation?.netPnl)}</strong>
                </div>
                <div className="list-row">
                  <span>New version live evaluation net PnL</span>
                  <strong>{formatCurrency(selectedModel.candidate?.shadowEvaluation?.netPnl)}</strong>
                </div>
                <div className="list-row">
                  <span>New version replay drawdown</span>
                  <strong>
                    {selectedModel.candidate?.evaluation?.maxDrawdownPct !== undefined
                      ? `${selectedModel.candidate.evaluation.maxDrawdownPct}%`
                      : EMPTY_VALUE}
                  </strong>
                </div>
                <div className="list-row">
                  <span>New version live evaluation drawdown</span>
                  <strong>
                    {selectedModel.candidate?.shadowEvaluation?.maxDrawdownPct !== undefined
                      ? `${selectedModel.candidate.shadowEvaluation.maxDrawdownPct}%`
                      : EMPTY_VALUE}
                  </strong>
                </div>
                <div className="list-row">
                  <span>New version best validation loss</span>
                  <strong>{selectedModel.candidate?.bestValLoss?.toFixed?.(4) ?? EMPTY_VALUE}</strong>
                </div>
                <div className="list-row">
                  <span>Feedback matches used</span>
                  <strong>{selectedModel.candidate?.feedbackMatches ?? EMPTY_VALUE}</strong>
                </div>
                <div className="list-row">
                  <span>Previous live model version</span>
                  <strong>{selectedModel.activeBefore?.version ?? EMPTY_VALUE}</strong>
                </div>
                <div className="list-row">
                  <span>Previous live model replay score</span>
                  <strong>{selectedModel.activeBefore?.evaluation?.score ?? EMPTY_VALUE}</strong>
                </div>
                <div className="list-row">
                  <span>Previous live model evaluation score</span>
                  <strong>{selectedModel.activeBefore?.shadowEvaluation?.score ?? EMPTY_VALUE}</strong>
                </div>
                <div className="list-row">
                  <span>Live model after decision</span>
                  <strong>{selectedModel.activeAfter?.version ?? EMPTY_VALUE}</strong>
                </div>
                <div className="list-row">
                  <span>Portfolio PnL snapshot</span>
                  <strong>{formatCurrency(selectedModel.portfolioAtDecision?.totalPnl)}</strong>
                </div>
              </div>
            ) : (
              <div className="empty-state">Pick a model from the ledger to inspect its metrics.</div>
            )}
          </Panel>
        </div>
      </main>

      <section className="model-live-grid">
        {orderedTraces.length ? orderedTraces.map((trace) => (
          <ModelPnLCard key={trace.modelLabel} trace={trace} />
        )) : (
          <div className="empty-state">Model traces will appear here when the backend records PnL snapshots.</div>
        )}
      </section>
    </div>
  );
}

function DashboardApp() {
  const chartContainerRef = useRef(null);
  const chartRef = useRef(null);
  const candleSeriesRef = useRef(null);
  const reconnectTimeoutRef = useRef(null);
  const mountedRef = useRef(false);
  const socketRef = useRef(null);
  const historyCountRef = useRef(0);

  const [dashboardState, setDashboardState] = useState(INITIAL_STATE);
  const [connectionState, setConnectionState] = useState('connecting');
  const [budgetDraft, setBudgetDraft] = useState(Number(INITIAL_STATE.simulation.initialEquity));
  const [ignoreFeesDraft, setIgnoreFeesDraft] = useState(INITIAL_STATE.simulation.ignoreFees);
  const [isStartingSimulation, setIsStartingSimulation] = useState(false);
  const [networkEnabledDraft, setNetworkEnabledDraft] = useState(INITIAL_STATE.networkTest.enabled);
  const [latencyDraft, setLatencyDraft] = useState(INITIAL_STATE.networkTest.latencyMs);
  const [jitterDraft, setJitterDraft] = useState(INITIAL_STATE.networkTest.jitterMs);
  const [packetLossDraft, setPacketLossDraft] = useState(INITIAL_STATE.networkTest.packetLossPct);
  const [isSavingNetworkProfile, setIsSavingNetworkProfile] = useState(false);
  const [networkAutomation, setNetworkAutomation] = useState(INITIAL_STATE.networkAutomation);
  const [networkAutomationReport, setNetworkAutomationReport] = useState(null);
  const [isStartingAutomation, setIsStartingAutomation] = useState(false);
  const [isStoppingAutomation, setIsStoppingAutomation] = useState(false);
  const [comparisonModelFilter, setComparisonModelFilter] = useState('all');
  const httpBaseUrl = getHttpBaseUrl();
  const wsUrl = getWsUrl();

  const latestPrice = dashboardState.candle?.close ?? dashboardState.history.at(-1)?.close ?? null;
  const prediction = dashboardState.prediction;
  const portfolio = dashboardState.portfolio;
  const telemetry = dashboardState.telemetry;
  const simulation = dashboardState.simulation ?? INITIAL_STATE.simulation;
  const networkTest = dashboardState.networkTest ?? INITIAL_STATE.networkTest;
  const simulationSummary = dashboardState.simulationSummary ?? INITIAL_STATE.simulationSummary;
  const liveTraining = dashboardState.liveTraining ?? INITIAL_STATE.liveTraining;
  const activity = dashboardState.activity ?? [];
  const blotter = dashboardState.blotter ?? [];

  const activePatterns = useMemo(
    () => Object.entries(dashboardState.patterns).filter(([, active]) => active).map(([name]) => name),
    [dashboardState.patterns],
  );

  const performanceTone = portfolio.totalPnl > 0 ? 'positive' : portfolio.totalPnl < 0 ? 'negative' : 'neutral';
  const actionTone = prediction.action === 'BUY' ? 'positive' : prediction.action === 'SELL' ? 'negative' : 'neutral';

  useEffect(() => {
    setBudgetDraft(Number(simulation.initialEquity));
    setIgnoreFeesDraft(Boolean(simulation.ignoreFees));
  }, [simulation.initialEquity, simulation.ignoreFees]);

  useEffect(() => {
    setNetworkEnabledDraft(Boolean(networkTest.enabled));
    setLatencyDraft(Number(networkTest.latencyMs ?? 0));
    setJitterDraft(Number(networkTest.jitterMs ?? 0));
    setPacketLossDraft(Number(networkTest.packetLossPct ?? 0));
  }, [networkTest.enabled, networkTest.latencyMs, networkTest.jitterMs, networkTest.packetLossPct]);

  useEffect(() => {
    let active = true;

    const refreshAutomation = async () => {
      try {
        const [statusResponse, reportResponse] = await Promise.all([
          fetch(`${httpBaseUrl}/network-test/status`),
          fetch(`${httpBaseUrl}/network-test/report`),
        ]);

        if (!statusResponse.ok || !reportResponse.ok || !active) {
          return;
        }

        const statusPayload = await statusResponse.json();
        const reportPayload = await reportResponse.json();
        if (!active) {
          return;
        }

        setNetworkAutomation({
          running: Boolean(statusPayload.running),
          currentPhase: statusPayload.currentPhase ?? null,
          currentModel: statusPayload.currentModel ?? null,
          startedAt: statusPayload.startedAt ?? null,
          endedAt: statusPayload.endedAt ?? null,
          fetchLoops: Number(statusPayload.fetchLoops ?? 1),
          pollSeconds: Number(statusPayload.pollSeconds ?? 2),
          sampleCount: Number(statusPayload.sampleCount ?? 0),
          completedRuns: Number(statusPayload.completedRuns ?? 0),
          totalRuns: Number(statusPayload.totalRuns ?? 12),
          currentSamples: Number(statusPayload.currentSamples ?? 0),
          estimatedSamples: Number(statusPayload.estimatedSamples ?? 0),
          phases: Array.isArray(statusPayload.phases) ? statusPayload.phases : [],
        });
        setNetworkAutomationReport(reportPayload.report ?? null);
      } catch (_) {
        // keep prior state on transient backend issues
      }
    };

    refreshAutomation().catch(() => {});
    const interval = window.setInterval(() => {
      refreshAutomation().catch(() => {});
    }, 5000);

    return () => {
      active = false;
      window.clearInterval(interval);
    };
  }, [httpBaseUrl]);

  useEffect(() => {
    mountedRef.current = true;

    const container = chartContainerRef.current;
    if (!container) {
      return () => {
        mountedRef.current = false;
      };
    }

    const chart = createChart(container, {
      autoSize: true,
      height: 520,
      layout: {
        background: { color: '#07111f' },
        textColor: '#9fb3c8',
        attributionLogo: false,
      },
      grid: {
        vertLines: { color: 'rgba(128, 157, 184, 0.08)' },
        horzLines: { color: 'rgba(128, 157, 184, 0.08)' },
      },
      rightPriceScale: {
        borderColor: 'rgba(128, 157, 184, 0.18)',
        autoScale: true,
      },
      timeScale: {
        borderColor: 'rgba(128, 157, 184, 0.18)',
        timeVisible: true,
        secondsVisible: false,
      },
      crosshair: {
        vertLine: { color: 'rgba(255, 255, 255, 0.12)' },
        horzLine: { color: 'rgba(255, 255, 255, 0.12)' },
      },
      localization: {
        priceFormatter: (price) => formatCurrency(price),
      },
    });

    const candleSeries = chart.addSeries(CandlestickSeries, {
      upColor: '#21c58b',
      downColor: '#f45b69',
      wickUpColor: '#21c58b',
      wickDownColor: '#f45b69',
      borderUpColor: '#21c58b',
      borderDownColor: '#f45b69',
      priceFormat: {
        type: 'price',
        precision: 2,
        minMove: 0.01,
      },
    });

    chartRef.current = chart;
    candleSeriesRef.current = candleSeries;

    const resizeObserver = new ResizeObserver(() => {
      chart.timeScale().fitContent();
    });
    resizeObserver.observe(container);

    return () => {
      mountedRef.current = false;
      resizeObserver.disconnect();
      chart.remove();
      chartRef.current = null;
      candleSeriesRef.current = null;
    };
  }, []);

  useEffect(() => {
    const candleSeries = candleSeriesRef.current;
    const chart = chartRef.current;

    if (!candleSeries || !chart) {
      return;
    }

    try {
      if (dashboardState.history.length && dashboardState.history.length !== historyCountRef.current) {
        candleSeries.setData(dashboardState.history);
        historyCountRef.current = dashboardState.history.length;
        chart.timeScale().fitContent();
      }

      if (dashboardState.candle) {
        const latestHistoryTime = dashboardState.history.at(-1)?.time ?? null;
        if (!latestHistoryTime || dashboardState.candle.time > latestHistoryTime) {
          candleSeries.update(dashboardState.candle);
        }
        chart.timeScale().scrollToRealTime();
      }

    } catch {
      setConnectionState('degraded');
    }
  }, [dashboardState.history, dashboardState.candle]);

  useEffect(() => {
    let manuallyClosed = false;

    const bootstrap = async () => {
      try {
        const response = await fetch(`${httpBaseUrl}/health`);
        if (!response.ok) {
          return;
        }

        const payload = await response.json();
        if (!mountedRef.current || !payload?.latest_state) {
          return;
        }

        setDashboardState((previous) => ({
          ...previous,
          ...payload.latest_state,
          history: sanitizeHistory(payload.latest_state.history, previous.history),
          candle: sanitizeCandle(payload.latest_state.candle, previous.candle),
          logs: Array.isArray(payload.latest_state.logs) ? payload.latest_state.logs : previous.logs,
        }));
      } catch {
        // Websocket reconnect logic will handle unavailable backend cases.
      }
    };

    const connect = () => {
      if (!mountedRef.current) {
        return;
      }

      setConnectionState('connecting');
      const socket = new WebSocket(wsUrl);
      socketRef.current = socket;

      socket.onopen = () => {
        if (!mountedRef.current) {
          return;
        }

        setConnectionState('online');
      };

      socket.onmessage = (event) => {
        if (!mountedRef.current) {
          return;
        }

        try {
          const payload = JSON.parse(event.data);
          setDashboardState((previous) => ({
            ...previous,
            ...payload,
            history: sanitizeHistory(payload.history, previous.history),
            candle: sanitizeCandle(payload.candle, previous.candle),
            logs: Array.isArray(payload.logs) ? payload.logs : previous.logs,
          }));
        } catch {
          setConnectionState('degraded');
        }
      };

      socket.onerror = () => {
        if (!mountedRef.current) {
          return;
        }

        setConnectionState('degraded');
      };

      socket.onclose = () => {
        if (!mountedRef.current || manuallyClosed) {
          return;
        }

        setConnectionState('offline');
        reconnectTimeoutRef.current = window.setTimeout(connect, 2500);
      };
    };

    bootstrap();
    connect();

    return () => {
      manuallyClosed = true;
      if (reconnectTimeoutRef.current) {
        window.clearTimeout(reconnectTimeoutRef.current);
      }
      const socket = socketRef.current;
      if (socket && socket.readyState === WebSocket.OPEN) {
        socket.close();
      }
    };
  }, [httpBaseUrl, wsUrl]);

  const heroStats = [
    {
      icon: TrendingUp,
      label: 'Portfolio PnL',
      value: formatCurrency(portfolio.totalPnl),
      hint: `${formatCurrency(portfolio.realizedPnl)} realized / ${formatCurrency(portfolio.unrealizedPnl)} unrealized`,
      tone: performanceTone,
    },
    {
      icon: BrainCircuit,
      label: 'Signal Confidence',
      value: `${Math.round(prediction.confidence * 100)}%`,
      hint: `${prediction.action} signal with ${Math.round(prediction.confidence * 100)}% confidence`,
      tone: actionTone,
    },
    {
      icon: Gauge,
      label: 'Decision Delay',
      value: `${telemetry.deltaMs} ms`,
      hint: `Tick ${formatTimestamp(telemetry.tickTime)} to signal ${formatTimestamp(telemetry.signalTime)}`,
      tone: telemetry.deltaMs > 500 ? 'negative' : 'neutral',
    },
    {
      icon: Wallet,
      label: 'Capital At Risk',
      value: formatCurrency(portfolio.equity),
      hint: `${formatCurrency(portfolio.initialEquity)} start, ${formatCurrency(portfolio.positionNotional)} currently deployed`,
      tone: portfolio.signalAccuracy >= 0.5 ? 'positive' : 'neutral',
    },
  ];

  const modelStats = [
    ['Market price', formatCurrency(latestPrice)],
    ['Available cash', formatCurrency(portfolio.availableCash)],
    ['Current notional', formatCurrency(portfolio.positionNotional)],
    ['Allocation', formatPercent(portfolio.allocationPct)],
    ['Confluence score', telemetry.confluenceScore.toFixed(2)],
    ['Trend bias', formatPercent(telemetry.trendBiasPct)],
    ['Volatility (20)', formatPercent(telemetry.volatilityPct)],
    ['Candle range', formatPercent(telemetry.rangePct)],
    ['Update throughput', `${telemetry.updatesPerMinute}/min`],
    ['Packet loss estimate', `${telemetry.packetLossPct.toFixed(2)}%`],
    ['Simulated network delay', `${(telemetry.simulatedDelayMs ?? 0).toFixed(0)} ms`],
    ['Processed candles', telemetry.processedCandles ?? 0],
    ['Dropped candles', telemetry.droppedCandles ?? 0],
  ];

  const automationSummaryRows = Array.isArray(networkAutomationReport?.summary_rows)
    ? networkAutomationReport.summary_rows
    : [];
  const comparisonModels = Array.isArray(networkAutomationReport?.models)
    ? networkAutomationReport.models
    : [];
  const filteredAutomationRows = comparisonModelFilter === 'all'
    ? automationSummaryRows
    : automationSummaryRows.filter((row) => row.modelKey === comparisonModelFilter);
  // The backend evaluates all models together on the same replay.
  // This client-side filter only changes which rows are shown in the table.

  const startSimulation = async () => {
    const parsedBudget = Number(budgetDraft);
    if (!Number.isFinite(parsedBudget) || parsedBudget <= 0) {
      return;
    }
    setIsStartingSimulation(true);
    try {
      const response = await fetch(`${httpBaseUrl}/simulation/start`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          budget: parsedBudget,
          ignoreFees: ignoreFeesDraft,
        }),
      });

      if (!response.ok) {
        return;
      }

      const payload = await response.json();
      if (payload?.latest_state) {
        setDashboardState((previous) => ({
          ...previous,
          ...payload.latest_state,
          history: sanitizeHistory(payload.latest_state.history, previous.history),
          candle: sanitizeCandle(payload.latest_state.candle, previous.candle),
          logs: Array.isArray(payload.latest_state.logs) ? payload.latest_state.logs : previous.logs,
        }));
      }
    } finally {
      setIsStartingSimulation(false);
    }
  };

  const saveNetworkProfile = async () => {
    setIsSavingNetworkProfile(true);
    try {
      const response = await fetch(`${httpBaseUrl}/network/profile`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          enabled: networkEnabledDraft,
          latencyMs: Number(latencyDraft),
          jitterMs: Number(jitterDraft),
          packetLossPct: Number(packetLossDraft),
        }),
      });

      if (!response.ok) {
        return;
      }

      const payload = await response.json();
      if (payload?.latest_state) {
        setDashboardState((previous) => ({
          ...previous,
          ...payload.latest_state,
          history: sanitizeHistory(payload.latest_state.history, previous.history),
          candle: sanitizeCandle(payload.latest_state.candle, previous.candle),
          logs: Array.isArray(payload.latest_state.logs) ? payload.latest_state.logs : previous.logs,
        }));
      }
    } finally {
      setIsSavingNetworkProfile(false);
    }
  };

  const startAutomatedNetworkTest = async () => {
    setIsStartingAutomation(true);
    try {
      const response = await fetch(`${httpBaseUrl}/network-test/start`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ fetchLoops: 1, pollSeconds: 2 }),
      });
      if (!response.ok) {
        return;
      }
      const payload = await response.json();
      setNetworkAutomation((prev) => ({
        ...prev,
        running: Boolean(payload.running),
        fetchLoops: Number(payload.fetchLoops ?? prev.fetchLoops ?? 1),
        pollSeconds: Number(payload.pollSeconds ?? 2),
        completedRuns: Number(payload.completedRuns ?? 0),
        totalRuns: Number(payload.totalRuns ?? prev.totalRuns ?? 12),
        currentSamples: Number(payload.currentSamples ?? 0),
        estimatedSamples: Number(payload.estimatedSamples ?? 0),
        phases: Array.isArray(payload.phases) ? payload.phases : prev.phases,
      }));
      setNetworkAutomationReport(null);
    } finally {
      setIsStartingAutomation(false);
    }
  };

  const stopAutomatedNetworkTest = async () => {
    setIsStoppingAutomation(true);
    try {
      const response = await fetch(`${httpBaseUrl}/network-test/stop`, {
        method: 'POST',
      });
      if (!response.ok) {
        return;
      }
      const payload = await response.json();
      setNetworkAutomation((prev) => ({ ...prev, running: false, currentPhase: null }));
      setNetworkAutomationReport(payload.report ?? null);
    } finally {
      setIsStoppingAutomation(false);
    }
  };

  return (
    <div className="shell">
      <div className="shell-backdrop" />

      {/* ── Compact top nav ──────────────────────────────────────────────── */}
      <header className="hero">
        <div className="hero-copy">
          <div className="hero-model-tags">
            <span className="hero-model-tag hero-model-primary">LSTM</span>
            <span className="hero-model-tag-sep">+</span>
            <span className="hero-model-tag hero-model-repair">SARIMA</span>
          </div>
          <h1>BTC Trading Bot</h1>
          <p className="hero-sub">Live paper trading on BTC/USDT with LSTM signals and SARIMA gap repair</p>
        </div>
        <div className="hero-side">
          <div className="hero-price">
            <span className="hero-price-label">BTC / USDT</span>
            <strong className="hero-price-value">{formatCurrency(latestPrice)}</strong>
            <div className={`signal-badge signal-${actionTone} hero-signal`}>
              {prediction.action === 'BUY' ? <ArrowUpRight size={13} /> : prediction.action === 'SELL' ? <ArrowDownRight size={13} /> : <Activity size={13} />}
              {prediction.action} · {Math.round(prediction.confidence * 100)}%
            </div>
          </div>
          <div className="hero-divider" />
          <div className={`status-pill status-${connectionState}`}>
            <Wifi size={14} />
            <span>{connectionState === 'online' ? 'Online' : connectionState === 'connecting' ? 'Connecting' : connectionState === 'degraded' ? 'Degraded' : 'Offline'}</span>
          </div>
          <a className="nav-link" href="/models" target="_blank" rel="noreferrer">
            <LineChart size={14} />
            <span>Model lab</span>
          </a>
        </div>
      </header>

      {/* ── Key stats ────────────────────────────────────────────────────── */}
      <section className="stats-grid">
        {heroStats.map((stat) => (
          <StatCard key={stat.label} {...stat} />
        ))}
      </section>

      {/* ── Main two-column grid ─────────────────────────────────────────── */}
      <main className="dashboard-grid">

        {/* ── PRIMARY: chart + trading details + logs ─────────────────── */}
        <div className="primary-column">

          {/* 1. Live market chart */}
          <Panel
            eyebrow="Live market"
            title="BTC/USDT live candlestick chart"
            action={
              <div className={`signal-badge signal-${actionTone}`}>
                {prediction.action === 'BUY' ? <ArrowUpRight size={15} /> : prediction.action === 'SELL' ? <ArrowDownRight size={15} /> : <Activity size={15} />}
                <span>{prediction.action}</span>
              </div>
            }
          >
            <div className="pattern-row">
              {activePatterns.length ? activePatterns.map((pattern) => (
                <span key={pattern} className="pattern-chip">{pattern.replace('_', ' ')}</span>
              )) : <span className="pattern-chip pattern-chip-muted">No active structure flags</span>}
            </div>
            <div ref={chartContainerRef} className="chart-shell" />
            <div className="chart-footer">
              <span>Close {latestPrice ? formatCurrency(latestPrice) : EMPTY_VALUE}</span>
              <span>Position {portfolio.position}</span>
            </div>
          </Panel>

          {/* 2. Portfolio + Signal telemetry (side by side) */}
          <div className="analysis-grid">
            <Panel eyebrow="Paper trading" title="Portfolio and execution">
              <div className="mini-grid">
                <div className="metric">
                  <span>Initial equity</span>
                  <strong>{formatCurrency(portfolio.initialEquity)}</strong>
                </div>
                <div className="metric">
                  <span>Equity</span>
                  <strong>{formatCurrency(portfolio.equity)}</strong>
                </div>
                <div className="metric">
                  <span>Available cash</span>
                  <strong>{formatCurrency(portfolio.availableCash)}</strong>
                </div>
                <div className="metric">
                  <span>Fees paid</span>
                  <strong>{formatCurrency(portfolio.feesPaid)}</strong>
                </div>
                <div className="metric">
                  <span>Position size</span>
                  <strong>{portfolio.positionSize ? `${portfolio.positionSize.toFixed(4)} BTC` : '0 BTC'}</strong>
                </div>
                <div className="metric">
                  <span>Position notional</span>
                  <strong>{formatCurrency(portfolio.positionNotional)}</strong>
                </div>
                <div className="metric">
                  <span>Deployed capital</span>
                  <strong>{formatCurrency(portfolio.deployedCapital)}</strong>
                </div>
                <div className="metric">
                  <span>Allocation</span>
                  <strong>{formatPercent(portfolio.allocationPct)}</strong>
                </div>
                <div className="metric">
                  <span>Entry price</span>
                  <strong>{portfolio.entryPrice ? formatCurrency(portfolio.entryPrice) : EMPTY_VALUE}</strong>
                </div>
              </div>
            </Panel>

            <Panel eyebrow="Model telemetry" title="Signal timing">
              <div className="mini-grid">
                <div className="metric">
                  <span>Tick received</span>
                  <strong>{formatTimestamp(telemetry.receivedTime)}</strong>
                </div>
                <div className="metric">
                  <span>Signal generated</span>
                  <strong>{formatTimestamp(telemetry.signalTime)}</strong>
                </div>
                <div className="metric">
                  <span>Backend latency</span>
                  <strong>{telemetry.latencyMs} ms</strong>
                </div>
                <div className="metric">
                  <span>Updates / min</span>
                  <strong>{telemetry.updatesPerMinute}</strong>
                </div>
                <div className="metric">
                  <span>Confluence score</span>
                  <strong>{telemetry.confluenceScore.toFixed(2)}</strong>
                </div>
                <div className="metric">
                  <span>Volatility (20)</span>
                  <strong>{formatPercent(telemetry.volatilityPct)}</strong>
                </div>
                <div className="metric">
                  <span>Processed candles</span>
                  <strong>{telemetry.processedCandles ?? 0}</strong>
                </div>
                <div className="metric">
                  <span>Dropped candles</span>
                  <strong>{telemetry.droppedCandles ?? 0}</strong>
                </div>
              </div>
            </Panel>
          </div>

          {/* 3. Activity log */}
          <Panel
            eyebrow="Simple activity"
            title="What the AI just did"
            action={<Activity size={16} className="panel-icon" />}
          >
            <div className="log-stream compact-stream">
              {activity.length ? activity.slice().reverse().map((entry, index) => (
                <div key={`${entry.time}-${index}`} className="log-entry log-info">
                  <div className="log-time">{entry.time}</div>
                  <div>{entry.text}</div>
                </div>
              )) : (
                <div className="empty-state">No trades or actions yet.</div>
              )}
            </div>
          </Panel>

          {/* 4. Trade blotter */}
          <Panel
            eyebrow="Trade blotter"
            title="Closed trades"
            action={<Wallet size={16} className="panel-icon" />}
          >
            <div className="blotter-table">
              <div className="blotter-head">
                <span>Side</span>
                <span>Size</span>
                <span>Entry</span>
                <span>Exit</span>
                <span>Net</span>
                <span>Reason</span>
              </div>
              {blotter.length ? blotter.map((trade) => (
                <div key={trade.id} className="blotter-row">
                  <span>{trade.side}</span>
                  <span>{trade.size.toFixed(4)} BTC</span>
                  <span>{formatCurrency(trade.entryPrice)}</span>
                  <span>{formatCurrency(trade.exitPrice)}</span>
                  <span className={trade.netPnl >= 0 ? 'pnl-positive' : 'pnl-negative'}>{formatCurrency(trade.netPnl)}</span>
                  <span>{trade.reason}</span>
                </div>
              )) : (
                <div className="empty-state">No closed trades yet.</div>
              )}
            </div>
          </Panel>

        </div>

        {/* ── SECONDARY: controls + models + network ───────────────────── */}
        <div className="secondary-column">

          {/* Group A: Simulation controls */}
          <div className="section-group">
            <div className="section-group-header">
              <SlidersHorizontal size={12} />
              <span>Simulation</span>
            </div>
            <Panel
              eyebrow="Paper trading controls"
              title="Start simulation"
            >
              <label className="metric metric-form">
                <span>Simulation budget</span>
                <input
                  type="number"
                  step="100"
                  value={budgetDraft}
                  onChange={(event) => setBudgetDraft(Number(event.target.value))}
                />
              </label>
              <div className="list-table compact-table">
                <div className="list-row">
                  <span>Active budget</span>
                  <strong>{formatCurrency(simulation.initialEquity)}</strong>
                </div>
              </div>
              <label className="metric-toggle" style={{ marginTop: 10 }}>
                <span>Ignore fees for testing</span>
                <input
                  type="checkbox"
                  checked={ignoreFeesDraft}
                  onChange={(event) => setIgnoreFeesDraft(event.target.checked)}
                />
              </label>
              <div className="list-table compact-table">
                <div className="list-row">
                  <span>Allocation range</span>
                  <strong>{`${simulationSummary.aiPolicy.allocationRangePct[0]}–${simulationSummary.aiPolicy.allocationRangePct[1]}%`}</strong>
                </div>
                <div className="list-row">
                  <span>Min confidence</span>
                  <strong>{`${simulationSummary.aiPolicy.minConfidencePct}%`}</strong>
                </div>
                <div className="list-row">
                  <span>Min notional</span>
                  <strong>{formatCurrency(simulationSummary.aiPolicy.minTradeNotional)}</strong>
                </div>
                <div className="list-row">
                  <span>Fees</span>
                  <strong>{simulationSummary.aiPolicy.ignoreFees ? 'Ignored' : `${simulationSummary.aiPolicy.roundTripCostPct}% round trip`}</strong>
                </div>
                <div className="list-row">
                  <span>Directions</span>
                  <strong>{`${simulationSummary.aiPolicy.allowLong ? 'Long' : ''}${simulationSummary.aiPolicy.allowLong && simulationSummary.aiPolicy.allowShort ? ' / ' : ''}${simulationSummary.aiPolicy.allowShort ? 'Short' : ''}`}</strong>
                </div>
              </div>
              <button className="save-button" onClick={startSimulation} disabled={isStartingSimulation}>
                <Save size={15} />
                <span>{isStartingSimulation ? 'Starting...' : 'Start simulation'}</span>
              </button>
            </Panel>
          </div>

          {/* Group B: Network testing */}
          <div className="section-group">
            <div className="section-group-header">
              <Wifi size={12} />
              <span>Network testing</span>
            </div>

            <Panel eyebrow="Condition simulation" title="Demo network conditions">
              <label className="metric-toggle">
                <span>Enable network simulation</span>
                <input
                  type="checkbox"
                  checked={networkEnabledDraft}
                  onChange={(event) => setNetworkEnabledDraft(event.target.checked)}
                />
              </label>
              <label className="slider-control">
                <span>Latency: {Math.round(latencyDraft)} ms</span>
                <input
                  type="range"
                  min="0"
                  max="1200"
                  step="10"
                  value={latencyDraft}
                  onChange={(event) => setLatencyDraft(Number(event.target.value))}
                />
              </label>
              <label className="slider-control">
                <span>Jitter: {Math.round(jitterDraft)} ms</span>
                <input
                  type="range"
                  min="0"
                  max="800"
                  step="10"
                  value={jitterDraft}
                  onChange={(event) => setJitterDraft(Number(event.target.value))}
                />
              </label>
              <label className="slider-control">
                <span>Packet loss: {packetLossDraft.toFixed(1)}%</span>
                <input
                  type="range"
                  min="0"
                  max="40"
                  step="0.5"
                  value={packetLossDraft}
                  onChange={(event) => setPacketLossDraft(Number(event.target.value))}
                />
              </label>
              <button className="save-button" onClick={saveNetworkProfile} disabled={isSavingNetworkProfile}>
                <Save size={15} />
                <span>{isSavingNetworkProfile ? 'Applying...' : 'Apply profile'}</span>
              </button>
            </Panel>

            <Panel eyebrow="Automated research" title="Multi-model comparison run">
              <div className="automation-actions">
                <button
                  className="save-button"
                  onClick={startAutomatedNetworkTest}
                  disabled={isStartingAutomation || networkAutomation.running}
                >
                  <PlayCircle size={15} />
                  <span>{isStartingAutomation ? 'Starting...' : 'Start test'}</span>
                </button>
                <button
                  className="save-button save-button-danger"
                  onClick={stopAutomatedNetworkTest}
                  disabled={isStoppingAutomation || !networkAutomation.running}
                >
                  <StopCircle size={15} />
                  <span>{isStoppingAutomation ? 'Stopping...' : 'Stop + report'}</span>
                </button>
              </div>
              <div className="metric-stack">
                <div className="metric">
                  <span>Status</span>
                  <strong>{networkAutomation.running ? 'Running' : 'Idle'}</strong>
                </div>
                <div className="metric">
                  <span>Current phase</span>
                  <strong>{networkAutomation.currentPhase ?? EMPTY_VALUE}</strong>
                </div>
                <div className="metric">
                  <span>Current model</span>
                  <strong>{networkAutomation.currentModel ?? EMPTY_VALUE}</strong>
                </div>
                <div className="metric">
                  <span>Runs completed</span>
                  <strong>{networkAutomation.completedRuns} / {networkAutomation.totalRuns}</strong>
                </div>
              </div>
            </Panel>

            <Panel eyebrow="Network evaluation" title="Condition indicators" action={<Sparkles size={16} className="panel-icon" />}>
              <div className="indicator-stack">
                <div className="indicator">
                  <div className="indicator-copy">
                    <span>Signal delay Δt</span>
                    <strong>{telemetry.deltaMs} ms</strong>
                  </div>
                  <div className="indicator-bar">
                    <div style={{ width: `${Math.min(100, telemetry.deltaMs / 8)}%` }} />
                  </div>
                </div>
                <div className="indicator">
                  <div className="indicator-copy">
                    <span>Volatility load</span>
                    <strong>{formatPercent(telemetry.volatilityPct)}</strong>
                  </div>
                  <div className="indicator-bar">
                    <div style={{ width: `${Math.min(100, telemetry.volatilityPct * 12)}%` }} />
                  </div>
                </div>
                <div className="indicator">
                  <div className="indicator-copy">
                    <span>Packet loss estimate</span>
                    <strong>{telemetry.packetLossPct.toFixed(2)}%</strong>
                  </div>
                  <div className="indicator-bar">
                    <div style={{ width: `${Math.min(100, telemetry.packetLossPct * 20)}%` }} />
                  </div>
                </div>
                <div className="indicator">
                  <div className="indicator-copy">
                    <span>Simulated delay</span>
                    <strong>{(telemetry.simulatedDelayMs ?? 0).toFixed(0)} ms</strong>
                  </div>
                  <div className="indicator-bar">
                    <div style={{ width: `${Math.min(100, (telemetry.simulatedDelayMs ?? 0) / 8)}%` }} />
                  </div>
                </div>
              </div>
            </Panel>
          </div>

          <div className="section-group">
            <div className="section-group-header">
              <ShieldCheck size={12} />
              <span>About</span>
            </div>
            <Panel
              eyebrow="System overview"
              title="What this site does"
              action={<Sparkles size={16} className="panel-icon" />}
            >
              <p className="site-summary-copy">
                Live paper trading on BTC/USDT. Streams 1-minute Binance candles, extracts market structure features,
                and routes them through LSTM for signals. SARIMA fills any gaps caused by network degradation.
              </p>
              <div className="site-summary-points">
                <div>
                  <span>Trading model</span>
                  <strong>LSTM produces BUY / SELL / HOLD signals from price structure features</strong>
                </div>
                <div>
                  <span>Network resilience</span>
                  <strong>SARIMA imputes missing candles so LSTM always receives clean input</strong>
                </div>
                <div>
                  <span>Evaluation goal</span>
                  <strong>Measure how network degradation affects signal quality and PnL</strong>
                </div>
              </div>
            </Panel>
          </div>

        </div>
      </main>

      {/* ── Full-width: network test comparison report ────────────────────── */}
      {automationSummaryRows.length ? (
        <section className="network-results-section">
          <Panel
            eyebrow="Multi-model network test"
            title="LSTM vs LSTM + SARIMA comparison results"
            action={<ExternalLink size={16} className="panel-icon" />}
          >
            <div className="metric-stack" style={{ marginBottom: 12 }}>
              <div className="metric">
                <span>Models compared</span>
                <strong>{comparisonModels.length || 3}</strong>
              </div>
              <div className="metric">
                <span>Replay candles</span>
                <strong>{networkAutomationReport?.replay_candles ?? EMPTY_VALUE}</strong>
              </div>
            </div>
            <div className="automation-actions" style={{ marginBottom: 10 }}>
              <button
                type="button"
                className={`save-button ${comparisonModelFilter === 'all' ? '' : 'save-button-secondary'}`}
                onClick={() => setComparisonModelFilter('all')}
              >
                All models
              </button>
              {comparisonModels.map((model) => (
                <button
                  type="button"
                  key={model.key}
                  className={`save-button ${comparisonModelFilter === model.key ? '' : 'save-button-secondary'}`}
                  onClick={() => setComparisonModelFilter(model.key)}
                >
                  {model.label}
                </button>
              ))}
            </div>
            <div className="network-report-scroll-outer">
              <div className="network-report-grid network-report-head">
                <span>Model</span>
                <span>Phase</span>
                <span>Samples</span>
                <span>Mean delay</span>
                <span>P95 delay</span>
                <span>Mean conf.</span>
                <span>Mean quality</span>
                <span>Imputed %</span>
                <span>High delay %</span>
                <span>Flip rate %</span>
                <span>Trades</span>
                <span>PnL</span>
                <span>PnL vs LSTM</span>
                <span>Drop rate</span>
                <span>Non-HOLD %</span>
                <span>Sig. acc.</span>
                <span>Max DD %</span>
              </div>
              {filteredAutomationRows.map((row) => (
                <div key={`${row.modelKey}-${row.phase}`} className="network-report-grid">
                  <span>{row.modelLabel}</span>
                  <span>{row.phase}</span>
                  <span>{row.samples}</span>
                  <span>{Number(row.mean_decision_delay_ms).toFixed(1)} ms</span>
                  <span>{Number(row.p95_decision_delay_ms).toFixed(1)} ms</span>
                  <span>{Number(row.mean_confidence).toFixed(3)}</span>
                  <span>{Number(row.mean_data_quality ?? 1).toFixed(3)}</span>
                  <span>{Number(row.mean_imputed_rate_pct ?? 0).toFixed(2)}%</span>
                  <span>{(Number(row.high_delay_rate ?? 0) * 100).toFixed(1)}%</span>
                  <span>{(Number(row.action_flip_rate ?? 0) * 100).toFixed(1)}%</span>
                  <span>{Number(row.trade_count_delta ?? 0)}</span>
                  <span>{formatPlainNumber(row.pnl_delta ?? 0)}</span>
                  <span>{formatPlainNumber(row.pnl_impact_vs_lstm ?? 0)}</span>
                  <span>{Number(row.drop_rate_pct_from_counter).toFixed(2)}%</span>
                  <span>{(Number(row.non_hold_rate ?? 0) * 100).toFixed(1)}%</span>
                  <span>{Number(row.signal_accuracy ?? 0).toFixed(3)}</span>
                  <span>{Number(row.max_drawdown_pct ?? 0).toFixed(2)}%</span>
                </div>
              ))}
            </div>
            {networkAutomationReport?.paragraph ? (
              <p className="control-copy" style={{ marginTop: 12 }}>{networkAutomationReport.paragraph}</p>
            ) : null}
          </Panel>
        </section>
      ) : null}

      {/* ── Full-width: prediction journal at the bottom ──────────────────── */}
      <section className="bottom-section">
        <Panel
          eyebrow="Learning loop"
          title="Prediction journal"
          action={<ShieldCheck size={16} className="panel-icon" />}
        >
          <div className="log-stream">
            {dashboardState.logs.length ? dashboardState.logs.slice().reverse().map((entry, index) => (
              <div key={`${entry.time}-${index}`} className={`log-entry log-${entry.level}`}>
                <div className="log-time">{entry.time}</div>
                <div>{entry.message}</div>
              </div>
            )) : (
              <div className="empty-state">Waiting for backend telemetry...</div>
            )}
          </div>
        </Panel>
      </section>
    </div>
  );
}

function App() {
  if (window.location.pathname.startsWith('/models')) {
    return <ModelLab />;
  }

  return <DashboardApp />;
}

export default App;
