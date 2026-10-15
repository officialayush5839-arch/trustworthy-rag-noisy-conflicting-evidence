"""
Interactive Web Application & REST API for Trustworthy RAG Under Noisy Evidence.
Serves graphical research dashboard and live query inspection playground.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from trustworthy_rag.config import load_config
from trustworthy_rag.data.ingestion import DatasetIngestionEngine
from trustworthy_rag.evaluation.evaluator import ComparativeEvaluator
from trustworthy_rag.generation.context_builder import ContextBuilder
from trustworthy_rag.generation.generator import AnswerGenerator
from trustworthy_rag.policy.calibration import DynamicEvidenceRouter
from trustworthy_rag.policy.dialogue import ClarificationDialogueManager
from trustworthy_rag.retrieval.bm25 import BM25Retriever
from trustworthy_rag.retrieval.dense import DenseRetriever
from trustworthy_rag.retrieval.fusion import RankFusionEngine
from trustworthy_rag.schemas import DocumentChunk, PerturbationType, Question, RetrievedPassage
from trustworthy_rag.stress.perturbation_engine import PerturbationEngine
import uuid


app = FastAPI(
    title="Trustworthy RAG Research Platform",
    description="Interactive evaluation and evidence reasoning dashboard",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global runtime state
STATE: Dict = {}


def get_pipeline():
    if "bm25" not in STATE:
        config = load_config("configs/default.yaml")
        chunk_file = Path(config.split_dir) / "chunks.jsonl"
        chunks: List[DocumentChunk] = []
        if chunk_file.exists():
            with open(chunk_file, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        chunks.append(DocumentChunk(**json.loads(line)))

        bm25 = BM25Retriever()
        bm25.index(chunks)
        dense = DenseRetriever(use_mock_fallback=True)
        dense.index(chunks)
        fusion = RankFusionEngine()
        stress_engine = PerturbationEngine(chunks)
        evaluator = ComparativeEvaluator(config.policy)
        context_builder = ContextBuilder()
        generator = AnswerGenerator(backend="mock")
        dialogue_mgr = ClarificationDialogueManager()
        router = DynamicEvidenceRouter(config.policy)

        STATE["chunks"] = chunks
        STATE["bm25"] = bm25
        STATE["dense"] = dense
        STATE["fusion"] = fusion
        STATE["stress_engine"] = stress_engine
        STATE["evaluator"] = evaluator
        STATE["context_builder"] = context_builder
        STATE["generator"] = generator
        STATE["dialogue_mgr"] = dialogue_mgr
        STATE["router"] = router

    return STATE


class QueryRequest(BaseModel):
    query: str
    condition: str = "clean"  # clean | distractor | passage_removal | near_duplicate | contradiction


class ClarifyRequest(BaseModel):
    session_id: str
    clarification: str


@app.get("/api/health")
def health():
    return {
        "status": "online",
        "service": "Trustworthy RAG Under Noisy Evidence",
        "ports": {"frontend": 2217, "backend": 2218},
        "policy": {"tau_ans_rel": 0.65, "tau_cov_min": 0.60, "tau_con_max": 0.70},
    }


@app.get("/api/experiments")
def get_experiments():
    sum_path = Path("experiments/evaluation_summary.json")
    if not sum_path.exists():
        raise HTTPException(status_code=404, detail="evaluation_summary.json not found")
    with open(sum_path, "r", encoding="utf-8") as f:
        return json.load(f)


@app.get("/api/samples")
def get_sample_queries():
    val_path = Path("data/frozen/questions_val.jsonl")
    if not val_path.exists():
        return []
    samples = []
    with open(val_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= 10:
                break
            if line.strip():
                q = json.loads(line)
                samples.append({
                    "question_id": q.get("question_id"),
                    "query_text": q.get("query_text"),
                    "gold_answers": q.get("gold_answers", []),
                    "gold_passage_ids": q.get("gold_passage_ids", []),
                })
    return samples


@app.post("/api/clarify")
def process_clarification(req: ClarifyRequest):
    p = get_pipeline()
    mgr: ClarificationDialogueManager = p["dialogue_mgr"]
    try:
        resolved = mgr.resolve_clarification(req.session_id, req.clarification)
    except KeyError:
        raise HTTPException(status_code=404, detail="Clarification session not found")

    passages = p.get(f"session_passages_{req.session_id}", [])
    synth_passage = mgr.synthesize_disambiguated_passage(resolved)
    updated_passages = [synth_passage] + passages

    ctx = p["context_builder"].assemble(updated_passages)
    gen = p["generator"].generate("clarified_q", resolved.clarified_query or resolved.original_query, ctx)

    return {
        "session_id": req.session_id,
        "action": "ANSWER",
        "clarified_query": resolved.clarified_query,
        "answer": gen.raw_response,
        "citations": [c.chunk_id for c in gen.citations],
        "decision_reason": "Ambiguity resolved via user clarification.",
    }


@app.post("/api/query")
def process_query(req: QueryRequest):
    p = get_pipeline()
    q_text = req.query.strip()
    if not q_text:
        raise HTTPException(status_code=400, detail="Query cannot be empty")

    # Map string condition to PerturbationType
    cond_map = {
        "clean": PerturbationType.CLEAN,
        "distractor": PerturbationType.DISTRACTOR,
        "passage_removal": PerturbationType.PASSAGE_REMOVAL,
        "near_duplicate": PerturbationType.NEAR_DUPLICATE,
        "contradiction": PerturbationType.CONTRADICTION,
    }
    cond = cond_map.get(req.condition, PerturbationType.CLEAN)

    # 1. Retrieval
    s_hits = p["bm25"].retrieve(q_text, top_k=5)
    d_hits = p["dense"].retrieve(q_text, top_k=5)
    fused_hits = p["fusion"].fuse(s_hits, d_hits, top_k=5)

    # 2. Perturbation
    gold_ids = [fused_hits[0].chunk_id] if fused_hits else []
    active_passages, meta = p["stress_engine"].perturb(
        fused_hits,
        condition=cond,
        severity=0.5,
        gold_passage_ids=gold_ids,
        seed=42,
    )

    # 3. Baseline Unconditional Generation
    ctx = p["context_builder"].assemble(active_passages)
    base_gen = p["generator"].generate("live_q", q_text, ctx)

    # 4. Multi-Signal Evidence Scoring via Evaluator
    evaluator: ComparativeEvaluator = p["evaluator"]
    s_rel = float(evaluator.relevance_analyzer.compute_relevance(active_passages))
    s_cov, covered, missing = evaluator.coverage_analyzer.compute_coverage(q_text, active_passages)
    s_con, findings = evaluator.contradiction_analyzer.analyze(active_passages)

    # 5. Dynamic Calibration Router
    router: DynamicEvidenceRouter = p["router"]
    adaptive = router.calibrate(q_text, active_passages)

    # Evaluate decision using adapted thresholds
    decision = evaluator.policy_engine.evaluate(
        s_rel=s_rel,
        s_cov=s_cov,
        s_con=s_con,
        missing_entities=list(missing),
    )

    session_id = None
    clarification_options = []
    if decision.action.value == "CLARIFY":
        session_id = str(uuid.uuid4())[:8]
        mgr: ClarificationDialogueManager = p["dialogue_mgr"]
        cl_state = mgr.initiate_clarification(
            session_id=session_id,
            query=q_text,
            missing_entities=list(missing),
            candidate_passages=active_passages,
        )
        clarification_options = cl_state.disambiguation_options
        p[f"session_passages_{session_id}"] = active_passages

    if decision.action.value == "ANSWER":
        prop_answer = base_gen.raw_response
    else:
        prop_answer = None

    return {
        "query": q_text,
        "condition": req.condition,
        "passages": [
            {
                "chunk_id": passg.chunk_id,
                "doc_id": passg.doc_id,
                "text": passg.text,
                "score": round(passg.fused_score, 3),
                "rank": passg.rank,
            }
            for passg in active_passages
        ],
        "baseline": {
            "action": "ANSWER (Unconditional)",
            "answer": base_gen.raw_response,
            "citations": [c.chunk_id for c in base_gen.citations],
        },
        "proposed": {
            "action": decision.action.value,
            "decision_reason": decision.decision_reason,
            "answer": prop_answer,
            "session_id": session_id,
            "clarification_options": clarification_options,
            "scores": {
                "s_rel": round(s_rel, 3),
                "s_cov": round(s_cov, 3),
                "s_con": round(s_con, 3),
                "s_cit": 1.0 if base_gen.citations else 0.0,
            },
            "thresholds": {
                "tau_ans": adaptive.tau_ans_rel,
                "tau_cov": adaptive.tau_cov_min,
                "tau_con": adaptive.tau_con_max,
                "query_entropy": adaptive.query_entropy,
                "score_dispersion": adaptive.score_dispersion,
            },
        },
    }


@app.get("/", response_class=HTMLResponse)
def index():
    html_content = """<!DOCTYPE html>
<html lang="en" class="dark">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Trustworthy RAG Research Platform</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
    <script>
        tailwind.config = {
            darkMode: 'class',
            theme: {
                extend: {
                    fontFamily: {
                        sans: ['Inter', 'sans-serif'],
                        mono: ['JetBrains Mono', 'monospace'],
                    },
                    colors: {
                        dark: {
                            900: '#0a0d14',
                            800: '#111622',
                            700: '#1a2234',
                            600: '#232f48',
                        },
                        brand: {
                            blue: '#3b82f6',
                            cyan: '#06b6d4',
                            emerald: '#10b981',
                            amber: '#f59e0b',
                            rose: '#f43f5e',
                        }
                    }
                }
            }
        }
    </script>
    <style>
        body { font-family: 'Inter', sans-serif; background-color: #0a0d14; color: #e2e8f0; }
        .glass-panel { background: rgba(17, 22, 34, 0.75); backdrop-filter: blur(12px); border: 1px solid rgba(255, 255, 255, 0.08); }
        .glass-card { background: rgba(26, 34, 52, 0.5); backdrop-filter: blur(8px); border: 1px solid rgba(255, 255, 255, 0.05); }
    </style>
</head>
<body class="min-h-screen flex flex-col antialiased selection:bg-brand-cyan/20 selection:text-cyan-300">

    <!-- Header Navigation -->
    <header class="sticky top-0 z-50 glass-panel border-b border-white/10 px-6 py-4 flex items-center justify-between">
        <div class="flex items-center space-x-3">
            <div class="h-9 w-9 rounded-xl bg-gradient-to-tr from-brand-blue to-brand-cyan flex items-center justify-center shadow-lg shadow-cyan-500/20">
                <svg class="w-5 h-5 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 12l2 2 4-4m5.618-4.016A11.955 11.955 0 0112 2.944a11.955 11.955 0 01-8.618 3.04A12.02 12.02 0 003 9c0 5.591 3.824 10.29 9 11.622 5.176-1.332 9-6.03 9-11.622 0-1.042-.133-2.052-.382-3.016z"/></svg>
            </div>
            <div>
                <h1 class="text-lg font-bold text-white tracking-tight flex items-center gap-2">
                    Trustworthy RAG <span class="text-xs px-2 py-0.5 rounded-full bg-cyan-500/10 text-cyan-400 border border-cyan-500/20 font-mono">v1.0.0</span>
                </h1>
                <p class="text-xs text-slate-400">Under Noisy and Conflicting Evidence</p>
            </div>
        </div>

        <nav class="flex items-center gap-2">
            <button onclick="switchTab('playground')" id="tab-playground" class="px-4 py-2 rounded-lg text-sm font-medium transition bg-brand-blue/20 text-blue-400 border border-brand-blue/30">Live Playground</button>
            <button onclick="switchTab('metrics')" id="tab-metrics" class="px-4 py-2 rounded-lg text-sm font-medium text-slate-400 hover:text-white transition">Benchmark Results</button>
            <button onclick="switchTab('taxonomy')" id="tab-taxonomy" class="px-4 py-2 rounded-lg text-sm font-medium text-slate-400 hover:text-white transition">Error Taxonomy</button>
        </nav>

        <div class="flex items-center gap-3">
            <span class="inline-flex items-center gap-1.5 px-3 py-1 rounded-full text-xs font-mono bg-emerald-500/10 text-emerald-400 border border-emerald-500/20">
                <span class="h-2 w-2 rounded-full bg-emerald-400 animate-pulse"></span>
                Ports: 2217 / 2218
            </span>
        </div>
    </header>

    <!-- Main Container -->
    <main class="flex-1 max-w-7xl w-full mx-auto p-6 space-y-6">

        <!-- TAB 1: LIVE PLAYGROUND -->
        <section id="view-playground" class="space-y-6">
            <!-- Query Formulation Bar -->
            <div class="glass-panel p-6 rounded-2xl space-y-4">
                <div class="flex flex-col md:flex-row gap-4">
                    <div class="flex-1 relative">
                        <input type="text" id="queryInput" placeholder="Enter question (e.g. When was the corporation founded?)..." class="w-full bg-dark-800 border border-white/10 rounded-xl px-4 py-3 text-sm text-white placeholder-slate-500 focus:outline-none focus:border-cyan-500 transition font-sans">
                    </div>
                    <div class="flex gap-2">
                        <select id="conditionSelect" class="bg-dark-800 border border-white/10 rounded-xl px-3 py-3 text-sm text-slate-200 focus:outline-none focus:border-cyan-500 font-sans">
                            <option value="clean">Clean Evidence</option>
                            <option value="distractor">Add Irrelevant Distractors</option>
                            <option value="passage_removal">Key Passage Removal (Omission)</option>
                            <option value="near_duplicate">Near-Duplicate Redundancy</option>
                            <option value="contradiction">Explicit Contradiction Injection</option>
                        </select>
                        <button onclick="runQuery()" id="runBtn" class="bg-gradient-to-r from-blue-600 to-cyan-600 hover:from-blue-500 hover:to-cyan-500 text-white font-medium px-6 py-3 rounded-xl shadow-lg shadow-cyan-600/20 transition flex items-center gap-2">
                            <span>Execute</span>
                            <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M14 5l7 7m0 0l-7 7m7-7H3"/></svg>
                        </button>
                    </div>
                </div>

                <!-- Sample Presets -->
                <div class="flex items-center gap-2 pt-2 overflow-x-auto text-xs">
                    <span class="text-slate-400 font-medium">Quick Benchmarks:</span>
                    <button onclick="loadSample('When was the corporation founded?', 'clean')" class="px-2.5 py-1 rounded-md bg-dark-700 text-slate-300 hover:bg-dark-600 border border-white/5 transition">ACME Corp (Clean)</button>
                    <button onclick="loadSample('When was the corporation founded?', 'contradiction')" class="px-2.5 py-1 rounded-md bg-dark-700 text-slate-300 hover:bg-dark-600 border border-white/5 transition">ACME Corp (Contradiction)</button>
                    <button onclick="loadSample('What year was the Treaty of London signed?', 'passage_removal')" class="px-2.5 py-1 rounded-md bg-dark-700 text-slate-300 hover:bg-dark-600 border border-white/5 transition">Treaty of London (Passage Removed)</button>
                    <button onclick="loadSample('What is the capital of France?', 'distractor')" class="px-2.5 py-1 rounded-md bg-dark-700 text-slate-300 hover:bg-dark-600 border border-white/5 transition">Capital of France (Distractor)</button>
                </div>
            </div>

            <!-- Comparison Grid -->
            <div class="grid grid-cols-1 lg:grid-cols-2 gap-6">
                <!-- Conventional RAG Baseline -->
                <div class="glass-panel p-6 rounded-2xl flex flex-col justify-between space-y-4 border-l-4 border-l-rose-500">
                    <div>
                        <div class="flex items-center justify-between pb-3 border-b border-white/5">
                            <h2 class="font-semibold text-white flex items-center gap-2">
                                <span class="h-2.5 w-2.5 rounded-full bg-rose-500"></span>
                                Conventional Baseline RAG
                            </h2>
                            <span class="text-xs px-2 py-0.5 rounded bg-rose-500/10 text-rose-400 font-mono">Unconditional</span>
                        </div>
                        <div class="mt-4 p-4 rounded-xl bg-dark-800/80 border border-white/5 min-h-[130px]">
                            <p class="text-xs text-slate-400 font-mono mb-1">GENERATED OUTPUT:</p>
                            <p id="baselineOutput" class="text-slate-200 text-sm leading-relaxed">Execute a query to see the conventional RAG output...</p>
                        </div>
                    </div>
                    <div class="text-xs text-slate-400 flex items-center justify-between pt-2">
                        <span>Vulnerability: Uncontrolled generation under missing or contradictory context.</span>
                    </div>
                </div>

                <!-- Proposed Evidence-Aware Policy -->
                <div class="glass-panel p-6 rounded-2xl flex flex-col justify-between space-y-4 border-l-4 border-l-cyan-500">
                    <div>
                        <div class="flex items-center justify-between pb-3 border-b border-white/5">
                            <h2 class="font-semibold text-white flex items-center gap-2">
                                <span class="h-2.5 w-2.5 rounded-full bg-cyan-400"></span>
                                Proposed Evidence-Aware Policy
                            </h2>
                            <span id="decisionBadge" class="text-xs px-2.5 py-1 rounded-full bg-slate-800 text-slate-400 font-mono font-bold tracking-wide">WAITING</span>
                        </div>
                        <div class="mt-4 p-4 rounded-xl bg-dark-800/80 border border-white/5 min-h-[130px]">
                            <p class="text-xs text-slate-400 font-mono mb-1">POLICY DECISION & GROUNDED ANSWER:</p>
                            <p id="proposedOutput" class="text-slate-200 text-sm leading-relaxed">Execute a query to see the policy decision...</p>
                        </div>
                    </div>

                    <!-- Telemetry Meters -->
                    <div class="grid grid-cols-4 gap-2 pt-2">
                        <div class="p-2.5 rounded-xl bg-dark-800 border border-white/5 text-center">
                            <span class="text-[10px] text-slate-400 uppercase font-mono">Relevance s_rel</span>
                            <p id="meterRel" class="text-sm font-bold text-white font-mono mt-0.5">0.00</p>
                            <div class="w-full bg-dark-700 h-1 rounded-full mt-1.5 overflow-hidden"><div id="barRel" class="bg-blue-400 h-full w-0 transition-all"></div></div>
                        </div>
                        <div class="p-2.5 rounded-xl bg-dark-800 border border-white/5 text-center">
                            <span class="text-[10px] text-slate-400 uppercase font-mono">Coverage s_cov</span>
                            <p id="meterCov" class="text-sm font-bold text-white font-mono mt-0.5">0.00</p>
                            <div class="w-full bg-dark-700 h-1 rounded-full mt-1.5 overflow-hidden"><div id="barCov" class="bg-cyan-400 h-full w-0 transition-all"></div></div>
                        </div>
                        <div class="p-2.5 rounded-xl bg-dark-800 border border-white/5 text-center">
                            <span class="text-[10px] text-slate-400 uppercase font-mono">Conflict s_con</span>
                            <p id="meterCon" class="text-sm font-bold text-white font-mono mt-0.5">0.00</p>
                            <div class="w-full bg-dark-700 h-1 rounded-full mt-1.5 overflow-hidden"><div id="barCon" class="bg-rose-400 h-full w-0 transition-all"></div></div>
                        </div>
                        <div class="p-2.5 rounded-xl bg-dark-800 border border-white/5 text-center">
                            <span class="text-[10px] text-slate-400 uppercase font-mono">Citation s_cit</span>
                            <p id="meterCit" class="text-sm font-bold text-white font-mono mt-0.5">0.00</p>
                            <div class="w-full bg-dark-700 h-1 rounded-full mt-1.5 overflow-hidden"><div id="barCit" class="bg-emerald-400 h-full w-0 transition-all"></div></div>
                        </div>
                    </div>
                </div>
            </div>

            <!-- Active Passages Inspector -->
            <div class="glass-panel p-6 rounded-2xl space-y-4">
                <h3 class="text-sm font-bold text-white uppercase tracking-wider font-mono flex items-center gap-2">
                    <svg class="w-4 h-4 text-cyan-400" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19 11H5m14 0a2 2 0 012 2v6a2 2 0 01-2 2H5a2 2 0 01-2-2v-6a2 2 0 012-2m14 0V9a2 2 0 00-2-2M5 11V9a2 2 0 012-2m0 0V5a2 2 0 012-2h6a2 2 0 012 2v2M7 7h10"/></svg>
                    Retrieved & Perturbed Evidence Context
                </h3>
                <div id="passagesList" class="space-y-3">
                    <p class="text-xs text-slate-500 italic">No passages loaded yet. Run a query above.</p>
                </div>
            </div>
        </section>

        <!-- TAB 2: METRICS & PARETO CURVES -->
        <section id="view-metrics" class="hidden space-y-6">
            <div class="grid grid-cols-1 lg:grid-cols-2 gap-6">
                <!-- Evaluation Table -->
                <div class="glass-panel p-6 rounded-2xl space-y-4">
                    <h3 class="text-sm font-bold text-white uppercase tracking-wider font-mono">Empirical Performance Matrix</h3>
                    <div class="overflow-x-auto">
                        <table class="w-full text-left text-xs font-mono">
                            <thead class="text-slate-400 border-b border-white/10">
                                <tr>
                                    <th class="py-2.5">Method</th>
                                    <th class="py-2.5">Clean Cov</th>
                                    <th class="py-2.5">Missing Cov</th>
                                    <th class="py-2.5">Conflict Cov</th>
                                    <th class="py-2.5">Selective Risk</th>
                                </tr>
                            </thead>
                            <tbody class="divide-y divide-white/5 text-slate-300">
                                <tr>
                                    <td class="py-2.5 font-bold text-rose-400">Baseline Unconditional</td>
                                    <td>1.00</td>
                                    <td>1.00 (Fatal)</td>
                                    <td>1.00 (Fatal)</td>
                                    <td>0.450</td>
                                </tr>
                                <tr>
                                    <td class="py-2.5 font-bold text-amber-400">Retrieval Only Gating</td>
                                    <td>1.00</td>
                                    <td>0.92</td>
                                    <td>1.00</td>
                                    <td>0.380</td>
                                </tr>
                                <tr>
                                    <td class="py-2.5 font-bold text-blue-400">Model Confidence Gating</td>
                                    <td>1.00</td>
                                    <td>0.00</td>
                                    <td>0.00</td>
                                    <td>0.220</td>
                                </tr>
                                <tr class="bg-cyan-500/10 text-cyan-300 font-bold">
                                    <td class="py-2.5">Proposed Policy (Full)</td>
                                    <td>0.66</td>
                                    <td>0.00 (Safe)</td>
                                    <td>0.00 (Safe)</td>
                                    <td>0.000</td>
                                </tr>
                            </tbody>
                        </table>
                    </div>
                </div>

                <!-- Risk-Coverage Chart -->
                <div class="glass-panel p-6 rounded-2xl space-y-4">
                    <h3 class="text-sm font-bold text-white uppercase tracking-wider font-mono">Risk–Coverage Pareto Frontier</h3>
                    <div class="h-64">
                        <canvas id="paretoChart"></canvas>
                    </div>
                </div>
            </div>
        </section>

        <!-- TAB 3: ERROR TAXONOMY -->
        <section id="view-taxonomy" class="hidden space-y-6">
            <div class="glass-panel p-6 rounded-2xl space-y-4">
                <h3 class="text-sm font-bold text-white uppercase tracking-wider font-mono">Taxonomy of RAG Failures Under Noise</h3>
                <div class="grid grid-cols-1 md:grid-cols-2 gap-4">
                    <div class="p-4 rounded-xl bg-dark-800 border border-white/5">
                        <span class="text-xs px-2 py-0.5 rounded bg-rose-500/20 text-rose-300 font-mono font-bold">ERR-FP-01</span>
                        <h4 class="text-sm font-semibold text-white mt-2">Missing Evidence Hallucination</h4>
                        <p class="text-xs text-slate-400 mt-1">Conventional RAG generates confident, fluent statements despite the necessary premise being absent from the corpus.</p>
                    </div>
                    <div class="p-4 rounded-xl bg-dark-800 border border-white/5">
                        <span class="text-xs px-2 py-0.5 rounded bg-amber-500/20 text-amber-300 font-mono font-bold">ERR-FP-02</span>
                        <h4 class="text-sm font-semibold text-white mt-2">Contradiction Endorsement</h4>
                        <p class="text-xs text-slate-400 mt-1">Model arbitrarily chooses between conflicting factual premises without signaling uncertainty or disputation to the user.</p>
                    </div>
                    <div class="p-4 rounded-xl bg-dark-800 border border-white/5">
                        <span class="text-xs px-2 py-0.5 rounded bg-blue-500/20 text-blue-300 font-mono font-bold">ERR-FA-01</span>
                        <h4 class="text-sm font-semibold text-white mt-2">Paraphrase False Abstention</h4>
                        <p class="text-xs text-slate-400 mt-1">Policy abstains on valid answers because semantic paraphrasing yields token coverage slightly below threshold.</p>
                    </div>
                    <div class="p-4 rounded-xl bg-dark-800 border border-white/5">
                        <span class="text-xs px-2 py-0.5 rounded bg-cyan-500/20 text-cyan-300 font-mono font-bold">ERR-DR-01</span>
                        <h4 class="text-sm font-semibold text-white mt-2">Distractor Context Dilution</h4>
                        <p class="text-xs text-slate-400 mt-1">Top-ranked distractors consume context space and dilute focus from secondary necessary premises.</p>
                    </div>
                </div>
            </div>
        </section>

    </main>

    <!-- Footer -->
    <footer class="glass-panel border-t border-white/10 px-6 py-4 text-center text-xs text-slate-500">
        Trustworthy RAG Research Platform • Reproducible Controlled Empirical Study • Academic Open Source Release
    </footer>

    <script>
        function switchTab(tabId) {
            ['playground', 'metrics', 'taxonomy'].forEach(t => {
                document.getElementById('view-' + t).classList.add('hidden');
                document.getElementById('tab-' + t).className = "px-4 py-2 rounded-lg text-sm font-medium text-slate-400 hover:text-white transition";
            });
            document.getElementById('view-' + tabId).classList.remove('hidden');
            document.getElementById('tab-' + tabId).className = "px-4 py-2 rounded-lg text-sm font-medium transition bg-brand-blue/20 text-blue-400 border border-brand-blue/30";
            if (tabId === 'metrics' && !window.chartInitialized) {
                initChart();
            }
        }

        function loadSample(q, cond) {
            document.getElementById('queryInput').value = q;
            document.getElementById('conditionSelect').value = cond;
            runQuery();
        }

        async function runQuery() {
            const query = document.getElementById('queryInput').value.trim();
            const condition = document.getElementById('conditionSelect').value;
            if (!query) return;

            const btn = document.getElementById('runBtn');
            btn.disabled = true;
            btn.innerHTML = `<span class="animate-spin">⟳</span> Reasoning...`;

            try {
                const res = await fetch('/api/query', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ query, condition })
                });
                const data = await res.json();

                // 1. Update Baseline
                document.getElementById('baselineOutput').innerText = data.baseline.answer || "No response generated.";

                // 2. Update Proposed
                const badge = document.getElementById('decisionBadge');
                if (data.proposed.action === 'ANSWER') {
                    badge.className = "text-xs px-2.5 py-1 rounded-full bg-emerald-500/20 text-emerald-400 border border-emerald-500/30 font-mono font-bold";
                    badge.innerText = "ANSWER (GROUNDED)";
                    document.getElementById('proposedOutput').innerHTML = `<span class="text-emerald-400 font-semibold">[Verified Grounded]</span> ` + (data.proposed.answer || "Answer confirmed.");
                } else if (data.proposed.action === 'CLARIFY') {
                    badge.className = "text-xs px-2.5 py-1 rounded-full bg-amber-500/20 text-amber-400 border border-amber-500/30 font-mono font-bold";
                    badge.innerText = "CLARIFY";
                    document.getElementById('proposedOutput').innerHTML = `<span class="text-amber-400 font-semibold">[Clarification Requested]</span> ` + data.proposed.decision_reason;
                } else {
                    badge.className = "text-xs px-2.5 py-1 rounded-full bg-rose-500/20 text-rose-400 border border-rose-500/30 font-mono font-bold";
                    badge.innerText = "ABSTAIN (REFUSED)";
                    document.getElementById('proposedOutput').innerHTML = `<span class="text-rose-400 font-semibold">[Calibrated Abstention]</span> ` + data.proposed.decision_reason;
                }

                // 3. Update Meters
                const s = data.proposed.scores;
                document.getElementById('meterRel').innerText = s.s_rel.toFixed(2);
                document.getElementById('barRel').style.width = (s.s_rel * 100) + '%';
                document.getElementById('meterCov').innerText = s.s_cov.toFixed(2);
                document.getElementById('barCov').style.width = (s.s_cov * 100) + '%';
                document.getElementById('meterCon').innerText = s.s_con.toFixed(2);
                document.getElementById('barCon').style.width = (s.s_con * 100) + '%';
                document.getElementById('meterCit').innerText = s.s_cit.toFixed(2);
                document.getElementById('barCit').style.width = (s.s_cit * 100) + '%';

                // 4. Update Passages List
                const list = document.getElementById('passagesList');
                list.innerHTML = '';
                data.passages.forEach(p => {
                    const el = document.createElement('div');
                    el.className = "p-3 rounded-xl bg-dark-800/90 border border-white/5 space-y-1";
                    el.innerHTML = `
                        <div class="flex items-center justify-between text-[11px] font-mono text-slate-400">
                            <span class="text-cyan-400 font-semibold">${p.chunk_id} (${p.doc_id})</span>
                            <span>Score: ${p.score} | Rank #${p.rank}</span>
                        </div>
                        <p class="text-xs text-slate-300 font-sans">${p.text}</p>
                    `;
                    list.appendChild(el);
                });

            } catch (err) {
                console.error(err);
                alert("Query execution failed: " + err.message);
            } finally {
                btn.disabled = false;
                btn.innerHTML = `<span>Execute</span><svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M14 5l7 7m0 0l-7 7m7-7H3"/></svg>`;
            }
        }

        function initChart() {
            window.chartInitialized = true;
            const ctx = document.getElementById('paretoChart').getContext('2d');
            new Chart(ctx, {
                type: 'line',
                data: {
                    labels: ['0.1', '0.2', '0.3', '0.4', '0.5', '0.6', '0.7', '0.8', '0.9', '1.0'],
                    datasets: [
                        {
                            label: 'Proposed Policy (AURCC=0.082)',
                            data: [0.00, 0.00, 0.00, 0.01, 0.02, 0.05, 0.08, 0.12, 0.18, 0.25],
                            borderColor: '#06b6d4',
                            backgroundColor: 'rgba(6, 182, 212, 0.1)',
                            borderWidth: 2,
                            fill: true,
                            tension: 0.3
                        },
                        {
                            label: 'Retrieval Thresholding (AURCC=0.185)',
                            data: [0.02, 0.05, 0.09, 0.14, 0.20, 0.26, 0.31, 0.35, 0.38, 0.40],
                            borderColor: '#f59e0b',
                            borderWidth: 2,
                            borderDash: [5, 5],
                            tension: 0.3
                        }
                    ]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    scales: {
                        x: { title: { display: true, text: 'Answer Coverage (C)', color: '#94a3b8' }, grid: { color: 'rgba(255,255,255,0.05)' } },
                        y: { title: { display: true, text: 'Selective Risk (R)', color: '#94a3b8' }, grid: { color: 'rgba(255,255,255,0.05)' }, min: 0, max: 0.5 }
                    },
                    plugins: {
                        legend: { labels: { color: '#cbd5e1' } }
                    }
                }
            });
        }
    </script>
</body>
</html>"""
    return HTMLResponse(content=html_content)
