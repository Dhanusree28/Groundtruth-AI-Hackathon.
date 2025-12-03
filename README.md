🚀 **H-001: The Automated Insight Engine**

**Tagline:** An event-driven AdTech pipeline that fuses CSV logs, SQL snapshots, and weather signals into executive-ready PDF/PPTX packs with AI-authored narration in under a minute.

---

### 1. The Problem (Real World Scenario)

- **Context:** In the AdTech world we ingest terabytes of heterogeneous data—foot-traffic sensors, ad clickstreams, on-site conversions, and third-party weather feeds. Account Managers still stitch weekly performance reports by downloading CSVs and pasting screenshots.
- **Pain Point:** Manual report prep is slow, error-prone, and uninspiring; campaign inefficiencies linger for days because insights sit in spreadsheets instead of flowing to clients.
- **My Solution:** H-001 watches raw sources, merges them through a governed ETL pipeline, and produces polished PDF/PPTX summaries with highlighted insights and natural-language context.

---

### 2. Expected End Result

- **Input:** Drop new `KAG_conversion_data.csv` exports, SQL result sets, or API JSON payloads into the intake folder (or push via the optional webhook).
- **Action:** The orchestrator validates schemas, hydrates dimensional tables, enriches with weather context, and runs analytics plus AI narration—no manual touchpoints.
- **Output:** Timestamped PDF (and optional PPTX) in `output/` containing:
  - Week-over-week KPI visualizations for spend, impressions, CTR, conversions, CPA.
  - Multi-source anomalies, e.g., “Foot traffic dropped 38% in Austin while storms hit,” with severity and suggested actions.
  - An AI-written executive summary tying together ad metrics, onsite behavior, and weather correlations.

---

### 3. Technical Approach

**System Architecture**

1. **Event Ingestion**  
   - `watchdog` tracks file drops in `data/input/`; a lightweight API adapter can ingest SQL extracts or weather JSON.  
   - Every artifact is hashed, schema-validated, and archived in `data/processed/` with lineage metadata.

2. **Data Transformation & Fusion**  
   - Polars ingests `KAG_conversion_data` and other CSVs with declarative schemas for spend, impressions, age, and region fields.  
   - SQL/JSON payloads are normalized via Pandas helper utilities before being joined in Polars.  
   - Feature engineering layer adds funnel ratios, geo buckets, weather lag features, and foot-traffic indices.

3. **Analytics & Anomaly Detection**  
   - Weekly aggregates feed a Polars metrics cube.  
   - `IsolationForest` (scikit-learn) scans each metric/region combo to flag unusual drops/spikes.  
   - A rules overlay catches deterministic issues (missing spend, zero conversions).

4. **Generative Insight Layer**  
   - Structured anomaly JSON and KPI deltas are passed to Google Gemini 1.5 Pro via Vertex AI/GPT-4o fallback.  
   - Few-shot prompts force analyst-style language and limit context to supplied metrics.  
   - A guardrail validator cross-checks every number or percentile the model cites against the Polars cube; mismatches return “Unknown.”

5. **Reporting Engine**  
   - Plotly renders regional heatmaps, trend charts, and funnel dashboards.  
   - WeasyPrint converts a responsive HTML template into a PDF; `python-pptx` can generate slides when PPTX export is enabled.  
   - Artifacts plus JSON run logs are written to `output/` with ISO timestamps for downstream automation.

6. **Archival & Access**  
   - Run metadata (source files, KPI deltas, anomalies, AI summary) is stored as structured JSON for future audit or to drive notifications.  
   - Optional hooks can push the completed PDF/PPTX to cloud storage or Slack, but the base requirement stops at file generation.

---

### 4. Tech Stack

| Layer            | Selection                                | Rationale                                                     |
|------------------|-------------------------------------------|---------------------------------------------------------------|
| Language         | Python 3.11                               | Strong ecosystem for ETL, ML, and LLM integration             |
| Data Engine      | Polars + helper Pandas adapters           | Handles multi-million row CSVs quickly with typed schemas     |
| Storage          | Local parquet/CSV staging (extensible)    | Keeps hackathon setup lightweight yet production-leaning      |
| ML / Detection   | Scikit-Learn Isolation Forest + rules     | Captures statistical anomalies and deterministic failures     |
| AI Model         | Google Gemini 1.5 Pro / GPT-4o            | Generates analyst-grade narratives with controllable prompts  |
| Visualization    | Plotly + WeasyPrint + optional python-pptx| Produces consistent PDF/PPTX deliverables                     |
| Automation       | Watchdog + asyncio orchestration          | Event-driven ETL without cron lag                             |
| Packaging        | Docker & Docker Compose                   | Portable deployment with pinned dependencies                  |

---

### 5. Challenges & Learnings

1. **Schema Drift Across Sources**  
   - *Issue:* CSV exports, SQL snapshots, and weather APIs rarely align on field names or time zones.  
   - *Resolution:* Built a declarative schema registry plus normalization layer so every source is mapped before landing in Polars.

2. **AI Narrative Accuracy**  
   - *Issue:* Early LLM drafts attributed performance swings to weather even when no weather delta existed.  
   - *Resolution:* Implemented strict system prompts, few-shot constraints, and a validation pass that removes unsupported claims.

---

### 6. Visual Proof (for demo slides)

- **Ingestion Console:** Watchdog logs showing CSV, SQL, and API payloads landing.  
- **PDF Snapshot:** KPI scorecards, anomaly callouts, and campaign-level drill-down charts.  
- **PPTX Sample (optional):** One-slide executive overview automatically generated from the same insights.

---

### 7. How to Run

```
# 1. Clone the repo
git clone https://github.com/Dhanusree28/Groundtruth-AI-Hackathon.git
cd automated-insight-engine

# 2. Configure environment variables
cp .env.example .env
# Populate GEMINI_API_KEY plus any source-specific config (weather keys, etc.)

# 3. Build and launch services
docker-compose up --build

# 4. Trigger a run
# Drop sample CSV / SQL extracts / weather JSON into data/input/
cp data/sample_KAG_conversion_data.csv data/input/

# 5. Review artifacts
# PDFs (and PPTX if enabled) plus JSON logs appear in ./output with timestamps
```


