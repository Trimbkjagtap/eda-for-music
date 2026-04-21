# GhostTrack Codebase State Report
**Generated:** 2026-04-21 | **Branch:** main | **Purpose:** Targeted-change reconnaissance

---

## SECTION 1: FRONTEND STRUCTURE

### 1. `static/index.html` — Full Dump

```html
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1.0"/>
  <title>GhostTrack | Ghost Artist Detection Framework</title>
  <link rel="icon" href="data:image/svg+xml,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 100 100'><text y='.9em' font-size='90'>👻</text></svg>"/>
  <link rel="preconnect" href="https://fonts.googleapis.com"/>
  <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800;900&display=swap" rel="stylesheet"/>
  <link rel="stylesheet" href="/static/css/styles.css"/>
</head>
<body>

  <!-- ── Fixed Navbar ── -->
  <nav id="navbar">
    <div class="nav-logo" onclick="navigate('home')">
      <div class="nav-logo-icon">♪</div>
      <span class="nav-logo-text">GhostTrack</span>
    </div>
    <div class="nav-links">
      <span class="nav-link active" data-page="home"     onclick="navigate('home')">Home</span>
      <span class="nav-link"        data-page="about"    onclick="navigate('about')">About</span>
      <span class="nav-link"        data-page="gallery"  onclick="navigate('gallery')">Gallery</span>
      <span class="nav-link"        data-page="analyzer" onclick="navigate('analyzer')">Analyzer</span>
      <span class="nav-link"        data-page="network"  onclick="navigate('network')">Network</span>
      <span class="nav-link"        data-page="ai"       onclick="navigate('ai')">AI Assistant</span>
    </div>
    <button class="nav-demo-btn" onclick="navigate('analyzer')">▶&nbsp; Demo</button>
  </nav>

  <!-- ── Page Container ── -->
  <main id="page-root"></main>

  <script src="/static/js/app.js"></script>
</body>
</html>
```

### 2. `static/js/app.js` — Full Dump (849 lines)

```javascript
/* ── GhostTrack SPA ────────────────────────────────────────────────────────
   Pure vanilla JS single-page app.
   navigate(page) swaps content inside #page-root, no reload.
   API calls go to FastAPI backend at /api/* (same origin).
──────────────────────────────────────────────────────────────────────────── */

const API = '';   // same origin; prefix all calls with /

// ── Static data ──────────────────────────────────────────────────────────────

const KNOWN_ARTISTS = [
  { label: 'Relaxing White Noise (ghost)',    id: '6bo3atMVp3qFECNALVwq9N', name: 'Relaxing White Noise' },
  { label: 'Meditation Relax Club (ghost)',   id: '3BqBPFLxBkzKQTkuBPGMNF', name: 'Meditation Relax Club' },
  { label: 'Calmo (candidate)',               id: '4Wx3ZL6d6p1gVMtwQ2YWsz', name: 'Calmo' },
  { label: 'Nils Frahm (organic)',            id: '5hVghJ3sCFHFJoLnSHySjL', name: 'Nils Frahm' },
];

const CROSS_PLATFORM = [
  { name: 'Relaxing White Noise',  id: '6bo3atMVp3qFECNALVwq9N', yt: 353_775_028, apple: true,  verdict: 'LIKELY_GHOST',   s6: 0.716 },
  { name: 'Meditation Relax Club', id: '3BqBPFLxBkzKQTkuBPGMNF', yt: 157_581_269, apple: true,  verdict: 'LIKELY_GHOST',   s6: 0.560 },
  { name: 'Calmo',                 id: '4Wx3ZL6d6p1gVMtwQ2YWsz', yt: 155,         apple: false, verdict: 'SUSPICIOUS',     s6: 0.446 },
  { name: 'Nils Frahm',            id: '5hVghJ3sCFHFJoLnSHySjL', yt: 9_107_596,  apple: true,  verdict: 'LIKELY_ORGANIC',  s6: 0.000 },
];

const FRAMEWORK_LAYERS = [
  { n:'1', name:'Catalog Coherence',   tag:'Kaggle + Spotify',  desc:'Audio feature variance per artist reveals unnaturally low variance.' },
  { n:'2', name:'Playlist Entropy',    tag:'Spotify /playlists',desc:'Shannon entropy of playlist feature distributions.' },
  { n:'3', name:'ISRC Attribution',    tag:'Spotify /tracks',   desc:'Production company identification via ISRC prefix.' },
  { n:'4', name:'Release Cadence',     tag:'Spotify /albums',   desc:'Statistical analysis of release date spacing patterns.' },
  { n:'5', name:'Metadata Similarity', tag:'NLP Embeddings',    desc:'Track/artist name reuse detection with minor variations.' },
  { n:'6', name:'Graph Centrality',    tag:'Neo4j Graph',       desc:'Co-appearance network analysis reveals isolated clusters.' },
  { n:'7', name:'Aggregate Score',     tag:'All Layers',        desc:'Weighted combination into final ghost probability score.' },
];

const FIGURES = [
  { file:'fig1_catalog_coherence.png',     ex:'Exercise 1', sig:'Catalog Variance',  title:'Figure 1: Catalog Coherence in Audio Feature Space', caption:'Same-genre comparison of N=13 ghost ambient vs N=75 organic ambient artists. Ghost within-catalog variance collapses relative to organic. Levene W=15.7, p=0.0002; Cohen\'s d=−1.45 to −2.08 per feature (large effect). Genre confound eliminated — separation is not an artifact of ambient genre.' },
  { file:'fig2_playlist_entropy.png',      ex:'Exercise 2', sig:'Playlist Entropy',  title:'Figure 2: Playlist Aesthetic Coherence', caption:'7-feature marginal entropy across 30 Kaggle-proxy playlists (editorial, fan-curated, ghost-suspect). ANOVA F=0.25, p=0.78 — no significant entropy difference between groups (honest negative result).' },
  { file:'fig3_isrc_join.png',             ex:'Exercise 3', sig:'ISRC Attribution',  title:'Figure 3: Artist to Production Company Attribution via ISRC', caption:'Expanded bipartite graph: 3 ghost + 17 organic artists across 27 registrant codes. Registrant-type classification shows all ghost artists use CUSTOM_REGISTRANT (small, non-public registrants), while organic artists use known aggregators (TuneCore, DistroKid) or labels. Categorical distinction — not just HHI magnitude — is the cleaner fraud signal.' },
  { file:'fig4_bipartite_neighborhood.png',ex:'Exercise 4', sig:'Graph Centrality',  title:'Figure 4: ISRC Registrant HHI Distribution', caption:'Real HHI from ISRC data: RWN=0.672, MRC=0.515, Calmo=0.452. Mann-Whitney p=0.003, r=1.000 vs 30 organic artists. Youden threshold: HHI ≥ 0.353.' },
  { file:'fig5_recommendation_walk.png',   ex:'Exercise 5', sig:'Release Cadence',   title:'Figure 5: Recommendation Walk — Release Cadence as Walk Closure Signal', caption:'Release cadence closure rate: ghost (N=14) vs organic baseline (N=1031 across 5 genres). KS D=1.000, p<0.001. Cohen\'s d=3.44 (very large effect). Sensitivity across 1d–14d thresholds: 100% TPR, 0% FPR. Prolific organic artists (Buckethead, King Gizzard, Merzbow, GBV) absent from Kaggle — documented limitation.' },
  { file:'fig6_signal_radar.png',          ex:'Exercise 6', sig:'Aggregate Score',   title:'Figure 6: Seven-Signal Ghost Artist Detection Radar', caption:'Grouped bar chart of all 7 signals with 95% bootstrap CIs. S1/S3/S6/S7 explicitly marked N/A (not zero) — require API access unavailable at scale. Post-audit: S2 Cadence (d=3.44) and S5 Metadata (d=−0.91, direction documented) are discriminative. S4 Catalog Density (d=0.32) is below threshold.' },
  { file:'fig6b_signal_heatmap.png',       ex:'Exercise 6', sig:'Signal Heatmap',    title:'Figure 6b: Signal Report Card Heatmap', caption:'Inter-signal Pearson correlation matrix (S2, S4, S5 — the 3 Kaggle-computable signals). S2 and S4 are effectively uncorrelated (r=−0.01), confirming they measure independent phenomena. Significant correlation between S2 and S4 would indicate multicollinearity — important for feature selection.' },
  { file:'fig7_gnn_performance.png',       ex:'Exercise 7', sig:'GNN Model',         title:'Figure 7: GNN Ghost Artist Detection Performance', caption:'GAT vs GCN training curves, ROC curves (AUC=1.000), confusion matrix, SHAP feature importance. Dataset: 65 nodes (14 ghost, 51 organic), 692 edges. Top features: total_variance, closure_rate, tracks_per_day.' },
];

const SIGNAL_NAMES = {
  signal_1: 'Audio Fingerprint',
  signal_2: 'Release Cadence',
  signal_3: 'Playlist Co-occurrence',
  signal_4: 'Catalog Density',
  signal_5: 'Metadata Similarity',
  signal_6: 'Graph / HHI',
  signal_7: 'Cross-Platform',
};

// ── Router ────────────────────────────────────────────────────────────────────

let _currentPage = 'home';

function navigate(page) {
  _currentPage = page;
  document.querySelectorAll('.nav-link').forEach(el => {
    el.classList.toggle('active', el.dataset.page === page);
  });
  const root = document.getElementById('page-root');
  root.innerHTML = '';
  const renderers = {
    home:     renderHome,
    gallery:  renderGallery,
    analyzer: renderAnalyzer,
    network:  renderNetwork,
    cross:    renderCrossPlatform,
    ai:       renderAI,
    about:    renderAbout,
  };
  (renderers[page] || renderHome)(root);
  window.scrollTo(0, 0);
}
// ... [renderHome, renderGallery, renderAnalyzer, renderNetwork, renderCrossPlatform, renderAI, renderAbout]
// Full functions documented in Sections 3–5 below.
```

### 3. `static/css/styles.css` — Key Sections (695 lines total)

**Demo Button (`.nav-demo-btn`, lines 70–79):**
```css
.nav-demo-btn {
  display: flex; align-items: center; gap: 8px;
  background: transparent; border: 1px solid #444;
  border-radius: 24px; padding: 8px 22px;
  color: var(--white); font-size: 0.85rem; font-weight: 600;
  cursor: pointer; font-family: var(--font);
  transition: border-color 0.15s, background 0.15s;
}
.nav-demo-btn:hover { border-color: #666; background: rgba(255,255,255,0.05); }
```

**Hero/Home Layout (lines 126–168):**
```css
.hero { padding: 0 0 64px; }
.hero-wrap { position: relative; }
.hero h1 { font-size: clamp(3rem, 6vw, 5rem); margin-bottom: 28px; animation: fadeUp 0.6s 0.1s ease both; }
.hero h1 .accent { color: var(--green); }
.hero-subtitle { font-size: 1.05rem; color: var(--gray3); max-width: 520px; line-height: 1.8; margin-bottom: 40px; }
.hero-btns { display: flex; gap: 14px; flex-wrap: wrap; }
```

**Navigation Bar (`#navbar`, lines 37–79):**
```css
#navbar {
  position: fixed; top: 0; left: 0; right: 0;
  height: var(--navbar-h);
  background: rgba(10,10,10,0.96);
  backdrop-filter: blur(16px);
  border-bottom: 1px solid var(--border);
  display: flex; align-items: center; justify-content: space-between;
  padding: 0 48px; z-index: 1000;
}
.nav-links { display: flex; align-items: center; gap: 36px; }
.nav-link { color: var(--gray3); font-size: 0.875rem; font-weight: 500; cursor: pointer; transition: color 0.15s; }
.nav-link:hover  { color: var(--white); }
.nav-link.active { color: var(--white); font-weight: 600; }
```

**Analyzer Page (lines 517–558):**
```css
.search-box { display: flex; gap: 10px; margin-bottom: 20px; }
.search-input { flex: 1; background: var(--bg2); border: 1px solid #333; border-radius: 10px; padding: 14px 18px; color: var(--white); font-size: 0.92rem; }
.search-input:focus { border-color: var(--green); box-shadow: 0 0 0 2px rgba(0,255,136,0.1); }
.quick-picks { display: flex; gap: 8px; flex-wrap: wrap; margin-bottom: 36px; }
.quick-pill { background: var(--bg3); border: 1px solid #333; color: var(--gray2); border-radius: 20px; padding: 7px 16px; font-size: 0.8rem; cursor: pointer; }
.quick-pill:hover { border-color: var(--green); color: var(--green); }
.verdict-banner { border-radius: 12px; padding: 22px 28px; margin-bottom: 24px; display: flex; align-items: center; gap: 18px; }
.verdict-ghost       { background: #1a0505; border: 1px solid #5a1515; }
.verdict-suspicious  { background: #1a1005; border: 1px solid #5a3a0a; }
.verdict-organic     { background: var(--green-dim); border: 1px solid var(--green-mid); }
.signal-row { display: flex; align-items: center; gap: 16px; background: var(--bg2); border: 1px solid var(--border); border-radius: 10px; padding: 14px 18px; }
```

### 4. Static Image Folders

**No `image/`, `avatar/`, or `hero-image/` folder exists under `static/`.** Only subdirectories are `static/css/` and `static/js/`. The hero "image" is a pure-CSS atmospheric effect (matrix rain text, CSS clip-path silhouette, scan-line animation) — zero real image files. All figures served from `paper/figures/` via the `/figures` FastAPI mount.

---

## SECTION 2: BACKEND STRUCTURE

### 5. `backend/main.py` — Full Dump

```python
"""
EDA-Music FastAPI Backend
Start: uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload
Docs:  http://localhost:8000/docs
"""
# [668 lines — key sections below]

app = FastAPI(title="EDA for Music — Ghost Artist Detection API", version="1.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], ...)

STATIC_DIR = ROOT / "static"
FIGURES_DIR = ROOT / "paper" / "figures"
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
app.mount("/figures", StaticFiles(directory=str(FIGURES_DIR)), name="figures")
```

### 6. All Routes

| Method | Path | Returns |
|--------|------|---------|
| GET | `/` | `FileResponse(index.html)` — SPA root |
| GET | `/app`, `/app/{path}` | `FileResponse(index.html)` — SPA catch-all |
| GET | `/health` | `{"status":"ok","project":"EDA for Music","version":"1.0.0"}` |
| GET | `/health` *(duplicate — bug)* | Neo4j + signal module checks (status: ok/degraded) |
| POST | `/analyze` | `AnalyzeResponse` — full 7-signal pipeline (study panel only, 4 artists) |
| GET | `/search?q=&limit=` | `{"results":[...]}` — Spotify artist search |
| POST | `/analyze-live` | `AnalyzeResponse` — live 3-signal analysis for any Spotify artist |
| GET | `/artist/{artist_id}/signals` | Per-signal score breakdown |
| GET | `/graph/stats` | Neo4j node/relationship counts |
| GET | `/graph/neighborhood/{artist_id}` | Bipartite nodes/edges (artist + production companies) |
| GET | `/graph/isrc-clusters` | Production company cluster list |
| GET | `/exercises/summary` | Static findings summary for all 5 exercises |
| GET | `/model/info` | GAT model architecture + training stats |
| POST | `/chat` | AI research assistant response via GPT-4o |
| GET | `/artists` | All artists in Neo4j with ghost labels |

**NOTE:** Two routes registered at `/health` — FastAPI uses the last-registered handler. This is a routing bug (line 117 vs line 122 of `backend/main.py`).

### 7. Spotify API Files

**`src/api/spotify_client.py`** — uses `spotipy` with `SpotifyClientCredentials`. 30-call hard session limit, MD5-hash cache in `data/raw/cache/` (3,086 files). All calls wrapped with `@with_retry()` (5 retries, exponential backoff for 429/5xx; 401/403 raise immediately).

| Function | Endpoint | Returns |
|----------|----------|---------|
| `get_artist(id)` | `GET /artists/{id}` | name, id, images, external_urls |
| `get_artist_albums(id)` | `GET /artists/{id}/albums` | album list (limit 10 as of Apr 2026) |
| `get_album_tracks(id)` | `GET /albums/{id}/tracks` | track list |
| `get_track(id)` | `GET /tracks/{id}` | full track with `external_ids.isrc` |
| `search_tracks(q)` | `GET /search?type=track` | track items |
| `search_artists(q)` | `GET /search?type=artist` | artist items |
| `search_playlists(q)` | `GET /search?type=playlist` | playlist metadata only |
| `get_playlist(id)` | `GET /playlists/{id}` | metadata (editorial 37i9… → 404) |
| `get_playlist_tracks(id)` | `GET /playlists/{id}/tracks` | track items (user playlists only) |

**`src/ingest/live_ingest.py`** — calls `search_artists()`, `get_artist()`, `get_artist_albums()`, `get_album_tracks()` for the `/search` and `/analyze-live` routes.

### 8. Third-Party API Integrations

#### YouTube Data API v3
**`src/api/youtube_client.py`**
```python
YOUTUBE_SEARCH_URL = "https://www.googleapis.com/youtube/v3/search"
YOUTUBE_VIDEOS_URL = "https://www.googleapis.com/youtube/v3/videos"
# GET /youtube/v3/search?part=snippet&q={artist} {track} official&type=video&maxResults=1&key={YOUTUBE_API_KEY}
# GET /youtube/v3/videos?part=statistics&id={video_id}&key={YOUTUBE_API_KEY}
# Returns: int view count. If YOUTUBE_API_KEY not set → silently returns 0.
```
Used by Signal 7 (`src/signals/cross_platform.py`).

#### iTunes / Apple Music
**`src/api/apple_music_client.py`**
```python
ITUNES_SEARCH_URL = "https://itunes.apple.com/search"
# GET https://itunes.apple.com/search?term={artist} {track}&media=music&entity=song&limit=5
# No API key required. Returns: bool (is_on_apple_music) or int (track count).
```
Used by Signal 7.

#### OpenAI API
**`backend/main.py` lines 614–646:**
```python
from openai import OpenAI
client = OpenAI()
resp = client.chat.completions.create(
    model="gpt-4o", messages=messages, temperature=0.4, max_tokens=1500
)
```
Requires `OPENAI_API_KEY` env var. Also used in `src/agents/crew.py` (crewai + GPT-4o).

#### MusicBrainz
Not integrated anywhere in the codebase.

---

## SECTION 3: ANALYZER CURRENT STATE

### 9. Analyzer Render Function — Full Dump

```javascript
function renderAnalyzer(root) {
  root.innerHTML = `
    <div class="page-header">
      <div class="eyebrow">Artist Analyzer</div>
      <h1>Ghost Detection Tool</h1>
      <p>Search any artist by name — or pick from the study panel below.</p>
    </div>
    <div class="search-box" style="position:relative">
      <input id="artist-input" class="search-input" type="text"
             placeholder="Artist name  e.g. Arijit Singh, Nils Frahm…"
             autocomplete="off"
             oninput="onArtistInput(this.value)"
             onkeydown="if(event.key==='Enter') runAnalysis()"/>
      <button class="btn-primary" onclick="runAnalysis()">Analyze →</button>
      <div id="search-dropdown" style="display:none;position:absolute;top:100%;left:0;right:80px;
           background:#111;border:1px solid var(--border);border-radius:8px;z-index:100;
           overflow:hidden;margin-top:4px;box-shadow:0 8px 24px rgba(0,0,0,0.6)"></div>
    </div>
    <div class="quick-picks">
      ${KNOWN_ARTISTS.map(a => `
        <button class="quick-pill" onclick="setAndAnalyze('${a.id}','${a.name}')">${a.label}</button>
      `).join('')}
    </div>
    <div id="analyzer-result"></div>
    <footer class="site-footer">GhostTrack | INFO 7390 - Spring 2026 | By Trimbkeshwar Jagtap</footer>
  `;
}
```

**`runAnalysis()`** routing logic:
- If artist ID is in `KNOWN_ARTISTS` → `POST /analyze` (full 7-signal, Neo4j cached)
- Otherwise → `POST /analyze-live` (3-signal, live Spotify fetch)

**`renderAnalysisResult(container, d)`** shows:
1. Verdict banner: icon (👻/⚠️/✓), verdict label, artist name, score%, confidence%
2. Expert Analysis box: `d.explanation` text, GNN score if available
3. Signal Breakdown: 7 animated bars (Audio Fingerprint, Release Cadence, Playlist Co-occurrence, Catalog Density, Metadata Similarity, Graph/HHI, Cross-Platform)
4. Info box: timing (cached mode) or live-mode caveat (live mode)

### 10. Fields Shown for an Artist

- `d.verdict_label` — LIKELY_GHOST / SUSPICIOUS / LIKELY_ORGANIC
- `d.verdict_score` — 0.0–1.0 ghost probability
- `d.confidence` — 0.0–1.0
- `d.artist_name`
- `d.explanation` — text paragraph
- `d.signals` — dict of 7 signal scores (null for unavailable)
- `d.gnn_score`, `d.rule_based_score`, `d.gnn_available` — GNN augmentation
- `d.timing_seconds`

### 11. Tracks Display

**No track-level display.** The Analyzer shows only artist-level scores. Track data is used internally (for S5 title similarity, cross-platform lookups) but never rendered in the UI. The closest available track data is `data/processed/neo4j_full_graph.csv` (110K, flat artist+album+track+ISRC export).

### 12. Neo4j Integration in Analyzer

Used indirectly — `/analyze` calls `src.agents.crew.run_analysis()` which pulls from Neo4j. The KNOWN_IDS whitelist (4 artists) gates access. The neighborhood API (`GET /graph/neighborhood/{id}`) is available but **not called by the frontend** — the Network page only shows static `CROSS_PLATFORM` array data.

---

## SECTION 4: AI ASSISTANT CURRENT STATE

### 13. AI Assistant Render Function — Full Dump

```javascript
const QUICK_QUESTIONS = [
  "What makes Relaxing White Noise a ghost artist?",
  "Why is cross-platform presence unreliable for the relaxation genre?",
  "Compare the signal profiles of all 3 ghost artists",
  "What would it take to definitively classify an unknown artist?",
  "Summarize the key findings from all 7 exercises",
  "What are the main limitations of this analysis?",
  "Explain the 7-layer framework to a non-technical audience",
  "Draft an abstract for the paper",
];

function renderAI(root) {
  root.innerHTML = `
    <div class="page-header">
      <div class="eyebrow">AI Research Assistant</div>
      <h1>Ask the Framework</h1>
      <p>PhD-level research assistant with full context of all 7 exercises and findings.</p>
    </div>
    <div class="quick-qs">
      ${QUICK_QUESTIONS.map(q => `<button class="quick-q" onclick="askQuestion(this)">${q}</button>`).join('')}
    </div>
    <div class="chat-window" id="chat-window">
      <div class="chat-msg assistant">
        <div class="chat-role">GhostTrack AI</div>
        <div class="chat-bubble">Hello! I'm a PhD-level research assistant for the GhostTrack project.
        I have full context of all 7 analysis layers, signal scores, and findings. Ask me anything
        about ghost artist detection, the methodology, or the results.</div>
      </div>
    </div>
    <div class="chat-input-row">
      <textarea id="chat-input" class="chat-input" placeholder="Ask a research question…"
        onkeydown="if(event.key==='Enter'&&!event.shiftKey){event.preventDefault();sendChat()}"></textarea>
      <button class="btn-primary" onclick="sendChat()">Send</button>
    </div>
    <footer class="site-footer" style="margin-top:40px">GhostTrack | INFO 7390 - Spring 2026 | By Trimbkeshwar Jagtap</footer>
  `;
}
```

### 14. Backend Endpoint — `/chat`

`POST /chat` in `backend/main.py` lines 614–646.
- Request: `{"question": str, "history": [{"user": str, "assistant": str}]}`
- Response: `{"answer": str}`
- On failure: `{"answer": "AI unavailable: {error}"}` (never raises HTTP error)

### 15. LLM and System Prompt (Verbatim)

**LLM:** `gpt-4o` · `temperature=0.4` · `max_tokens=1500`

**System prompt (verbatim from `backend/main.py` lines 620–637):**
```
You are a PhD-level research assistant for the GhostTrack project.
PROJECT: EDA for Music — Ghost Artist Detection on Spotify | INFO 7390, Spring 2026

EXERCISE RESULTS:
- Ex1: Ghost catalog variance 12.5x lower (Levene p<0.001)
- Ex2: Shannon entropy: editorial=2.59, fan=2.89, ghost-suspect=2.51 bits
- Ex3: 8 production companies, 490 tracks, 0 cross-artist ISRC sharing
- Ex4: HHI: RWN=0.88, MRC=0.66, Calmo=0.54
- Ex5: Walk closure: RWN=81%, MRC=95%, Calmo=32%, Nils Frahm=0%
- Ex6: S2 cadence + S4 catalog density + S6 HHI most discriminative
- Ex7: GAT 100% test accuracy on 65-node graph (14 ghost, 51 organic)

SIGNAL SCORES: RWN→GHOST(0.771), MRC→SUSPICIOUS, Calmo→ORGANIC(rule), NF→ORGANIC
CROSS-PLATFORM: RWN=353M YT views, MRC=157M, Calmo=155, NF=9M
KEY INSIGHT: Ghost behavior = Spotify-economic stream farming, NOT cross-platform absence.

Respond at PhD level. Use markdown.
```

**NOTE:** The system prompt still has old proxy HHI values (RWN=0.88, MRC=0.66, Calmo=0.54). The real ISRC-derived HHI values (RWN=0.672, MRC=0.515, Calmo=0.452) from Audit 1 have not been updated here.

### 16. "PhD-level research assistant" — Exact Locations

| File | Line | Context |
|------|------|---------|
| `backend/main.py` | 637 | System prompt string in `/chat` endpoint |
| `static/js/app.js` | 720 | Initial chat bubble in `renderAI()` |
| `frontend/app.py` | ~791 | Old Streamlit system prompt (superseded) |

---

## SECTION 5: NAVIGATION & ROUTING

### 17. Navigation Menu

Defined in `static/index.html` (not a JS array):
```html
Home | About | Gallery | Analyzer | Network | AI Assistant
```

The `renderers` map in `navigate()` also includes `cross` (Cross-Platform page) — no navbar link, only accessible via `navigate('cross')`.

### 18. Routing Method

**innerHTML swap — no browser URL changes.** `navigate(page)` clears `#page-root` and calls the renderer. No `history.pushState()`, no hash routing, no page reloads. The URL stays at `http://localhost:8000/` regardless of page. Deep-linking impossible. FastAPI has `/app/{path}` catch-all for direct URL access but SPA never uses it.

### 19. Demo Button

- **`static/index.html` line 28:** `<button class="nav-demo-btn" onclick="navigate('analyzer')">▶&nbsp; Demo</button>`
- **`static/css/styles.css` lines 70–79:** `.nav-demo-btn` — pill-shaped, transparent, `border: 1px solid #444`
- **In `app.js`:** No JS definition — it's pure HTML onclick, calls `navigate('analyzer')`.

---

## SECTION 6: DATA AVAILABLE OFFLINE

### 20. CSV Files

| File | Size |
|------|------|
| `data/kaggle/dataset.csv` | 19 MB — 114K tracks, 114 genres, audio features |
| `data/playlists/all_playlist_tracks.csv` | 316 KB — 30 proxy playlists, track features |
| `data/processed/organic_controls_kaggle.csv` | 169 KB — organic artist baselines |
| `data/processed/neo4j_full_graph.csv` | 110 KB — flat: artist+album+track+ISRC |
| `data/processed/exercise4_full_data.csv` | 110 KB — bipartite graph export |
| `data/processed/ex1_catalog_features.csv` | 26 KB — per-track audio features for Ex1 |
| `data/processed/high_variance_artists.csv` | 15 KB |
| `data/processed/low_variance_artists.csv` | 14 KB |
| `data/ground_truth/missing_ids.csv` | 14 KB |
| `data/processed/ex2_playlist_features.csv` | 13 KB |
| `data/ground_truth/organic_artists.csv` | 12 KB |
| `data/playlists/playlist_stats.csv` | 11 KB |
| `data/ground_truth/ghost_artists.csv` | 7.4 KB |
| `data/reference/known_aggregators.csv` | 6.4 KB |
| `data/playlists/playlist_sources.csv` | 5.4 KB |
| `data/processed/isrc_classified.csv` | 2.0 KB — 3 ghost artists, ISRC HHI data |
| `data/processed/ghost_candidates_kaggle.csv` | 1.9 KB |
| `data/processed/ex1_variance_table.csv` | 833 B |
| `data/processed/exercise5_walk_metrics.csv` | 491 B |
| `data/processed/ex6_signal_discrimination.csv` | 441 B |
| `data/processed/ex6_signal_report_card.csv` | 376 B |
| `data/processed/exercise4_metrics.csv` | 328 B |

**First 3 rows — key files:**

`data/ground_truth/ghost_artists.csv`:
```csv
spotify_artist_id,name,source,confidence,notes
4Wx3ZL6d6p1gVMtwQ2YWsz,Calmo,Smith,high,Named in DOJ indictment; ISRC ITIWE/CH654
6bo3atMVp3qFECNALVwq9N,Relaxing White Noise,Smith,high,Named in DOJ indictment; ISRC DEPI8/DE1QW
```

`data/processed/neo4j_full_graph.csv` (track-level, best source for Analyzer tracks display):
```csv
artist_id,artist_name,album_id,album_name,release_date,track_id,track_name,isrc,prefix,company_name
4Wx3ZL6d6p1gVMtwQ2YWsz,Calmo,69Eu3Q7AJHmee75X4sEKsS,Resto a galla,2020-07-17,0P12OoImjHRfC1rhyagBia,Resto a galla,CH6542083853,CH654,Unknown (CH654)
```

`data/processed/exercise5_walk_metrics.csv`:
```csv
Artist,Tracks,Span (days),Release rate (tracks/mo),Closure (≤1d gap %),Within 7d (%),Median gap (days),...
Calmo,38,2030,0.6,32.4,32.4,29.0,...
Meditation Relax Club,172,1240,4.2,94.7,95.3,0.0,...
Relaxing White Noise,280,1156,7.3,81.0,92.1,0.0,...
Nils Frahm (organic),56,7566,0.2,3.6,9.1,98.0,...
```

### 21. `data/cache/` Contents

- `data/cache/artist_resolution.json` — artist name→ID resolution cache
- `data/cache/playlists/` — (subdirectory, contents not catalogued)
- `data/raw/cache/` — **3,086 MD5-hashed JSON files** — cached Spotify API responses (artist, album, track, search, playlist calls). This is the primary data source for study-panel analysis.

### 22. Track-Level Data

No dedicated `tracks.csv`. Track data lives in:
- `data/processed/neo4j_full_graph.csv` (110 KB) — best source, flat artist+album+track+ISRC
- `data/playlists/all_playlist_tracks.csv` (316 KB) — playlist-context track features
- `data/kaggle/dataset.csv` (19 MB) — 114K tracks with audio features (no ISRC)

---

## SECTION 7: SPOTIFY API STATUS

### 23. Error Handling on 401/403

The `@with_retry()` decorator in `src/utils/rate_limiter.py` retries on 429/5xx but **immediately raises on 401/403** (the error string doesn't match any retry branch). The backend `/search` and `/analyze-live` routes catch the exception and return `HTTPException(status_code=500, detail=str(e))`. The frontend shows a styled amber error box.

The 30-call hard limit in `_checked_call()` raises `RuntimeError` (also → 500 to frontend).

### 24. Spotify Endpoints Used

| Endpoint | Status (Apr 2026) |
|----------|-------------------|
| `GET /artists/{id}` | Working (followers/genres/popularity stripped) |
| `GET /artists/{id}/albums` | Working (limit now 10, was 50) |
| `GET /albums/{id}/tracks` | Working |
| `GET /tracks/{id}` | Working (ISRC in external_ids confirmed) |
| `GET /search?type=track` | Working |
| `GET /search?type=artist` | Working |
| `GET /search?type=playlist` | Working (metadata only, no tracks) |
| `GET /playlists/{id}` | 404 for editorial (37i9…) |
| `GET /playlists/{id}/tracks` | Works for public user playlists |
| `GET /artists/{id}/related-artists` | **403 Blocked** |
| `GET /audio-features/{id}` | **403 Blocked** |
| popularity, followers, genres fields | **Stripped from all responses** |

---

## SECTION 8: NEO4J STATUS

### 25. Connection

**AuraDB (cloud).** `.env.example` format:
```
NEO4J_URI=neo4j+s://xxxxx.databases.neo4j.io
NEO4J_USER=neo4j
NEO4J_PASSWORD=[redacted]
```
`neo4j+s://` = AuraDB TLS protocol. Driver: `GraphDatabase.driver()` (lazy singleton in `backend/main.py`).

### 26. All Cypher Queries

```cypher
-- Schema constraints (src/graph/neo4j_client.py)
CREATE CONSTRAINT IF NOT EXISTS FOR (a:Artist) REQUIRE a.spotify_id IS UNIQUE
CREATE CONSTRAINT IF NOT EXISTS FOR (t:Track) REQUIRE t.spotify_id IS UNIQUE
CREATE CONSTRAINT IF NOT EXISTS FOR (al:Album) REQUIRE al.spotify_id IS UNIQUE
CREATE CONSTRAINT IF NOT EXISTS FOR (p:Playlist) REQUIRE p.spotify_id IS UNIQUE
CREATE CONSTRAINT IF NOT EXISTS FOR (c:ProductionCompany) REQUIRE c.isrc_prefix IS UNIQUE

-- Upserts
MERGE (a:Artist {spotify_id: $id}) SET a.name=$name, a.followers=$followers, a.is_ghost=$is_ghost, a.label=$label
MERGE (t:Track {spotify_id: $id}) SET t.name=$name, t.isrc=$isrc, t.duration_ms=$duration_ms, t.release_date=$release_date
MERGE (al:Album {spotify_id: $id}) SET al.name=$name, al.release_date=$release_date, al.isrc_prefix=$isrc_prefix
MERGE (p:Playlist {spotify_id: $id}) SET p.name=$name, p.owner=$owner
MERGE (c:ProductionCompany {isrc_prefix: $prefix}) SET c.name=$name

-- Relationships
MATCH (a:Artist {spotify_id: $aid}) MATCH (al:Album {spotify_id: $alid}) MERGE (a)-[:RELEASED]->(al)
MATCH (al:Album {spotify_id: $alid}) MATCH (t:Track {spotify_id: $tid}) MERGE (al)-[:CONTAINS]->(t)
MATCH (a:Artist {spotify_id: $aid}) MATCH (b:Artist {spotify_id: $bid}) MERGE (a)-[:RELATED_TO]->(b)
MATCH (t:Track {spotify_id: $tid}) MATCH (c:ProductionCompany {isrc_prefix: $prefix}) MERGE (t)-[:REGISTERED_WITH]->(c)

-- Artist neighborhood (/graph/neighborhood/{id})
MATCH (a:Artist {spotify_id: $id})-[:RELEASED]->(al:Album)
      -[:CONTAINS]->(t:Track)-[:REGISTERED_WITH]->(c:ProductionCompany)
RETURN c.isrc_prefix AS prefix, c.name AS company, count(t) AS track_count
ORDER BY track_count DESC

-- ISRC clusters (/graph/isrc-clusters)
MATCH (a:Artist)-[:RELEASED]->(al:Album)-[:CONTAINS]->(t:Track)
      -[:REGISTERED_WITH]->(c:ProductionCompany)
WITH c, collect(DISTINCT a.name) AS artists, count(DISTINCT t) AS track_count
RETURN c.isrc_prefix AS prefix, c.name AS company_name, artists, size(artists) AS artist_count, track_count
ORDER BY track_count DESC

-- Validate artist existence (/analyze)
MATCH (a:Artist {spotify_id: $id}) RETURN a.name AS name

-- Relationship stats (/graph/stats)
MATCH ()-[r]->() RETURN type(r) AS rel_type, count(r) AS cnt ORDER BY cnt DESC

-- List artists (/artists)
MATCH (a:Artist)
OPTIONAL MATCH (a)-[:RELEASED]->(al:Album)
WITH a, count(al) AS album_count
RETURN a.spotify_id AS id, a.name AS name, a.is_ghost AS is_ghost, a.label AS label, album_count
ORDER BY a.name

-- Signal-level queries (src/signals/*.py)
MATCH (a:Artist {spotify_id: $id})-[:RELEASED]->(al:Album) WHERE al.release_date IS NOT NULL RETURN al.release_date AS dt
MATCH (a:Artist {spotify_id: $id})-[:RELEASED]->(al:Album)-[:CONTAINS]->(t:Track)-[:REGISTERED_WITH]->(c:ProductionCompany) RETURN c.isrc_prefix AS prefix, count(t) AS track_count
MATCH (a:Artist {spotify_id: $id})-[:RELEASED]->(al:Album)-[:CONTAINS]->(t:Track) RETURN collect(t.name) AS track_names
MATCH (a:Artist {spotify_id: $id})-[:RELEASED]->(al:Album)-[:CONTAINS]->(t:Track) RETURN t.name AS name LIMIT 1
```

### 27. Live Graph Visualization

**None.** The Network Explorer page (`renderNetwork`) shows only a static HTML table from the hardcoded `CROSS_PLATFORM` JS array. No graph library (vis.js, cytoscape.js, neovis.js, D3.js, sigma.js) is imported anywhere. The only graph-like visual is a static inline SVG in the Home page (pure decorative markup). The `/graph/neighborhood/{id}` API returns nodes+edges JSON but no frontend calls it.

---

## KEY OBSERVATIONS

- **The `/chat` system prompt has stale HHI values** — still shows RWN=0.88, MRC=0.66, Calmo=0.54 (proxy values). The real ISRC-derived values from Audit 1 (RWN=0.672, MRC=0.515, Calmo=0.452) need to be updated in `backend/main.py` lines 626–627.

- **MRC Spotify ID mismatch between frontend and backend** — `KNOWN_ARTISTS` in `app.js` uses `3BqBPFLxBkzKQTkuBPGMNF` for Meditation Relax Club, but `KNOWN_IDS` in `backend/main.py` uses `39t4EeLBfpT72UQJVkIeuj`. Clicking the MRC quick-pick would fall through to Neo4j lookup and potentially 404.

- **Two `/health` routes registered** at the same path in `backend/main.py` (lines 117 and 122). FastAPI silently uses one and the Swagger `/docs` shows both. Minor but confusing.

- **The Network page is a dead end** — it renders only hardcoded `CROSS_PLATFORM` data and says "Full graph visualization requires Neo4j." The `/graph/neighborhood/{id}` and `/graph/isrc-clusters` API endpoints exist and return real Neo4j data but are never called by the frontend.

- **The raw Spotify cache (`data/raw/cache/`, 3,086 files) is the real data backbone** — all study-panel analysis runs from this cache. The `/analyze` endpoint is effectively offline-capable for the 4 known artists; Spotify API health only matters for the `/search` and `/analyze-live` (live mode) flows.
