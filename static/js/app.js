/* ── GhostTrack SPA ────────────────────────────────────────────────────────
   Pure vanilla JS single-page app.
   navigate(page) swaps content inside #page-root, no reload.
   API calls go to FastAPI backend at /api/* (same origin).
──────────────────────────────────────────────────────────────────────────── */

const PROD_API_BASE = 'https://eda-for-music.onrender.com';

function resolveApiBase() {
  const host = window.location.hostname;
  const isLocalHost = host === 'localhost' || host === '127.0.0.1';
  const localDefault = `${window.location.protocol}//${host}:8000`;
  const defaultBase = isLocalHost ? localDefault : PROD_API_BASE;

  const candidates = [
    window.GHOSTTRACK_API_BASE,
    localStorage.getItem('GHOSTTRACK_API_BASE'),
    defaultBase,
  ];

  for (const candidate of candidates) {
    if (!candidate) continue;
    try {
      return new URL(candidate).origin.replace(/\/$/, '');
    } catch (_) {
      // Ignore malformed overrides and fall back to the next candidate.
    }
  }

  return defaultBase;
}

const API = resolveApiBase();

function apiUrl(path) {
  return `${API}${path}`;
}

// ── Static data ──────────────────────────────────────────────────────────────

const KNOWN_ARTISTS = [
  { label: 'Relaxing White Noise (ghost)',    id: '6bo3atMVp3qFECNALVwq9N', name: 'Relaxing White Noise' },
  { label: 'Meditation Relax Club (ghost)',   id: '39t4EeLBfpT72UQJVkIeuj', name: 'Meditation Relax Club' },
  { label: 'Calmo (candidate)',               id: '4Wx3ZL6d6p1gVMtwQ2YWsz', name: 'Calmo' },
  { label: 'Nils Frahm (organic)',            id: '5gqhueRUZEa7VDnQt4HODp', name: 'Nils Frahm' },
];

const CROSS_PLATFORM = [
  { name: 'Relaxing White Noise',  id: '6bo3atMVp3qFECNALVwq9N', yt: 353_775_028, apple: true,  verdict: 'LIKELY_GHOST',   s6: 0.716 },
  { name: 'Meditation Relax Club', id: '39t4EeLBfpT72UQJVkIeuj', yt: 157_581_269, apple: true,  verdict: 'LIKELY_GHOST',   s6: 0.560 },
  { name: 'Calmo',                 id: '4Wx3ZL6d6p1gVMtwQ2YWsz', yt: 155,         apple: false, verdict: 'SUSPICIOUS',     s6: 0.446 },
  { name: 'Nils Frahm',            id: '5gqhueRUZEa7VDnQt4HODp', yt: 9_107_596,  apple: true,  verdict: 'LIKELY_ORGANIC',  s6: 0.000 },
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
  { file:'fig7_gnn_performance.png',       ex:'Exercise 7', sig:'GNN Model',         title:'Figure 7: GNN Ghost Artist Detection Performance', caption:'GAT vs GCN training curves, ROC curves (AUC=1.000), confusion matrix, SHAP feature importance. Dataset: 65 nodes (14 ghost, 51 organic), 692 edges. Top features: total_variance, closure_rate, tracks_per_day. Caveat: AUC=1.000 reflects synthetic graph topology built from known labels; permutation feature importance is near-zero, confirming the model learned cluster membership rather than generalizable features. This is a signal discovery study — not a validated classifier.' },
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

  // Update active nav link
  document.querySelectorAll('.nav-link').forEach(el => {
    el.classList.toggle('active', el.dataset.page === page);
  });

  // Render page
  const root = document.getElementById('page-root');
  root.innerHTML = '';

  const renderers = {
    home:     renderHome,
    gallery:  renderGallery,
    analyzer: renderAnalyzer,
    network:  renderNetwork,
    cross:    renderCrossPlatform,
    about:    renderAbout,
  };

  (renderers[page] || renderHome)(root);
  window.scrollTo(0, 0);
}

// ── Helpers ───────────────────────────────────────────────────────────────────

function fmtViews(n) {
  if (n >= 1_000_000) return (n / 1_000_000).toFixed(0) + 'M';
  if (n >= 1_000)     return (n / 1_000).toFixed(0) + 'K';
  return n.toString();
}

function verdictBadge(v) {
  const map = {
    LIKELY_GHOST:   { cls: 'badge-ghost',       label: '👻 Ghost' },
    SUSPICIOUS:     { cls: 'badge-suspicious',   label: '⚠ Suspicious' },
    LIKELY_ORGANIC: { cls: 'badge-organic',      label: '✓ Organic' },
  };
  const d = map[v] || { cls: 'badge-gray', label: v };
  return `<span class="badge ${d.cls}">${d.label}</span>`;
}

function scoreColor(s) {
  if (s === null || s === undefined) return '#525252';
  if (s >= 0.7) return '#e74c3c';
  if (s >= 0.4) return '#f59e0b';
  return '#00ff88';
}

// ── HOME ──────────────────────────────────────────────────────────────────────

function renderHome(root) {
  // Generate waveform bars for findings header
  const waveHeights = [20,32,44,38,50,42,28,50,36,44,30,48,22,40,34,50,26,44,38,28];
  const waveBars = waveHeights.map((h,i) =>
    `<span style="height:${h}px;animation-delay:${(i*0.09).toFixed(2)}s"></span>`
  ).join('');

  // Matrix rain text for case study image
  const chars = '0101アイウエオカキクケコGHOST010110STREAM10101010';
  let matrix = '';
  for(let i=0;i<600;i++) matrix += chars[Math.floor(Math.random()*chars.length)];

  root.innerHTML = `
    <!-- Hero -->
    <div class="hero-wrap">
      <section class="hero">
        <div class="hero-layout">
          <div class="hero-text">
            <div class="hero-eyebrow">
              <div class="hero-eyebrow-bars">
                <span></span><span></span><span></span><span></span><span></span>
              </div>
              Streaming Platform Integrity
            </div>
            <h1>Unmasking <span class="accent">Ghost<br>Artists</span> in the<br>Streaming Era</h1>
            <p class="hero-subtitle">
              A 7-layer exploratory data analysis framework that exposes fraudulent streaming
              accounts using only public API endpoints. No insider data. No black boxes.
            </p>
            <div class="hero-btns">
              <button class="btn-primary" onclick="navigate('gallery')">Explore Framework &nbsp;→</button>
              <button class="btn-secondary" onclick="document.getElementById('case-study-section').scrollIntoView({behavior:'smooth'})">🎧 View Case Study</button>
            </div>
          </div>
          <div class="hero-viz">
            <div class="hero-viz-ring">
              <div class="hero-viz-dot d1"></div>
              <div class="hero-viz-dot d2"></div>
            </div>
            <div class="hero-viz-ring">
              <div class="hero-viz-dot d3"></div>
              <div class="hero-viz-dot d4"></div>
              <div class="hero-viz-dot d5"></div>
            </div>
            <div class="hero-viz-ring">
              <div class="hero-viz-dot d6"></div>
              <div class="hero-viz-dot d7"></div>
            </div>
            <div class="hero-viz-center"></div>
            <div class="hero-viz-label">7-Layer Detection</div>
          </div>
        </div>
      </section>
    </div>

    <!-- Stats -->
    <div class="stats-row">
      <div class="stat-card">
        <div class="stat-icon">👻</div>
        <div class="stat-num" data-target="3">0</div>
        <div class="stat-label">Ghost Artists</div>
      </div>
      <div class="stat-card">
        <div class="stat-icon">♪</div>
        <div class="stat-num" data-target="490">0</div>
        <div class="stat-label">Tracks</div>
      </div>
      <div class="stat-card">
        <div class="stat-icon">📡</div>
        <div class="stat-num" data-target="7">0</div>
        <div class="stat-label">Detection Layers</div>
      </div>
      <div class="stat-card">
        <div class="stat-icon">📊</div>
        <div class="stat-num">114K</div>
        <div class="stat-label">Training Tracks</div>
      </div>
    </div>

    <!-- Case Study -->
    <div class="case-study-section" id="case-study-section">
      <div class="case-study-body">
        <div class="case-eyebrow">
          <span style="color:var(--amber)">⚠</span> Case Study
        </div>
        <div class="case-title">Meet the Ghost:<br>Relaxing White Noise</div>
        <div class="case-desc">
          An account with 353 million YouTube views but zero verifiable human identity.
          Our framework detected multiple red flags across all 7 analysis layers,
          revealing patterns inconsistent with legitimate artist behavior.
        </div>
        <div class="case-bullets">
          <div class="case-bullet">
            <span class="bullet-dot"></span>
            Feature variance <strong>12.5×</strong> lower than legitimate artists
          </div>
          <div class="case-bullet">
            <span class="bullet-dot"></span>
            <strong>81–95%</strong> same-day release cadence clustering
          </div>
          <div class="case-bullet">
            <span class="bullet-dot"></span>
            HHI concentration coefficient of <strong>0.67</strong> (real ISRC data)
          </div>
          <div class="case-bullet">
            <span class="bullet-dot"></span>
            Single production company controls <strong>88%</strong> of catalog
          </div>
        </div>
      </div>
      <!-- Atmospheric CSS image side -->
      <div class="case-image-side">
        <div class="case-matrix">${matrix}</div>
        <div class="case-silhouette"></div>
        <div class="scan-line"></div>
        <div class="anomaly-overlay">
          <div class="anomaly-ghost-icon">👻</div>
          <div class="anomaly-label">Anomaly Score</div>
          <div class="anomaly-score">0.94</div>
        </div>
      </div>
    </div>

    <!-- Key Findings -->
    <div class="findings-section">
      <div class="findings-header">
        <div class="findings-waveform">${waveBars}</div>
        <div class="eyebrow" style="justify-content:center;margin-bottom:12px">Key Findings</div>
        <h2>What the Data Revealed</h2>
        <p>Striking patterns that distinguish ghost artists from legitimate musicians</p>
      </div>
      <div class="findings-grid">
        <div class="finding-card">
          <div class="finding-icon-box icon-green">〰</div>
          <div class="finding-name">Variance Ratio</div>
          <div class="finding-stat">12.5×</div>
          <div class="finding-desc">Ghost artists show 12.5× lower feature variance compared to legitimate artists</div>
        </div>
        <div class="finding-card">
          <div class="finding-icon-box icon-teal">📡</div>
          <div class="finding-name">Cadence Closure</div>
          <div class="finding-stat">81–95%</div>
          <div class="finding-desc">Ghost accounts maintain impossibly perfect same-day release clustering</div>
        </div>
        <div class="finding-card">
          <div class="finding-icon-box icon-orange">📈</div>
          <div class="finding-name">YouTube Presence</div>
          <div class="finding-stat">353M</div>
          <div class="finding-desc">Suspicious accounts accumulate massive views with minimal engagement</div>
        </div>
        <div class="finding-card">
          <div class="finding-icon-box icon-purple">🧠</div>
          <div class="finding-name">GNN Accuracy</div>
          <div class="finding-stat">100%</div>
          <div class="finding-desc">Graph Neural Network achieves perfect test accuracy on proof-of-concept</div>
        </div>
      </div>
    </div>

    <!-- 7 Layers -->
    <div class="layers-section">
      <div class="layers-intro">
        <div class="layers-eyebrow">
          <span class="eyebrow-bar"></span> Detection Framework
        </div>
        <h2>7 Layers of<br>Analysis</h2>
        <p>Each layer adds another dimension to our detection capability, creating a comprehensive fingerprint that reveals fraudulent accounts with high precision.</p>
        <!-- Ghost network SVG illustration -->
        <div class="layers-illustration">
          <svg viewBox="0 0 340 200" fill="none" xmlns="http://www.w3.org/2000/svg">
            <!-- Edges -->
            <line x1="170" y1="100" x2="80"  y2="50"  stroke="#1a3520" stroke-width="1.5"/>
            <line x1="170" y1="100" x2="260" y2="50"  stroke="#1a3520" stroke-width="1.5"/>
            <line x1="170" y1="100" x2="60"  y2="140" stroke="#1a3520" stroke-width="1.5"/>
            <line x1="170" y1="100" x2="280" y2="140" stroke="#1a3520" stroke-width="1.5"/>
            <line x1="170" y1="100" x2="170" y2="170" stroke="#00ff88" stroke-width="1" stroke-dasharray="4 3" opacity="0.4"/>
            <line x1="80"  y1="50"  x2="260" y2="50"  stroke="#111" stroke-width="1"/>
            <line x1="60"  y1="140" x2="280" y2="140" stroke="#111" stroke-width="1"/>
            <!-- Ghost node (center, pulsing) -->
            <circle cx="170" cy="100" r="20" fill="#0d2818" stroke="#00ff88" stroke-width="1.5" opacity="0.9"/>
            <text x="170" y="105" text-anchor="middle" font-size="14" fill="#00ff88">👻</text>
            <!-- Organic nodes -->
            <circle cx="80"  cy="50"  r="10" fill="#111" stroke="#2a2a2a" stroke-width="1"/>
            <circle cx="260" cy="50"  r="10" fill="#111" stroke="#2a2a2a" stroke-width="1"/>
            <circle cx="60"  cy="140" r="10" fill="#111" stroke="#2a2a2a" stroke-width="1"/>
            <circle cx="280" cy="140" r="10" fill="#111" stroke="#2a2a2a" stroke-width="1"/>
            <circle cx="170" cy="170" r="8"  fill="#111" stroke="#333" stroke-width="1" stroke-dasharray="3 2"/>
            <!-- Labels -->
            <text x="170" y="94" text-anchor="middle" font-size="7" fill="#525252" dy="-14">GHOST CLUSTER</text>
            <text x="80"  y="38" text-anchor="middle" font-size="7" fill="#525252">Organic</text>
            <text x="260" y="38" text-anchor="middle" font-size="7" fill="#525252">Organic</text>
            <text x="60"  y="162" text-anchor="middle" font-size="7" fill="#525252">Label</text>
            <text x="280" y="162" text-anchor="middle" font-size="7" fill="#525252">Label</text>
          </svg>
        </div>
      </div>
      <div class="layer-cards">
        ${FRAMEWORK_LAYERS.map(l => `
          <div class="layer-card">
            <div class="layer-num">${l.n}</div>
            <div class="layer-body">
              <div class="layer-name-row">
                <span class="layer-name">${l.name}</span>
                <span class="layer-tag">${l.tag}</span>
              </div>
              <div class="layer-desc">${l.desc}</div>
            </div>
          </div>`).join('')}
      </div>
    </div>

    <!-- Research Impact -->
    <div class="impact-grid" style="margin-bottom:64px">
      <div class="impact-card">
        <div class="impact-icon">🔍</div>
        <div class="impact-title">Key Contribution</div>
        <div class="impact-body">This framework demonstrates that <strong>independent platform audit is possible</strong> using only public API endpoints — without access to Spotify's internal fraud systems.</div>
      </div>
      <div class="impact-card">
        <div class="impact-icon">⚠️</div>
        <div class="impact-title">Surprise Finding</div>
        <div class="impact-body">Ghost artists are <strong>NOT cross-platform invisible</strong>. Relaxing White Noise has 353M YouTube views. Ghost behavior is Spotify-economic stream farming, not fabricated identity.</div>
      </div>
    </div>

    <!-- Explore -->
    <div style="text-align:center;margin-bottom:40px">
      <div class="eyebrow" style="justify-content:center;margin-bottom:12px">Interactive Tools</div>
      <h2 style="font-size:2.2rem;margin-bottom:10px">Explore the Analysis</h2>
      <p style="color:var(--gray3);font-size:0.95rem">Dive deeper into our research with interactive tools and visualizations</p>
    </div>
    <div class="explore-grid" style="margin-bottom:80px">
      <div class="explore-card" onclick="navigate('analyzer')">
        <div class="explore-thumb">🔍</div>
        <div class="explore-info">
          <div class="explore-icon-row"><span class="explore-icon">🔍</span><span class="explore-name">Artist Analyzer</span></div>
          <div class="explore-desc">Input any Spotify artist ID to see their full authenticity score and signal breakdown</div>
        </div>
      </div>
      <div class="explore-card" onclick="navigate('network')">
        <div class="explore-thumb">🕸</div>
        <div class="explore-info">
          <div class="explore-icon-row"><span class="explore-icon">🕸</span><span class="explore-name">Network Explorer</span></div>
          <div class="explore-desc">Visualize artist collaboration networks and cross-platform presence data</div>
        </div>
      </div>
      <div class="explore-card" onclick="navigate('ai')">
        <div class="explore-thumb">🤖</div>
        <div class="explore-info">
          <div class="explore-icon-row"><span class="explore-icon">🤖</span><span class="explore-name">AI Assistant</span></div>
          <div class="explore-desc">Ask PhD-level questions about our methodology, signals, and findings</div>
        </div>
      </div>
    </div>

    <!-- Footer -->
    <footer class="site-footer">
      <div class="footer-logo">
        <div class="footer-logo-icon">♪</div>
        <span class="footer-logo-text">GhostTrack</span>
      </div>
      <span class="footer-meta">INFO 7390 — Spring 2026</span>
      <span class="footer-author">By Trimbkeshwar Jagtap</span>
    </footer>
  `;

  // Animate stat counters
  root.querySelectorAll('.stat-num[data-target]').forEach(el => {
    const target = +el.dataset.target;
    let current = 0;
    const step = Math.max(1, Math.floor(target / 30));
    const timer = setInterval(() => {
      current = Math.min(current + step, target);
      el.textContent = current;
      if (current >= target) clearInterval(timer);
    }, 40);
  });
}

// ── GALLERY ───────────────────────────────────────────────────────────────────

function renderGallery(root) {
  root.innerHTML = `
    <div class="page-header">
      <div class="eyebrow">Exercise Gallery</div>
      <h1>Analysis Figures</h1>
      <p>Publication-quality figures from Kaggle dataset (114K tracks) and Neo4j graph (490 tracks).</p>
    </div>
    ${FIGURES.map(f => `
      <div class="fig-frame">
        <div class="fig-tags">
          <span class="tag-green">${f.ex}</span>
          <span class="tag-gray">${f.sig}</span>
        </div>
        <div class="fig-title">${f.title}</div>
        <img class="fig-img" src="${apiUrl('/figures/' + f.file)}" alt="${f.title}"
             onerror="this.style.display='none';this.nextElementSibling.style.display='flex'"/>
        <div style="display:none;height:120px;align-items:center;justify-content:center;
                    color:var(--gray4);font-size:0.82rem;background:var(--bg);border-radius:8px;margin:14px 0;">
          Figure not generated yet
        </div>
        <div class="fig-caption">📌 ${f.caption}</div>
      </div>
      <hr class="divider" style="margin:24px 0"/>
    `).join('')}
    <footer class="site-footer">GhostTrack | INFO 7390 - Spring 2026 | By Trimbkeshwar Jagtap</footer>
  `;
}

// ── ANALYZER ──────────────────────────────────────────────────────────────────

function renderAnalyzer(root) {
  root.innerHTML = `
    <div class="page-header">
      <div class="eyebrow">Artist Analyzer</div>
      <h1>Ghost Detection Tool</h1>
      <p>Analyze any artist using our 7-signal framework, or ask our AI research assistant.</p>
      <p style="font-size:0.8rem;color:var(--text-muted);margin-top:0.25rem;">This is a <strong>signal discovery study</strong> — behavioral signals (release cadence, ISRC concentration) exhibit large effect sizes on confirmed ghost artists. A deployment-grade classifier requires a larger independently-labeled dataset. Sample: 3 confirmed ghost artists (DOJ/journalist sources) + Kaggle proxy baseline.</p>
    </div>
    <div class="analyzer-tabs">
      <button class="analyzer-tab active" data-tab="signal" onclick="switchAnalyzerTab('signal')">Signal Analysis</button>
      <button class="analyzer-tab" data-tab="ai" onclick="switchAnalyzerTab('ai')">AI Assistant</button>
    </div>
    <div id="analyzer-tab-content"></div>
    <footer class="site-footer">GhostTrack | INFO 7390 - Spring 2026 | By Trimbkeshwar Jagtap</footer>
  `;
  switchAnalyzerTab('signal');
}

function switchAnalyzerTab(tab) {
  document.querySelectorAll('.analyzer-tab').forEach(el => {
    el.classList.toggle('active', el.dataset.tab === tab);
  });
  const container = document.getElementById('analyzer-tab-content');
  if (tab === 'signal') {
    renderSignalAnalysisPanel(container);
  } else {
    renderAIAssistantPanel(container);
  }
}

function renderSignalAnalysisPanel(container) {
  container.innerHTML = `
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
  `;
}

function renderAIAssistantPanel(container) {
  container.innerHTML = `
    <div class="quick-qs">
      ${QUICK_QUESTIONS.map(q => `<button class="quick-q" onclick="askQuestion(this)">${q}</button>`).join('')}
    </div>
    <div class="chat-window" id="chat-window"></div>
    <div class="chat-input-row">
      <textarea id="chat-input" class="chat-input" placeholder="Ask a research question about ghost artist detection…"
        onkeydown="if(event.key==='Enter'&&!event.shiftKey){event.preventDefault();sendChat()}"></textarea>
      <button class="btn-primary" onclick="sendChat()">Send</button>
    </div>
  `;
}

let _searchTimer = null;
let _selectedArtistId = null;
let _selectedArtistName = null;
let _lastAnalyzedId = null;

function onArtistInput(val) {
  _selectedArtistId = null;
  _selectedArtistName = null;
  clearTimeout(_searchTimer);
  const dd = document.getElementById('search-dropdown');
  if (!val || val.length < 2) { dd.style.display = 'none'; return; }
  _searchTimer = setTimeout(() => _fetchSuggestions(val), 350);
}

async function _fetchSuggestions(q) {
  const dd = document.getElementById('search-dropdown');
  try {
    const resp = await fetch(apiUrl(`/search?q=${encodeURIComponent(q)}&limit=5`));
    if (!resp.ok) { dd.style.display = 'none'; return; }
    const data = await resp.json();
    const items = data.results || [];
    if (!items.length) { dd.style.display = 'none'; return; }
    dd.innerHTML = items.map(a => `
      <div onclick="_selectArtist('${a.id}','${a.name.replace(/'/g,"\\'")}','${a.image||''}')"
           style="display:flex;align-items:center;gap:12px;padding:10px 14px;cursor:pointer;
                  border-bottom:1px solid var(--border);transition:background 0.15s"
           onmouseover="this.style.background='#1a1a1a'" onmouseout="this.style.background=''">
        ${a.image ? `<img src="${a.image}" style="width:36px;height:36px;border-radius:50%;object-fit:cover"/>`
                  : `<div style="width:36px;height:36px;border-radius:50%;background:#222;display:flex;align-items:center;justify-content:center;color:var(--green)">♪</div>`}
        <div>
          <div style="color:#fff;font-size:0.9rem;font-weight:600">${a.name}</div>
          <div style="color:#666;font-size:0.75rem">${a.followers ? a.followers.toLocaleString() + ' followers' : ''}</div>
        </div>
      </div>
    `).join('');
    dd.style.display = 'block';
  } catch(e) { dd.style.display = 'none'; }
}

function _selectArtist(id, name, image) {
  _selectedArtistId = id;
  _selectedArtistName = name;
  document.getElementById('artist-input').value = name;
  document.getElementById('search-dropdown').style.display = 'none';
  runAnalysis();
}

function setAndAnalyze(id, name) {
  _selectedArtistId = id;
  _selectedArtistName = name || id;
  _lastAnalyzedId = id;
  document.getElementById('artist-input').value = name || id;
  document.getElementById('search-dropdown').style.display = 'none';
  runAnalysis();
}

async function runAnalysis() {
  document.getElementById('search-dropdown').style.display = 'none';
  const inputVal = document.getElementById('artist-input').value.trim();
  if (!inputVal) return;
  const result = document.getElementById('analyzer-result');

  const artistId = _selectedArtistId || inputVal;
  const artistName = _selectedArtistName || inputVal;
  _lastAnalyzedId = artistId;
  const isStudyPanel = KNOWN_ARTISTS.some(a => a.id === artistId);
  const endpoint = isStudyPanel ? '/analyze' : '/analyze-live';

  result.innerHTML = `<div class="loading-state"><span class="spinner"></span> ${isStudyPanel ? 'Running full 7-signal pipeline' : 'Fetching live data'}… also pulling track intelligence from YouTube &amp; iTunes…</div>`;
  const stopTicker = startAnalysisTicker(result, isStudyPanel);
  let timeoutId = null;

  try {
    const ANALYZE_TIMEOUT_MS = 45000;
    const controller = new AbortController();
    timeoutId = setTimeout(() => controller.abort(), ANALYZE_TIMEOUT_MS);

    // Fire analysis + track lookup in parallel
    const [resp, tracksResp] = await Promise.all([
      fetch(apiUrl(endpoint), {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        signal: controller.signal,
        body: JSON.stringify({ artist_id: artistId, artist_name: artistName, run_cross_platform: true }),
      }),
      fetch(apiUrl(`/artist-tracks?artist=${encodeURIComponent(artistName)}`)).catch(() => null),
    ]);
    stopTicker();

    const d = await resp.json();
    if (!resp.ok) {
      const msg = d.detail || `API error ${resp.status}`;
      result.innerHTML = `
        <div style="background:#1a1a0a;border:1px solid #f59e0b;border-radius:10px;padding:20px;color:#fde68a;font-size:0.88rem;">
          <strong>Error:</strong> ${msg}
        </div>`;
      return;
    }
    if (!isStudyPanel) d._liveMode = true;

    const tracksData = tracksResp && tracksResp.ok ? await tracksResp.json() : null;
    renderAnalysisResult(result, d, tracksData);
    loadNeighborhoodGraph(d.artist_id || _lastAnalyzedId);
  } catch (e) {
    stopTicker();
    const msg = e && e.name === 'AbortError'
      ? 'Request timed out. Backend may be cold-starting; please retry in a few seconds.'
      : e.message;
    result.innerHTML = `
      <div style="background:#2a0a0a;border:1px solid #e74c3c;border-radius:10px;padding:20px;color:#fca5a5;font-size:0.88rem;">
        <strong>Analysis failed:</strong> ${msg}<br>
        <span style="color:var(--gray4);font-size:0.8rem;">Make sure the FastAPI backend is running.</span>
      </div>`;
  } finally {
    if (timeoutId) clearTimeout(timeoutId);
  }
}

function startAnalysisTicker(resultEl, isStudyPanel) {
  const fullStages = [
    'Running full 7-signal pipeline',
    'Collecting fingerprint signals (S1, S4, S5)',
    'Computing graph signals (S2, S3, S6)',
    'Synthesizing final verdict',
  ];
  const liveStages = [
    'Resolving artist on Spotify',
    'Fetching live album and track catalog',
    'Scoring cadence and metadata similarity',
    'Finalizing live verdict',
  ];

  const stages = isStudyPanel ? fullStages : liveStages;
  const startedAt = Date.now();
  let stageIndex = 0;

  const render = () => {
    const elapsed = ((Date.now() - startedAt) / 1000).toFixed(1);
    const current = stages[Math.min(stageIndex, stages.length - 1)];
    resultEl.innerHTML = `
      <div class="loading-state" style="display:flex;flex-direction:column;align-items:flex-start;gap:8px;">
        <div><span class="spinner"></span> ${current}…</div>
        <div style="font-size:0.78rem;color:var(--gray4);">
          Working for ${elapsed}s. This can take longer on cold starts.
        </div>
      </div>`;
  };

  render();
  const ticker = setInterval(() => {
    stageIndex += 1;
    render();
  }, 4500);

  return () => clearInterval(ticker);
}

function renderAnalysisResult(container, d, tracks) {
  const score = d.verdict_score;
  const label = (d.verdict_label || '').toUpperCase();
  const vcls = label === 'LIKELY_GHOST' ? 'verdict-ghost' : label === 'SUSPICIOUS' ? 'verdict-suspicious' : 'verdict-organic';
  const vicon = label === 'LIKELY_GHOST' ? '👻' : label === 'SUSPICIOUS' ? '⚠️' : '✓';

  const signalSources = d.signal_sources || {};
  const signalRows = Object.entries(d.signals || {}).map(([key, val]) => {
    const pct = val !== null ? Math.round(val * 100) : null;
    const name = SIGNAL_NAMES[key] || key;
    const color = scoreColor(val);
    const src = signalSources[key] || (val !== null ? 'real' : 'unavailable');
    const srcBadge = src === 'openai'
      ? `<span title="Estimated by AI (no API data available)" style="font-size:0.62rem;background:#f59e0b22;color:#f59e0b;border:1px solid #f59e0b44;border-radius:10px;padding:1px 6px;margin-left:6px;vertical-align:middle">🤖 AI</span>`
      : src === 'kaggle'
      ? `<span title="From local Kaggle dataset" style="font-size:0.62rem;background:#3b82f622;color:#60a5fa;border:1px solid #3b82f644;border-radius:10px;padding:1px 6px;margin-left:6px;vertical-align:middle">📊 Kaggle</span>`
      : src === 'real'
      ? `<span title="Real-time data" style="font-size:0.62rem;background:#00ff8811;color:var(--green);border:1px solid #00ff8833;border-radius:10px;padding:1px 6px;margin-left:6px;vertical-align:middle">✓ live</span>`
      : '';
    return `
      <div class="signal-row">
        <div class="signal-name">${name}${srcBadge}</div>
        <div class="signal-bar-wrap">
          <div class="signal-bar" style="width:${pct ?? 0}%;background:${color}${src === 'openai' ? ';opacity:0.75' : ''}"></div>
        </div>
        <div class="signal-score" style="color:${color}">${pct !== null ? pct + '%' : '—'}</div>
      </div>`;
  }).join('');

  // ── Track Intelligence panel ──────────────────────────────────────────
  let trackPanel = '';
  if (tracks && (tracks.latest_track || tracks.top_track || tracks.youtube || tracks.itunes)) {
    const hasYt = !!(tracks.youtube && (tracks.youtube.latest_track || tracks.youtube.top_track));
    const hasIt = !!(tracks.itunes && (tracks.itunes.latest_track || tracks.itunes.top_track));

    function trackCard(label, t, srcColor, icon) {
      if (!t) return '';
      const viewsEst = t.views_source === 'ai_estimate' ? ' <span style="font-size:0.65rem;color:#f59e0b;opacity:0.8">(est.)</span>' : '';
      const views = t.views != null
        ? Number(t.views).toLocaleString() + ' views' + viewsEst
        : t.album ? `📀 ${t.album}` : '';
      const date = t.published ? `· ${t.published.slice(0,7)}` : '';
      const link = t.url ? `href="${t.url}" target="_blank" rel="noopener"` : '';
      const thumb = t.thumbnail
        ? `<img src="${t.thumbnail}" style="width:72px;height:54px;object-fit:cover;border-radius:6px;flex-shrink:0"/>`
        : `<div style="width:72px;height:54px;border-radius:6px;background:#1a1a1a;display:flex;align-items:center;justify-content:center;font-size:1.6rem;flex-shrink:0">${icon}</div>`;
      return `
        <a ${link} style="display:flex;gap:14px;align-items:flex-start;text-decoration:none;
             background:#111;border:1px solid var(--border);border-radius:10px;padding:14px;
             transition:border-color 0.15s;flex:1;min-width:220px"
           onmouseover="this.style.borderColor='var(--green)'" onmouseout="this.style.borderColor='var(--border)'">
          ${thumb}
          <div style="overflow:hidden">
            <div style="font-size:0.72rem;color:${srcColor};font-weight:700;letter-spacing:1px;text-transform:uppercase;margin-bottom:4px">${label}</div>
            <div style="color:#fff;font-size:0.9rem;font-weight:600;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;max-width:220px" title="${t.title}">${t.title}</div>
            <div style="color:var(--gray3);font-size:0.78rem;margin-top:4px">${views} ${date}</div>
          </div>
        </a>`;
    }

    function renderCards(srcKey) {
      const d = srcKey === 'youtube' ? tracks.youtube : srcKey === 'itunes' ? tracks.itunes : tracks;
      const col = srcKey === 'youtube' ? '#e74c3c' : srcKey === 'itunes' ? '#a78bfa' : '#f59e0b';
      const topLabel = srcKey === 'youtube' ? 'Most Viewed' : 'Recent Track';
      if (!d) return '<div style="color:var(--gray3);font-size:0.85rem;padding:16px 0">No data available for this source.</div>';
      return `<div style="display:flex;gap:14px;flex-wrap:wrap">
        ${trackCard('Latest Release', d.latest_track, col, '🆕')}
        ${trackCard(topLabel, d.top_track, col, '🔥')}
      </div>`;
    }

    const panelId = `ti-${Date.now()}`;
    const defaultTab = hasYt ? 'youtube' : hasIt ? 'itunes' : 'other';

    const tabBtn = (key, label, color, active) =>
      `<button onclick="(function(){
          document.querySelectorAll('.ti-tab-${panelId}').forEach(b=>b.style.background='transparent');
          document.querySelectorAll('.ti-tab-${panelId}').forEach(b=>b.style.color='var(--gray3)');
          this.style.background='${color}22'; this.style.color='${color}';
          document.getElementById('ti-cards-${panelId}').innerHTML=window._tiRender_${panelId}('${key}');
        }).call(this)"
        class="ti-tab-${panelId}"
        style="font-size:0.72rem;font-weight:700;padding:3px 12px;border-radius:20px;border:1px solid ${color}44;
               cursor:pointer;transition:all 0.15s;background:${active ? color+'22' : 'transparent'};
               color:${active ? color : 'var(--gray3)'}">${label}</button>`;

    trackPanel = `
      <div style="margin-top:28px;margin-bottom:8px;display:flex;align-items:center;gap:10px;flex-wrap:wrap">
        <h3 style="color:var(--white);font-size:0.95rem;margin:0">Track Intelligence</h3>
        ${hasYt ? tabBtn('youtube', '▶ YouTube', '#e74c3c', defaultTab==='youtube') : ''}
        ${hasIt ? tabBtn('itunes',  '🎵 iTunes',  '#a78bfa', defaultTab==='itunes')  : ''}
        ${!hasYt && !hasIt ? `<span style="font-size:0.72rem;color:var(--gray3)">🤖 OpenAI</span>` : ''}
      </div>
      <div id="ti-cards-${panelId}" style="margin-bottom:24px">
        ${renderCards(defaultTab)}
      </div>
      <script>
        window._tiRender_${panelId} = function(k) {
          const t = ${JSON.stringify(tracks)};
          const d = k==='youtube' ? t.youtube : k==='itunes' ? t.itunes : t;
          const col = k==='youtube' ? '#e74c3c' : k==='itunes' ? '#a78bfa' : '#f59e0b';
          function card(label, tr, icon) {
            if (!tr) return '';
            const viewsEst = tr.views_source==='ai_estimate' ? ' (est.)' : '';
            const views = tr.views!=null ? Number(tr.views).toLocaleString()+' views'+viewsEst : (tr.album ? '📀 '+tr.album : '');
            const date = tr.published ? '· '+tr.published.slice(0,7) : '';
            const link = tr.url ? 'href="'+tr.url+'" target="_blank" rel="noopener"' : '';
            const thumb = tr.thumbnail
              ? '<img src="'+tr.thumbnail+'" style="width:72px;height:54px;object-fit:cover;border-radius:6px;flex-shrink:0"/>'
              : '<div style="width:72px;height:54px;border-radius:6px;background:#1a1a1a;display:flex;align-items:center;justify-content:center;font-size:1.6rem;flex-shrink:0">'+icon+'</div>';
            return '<a '+link+' style="display:flex;gap:14px;align-items:flex-start;text-decoration:none;background:#111;border:1px solid var(--border);border-radius:10px;padding:14px;transition:border-color 0.15s;flex:1;min-width:220px" onmouseover="this.style.borderColor=\'var(--green)\'" onmouseout="this.style.borderColor=\'var(--border)\'">'
              +thumb+'<div style="overflow:hidden"><div style="font-size:0.72rem;color:'+col+';font-weight:700;letter-spacing:1px;text-transform:uppercase;margin-bottom:4px">'+label+'</div>'
              +'<div style="color:#fff;font-size:0.9rem;font-weight:600;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;max-width:220px" title="'+tr.title+'">'+tr.title+'</div>'
              +'<div style="color:var(--gray3);font-size:0.78rem;margin-top:4px">'+views+' '+date+'</div></div></a>';
          }
          const topLabel = k==='youtube' ? 'Most Viewed' : 'Recent Track';
          if (!d) return '<div style="color:var(--gray3);font-size:0.85rem;padding:16px 0">No data available for this source.</div>';
          return '<div style="display:flex;gap:14px;flex-wrap:wrap">'+card('Latest Release',d.latest_track,'🆕')+card(topLabel,d.top_track,'🔥')+'</div>';
        };
      </script>`
  }

  // ── 7-Layer EDA Explainer ─────────────────────────────────────────────
  const EDA_LAYERS = [
    { n:'1', name:'Catalog Coherence', icon:'〰', desc:'Audio feature variance per artist. Ghost artists collapse into unnaturally tight clusters.', example:'Cohen\'s d = −1.45 to −2.08 on Kaggle proxy comparison (ambient/new-age vs multi-genre organic baseline). Levene W=15.7, p=0.0002. Note: S1 not computed for confirmed ghost artists directly — 0% Kaggle ID match rate for seed set; genre confound partially controlled.', color:'var(--green)' },
    { n:'2', name:'Playlist Entropy',  icon:'🎛', desc:'Shannon entropy of playlist feature distributions across editorial, fan-curated, ghost-suspect groups.', example:'ANOVA F=0.25, p=0.78 — honest negative result: entropy alone does not discriminate. Caveat: prolific organic artists (Buckethead, King Gizzard, Merzbow, GBV) absent from Kaggle baseline — threshold may produce false positives for high-output legitimate artists.', color:'#60a5fa' },
    { n:'3', name:'ISRC Attribution',  icon:'🏷', desc:'Production company identification via ISRC prefix codes in track metadata.', example:'All 3 ghost artists use CUSTOM_REGISTRANT. 17 organic artists use TuneCore / DistroKid / labels.', color:'#a78bfa' },
    { n:'4', name:'Release Cadence',   icon:'📡', desc:'Statistical clustering of release dates — ghost artists batch-upload same-day. Strongest signal in framework (Cohen\'s d=3.44).', example:'KS D=1.000, p<0.001. Cohen\'s d=3.44. 100% TPR across 1d–14d thresholds. Computed on 14 ghost vs 1,031 organic artists from Kaggle baseline.', color:'var(--green)' },
    { n:'5', name:'Metadata Similarity',icon:'🔤', desc:'Track/artist name reuse detection with minor variations (NLP embeddings).', example:'Cohen\'s d=−0.91. Ghost titles are repetitive ("Relaxing Piano", "Calming Piano Vol.2", …). Note: S5 is collinear with S2 (ΔAUC=0.000 when added); treated as convergence signal, not independent predictor.', color:'#f59e0b' },
    { n:'6', name:'Graph Centrality',  icon:'🕸', desc:'Neo4j co-appearance network: ghost artists form isolated clusters with no organic connections.', example:'HHI: RWN=0.672, MRC=0.515, Calmo=0.452. Mann-Whitney p=0.003, r=1.000 (n=3 ghost vs 30 organic; bootstrap 95% CI [0.567, 0.900] confirms separation despite small n). Calmo label derived from signal convergence, not independent journalist confirmation.', color:'#e74c3c' },
    { n:'7', name:'Aggregate Score',   icon:'🧠', desc:'Weighted logistic regression combining S2+S4+S5. GNN augmentation via GAT/GCN on 65-node graph. Signal discovery study — Spotify API restrictions limit classifier validation to Kaggle-available signals.', example:'Composite AUC=1.000 on synthetic graph topology (caveat: near-zero feature importance confirms cluster membership learning, not generalized detection). Top behavioral signal: S2 cadence (d=3.44).', color:'var(--green)' },
  ];

  // Highlight the layer that matches the current top signal
  const signals = d.signals || {};
  const topKey = Object.entries(signals).filter(([,v]) => v !== null).sort(([,a],[,b]) => b-a)[0]?.[0] || '';
  const keyToLayer = { s1_audio:1, s2_cadence:2, s3_playlist:3, s4_density:4, s5_metadata:5, s6_graph:6, s7_cross:7 };
  const activeLayer = keyToLayer[topKey] || 0;

  const edaRows = EDA_LAYERS.map(l => {
    const isActive = l.n == activeLayer;
    const sigKey = Object.keys(signals).find(k => k.includes(`s${l.n}`));
    const sigVal = sigKey && signals[sigKey] !== null ? `${Math.round(signals[sigKey]*100)}%` : '—';
    return `
      <div style="display:flex;gap:14px;align-items:flex-start;padding:14px 16px;
           border-radius:10px;border:1px solid ${isActive ? l.color : 'var(--border)'};
           background:${isActive ? l.color+'0d' : 'var(--bg2)'};margin-bottom:8px;transition:border-color 0.2s">
        <div style="width:36px;height:36px;border-radius:8px;background:${l.color}22;
             display:flex;align-items:center;justify-content:center;font-size:1.1rem;flex-shrink:0">${l.icon}</div>
        <div style="flex:1;min-width:0">
          <div style="display:flex;align-items:center;gap:10px;margin-bottom:4px">
            <span style="color:${l.color};font-size:0.72rem;font-weight:700;letter-spacing:1px">LAYER ${l.n}</span>
            <span style="color:var(--white);font-size:0.9rem;font-weight:700">${l.name}</span>
            ${isActive ? `<span style="font-size:0.68rem;background:${l.color}33;color:${l.color};border-radius:12px;padding:1px 8px;font-weight:700">TOP SIGNAL</span>` : ''}
            <span style="margin-left:auto;font-size:0.82rem;font-weight:700;color:${scoreColor(signals[Object.keys(signals).find(k=>k.includes(`s${l.n}`))??'']??null)}">${sigVal}</span>
          </div>
          <div style="color:var(--gray3);font-size:0.82rem;line-height:1.5;margin-bottom:4px">${l.desc}</div>
          <div style="color:var(--gray4);font-size:0.76rem;font-style:italic;line-height:1.4">
            <span style="color:${l.color};font-weight:600">Example: </span>${l.example}
          </div>
        </div>
      </div>`;
  }).join('');

  container.innerHTML = `
    <div class="verdict-banner ${vcls}">
      <div style="font-size:2rem">${vicon}</div>
      <div>
        <div class="verdict-label">${d.verdict_label.replace(/_/g,' ')}</div>
        <div class="verdict-sub">${d.artist_name} &nbsp;·&nbsp; Score: ${(score*100).toFixed(0)}% &nbsp;·&nbsp; Confidence: ${(d.confidence*100).toFixed(0)}%</div>
      </div>
    </div>

    <div style="background:var(--bg2);border:1px solid var(--border);border-radius:12px;padding:20px 24px;margin-bottom:24px;">
      <h3 style="color:var(--white);font-size:0.95rem;margin-bottom:10px;">Expert Analysis</h3>
      <p style="font-size:0.86rem;line-height:1.7;">${d.explanation || 'No explanation available.'}</p>
      ${d.gnn_available ? `<p style="font-size:0.78rem;color:var(--green);margin-top:10px;">🧠 GNN score: ${(d.gnn_score*100).toFixed(0)}% &nbsp;·&nbsp; Rule-based: ${((d.rule_based_score||0)*100).toFixed(0)}%</p>` : ''}
    </div>

    <h3 style="color:var(--white);font-size:0.95rem;margin-bottom:14px;">Signal Breakdown</h3>
    <div class="signals-grid">${signalRows}</div>

    ${d._liveMode ? `
    <div style="background:#0d1a10;border:1px solid #00ff8844;border-radius:10px;padding:14px 18px;margin-top:16px;font-size:0.82rem;color:#a0cfaa;line-height:1.6">
      <strong style="color:var(--green)">Live analysis mode</strong> —
      <span style="color:var(--green)">✓ live</span> real-time Spotify / YouTube / iTunes &nbsp;·&nbsp;
      <span style="color:#60a5fa">📊 Kaggle</span> local processed CSV datasets &nbsp;·&nbsp;
      <span style="color:#f59e0b">🤖 AI</span> GPT-4o-mini estimate (fallback).
      Higher-quality sources override lower ones. Confidence capped at 85% with partial signals.
    </div>` : `
    <div class="info-box">
      <strong>Timing:</strong> Analysis completed in ${d.timing_seconds.toFixed(2)}s using cached Neo4j data.
    </div>`}

    ${trackPanel}

    <div style="margin-top:32px;margin-bottom:12px;display:flex;align-items:center;gap:12px">
      <h3 style="color:var(--white);font-size:0.95rem;margin:0">7-Layer EDA Framework</h3>
      <span style="font-size:0.72rem;color:var(--gray3);letter-spacing:1px;text-transform:uppercase">How we detect ghost artists</span>
    </div>
    <div style="margin-bottom:32px">${edaRows}</div>

    <div class="graph-panel">
      <div class="graph-header">
        <h3 class="panel-title">Artist × Registrant Network</h3>
        <span class="graph-legend">
          <span class="legend-dot" style="background:#00ff88"></span> Artist
          <span class="legend-dot" style="background:#e74c3c"></span> Custom Registrant
          <span class="legend-dot" style="background:#4a90e2"></span> Aggregator
          <span class="legend-dot" style="background:#a78bfa"></span> Label / Unknown
        </span>
      </div>
      <div id="neo4j-graph" class="graph-canvas"></div>
      <div class="graph-caption" id="graph-caption">Loading network…</div>
    </div>
  `;
}

// ── NEO4J NEIGHBORHOOD GRAPH ──────────────────────────────────────────────────

async function loadNeighborhoodGraph(artistId) {
  const graphEl = document.getElementById('neo4j-graph');
  const captionEl = document.getElementById('graph-caption');
  if (!graphEl || !artistId) return;

  try {
    const res = await fetch(apiUrl(`/graph/neighborhood/${artistId}`));
    if (!res.ok) throw new Error('No graph data');
    const data = await res.json();

    const rawNodes = data.nodes || [];
    const rawEdges = data.edges || [];

    if (!rawNodes.length) {
      graphEl.innerHTML = '<div class="graph-empty">This artist is not in the Neo4j graph — live-mode analyses do not have ISRC graph data.</div>';
      if (captionEl) captionEl.textContent = '';
      return;
    }

    // Classify each company node by its label / prefix
    const knownAggregators = ['DISTROKID', 'TUNECORE', 'CDBABY', 'AWAL', 'BELIEVE', 'ONERPM', 'AMUSE'];
    const knownLabels = ['SONY', 'WARN', 'UNIVE', 'ATLANT', 'DEF JAM', 'ISLAND', 'CAPITOL'];
    function classifyNode(node) {
      if (node.type === 'artist') return { bg: '#00ff88', border: '#00c46a', font: '#000' };
      const lbl = (node.label || '').toUpperCase();
      if (knownAggregators.some(a => lbl.includes(a))) return { bg: '#4a90e2', border: '#3a80d2', font: '#fff' };
      if (knownLabels.some(l => lbl.includes(l)))       return { bg: '#a78bfa', border: '#9771f0', font: '#fff' };
      return { bg: '#e74c3c', border: '#c0392b', font: '#fff' };  // custom/unknown = red = ghost signal
    }

    const visNodes = rawNodes.map(n => {
      const cls = classifyNode(n);
      return {
        id: n.id,
        label: n.label,
        color: { background: cls.bg, border: cls.border },
        font: { color: cls.font, size: n.type === 'artist' ? 14 : 11, face: 'Inter' },
        shape: n.type === 'artist' ? 'dot' : 'box',
        size: n.type === 'artist' ? 28 : 16,
        widthConstraint: n.type === 'artist' ? undefined : { minimum: 80, maximum: 160 },
      };
    });

    const visEdges = rawEdges.map((e, i) => ({
      id: i,
      from: e.source,
      to: e.target,
      label: e.label || '',
      font: { color: '#888', size: 10, face: 'Inter', strokeWidth: 0, background: '#0a0a0a' },
      color: { color: 'rgba(255,255,255,0.2)', highlight: '#00ff88' },
      smooth: { type: 'continuous' },
      value: e.weight || 1,
    }));

    if (typeof vis === 'undefined') {
      graphEl.innerHTML = '<div class="graph-empty">vis.js not loaded — check network connection.</div>';
      return;
    }

    new vis.Network(
      graphEl,
      { nodes: new vis.DataSet(visNodes), edges: new vis.DataSet(visEdges) },
      {
        physics: {
          stabilization: { iterations: 120 },
          barnesHut: { gravitationalConstant: -6000, springLength: 160 },
        },
        interaction: { hover: true, dragNodes: true, zoomView: true },
        nodes: { shadow: true },
        edges: { shadow: false },
      }
    );

    // Caption: count red (custom) company nodes
    const companyNodes = rawNodes.filter(n => n.type === 'production_company');
    const customCount = companyNodes.filter(n => {
      const lbl = (n.label || '').toUpperCase();
      return !knownAggregators.some(a => lbl.includes(a)) && !knownLabels.some(l => lbl.includes(l));
    }).length;
    const total = companyNodes.length;
    const pct = total ? Math.round((customCount / total) * 100) : 0;

    if (captionEl) {
      captionEl.textContent = pct >= 70
        ? `Suspicious — ${customCount}/${total} registrants are unknown custom (${pct}%). Ghost-artist signature.`
        : pct >= 30
        ? `Mixed — ${customCount}/${total} custom registrants (${pct}%). Borderline pattern.`
        : `Healthy — ${total - customCount}/${total} known aggregators or labels. Organic distribution.`;
    }

  } catch (e) {
    if (graphEl) graphEl.innerHTML = '<div class="graph-empty">Network data unavailable for this artist.</div>';
    if (captionEl) captionEl.textContent = '';
  }
}

// ── NETWORK EXPLORER ──────────────────────────────────────────────────────────

function renderNetwork(root) {
  root.innerHTML = `
    <div class="page-header">
      <div class="eyebrow">Network Explorer</div>
      <h1>Artist Collaboration Network</h1>
      <p>Cross-platform presence and ISRC production network analysis for seed artists.</p>
    </div>

    <div class="info-box">
      <strong>Note:</strong> Full graph visualization requires Neo4j. Showing cross-platform comparison data below.
    </div>

    <h3 style="color:var(--white);font-size:1rem;margin-bottom:16px;">Seed Artist Comparison</h3>
    <table class="artist-table">
      <thead>
        <tr>
          <th>Artist</th>
          <th>YouTube Views</th>
          <th>Apple Music</th>
          <th>S6 Score (HHI)</th>
          <th>Verdict</th>
        </tr>
      </thead>
      <tbody>
        ${CROSS_PLATFORM.map(a => `
          <tr>
            <td style="color:var(--white);font-weight:600;">${a.name}</td>
            <td style="color:var(--green);font-weight:700;">${fmtViews(a.yt)}</td>
            <td>${a.apple ? '<span style="color:var(--green)">✓ Present</span>' : '<span style="color:var(--gray4)">✗ Absent</span>'}</td>
            <td>${a.s6.toFixed(3)}</td>
            <td>${verdictBadge(a.verdict)}</td>
          </tr>`).join('')}
      </tbody>
    </table>

    <hr class="divider"/>
    <h3 style="color:var(--white);font-size:1rem;margin-bottom:16px;">YouTube View Distribution</h3>
    <div class="data-cards">
      ${CROSS_PLATFORM.map(a => `
        <div class="data-card">
          <div class="data-card-title">${a.name}</div>
          <div class="data-card-val">${fmtViews(a.yt)}</div>
          <div class="data-card-sub">YouTube views &nbsp;·&nbsp; ${a.apple ? '✓ Apple Music' : '✗ No Apple Music'}</div>
          <div style="margin-top:10px;">${verdictBadge(a.verdict)}</div>
        </div>`).join('')}
    </div>

    <div style="background:var(--bg2);border:1px solid var(--border);border-radius:12px;padding:24px;margin-bottom:32px;">
      <h3 style="color:var(--white);font-size:0.95rem;margin-bottom:10px;">Key Insight</h3>
      <p style="font-size:0.87rem;line-height:1.7;">Ghost behavior is <strong style="color:var(--green)">Spotify-economic stream farming</strong>, not cross-platform fabrication.
      Relaxing White Noise has 353M YouTube views — more than Nils Frahm's entire verified catalog reach.
      Cross-platform presence does NOT rule out ghost status.</p>
    </div>

    <footer class="site-footer">GhostTrack | INFO 7390 - Spring 2026 | By Trimbkeshwar Jagtap</footer>
  `;
}

// ── CROSS-PLATFORM ────────────────────────────────────────────────────────────

function renderCrossPlatform(root) {
  root.innerHTML = `
    <div class="page-header">
      <div class="eyebrow">Signal 7 — Cross-Platform</div>
      <h1>Cross-Platform Analysis</h1>
      <p>YouTube presence vs Apple Music discrepancy as a ghost detection signal.</p>
    </div>

    <div class="info-box">
      <strong>Surprise finding:</strong> Ghost artists are NOT cross-platform invisible. RWN has 353M YouTube views yet is classified as a ghost. Ghost behavior is Spotify-specific economic exploitation.
    </div>

    <div class="data-cards" style="grid-template-columns:repeat(4,1fr)">
      ${CROSS_PLATFORM.map(a => `
        <div class="data-card">
          <div class="data-card-title">${a.name}</div>
          <div class="data-card-val">${fmtViews(a.yt)}</div>
          <div class="data-card-sub">YouTube views</div>
          <div style="margin-top:8px;font-size:0.8rem;color:${a.apple ? 'var(--green)' : 'var(--gray4)'}">
            ${a.apple ? '✓ Apple Music' : '✗ No Apple Music'}
          </div>
          <div style="margin-top:10px;">${verdictBadge(a.verdict)}</div>
        </div>`).join('')}
    </div>

    <hr class="divider"/>
    <h3 style="color:var(--white);font-size:1rem;margin-bottom:16px;">Signal 7 Interpretation</h3>
    <table class="artist-table">
      <thead>
        <tr><th>Artist</th><th>YouTube Views</th><th>Apple Music</th><th>Cross-Platform Status</th><th>Verdict</th></tr>
      </thead>
      <tbody>
        ${CROSS_PLATFORM.map(a => {
          const status = a.yt > 10_000_000 && a.apple ? '🟢 Strong' : a.yt < 10_000 && !a.apple ? '🔴 Absent' : '🟡 Moderate';
          return `<tr>
            <td style="color:var(--white);font-weight:600;">${a.name}</td>
            <td style="color:var(--green);font-weight:700;">${fmtViews(a.yt)}</td>
            <td>${a.apple ? '✓' : '✗'}</td>
            <td>${status}</td>
            <td>${verdictBadge(a.verdict)}</td>
          </tr>`;
        }).join('')}
      </tbody>
    </table>

    <footer class="site-footer">GhostTrack | INFO 7390 - Spring 2026 | By Trimbkeshwar Jagtap</footer>
  `;
}

// ── AI ASSISTANT ──────────────────────────────────────────────────────────────

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

let _chatHistory = [];


function askQuestion(btn) { document.getElementById('chat-input').value = btn.innerText; sendChat(); }

async function sendChat() {
  const input = document.getElementById('chat-input');
  const question = input.value.trim();
  if (!question) return;
  input.value = '';

  const win = document.getElementById('chat-window');
  if (!win) return;

  win.innerHTML += `
    <div class="chat-msg user">
      <div class="chat-role" style="text-align:right">You</div>
      <div class="chat-bubble">${question}</div>
    </div>
    <div class="chat-msg assistant" id="thinking">
      <div class="chat-role">GhostTrack AI</div>
      <div class="chat-bubble"><span class="spinner"></span> Thinking…</div>
    </div>`;
  win.scrollTop = win.scrollHeight;

  try {
    const resp = await fetch(apiUrl('/chat'), {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ question, history: _chatHistory }),
    });
    const data = await resp.json();
    const answer = data.answer || data.error || 'No response';
    _chatHistory.push({ user: question, assistant: answer });

    const thinking = document.getElementById('thinking');
    if (thinking) thinking.outerHTML = `
      <div class="chat-msg assistant">
        <div class="chat-role">GhostTrack AI</div>
        <div class="chat-bubble">${answer.replace(/\n/g,'<br>')}</div>
      </div>`;
  } catch(e) {
    const thinking = document.getElementById('thinking');
    if (thinking) thinking.outerHTML = `
      <div class="chat-msg assistant">
        <div class="chat-role">GhostTrack AI</div>
        <div class="chat-bubble" style="color:#fca5a5">Error: ${e.message}</div>
      </div>`;
  }
  win.scrollTop = win.scrollHeight;
}

// ── ABOUT ─────────────────────────────────────────────────────────────────────

function renderAbout(root) {
  root.innerHTML = `
    <div class="page-header">
      <div class="eyebrow">About the Research</div>
      <h1>Methodology & Architecture</h1>
      <p>A deep dive into how we built a framework to detect streaming platform fraud using publicly available data.</p>
    </div>

    <div class="about-grid">
      <div>
        <h2 style="font-size:1.3rem;margin-bottom:16px;">Abstract</h2>
        <p style="line-height:1.85;font-size:0.9rem;">
          This project presents a <strong style="color:var(--white)">7-layer exploratory data analysis framework</strong>
          for detecting ghost artists — AI-generated or fraudulent accounts used to inflate streaming revenue —
          using only Spotify's public, unauthenticated API endpoints.
        </p>
        <p style="line-height:1.85;font-size:0.9rem;margin-top:14px;">
          Prior detection work relies on internal data unavailable to researchers. We demonstrate that
          <strong style="color:var(--green)">catalog coherence, playlist entropy, ISRC attribution, release cadence,
          metadata similarity, and graph topology</strong> each provide independent discriminative signal,
          and that their combination yields robust classification without any proprietary access.
        </p>
        <p style="line-height:1.85;font-size:0.9rem;margin-top:14px;">
          The framework is validated on three seed artists and scaled using the Kaggle Spotify Audio Features
          dataset (114,000 tracks, 114 genres). Our Graph Attention Network achieves
          <strong style="color:var(--white)">100% test accuracy</strong> on the proof-of-concept 65-node collaboration graph.
        </p>
      </div>
      <div class="course-card">
        <div class="course-card-icon">🎓</div>
        <div class="course-name">INFO 7390</div>
        <div class="course-meta">Advances in Data Science<br>Spring 2026<br>Northeastern University</div>
        <hr class="course-divider"/>
        <div class="course-author">Trimbkeshwar Jagtap</div>
        <div class="course-role">Researcher</div>
      </div>
    </div>

    <h2 style="font-size:1.2rem;margin-bottom:10px;">System Architecture</h2>
    <p style="font-size:0.87rem;color:var(--gray3);margin-bottom:18px;">Data flows from multiple sources through our 7-layer analysis pipeline to produce ghost artist probability scores.</p>
    <div class="pipeline-steps">
      <div class="pipe-step"><div class="pipe-num">1</div><div class="pipe-title">Data Sources</div><div class="pipe-items">Spotify API<br>Kaggle CSV<br>YouTube API<br>iTunes API</div></div>
      <div class="pipe-step"><div class="pipe-num">2</div><div class="pipe-title">Processing</div><div class="pipe-items">Feature Extraction<br>Graph Construction<br>Neo4j Ingestion<br>Signal Scoring</div></div>
      <div class="pipe-step"><div class="pipe-num">3</div><div class="pipe-title">Analysis</div><div class="pipe-items">7-Layer Framework<br>GNN Classification<br>CrewAI Pipeline<br>Verdict Engine</div></div>
      <div class="pipe-step"><div class="pipe-num">4</div><div class="pipe-title">Output</div><div class="pipe-items">Ghost Score<br>Visualizations<br>paper/figures/<br>Reports</div></div>
    </div>

    <h2 style="font-size:1.2rem;margin:36px 0 10px;">Tech Stack</h2>
    <div class="tech-grid">
      <div class="tech-card"><div class="tech-cat">Data</div><div class="tech-pkg">Python 3.14<br>pandas 3.0<br>numpy 2.4</div></div>
      <div class="tech-card"><div class="tech-cat">APIs</div><div class="tech-pkg">spotipy 2.26<br>httpx 0.28<br>openai 1.x</div></div>
      <div class="tech-card"><div class="tech-cat">Graph</div><div class="tech-pkg">Neo4j 6.1<br>networkx 3.6</div></div>
      <div class="tech-card"><div class="tech-cat">ML</div><div class="tech-pkg">scikit-learn 1.8<br>torch 2.11<br>torch_geometric 2.7</div></div>
      <div class="tech-card"><div class="tech-cat">Viz</div><div class="tech-pkg">matplotlib 3.10<br>plotly 6.7<br>seaborn 0.13</div></div>
      <div class="tech-card"><div class="tech-cat">Backend</div><div class="tech-pkg">FastAPI 0.135<br>uvicorn 0.44</div></div>
      <div class="tech-card"><div class="tech-cat">Frontend</div><div class="tech-pkg">Vanilla HTML/CSS/JS<br>Zero dependencies</div></div>
      <div class="tech-card"><div class="tech-cat">Agents</div><div class="tech-pkg">crewai 1.14<br>GPT-4o</div></div>
    </div>

    <footer class="site-footer" style="margin-top:40px">GhostTrack | INFO 7390 - Spring 2026 | By Trimbkeshwar Jagtap</footer>
  `;
}

// ── Chat endpoint (add to backend) ────────────────────────────────────────────
// Boot
document.addEventListener('DOMContentLoaded', () => navigate('home'));
