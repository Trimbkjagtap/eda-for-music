/* ── GhostTrack SPA ────────────────────────────────────────────────────────
   Pure vanilla JS single-page app.
   navigate(page) swaps content inside #page-root, no reload.
   API calls go to FastAPI backend at /api/* (same origin).
──────────────────────────────────────────────────────────────────────────── */

const API = '';   // same origin; prefix all calls with /

// ── Static data ──────────────────────────────────────────────────────────────

const KNOWN_ARTISTS = [
  { label: 'Relaxing White Noise (ghost)',    id: '6bo3atMVp3qFECNALVwq9N' },
  { label: 'Meditation Relax Club (ghost)',   id: '3BqBPFLxBkzKQTkuBPGMNF' },
  { label: 'Calmo (candidate)',               id: '4Wx3ZL6d6p1gVMtwQ2YWsz' },
  { label: 'Nils Frahm (organic)',            id: '5hVghJ3sCFHFJoLnSHySjL' },
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
  { file:'fig1_catalog_coherence.png',     ex:'Exercise 1', sig:'Catalog Variance',  title:'Figure 1: Catalog Coherence in Audio Feature Space', caption:'PCA projection of per-track audio features for ghost-like vs organic artists. Ghost-like artists cluster into tight compact ellipses. Data: Kaggle 114K-track dataset.' },
  { file:'fig2_playlist_entropy.png',      ex:'Exercise 2', sig:'Playlist Entropy',  title:'Figure 2: Playlist Aesthetic Coherence', caption:'Energy vs Valence scatter for three simulated playlist archetypes. TIGHT playlists show low Shannon entropy — hallmark of a fraud target zone.' },
  { file:'fig3_isrc_join.png',             ex:'Exercise 3', sig:'ISRC Attribution',  title:'Figure 3: Artist to Production Company Attribution via ISRC', caption:'Bipartite graph connecting artists to production companies via ISRC prefix. 3 seed artists, 490 tracks, 8 production companies.' },
  { file:'fig4_bipartite_neighborhood.png',ex:'Exercise 4', sig:'Graph Centrality',  title:'Figure 4: Artist × Production Company Bipartite Neighborhood', caption:'HHI scores: RWN=0.88, MRC=0.66, Calmo=0.54 — ghost artists show extreme ISRC concentration.' },
  { file:'fig5_recommendation_walk.png',   ex:'Exercise 5', sig:'Release Cadence',   title:'Figure 5: Recommendation Walk — Release Cadence as Walk Closure Signal', caption:'Ghost artists: RWN=81%, MRC=95% closure. Organic control (Nils Frahm): 0% closure, median gap 105 days.' },
  { file:'fig6_signal_radar.png',          ex:'Exercise 6', sig:'Aggregate Score',   title:'Figure 6: Seven-Signal Ghost Artist Detection Radar', caption:'S2 Release Cadence, S4 Catalog Density, and S6 Graph/HHI are the most discriminative signals.' },
  { file:'fig6b_signal_heatmap.png',       ex:'Exercise 6', sig:'Signal Heatmap',    title:'Figure 6b: Signal Report Card Heatmap', caption:'Heatmap of all 7 signal scores across 4 artists. S2/S4/S6 cleanly separate ghost from organic.' },
  { file:'fig7_gnn_performance.png',       ex:'Exercise 7', sig:'GNN Model',         title:'Figure 7: GNN Ghost Artist Detection Performance', caption:'GAT vs GCN training curves, ROC, confusion matrix. Dataset: 65 nodes (14 ghost, 51 organic), 692 edges.' },
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
    ai:       renderAI,
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
            HHI concentration coefficient of <strong>0.88</strong>
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
        <img class="fig-img" src="/figures/${f.file}" alt="${f.title}"
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
      <p>Enter a Spotify Artist ID to run the full 7-signal detection pipeline.</p>
    </div>

    <div class="search-box">
      <input id="artist-input" class="search-input" type="text"
             placeholder="Spotify Artist ID  e.g. 6bo3atMVp3qFECNALVwq9N"
             onkeydown="if(event.key==='Enter') runAnalysis()"/>
      <button class="btn-primary" onclick="runAnalysis()">Analyze →</button>
    </div>

    <div class="quick-picks">
      ${KNOWN_ARTISTS.map(a => `
        <button class="quick-pill" onclick="setAndAnalyze('${a.id}')">${a.label}</button>
      `).join('')}
    </div>

    <div id="analyzer-result"></div>
    <footer class="site-footer">GhostTrack | INFO 7390 - Spring 2026 | By Trimbkeshwar Jagtap</footer>
  `;
}

function setAndAnalyze(id) {
  document.getElementById('artist-input').value = id;
  runAnalysis();
}

async function runAnalysis() {
  const id = document.getElementById('artist-input').value.trim();
  if (!id) return;
  const result = document.getElementById('analyzer-result');
  result.innerHTML = `<div class="loading-state"><span class="spinner"></span> Running 7-signal pipeline…</div>`;

  try {
    const resp = await fetch(`/analyze`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ artist_id: id, run_cross_platform: false }),
    });
    if (!resp.ok) throw new Error(`API error ${resp.status}`);
    const d = await resp.json();
    renderAnalysisResult(result, d);
  } catch (e) {
    result.innerHTML = `
      <div style="background:#2a0a0a;border:1px solid #e74c3c;border-radius:10px;padding:20px;color:#fca5a5;font-size:0.88rem;">
        <strong>Analysis failed:</strong> ${e.message}<br>
        <span style="color:var(--gray4);font-size:0.8rem;">Make sure the FastAPI backend is running and Neo4j has this artist ingested.</span>
      </div>`;
  }
}

function renderAnalysisResult(container, d) {
  const score = d.verdict_score;
  const vcls = score >= 0.7 ? 'verdict-ghost' : score >= 0.4 ? 'verdict-suspicious' : 'verdict-organic';
  const vicon = score >= 0.7 ? '👻' : score >= 0.4 ? '⚠️' : '✓';

  const signalRows = Object.entries(d.signals || {}).map(([key, val]) => {
    const pct = val !== null ? Math.round(val * 100) : null;
    const name = SIGNAL_NAMES[key] || key;
    const color = scoreColor(val);
    return `
      <div class="signal-row">
        <div class="signal-name">${name}</div>
        <div class="signal-bar-wrap">
          <div class="signal-bar" style="width:${pct ?? 0}%;background:${color}"></div>
        </div>
        <div class="signal-score" style="color:${color}">${pct !== null ? pct + '%' : '—'}</div>
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

    <div class="info-box">
      <strong>Timing:</strong> Analysis completed in ${d.timing_seconds.toFixed(2)}s using cached Neo4j data.
    </div>
  `;
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
        <div class="chat-bubble">Hello! I'm a PhD-level research assistant for the GhostTrack project. I have full context of all 7 analysis layers, signal scores, and findings. Ask me anything about ghost artist detection, the methodology, or the results.</div>
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
    const resp = await fetch('/chat', {
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
