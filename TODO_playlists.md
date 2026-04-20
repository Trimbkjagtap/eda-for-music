# TODO: Real Spotify Playlist Data

## Status: BLOCKED (April 2026)

### What Was Attempted

`scripts/10_fetch_playlists.py` attempted to pull real Spotify playlist track lists
via the Spotify Web API. Two endpoints were tested:

| Endpoint | Result |
|---|---|
| `GET /playlists/{id}/tracks` (user-created) | **HTTP 401** — OAuth user auth required |
| `GET /playlists/{id}/tracks` (editorial 37i9…) | **HTTP 401** — OAuth user auth required |
| `search?type=playlist` | **Works** — returns playlist metadata only (no tracks) |

### Root Cause

The Spotify Client Credentials flow (app-only auth, no user login) does not grant
access to playlist track contents as of April 2026. The `/playlists/{id}/tracks`
endpoint requires the `playlist-read-public` OAuth scope, which needs a user to
authorize the app via the browser-based PKCE/Authorization Code flow.

Editorial playlists (IDs starting with `37i9dQZF1DX4...`) additionally require
special editorial access that is not available to standard developer apps.

### Workaround Applied

`scripts/10_fetch_playlists.py` builds 30 playlist proxies from the Kaggle 114K
dataset (`data/kaggle/dataset.csv`) using genre stratification:

- **Editorial proxies**: Random sample of 60 tracks per Kaggle genre
  (ambient, classical, new-age, piano, sleep, study, idm, guitar, chill, trip-hop)
- **Fan-curated proxies**: Random sample of 60 tracks per Kaggle genre
  (indie, folk, acoustic, singer-songwriter, alternative, indie-pop, jazz, soul, blues, r-n-b)
- **Ghost-suspect proxies**: 
  - 3 DOJ-indicted artist catalog proxies (Relaxing White Noise, Meditation Relax Club, Calmo)
    — not in Kaggle, so built by sampling lowest-variance tracks from their genre pools
  - 7 Kaggle genre pools filtered to low-variance artists (total_variance < 0.010)

### How to Resolve

To get real Spotify playlist data when this block is lifted:

1. Implement OAuth Authorization Code + PKCE flow in `src/api/spotify_client.py`
2. Add `search_editorial_playlists()` using the category/playlist browse endpoints:
   - `GET /browse/categories/{category_id}/playlists`
   - `GET /browse/featured-playlists`
3. For fan-curated: search public playlists by keyword and filter by owner ≠ "spotify"
4. Re-run `scripts/10_fetch_playlists.py` with `USE_API=True` flag (add this env var)
5. Re-run `scripts/11_regenerate_fig2.py` to regenerate figure with real data

Known real editorial playlist IDs (for when API access is restored):
- `37i9dQZF1DX4sWSpwq3LiO` — Peaceful Piano
- `37i9dQZF1DWZeKCadgRdKQ` — Deep Focus
- `37i9dQZF1DWS9BryBdGDsm` — Sleep
- `37i9dQZF1DX3Oog6HxrFTx` — Ambient Relaxation
- `37i9dQZF1DX0SM0LYsmbMT` — Calm Vibes

### Impact on Results

**Figure 2 (Playlist Entropy)**: ANOVA F=0.253, p=0.778 — NO significant entropy
difference between editorial, fan-curated, and ghost-suspect playlist groups.
All pairwise Tukey HSD comparisons are non-significant (p > 0.75).
Cohen's d ≤ 0.27 for all pairs (below the 0.5 threshold for practical significance).

**Signal S2 (playlist entropy component)**: NOT supported by this analysis.
The bulk-upload signature in ghost artists shows up clearly in their *artist-level*
audio variance (Signal S1), but NOT in playlist-level entropy when compared to
same-genre editorial content. This is an honest negative result.

See `scripts/11_regenerate_fig2.py` for full methodology and figure generation.
