# TODO: Dagens Nyheter 830 Ghost Artists List

## Status: Partial — 9 names confirmed, 821+ missing

## What We Searched

The following queries were run against public web sources (April 2026):

1. "Dagens Nyheter 830 fake artists Spotify list 2024"
2. "DN ghost artists Spotify fake streaming fraud list Swedish"
3. "Music Business Worldwide Dagens Nyheter fake artists names Johan Röhr"
4. "Johan Röhr pseudonyms Firefly Entertainment complete list"
5. "Firefly Entertainment AB fake artist names ISRC SEWA"

## What We Found

### Confirmed Artist: Johan Röhr (lead composer)
- Real name: Johan Röhr (Stockholm-based composer)
- Production label: Firefly Entertainment AB
- Scale: 656+ pseudonyms, 2,700+ songs, 15 billion streams, SEK 70M+ royalties
- Associated labels: Lucille AB, Tombola Music, QUeenstreet Content AB, Catfish Music Group

### 9 Confirmed Pseudonyms (publicly named in MBW/English coverage)
These 9 names were confirmed in English-language journalism:
1. Minik Knudsen
2. Mingmei Hsueh
3. Csizmazia Etel
4. Adelmar Borrego
5. Maya Åström
6. Ralph Kaler
7. Sherry Novak
8. Jospeh Turley
9. Miu Hayashi

### ISRC Prefix Fingerprints (from data/raw/cache + ground_truth.py)
Artists registered under these Swedish ISRC prefixes are likely Röhr pseudonyms:
- `SEWA` → Firefly Entertainment AB
- `SEWB` → Lucille AB
- `SEZZ` → Tombola Music
- `SEXX` → Catfish Music Group
- `SECC` → Calm and Collected Music Publishing

## Why the Full List Is Not Available

The original Dagens Nyheter investigation:
1. Was published in Swedish — English summaries omit the name list
2. Journalists deliberately withheld the full list to avoid "free advertising"
3. Spotify removed artists but did not publish which ones
4. The DN investigation archive requires a DN digital subscription + Swedish reading ability

Music Business Worldwide published English coverage but only named 9 examples.

## Next Steps to Obtain Full List

### Option A: DN Archive Access (Best)
- URL: https://www.dn.se/kultur-noje/falskt-konstnarsnamn/
- Requires: DN digital subscription (~SEK 129/month)
- Requires: Swedish reading ability or machine translation
- Expected yield: 50-200 additional names with Swedish naming patterns

### Option B: Spotify ISRC Reverse Lookup
- Query Spotify API for artists whose ISRCs start with SEWA, SEWB, SEZZ, SEXX, SECC
- Method: `SpotifyClient.find_similar_artists_by_isrc(prefix)` already implemented
- Constraint: ~25 API calls available; each prefix lookup = 1-5 calls
- Expected yield: 20-50 additional artist IDs matched to known prefixes

### Option C: Court Document Mining
- The Dagens Nyheter case led to a Swedish civil/criminal investigation
- Swedish court records may be publicly accessible via https://www.domstol.se/
- Search for: "Firefly Entertainment" + "Johan Röhr" in Swedish court databases

### Option D: Spotify Playlist Analysis
- The 495 ghost artists on curated playlists were identified by DN
- Strategy: find "fake" playlist curators on Spotify and examine track ISRCs
- Match ISRCs with known Firefly prefixes to surface artist names

## Current Impact on Dataset

With only 9 confirmed DN names:
- ghost_artists.csv currently uses source="DN" for these 9 entries (confidence=medium)
- 12 additional entries use source="Smith" (from US DOJ indictment examples)
- Bulk of ghost candidates use source="Kaggle-variance" (confidence=low — heuristic only)

## Recommended Priority

Run Option B (ISRC lookup) first — it uses existing tooling with no subscription required.
Then attempt Option A for higher-confidence labels.
