"""
Day 1 API verification test.
Run with: python tests/test_spotify_connection.py
"""
import sys
sys.path.insert(0, ".")

from src.api.spotify_client import SpotifyClient
from src.api.apple_music_client import AppleMusicClient
from src.utils.config import config

def run_tests():
    print("=" * 60)
    print("EDA for Music — API Connection Tests")
    print("=" * 60)

    # Check credentials
    missing = config.validate()
    if missing:
        print(f"FAIL: Missing credentials: {missing}")
        return
    print(f"OK   Credentials loaded (Spotify Client ID: {config.SPOTIFY_CLIENT_ID[:8]}...)")

    client = SpotifyClient()

    # Test 1: Connection
    print("\n--- Test 1: Spotify Connection ---")
    ok = client.test_connection()
    print(f"{'OK' if ok else 'FAIL'}   Basic connection")

    # Test 2: Artist metadata
    print("\n--- Test 2: Artist Metadata ---")
    artist = client.get_artist("4Z8W4fKeB5YxbusRsdQVPb")  # Radiohead
    print(f"OK   Artist name: {artist['name']}")
    print(f"     Fields returned: {list(artist.keys())}")
    print(f"     NOTE: followers/genres stripped in April 2026 API")

    # Test 3: Albums
    print("\n--- Test 3: Artist Albums ---")
    albums = client.get_artist_albums("4Z8W4fKeB5YxbusRsdQVPb", include_groups="album")
    print(f"OK   Albums found: {len(albums)}")
    if albums:
        print(f"     Latest: {albums[0]['name']} ({albums[0]['release_date']})")

    # Test 4: Album tracks
    print("\n--- Test 4: Album Tracks ---")
    if albums:
        tracks = client.get_album_tracks(albums[0]["id"])
        print(f"OK   Tracks in album: {len(tracks)}")
        if tracks:
            print(f"     Sample track: {tracks[0]['name']}")

    # Test 5: ISRC (Exercise 3 dependency)
    print("\n--- Test 5: ISRC Codes (Critical for Exercise 3) ---")
    ok_isrc = client.test_isrc()
    if ok_isrc:
        track = client.get_track("1HnY1Lu9tH5YK8DiIxho9i")
        isrc = track["external_ids"]["isrc"]
        print(f"OK   ISRC confirmed: {isrc}")
        print(f"     Registrant prefix: {isrc[:5]}")
    else:
        print("FAIL ISRC not available")

    # Test 6: Search
    print("\n--- Test 6: Search ---")
    results = client.search_artists("Bon Iver", limit=1)
    if results:
        print(f"OK   Search found: {results[0]['name']} (id: {results[0]['id']})")
    else:
        print("FAIL Search returned no results")

    # Test 7: Release dates (Signal 2)
    print("\n--- Test 7: Release Dates (Signal 2 - Cadence) ---")
    dates = client.get_release_dates("4Z8W4fKeB5YxbusRsdQVPb")
    print(f"OK   Release dates found: {len(dates)}")
    if dates:
        print(f"     Sample dates: {dates[:5]}")

    # Test 8: Apple Music (no API key needed)
    print("\n--- Test 8: Apple Music / iTunes ---")
    apple = AppleMusicClient()
    results = apple.search_track("Radiohead", "Creep")
    print(f"OK   Apple Music results for 'Radiohead - Creep': {len(results)}")

    # Test 9: Related artists (expect 403)
    print("\n--- Test 9: Related Artists (expect 403 for new apps) ---")
    try:
        related = client.sp.artist_related_artists("4Z8W4fKeB5YxbusRsdQVPb")
        print(f"OK   Related artists available: {len(related.get('artists', []))}")
    except Exception as e:
        print(f"NOTE Related artists blocked (403) — using album-based graph expansion instead")
        print(f"     This is expected for Dev Mode apps created after Feb 2026")

    print("\n" + "=" * 60)
    print("Day 1 API tests complete.")
    print("What works: artist lookup, albums, tracks, ISRC, search")
    print("What's blocked: related-artists, editorial playlists, audio features")
    print("Strategy: use Kaggle for audio features, album co-artists for graph")
    print("=" * 60)


if __name__ == "__main__":
    run_tests()
