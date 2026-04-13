"""
Neo4j AuraDB client — stores artist/track/playlist graph.
Schema:
  (:Artist {spotify_id, name, followers, genres, is_ghost, label})
  (:Track  {spotify_id, name, isrc, duration_ms, release_date})
  (:Album  {spotify_id, name, release_date, isrc_prefix})
  (:Playlist {spotify_id, name, owner})
  (:ProductionCompany {name, isrc_prefix})

  (Artist)-[:RELEASED]->(Album)
  (Album)-[:CONTAINS]->(Track)
  (Track)-[:APPEARS_ON]->(Playlist)
  (Artist)-[:RELATED_TO]->(Artist)
  (Track)-[:REGISTERED_WITH]->(ProductionCompany)
"""
from neo4j import GraphDatabase
from loguru import logger

from src.utils.config import config


class Neo4jClient:
    def __init__(self):
        self._driver = None

    @property
    def driver(self):
        if self._driver is None:
            self._driver = GraphDatabase.driver(
                config.NEO4J_URI,
                auth=(config.NEO4J_USER, config.NEO4J_PASSWORD),
            )
            logger.info("Neo4j driver initialized")
        return self._driver

    def close(self):
        if self._driver:
            self._driver.close()
            self._driver = None

    def run(self, query: str, **params):
        with self.driver.session() as session:
            return session.run(query, **params).data()

    # ----------------------------------------------------------- schema setup

    def setup_constraints(self):
        """Create uniqueness constraints — idempotent."""
        constraints = [
            "CREATE CONSTRAINT IF NOT EXISTS FOR (a:Artist) REQUIRE a.spotify_id IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (t:Track) REQUIRE t.spotify_id IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (al:Album) REQUIRE al.spotify_id IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (p:Playlist) REQUIRE p.spotify_id IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (c:ProductionCompany) REQUIRE c.isrc_prefix IS UNIQUE",
        ]
        for q in constraints:
            self.run(q)
        logger.info("Neo4j constraints ready")

    # ----------------------------------------------------------- upsert nodes

    def upsert_artist(self, artist_id: str, name: str, followers: int,
                      genres: list, is_ghost: bool = False, label: str = "unknown"):
        self.run(
            """
            MERGE (a:Artist {spotify_id: $id})
            SET a.name = $name,
                a.followers = $followers,
                a.genres = $genres,
                a.is_ghost = $is_ghost,
                a.label = $label
            """,
            id=artist_id, name=name, followers=followers,
            genres=genres, is_ghost=is_ghost, label=label,
        )

    def upsert_track(self, track_id: str, name: str, isrc: str,
                     duration_ms: int, release_date: str):
        self.run(
            """
            MERGE (t:Track {spotify_id: $id})
            SET t.name = $name,
                t.isrc = $isrc,
                t.duration_ms = $duration_ms,
                t.release_date = $release_date
            """,
            id=track_id, name=name, isrc=isrc or "",
            duration_ms=duration_ms, release_date=release_date,
        )

    def upsert_album(self, album_id: str, name: str, release_date: str, isrc_prefix: str = ""):
        self.run(
            """
            MERGE (al:Album {spotify_id: $id})
            SET al.name = $name,
                al.release_date = $release_date,
                al.isrc_prefix = $isrc_prefix
            """,
            id=album_id, name=name, release_date=release_date, isrc_prefix=isrc_prefix,
        )

    def upsert_playlist(self, playlist_id: str, name: str, owner: str):
        self.run(
            """
            MERGE (p:Playlist {spotify_id: $id})
            SET p.name = $name, p.owner = $owner
            """,
            id=playlist_id, name=name, owner=owner,
        )

    def upsert_production_company(self, isrc_prefix: str, name: str):
        self.run(
            """
            MERGE (c:ProductionCompany {isrc_prefix: $prefix})
            SET c.name = $name
            """,
            prefix=isrc_prefix, name=name,
        )

    # ----------------------------------------------------------- relationships

    def link_artist_album(self, artist_id: str, album_id: str):
        self.run(
            """
            MATCH (a:Artist {spotify_id: $aid})
            MATCH (al:Album {spotify_id: $alid})
            MERGE (a)-[:RELEASED]->(al)
            """,
            aid=artist_id, alid=album_id,
        )

    def link_album_track(self, album_id: str, track_id: str):
        self.run(
            """
            MATCH (al:Album {spotify_id: $alid})
            MATCH (t:Track {spotify_id: $tid})
            MERGE (al)-[:CONTAINS]->(t)
            """,
            alid=album_id, tid=track_id,
        )

    def link_related_artists(self, artist_id_a: str, artist_id_b: str):
        self.run(
            """
            MATCH (a:Artist {spotify_id: $aid})
            MATCH (b:Artist {spotify_id: $bid})
            MERGE (a)-[:RELATED_TO]->(b)
            """,
            aid=artist_id_a, bid=artist_id_b,
        )

    def link_track_playlist(self, track_id: str, playlist_id: str):
        self.run(
            """
            MATCH (t:Track {spotify_id: $tid})
            MATCH (p:Playlist {spotify_id: $pid})
            MERGE (t)-[:APPEARS_ON]->(p)
            """,
            tid=track_id, pid=playlist_id,
        )

    def link_track_company(self, track_id: str, isrc_prefix: str):
        self.run(
            """
            MATCH (t:Track {spotify_id: $tid})
            MATCH (c:ProductionCompany {isrc_prefix: $prefix})
            MERGE (t)-[:REGISTERED_WITH]->(c)
            """,
            tid=track_id, prefix=isrc_prefix,
        )

    # ----------------------------------------------------------- queries

    def get_artist_neighborhood(self, artist_id: str, hops: int = 2) -> dict:
        """Return subgraph around an artist (nodes + edges)."""
        result = self.run(
            """
            MATCH path = (a:Artist {spotify_id: $id})-[:RELATED_TO*1..$hops]-(b:Artist)
            RETURN DISTINCT b.spotify_id AS id, b.name AS name,
                   b.followers AS followers, b.is_ghost AS is_ghost
            """,
            id=artist_id, hops=hops,
        )
        return result

    def get_isrc_clusters(self) -> list[dict]:
        """
        Find production companies with multiple artists — Exercise 3.
        Returns: [{company_name, isrc_prefix, artist_count, artists: [name, ...]}]
        """
        return self.run(
            """
            MATCH (t:Track)-[:REGISTERED_WITH]->(c:ProductionCompany)
            MATCH (al:Album)-[:CONTAINS]->(t)
            MATCH (a:Artist)-[:RELEASED]->(al)
            WITH c, collect(DISTINCT a.name) AS artists
            WHERE size(artists) > 1
            RETURN c.name AS company_name,
                   c.isrc_prefix AS isrc_prefix,
                   size(artists) AS artist_count,
                   artists
            ORDER BY artist_count DESC
            """
        )

    def count_nodes(self) -> dict:
        """Return count of each node type."""
        labels = ["Artist", "Track", "Album", "Playlist", "ProductionCompany"]
        counts = {}
        for label in labels:
            result = self.run(f"MATCH (n:{label}) RETURN count(n) AS c")
            counts[label] = result[0]["c"] if result else 0
        return counts

    def test_connection(self) -> bool:
        """Quick connectivity test."""
        try:
            self.run("RETURN 1 AS ok")
            logger.info("Neo4j connection OK")
            return True
        except Exception as e:
            logger.error(f"Neo4j connection failed: {e}")
            return False
