import os, pathlib
import shutil, argparse
from datetime import datetime, timezone
import numpy as np
import pandas as pd
import duckdb
import kuzu

# ---- Paths
DATABASES_DIR = "./databases"
GRAPH_DB_PATH = os.path.join(DATABASES_DIR, "graph.kuzu")
TS_DB_PATH = os.path.join(DATABASES_DIR, "timeseries.duckdb")

# ---- Externalized Kùzu DDL / SEED data (loaded from text files)
BASE_DIR = pathlib.Path(__file__).resolve().parent
DDL_PATH = BASE_DIR / "kuzu_ddl.txt"
SEED_PATH = BASE_DIR / "kuzu_seed_data.txt"

def _read_text(p: pathlib.Path) -> str:
    try:
        return p.read_text(encoding="utf-8")
    except FileNotFoundError as e:
        raise FileNotFoundError(f"Required file not found: {p}") from e

DDL = _read_text(DDL_PATH)
SEED_DATA = _read_text(SEED_PATH)

class KGBuilder:
    """Knowledge Graph + Time-series database builder.

    On initialization it creates/opens both the Kùzu graph database and DuckDB
    time-series store. Optionally rebuilds (deletes & recreates) and seeds them.

    Attributes
    ----------
    kuzu_db : kuzu.Database
        Underlying Kùzu database instance.
    kuzu_conn : kuzu.Connection
        Connection to the Kùzu database (use for queries).
    duck_conn : duckdb.DuckDBPyConnection
        Connection to the DuckDB time-series store.
    """

    def __init__(
        self,
        rebuild: bool = False,
        graph_db_path: str = GRAPH_DB_PATH,
        ts_db_path: str = TS_DB_PATH,
        ddl: str = DDL,
        seed_data: str = SEED_DATA,
    ) -> None:
        self.graph_db_path = pathlib.Path(graph_db_path)
        self.ts_db_path = ts_db_path
        self._ddl = ddl
        self._seed_data = seed_data

        self.kuzu_db, self.kuzu_conn = self._init_kuzu(rebuild=rebuild)
        self.duck_conn = self._init_duckdb(rebuild=rebuild)

        # Always seed time-series data on rebuild for deterministic demo environment
        if rebuild:
            self.seed_duckdb_timeseries()

    # ---- Public helpers ----
    @property
    def connections(self):
        """Return (kuzu_connection, duckdb_connection) tuple for convenience."""
        return self.kuzu_conn, self.duck_conn

    # ---- Internal build methods ----
    def _kuzu_exec(self, q: str, conn: kuzu.Connection | None = None):
        """Execute potentially multi-statement Kùzu query text safely.

        Parameters
        ----------
        q : str
            One or more semicolon-delimited statements.
        conn : kuzu.Connection | None
            Explicit connection (used during early init). Falls back to self.kuzu_conn.
        """
        target_conn = conn or getattr(self, "kuzu_conn", None)
        if target_conn is None:
            raise RuntimeError("Kùzu connection not initialized yet.")
        for stmt in [s.strip() for s in q.split(";") if s.strip()]:
            try:
                target_conn.execute(stmt + ";")
            except Exception as e:
                snippet = (stmt[:200] + ("..." if len(stmt) > 200 else "")).replace("\n", " ")
                raise RuntimeError(
                    f"Kùzu exec failed for statement: {snippet}\nOriginal error: {e}"
                ) from e

    def _init_kuzu(self, rebuild: bool):
        db_path = self.graph_db_path
        if rebuild and db_path.exists():
            if db_path.is_dir():
                shutil.rmtree(db_path)
            else:
                db_path.unlink()
        db_path.parent.mkdir(parents=True, exist_ok=True)
        db = kuzu.Database(str(db_path))
        conn = kuzu.Connection(db)

        # Detect existing seed via sentinel node using local conn
        already_seeded = False
        try:
            res = conn.execute("MATCH (s:Site {id:'SITE1'}) RETURN s.id LIMIT 1;")
            already_seeded = len(list(res)) > 0
        except Exception:
            pass  # Will run DDL below

        if rebuild or not already_seeded:
            self._kuzu_exec(self._ddl, conn=conn)
            self._kuzu_exec(self._seed_data, conn=conn)
        else:
            print("Existing data detected; skipping seed. Use rebuild=True to recreate.")

        # Only now set instance attributes (ensures _kuzu_exec can still work via param)
        self.kuzu_conn = conn
        return db, conn

    def _init_duckdb(self, rebuild: bool):
        con = duckdb.connect(self.ts_db_path)
        if rebuild:
            con.execute("DROP SCHEMA IF EXISTS ts CASCADE;")
            con.execute("CREATE SCHEMA ts;")
            con.execute(
                """
                CREATE TABLE ts.timeseries(
                    tag_id TEXT,
                    ts TIMESTAMPTZ,
                    value DOUBLE,
                    quality SMALLINT
                );
                """
            )
            con.execute("CREATE INDEX idx_timeseries_tag_ts ON ts.timeseries(tag_id, ts);")
        return con

    # ---- Time-series seeding ----
    def simulate_series(
        self,
        start: datetime,
        end: datetime,
        base: float,
        noise: float,
        step_prob: float = 0.0,
        step_size: float = 0.0,
    ) -> pd.DataFrame:
        return simulate_series(start, end, base, noise, step_prob, step_size)

    def write_timeseries(self, tag_id: str, df: pd.DataFrame):
        write_timeseries(self.duck_conn, tag_id, df)

    def seed_duckdb_timeseries(self):
        seed_duckdb_timeseries(self.duck_conn)

def simulate_series(start: datetime, end: datetime, base: float, noise: float,
                    step_prob: float = 0.0, step_size: float = 0.0) -> pd.DataFrame:
    idx = pd.date_range(start, end, freq="1min", tz="UTC")
    v = np.full(len(idx), base, dtype=float)
    eps = np.random.normal(scale=noise, size=len(idx))
    for i in range(1, len(idx)):
        v[i] = 0.9 * v[i-1] + 0.1 * base + eps[i]
        if step_prob and np.random.rand() < step_prob:
            v[i] += np.random.choice([-1, 1]) * step_size
    return pd.DataFrame({"ts": idx, "value": v, "quality": 1}, columns=["ts","value","quality"])

def write_timeseries(con: duckdb.DuckDBPyConnection, tag_id: str, df: pd.DataFrame):
    # Ensure tz-aware to map to TIMESTAMPTZ
    if df["ts"].dt.tz is None:
        df["ts"] = df["ts"].dt.tz_localize("UTC")
    df2 = df.copy()
    df2.insert(0, "tag_id", tag_id)
    con.register("df_in", df2)
    con.execute("INSERT INTO ts.timeseries SELECT * FROM df_in;")
    con.unregister("df_in")

# ---------------- Demo wiring (module-level function retained for reuse) ----------------
def seed_duckdb_timeseries(con: duckdb.DuckDBPyConnection):
    # Simulate 24h to cover the seeded lab times
    start = datetime(2025, 9, 9, 12, 0, tzinfo=timezone.utc)
    end   = datetime(2025, 9, 10, 23, 59, tzinfo=timezone.utc)

    gas_s  = simulate_series(start, end, base=8.0,  noise=0.15, step_prob=0.002, step_size=0.6)
    lpg_c3 = simulate_series(start, end, base=38.0, noise=0.40, step_prob=0.002, step_size=1.5)
    gas_rvp = simulate_series(start, end, base=8.3, noise=0.08, step_prob=0.002, step_size=0.30)
    lco_rho = simulate_series(start, end, base=923.0, noise=0.30, step_prob=0.001, step_size=1.0)

    write_timeseries(con, "TAG:S_GAS_PPM", gas_s)
    write_timeseries(con, "TAG:PROPYLENE_MOLPCT", lpg_c3)
    write_timeseries(con, "TAG:RVP_GAS_PSI",   gas_rvp)
    write_timeseries(con, "TAG:LCO_RHO_KGPM3", lco_rho)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="KGBuilder demo")
    parser.add_argument("--rebuild", action="store_true", help="Recreate both databases and reseed graph + timeseries (timeseries always seeded on rebuild)")
    args = parser.parse_args()

    builder = KGBuilder(rebuild=args.rebuild)
    kconn, dconn = builder.connections
    print("Kùzu and DuckDB connections ready (rebuild=%s)" % args.rebuild)
    print("  Kùzu connection:", kconn)
    print("  DuckDB connection:", dconn)
