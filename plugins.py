from semantic_kernel.functions import kernel_function
from typing import Annotated, TypedDict
import pandas as pd
import duckdb
import kuzu


class Tag(TypedDict):
    tag_id: str
    tag_name: str

class LabVsSpec(TypedDict):
    stream_id: str
    analyte: str
    test_id: str
    lab_value: float
    unit: str
    analyzed_time: str
    draw_time: str
    limit_type: str
    lo: float | None
    hi: float | None
    basis: str

class TimeSeriesWindowMean(TypedDict):
    tag_id: str
    center_ts: str
    window_size_minutes: int
    mean_value: float

class KGAgentPlugin:
    def __init__(
            self,
            kuzu_connection: kuzu.Connection,
            duckdb_connection: duckdb.DuckDBPyConnection,
        ):
            self.kuzu_connection = kuzu_connection
            self.duckdb_connection = duckdb_connection
            # Pre-cache simple parsed schema artifacts (node labels, rel types) for LLM grounding
            try:
                self._schema_text = self._load_schema_text()
                self._schema_summary = self._summarize_schema(self._schema_text)
            except Exception:
                self._schema_text = ""
                self._schema_summary = ""

    # ---------------- Schema Helpers ----------------
    def _load_schema_text(self) -> str:
        """Attempt to load the external Kùzu DDL text if available.

        Looks for a sibling file named 'kuzu_ddl.txt' relative to this module.
        Fails silent (empty string) if not found so plugin construction never breaks.
        """
        import pathlib
        ddl_path = pathlib.Path(__file__).with_name("kuzu_ddl.txt")
        try:
            return ddl_path.read_text(encoding="utf-8")
        except Exception:
            return ""

    def _summarize_schema(self, ddl: str) -> str:
        """Return a concise natural-language summary of nodes and relationships.

        This is intentionally lightweight so it can be injected into a system
        prompt or retrieved by the model via a tool call rather than pushing the
        full DDL every time. The LLM can call a second function to fetch full
        DDL if needed.
        """
        if not ddl:
            return ""
        import re
        node_lines = [l for l in ddl.splitlines() if l.strip().upper().startswith("CREATE NODE TABLE")]
        rel_lines  = [l for l in ddl.splitlines() if l.strip().upper().startswith("CREATE REL TABLE")]
        def parse_nodes(lines):
            out = []
            for ln in lines:
                m = re.search(r"CREATE NODE TABLE (\w+)\(([^;]+)\)", ln)
                if m:
                    label = m.group(1)
                    cols = [c.strip().split()[0] for c in m.group(2).split(',') if c.strip() and not c.strip().upper().startswith('PRIMARY')]
                    out.append(f"{label}({', '.join(cols)})")
            return out
        def parse_rels(lines):
            out = []
            for ln in lines:
                m = re.search(r"CREATE REL TABLE (\w+)\(FROM (\w+) TO (\w+)\)", ln)
                if m:
                    out.append(f"{m.group(1)}: {m.group(2)} -> {m.group(3)}")
            return out
        nodes_str = "; ".join(parse_nodes(node_lines))
        rels_str  = "; ".join(parse_rels(rel_lines))
        return f"Nodes: {nodes_str}. Relationships: {rels_str}."

    @kernel_function(name="tag_for_stream_analyte", description="Get the tag ID for a given stream ID and analyte code.")
    def tag_for_stream_analyte(
         self,
         stream_id: Annotated[str, "The stream ID"],
         analyte_code: Annotated[str, "The analyte code"]
    ) -> Annotated[Tag | None, "The tag ID for the given stream ID and analyte code"]:
        q = f"""
        MATCH (:Stream {{id:'{stream_id}'}})<-[:MEASURES]-(:Analyzer)-[:HAS_TAG]->(t:Tag)<-[:SERVED_BY_TAG]-(an:Analyte {{code:'{analyte_code}'}})
        RETURN t.id AS tag_id, t.name AS tag_name LIMIT 1;
        """
        res = self.kuzu_connection.execute(q)
        rows = list(res)
        return Tag(tag_id=rows[0][0], tag_name=rows[0][1]) if rows else None

    @kernel_function(name="latest_lab_vs_spec", description="Get the latest lab value and corresponding spec limits for a given stream ID and analyte code.")
    def latest_lab_vs_spec(
        self,
        stream_id: Annotated[str, "The stream ID"],
        analyte_code: Annotated[str, "The analyte code"]
    ) -> Annotated[LabVsSpec | None, "The latest lab value and spec limits for the given stream ID and analyte code"]:
        q = f"""
        MATCH (s:Stream {{id:'{stream_id}'}})<-[:SAMPLE_OF]-(sm:Sample)<-[:RESULT_OF]-(tr:TestResult {{status:'Validated'}})
            -[:TR_USES_METHOD]->(:Method)-[:MEASURES_ANALYTE]->(an:Analyte {{code:'{analyte_code}'}})
        WITH tr, s, sm, an ORDER BY sm.draw_time DESC LIMIT 1
        MATCH (sl:SpecLimit)-[:APPLIES_TO]->(s), (sl)-[:FOR_ANALYTE]->(an)
        RETURN s.id AS stream_id, an.code AS analyte, tr.id AS test_id, tr.value AS lab_value,
            tr.unit AS unit, tr.analyzed_time AS analyzed_time, sm.draw_time AS draw_time,
            sl.limit_type AS limit_type, sl.lo AS lo, sl.hi AS hi, sl.basis AS basis;
        """
        res = self.kuzu_connection.execute(q)
        rows = list(res)
        if not rows:
            return None
        return LabVsSpec(
            stream_id=rows[0][0],
            analyte=rows[0][1],
            test_id=rows[0][2],
            lab_value=rows[0][3],
            unit=rows[0][4],
            analyzed_time=str(pd.Timestamp(rows[0][5], tz='UTC')),
            draw_time=str(pd.Timestamp(rows[0][6], tz='UTC')),
            limit_type=rows[0][7],
            lo=rows[0][8],
            hi=rows[0][9],
            basis=rows[0][10],
        )

    @kernel_function(name="timeseries_window_mean", description="Get the mean value of a time series data window for a given tag ID and center timestamp.")
    def timeseries_window_mean(
        self,
        tag_id: Annotated[str, "The tag ID"],
        center_ts: Annotated[pd.Timestamp, "The center timestamp"],
        minutes: Annotated[int, "The window size in minutes"] = 10
    ) -> Annotated[TimeSeriesWindowMean, "The mean value of the time series data window"]:
        start = (center_ts - pd.Timedelta(minutes=minutes)).tz_convert("UTC")
        end   = (center_ts + pd.Timedelta(minutes=minutes)).tz_convert("UTC")
        res = self.duckdb_connection.execute(
                """
                SELECT ts, value, quality
                FROM ts.timeseries
                WHERE tag_id = ? AND ts BETWEEN ? AND ?
                ORDER BY ts
                """,
                [tag_id, start.to_pydatetime(), end.to_pydatetime()],
            ).df()
        mean_value = float(res["value"].mean()) if not res.empty else 0.0
        return TimeSeriesWindowMean(
            tag_id=tag_id,
            center_ts=str(center_ts),
            window_size_minutes=minutes,
            mean_value=mean_value,
        )

    # ---------------- Generic Graph Query Execution ----------------
    @kernel_function(name="graph_query", description="Execute a read-only Cypher/GQL query against the knowledge graph and return rows as a list of dict objects. Use ONLY for MATCH/RETURN style queries. Always prefer specialized functions when available.")
    def graph_query(
        self,
        query: Annotated[str, "A read-only Cypher query (MATCH/RETURN). No writes."],
        limit: Annotated[int, "Optional enforced maximum rows to return"] = 50,
    ) -> Annotated[list[dict] | str, "List of result row dicts or error string"]:
        """Execute an arbitrary read-only graph query with safety guards.

        Safety constraints:
        - Reject queries containing keywords suggesting mutation (CREATE, MERGE, DELETE, SET, DROP, LOAD, COPY)
        - Auto-append LIMIT if user did not provide one and limit is set.
        - Hard cap rows fetched to 'limit'.
        Returns list of row dicts for structured consumption by the LLM.

        Usage Guidance for LLM:
        1. If unsure about schema, FIRST call graph_schema_summary.
        2. Formulate a MATCH ... RETURN query referencing correct labels and properties.
        3. Do not include modifying clauses or subqueries that alter the database.
        4. Avoid returning extremely wide cartesian products; specify pattern depth precisely.
        5. Provide human readable aliases using 'AS' to make output clearer.
        """
        q_lower = query.lower()
        forbidden = ["create ", "merge ", "delete ", "set ", "drop ", "load ", "copy "]
        if any(k in q_lower for k in forbidden):
            return "Rejected: only read-only MATCH/RETURN queries are permitted."
        if " match" not in q_lower and not q_lower.startswith("match"):
            # Simple heuristic to avoid arbitrary functions/DDL
            return "Rejected: query must begin with or contain a MATCH clause."
        # Enforce limit if not already present (naive check)
        if " limit " not in q_lower:
            query = query.rstrip().rstrip(';') + f" LIMIT {int(limit)};"
        try:
            res = self.kuzu_connection.execute(query)
            cols = [c for c in res.get_column_names()]
            out = []
            for i, row in enumerate(res):
                if i >= limit:
                    break
                out.append({cols[j]: row[j] for j in range(len(cols))})
            return out
        except Exception as e:
            return f"Error: {e}"

    # ---------------- Generic Time-series Query Execution ----------------
    @kernel_function(name="ts_query", description="Execute a read-only SQL query against the DuckDB time-series store (schema 'ts'). Use ONLY for SELECT style analytical queries. Always prefer specialized functions when available.")
    def ts_query(
        self,
        query: Annotated[str, "A read-only SQL query (SELECT). No writes or DDL."],
        limit: Annotated[int, "Optional enforced maximum rows to return"] = 500,
    ) -> Annotated[list[dict] | str, "List of result row dicts or error string"]:
        """Execute an arbitrary read-only DuckDB query with safety guards.

        Safety constraints:
        - Reject queries containing keywords suggesting mutation (INSERT, UPDATE, DELETE, COPY, CREATE, DROP, ALTER, REPLACE, ATTACH, DETACH, LOAD, EXPORT)
        - Reject PRAGMA statements
        - Require query to contain a SELECT clause
        - Auto-append LIMIT if user did not provide one and limit is set.
        - Hard cap rows fetched to 'limit'.
        Returns list of row dicts for structured consumption by the LLM.

        Usage Guidance for LLM:
        1. If unsure about available tables/columns, first run: SELECT * FROM information_schema.tables WHERE table_schema='ts';
        (Will be truncated by LIMIT; refine to specific table names.)
        2. Typical base table is ts.timeseries(tag_id, ts, value, quality).
        3. Always filter on tag_id and time ranges (ts BETWEEN ... AND ...) to avoid large scans.
        4. Provide human readable aliases using 'AS' to make output clearer.
        5. Avoid SELECT * on large time ranges; aggregate where possible (e.g., AVG(value) AS avg_value).
        """
        q_lower = query.lower()
        forbidden = [
            "insert ", "update ", "delete ", "copy ", "create ", "drop ", "alter ",
            "replace ", "attach ", "detach ", "pragma ", "export ", "load "
        ]
        if any(k in q_lower for k in forbidden):
            return "Rejected: only read-only SELECT queries are permitted."
        if " select" not in q_lower and not q_lower.startswith("select"):
            return "Rejected: query must begin with or contain a SELECT clause."
        # Enforce limit if not already present (naive check)
        if " limit " not in q_lower:
            query = query.rstrip().rstrip(';') + f" LIMIT {int(limit)};"
        try:
            res = self.duckdb_connection.execute(query)
            cols = [c for c in res.description]  # duckdb cursor description
            out = []
            for i, row in enumerate(res.fetchall()):
                if i >= limit:
                    break
                out.append({cols[j][0]: row[j] for j in range(len(cols))})
            return out
        except Exception as e:
            return f"Error: {e}"

    # ---------------- Schema Exposure Functions ----------------
    @kernel_function(name="graph_schema_summary", description="Get a concise natural language summary of the graph schema (node labels and relationships). Call this before constructing novel queries.")
    def graph_schema_summary(self) -> Annotated[str, "Summary of graph schema"]:
        return self._schema_summary or "Schema summary unavailable."

    @kernel_function(name="graph_schema_ddl", description="Retrieve the full raw DDL used to build the graph (for reference when crafting complex queries). Use sparingly due to length.")
    def graph_schema_ddl(self) -> Annotated[str, "Full graph DDL text"]:
        return self._schema_text or "DDL unavailable."

    # ---------------- Time-series Schema Exposure ----------------
    @kernel_function(name="ts_schema", description="Return a concise description of the DuckDB time-series schema (tables, columns, types). Call this before constructing novel ts_query statements if unsure.")
    def ts_schema(self) -> Annotated[str, "Summary of time-series schema"]:
        try:
            # Query information_schema for tables in 'ts' schema
            tables = self.duckdb_connection.execute(
                """
                SELECT table_name
                FROM information_schema.tables
                WHERE table_schema='ts'
                ORDER BY table_name
                """
            ).fetchall()
            if not tables:
                return "No tables found in schema 'ts'."
            summaries: list[str] = []
            for (tbl,) in tables:
                cols = self.duckdb_connection.execute(
                    f"""
                    SELECT column_name, data_type
                    FROM information_schema.columns
                    WHERE table_schema='ts' AND table_name='{tbl}'
                    ORDER BY ordinal_position
                    """
                ).fetchall()
                col_str = ", ".join(f"{c}:{t}" for c, t in cols)
                summaries.append(f"{tbl}({col_str})")
            return "Tables: " + "; ".join(summaries)
        except Exception as e:
            return f"Error retrieving ts schema: {e}"

    # ---------------- Utility Calculations ----------------
    @kernel_function(name="calculator", description="Safely evaluate a pure arithmetic Python expression (numbers, + - * / // % **, parentheses, and selected math functions like sin, cos, sqrt, log, exp, pi, e). Use for lightweight numeric calculations instead of external tools. Do NOT include assignments or variable definitions.")
    def calculator(
        self,
        expression: Annotated[str, "A pure arithmetic expression (e.g. '2 + 3*4', 'sin(pi/4)**2')"],
        precision: Annotated[int, "Optional number of decimal places to round the result to (0-12)"] = 6,
    ) -> Annotated[float | str, "Numeric result (float) or error string"]:
        """Evaluate a limited arithmetic expression securely.

        Safety / Allowed:
        - Literals: int, float
        - Operators: + - * / // % ** and parentheses
        - Unary +/-
        - Functions/constants from math: sin, cos, tan, asin, acos, atan, sqrt, log, log10, exp, pow, fabs, floor, ceil, pi, e
        Rejections:
        - Any names not in allowlist
        - Attribute access, indexing, comprehensions, lambdas, conditionals, boolean ops, comparisons
        - Assignments or statements (only a single expression is parsed in 'eval' mode)
        Returns an error string on rejection instead of raising.
        """
        import ast, math

        allow_names: dict[str, object] = {k: getattr(math, k) for k in [
            "sin", "cos", "tan", "asin", "acos", "atan", "sqrt", "log", "log10", "exp", "pow", "fabs", "floor", "ceil", "pi", "e"
        ]}

        try:
            tree = ast.parse(expression, mode="eval")
        except SyntaxError as se:
            return f"Error: invalid syntax ({se.msg})"

        allowed_nodes = (
            ast.Expression,
            ast.BinOp, ast.UnaryOp,
            ast.Add, ast.Sub, ast.Mult, ast.Div, ast.FloorDiv, ast.Mod, ast.Pow,
            ast.USub, ast.UAdd,
            ast.Call, ast.Name, ast.Load,
            ast.Constant,
        )

        def _check(node: ast.AST) -> bool:
            if not isinstance(node, allowed_nodes):
                return False
            # Restrict Call nodes
            if isinstance(node, ast.Call):
                if not isinstance(node.func, ast.Name):
                    return False
                if node.func.id not in allow_names:
                    return False
                # Disallow keyword args for simplicity
                if node.keywords:
                    return False
            # Restrict Name usage
            if isinstance(node, ast.Name):
                if node.id not in allow_names:
                    return False
            for child in ast.iter_child_nodes(node):
                if not _check(child):
                    return False
            return True

        if not _check(tree):
            return "Error: expression contains disallowed syntax. Use only arithmetic operators and approved math functions."

        try:
            result = eval(compile(tree, filename="<calc>", mode="eval"), {"__builtins__": {}}, allow_names)
        except ZeroDivisionError:
            return "Error: division by zero"
        except Exception as e:
            return f"Error during evaluation: {e}"

        # Normalize result to float
        try:
            precision = max(0, min(int(precision), 12))
        except Exception:
            precision = 6
        if isinstance(result, (int, float)):
            return float(round(result, precision))
        return "Error: non-numeric result produced (unexpected)."