import uuid

from enum import IntEnum, auto
from dataclasses import dataclass


# --------------------------------------------------------------------- #
#  Latency-stress Query Categories                                      #
# --------------------------------------------------------------------- #


class QueryType(IntEnum):
    """
    High-level *SELECT* shapes that exercise very different parts of the
    PostgreSQL execution engine.  Ordered by ever-growing complexity.
    """

    TABLE_SCAN = 0  # Simple LIMIT on the whole table
    WHERE_PREDICATE = auto()  # Single predicate on one column (=, <, > …)
    RANGE_PREDICATE = auto()  # BETWEEN / numeric range
    ORDER_LIMIT = auto()  # ORDER BY … LIMIT …
    MULTI_COLUMN_ORDER = auto()  # ORDER BY col1, col2  LIMIT …
    GROUP_BY = auto()  # GROUP BY one column
    GROUP_BY_HAVING = auto()  # GROUP BY + HAVING aggregate filter
    AGGREGATE = auto()  # Straight aggregate with predicate
    INNER_JOIN = auto()  # Self-join to force hash/merge join
    LEFT_JOIN = auto()  # Outer join
    SUBQUERY_IN = auto()  # Correlated / semi-join via IN (sub-select)
    WINDOW_FUNCTION = auto()  # Analytic function (e.g. running total)


@dataclass
class Request:
    """
    One live SQL request.

    Parameters
    ----------
    request_id : str
    query_type : QueryType
        Coarse category - useful for contextual bandits or curricula.
    sql : str
        The actual query string that will be executed.
    arrival_time : float
        Wall-clock when it entered the environment (``time.perf_counter()``).
    """

    request_id: str
    query_type: QueryType
    sql: str
    arrival_time: float


# --------------------------------------------------------------------- #
#  Query-generation utilities                                           #
# --------------------------------------------------------------------- #
import random
from typing import Callable, List, Tuple


def make_query_generator(
    table: str,
    columns: List[str],
    *,
    seed: int | None = None,
    numeric_cols: List[str] | None = None,
    text_cols: List[str] | None = None,
) -> Callable[[], Tuple[str, QueryType]]:
    """
    Return a zero-arg callable that yields ``(sql, QueryType)``.

    Parameters
    ----------
    table         : fully-qualified table name.
    columns       : every column in that table.
    numeric_cols  : subset of *numeric* columns (defaults to ``columns``).
    text_cols     : subset of *textual* columns (defaults to ``columns``).

    Notes
    -----
    • The generator is **stateless** - perfect for parallel Gym workers.
    • If you do not care about data types, just pass ``columns`` for all
      three arguments and every query will still be syntactically valid.
    """
    rng = random.Random(seed)
    numeric_cols = numeric_cols or columns
    text_cols = text_cols or columns

    OPS = ["=", "<", ">", "<=", ">="]
    DIRECTIONS = ["ASC", "DESC"]
    AGG_FUNCS = ["AVG", "COUNT", "SUM", "MAX", "MIN"]

    def _next_query() -> Tuple[str, QueryType]:
        qt: QueryType = rng.choice(list(QueryType))

        if qt is QueryType.TABLE_SCAN:
            lim = rng.randint(10, 500)
            sql = f"SELECT * FROM {table} LIMIT {lim};"

        elif qt is QueryType.WHERE_PREDICATE:
            col = rng.choice(columns)
            op = rng.choice(OPS)
            val = rng.randint(0, 1000)
            sql = f"SELECT * FROM {table} WHERE {col} {op} {val};"

        elif qt is QueryType.RANGE_PREDICATE:
            col = rng.choice(numeric_cols)
            lo = rng.randint(0, 400)
            hi = lo + rng.randint(50, 400)
            sql = f"SELECT * FROM {table} WHERE {col} BETWEEN {lo} AND {hi};"

        elif qt is QueryType.ORDER_LIMIT:
            col = rng.choice(columns)
            direc = rng.choice(DIRECTIONS)
            lim = rng.randint(10, 200)
            sql = f"SELECT * FROM {table} ORDER BY {col} {direc} LIMIT {lim};"

        elif qt is QueryType.MULTI_COLUMN_ORDER:
            col1, col2 = rng.sample(columns, 2)
            lim = rng.randint(10, 200)
            sql = (
                f"SELECT * FROM {table} "
                f"ORDER BY {col1} ASC, {col2} DESC "
                f"LIMIT {lim};"
            )

        elif qt is QueryType.GROUP_BY:
            col = rng.choice(columns)
            sql = f"SELECT {col}, COUNT(*) FROM {table} GROUP BY {col};"

        elif qt is QueryType.GROUP_BY_HAVING:
            grp = rng.choice(columns)
            num = rng.choice(numeric_cols)
            thr = rng.randint(10, 500)
            sql = (
                f"SELECT {grp}, AVG({num}) AS avg_val "
                f"FROM {table} "
                f"GROUP BY {grp} "
                f"HAVING AVG({num}) > {thr};"
            )

        elif qt is QueryType.AGGREGATE:
            func = rng.choice(AGG_FUNCS)
            col = rng.choice(numeric_cols)
            op = rng.choice(OPS)
            val = rng.randint(0, 1000)
            sql = f"SELECT {func}({col}) FROM {table} WHERE {col} {op} {val};"

        elif qt is QueryType.INNER_JOIN:
            a, b = "a", "b"
            col1 = rng.choice(columns)
            col2 = rng.choice(columns)
            lim = rng.randint(10, 200)
            sql = (
                f"SELECT {a}.{col1}, {b}.{col2} "
                f"FROM {table} {a} "
                f"JOIN {table} {b} ON {a}.{col1} = {b}.{col1} "
                f"LIMIT {lim};"
            )

        elif qt is QueryType.LEFT_JOIN:
            a, b = "l", "r"
            col1 = rng.choice(columns)
            col2 = rng.choice(columns)
            lim = rng.randint(10, 200)
            sql = (
                f"SELECT {a}.{col1}, {b}.{col2} "
                f"FROM {table} {a} "
                f"LEFT JOIN {table} {b} ON {a}.{col1} = {b}.{col1} "
                f"LIMIT {lim};"
            )

        elif qt is QueryType.SUBQUERY_IN:
            col = rng.choice(columns)
            op = rng.choice(OPS)
            val = rng.randint(0, 1000)
            sql = (
                f"SELECT * FROM {table} "
                f"WHERE {col} IN ("
                f"SELECT {col} FROM {table} WHERE {col} {op} {val} LIMIT 20"
                f");"
            )

        elif qt is QueryType.WINDOW_FUNCTION:
            num = rng.choice(numeric_cols)
            date = rng.choice(columns)
            lim = rng.randint(10, 200)
            sql = (
                f"SELECT {date}, {num}, "
                f"SUM({num}) OVER (ORDER BY {date}) AS running_total "
                f"FROM {table} "
                f"LIMIT {lim};"
            )

        else:  # never reached - keeps type-checkers happy
            sql = "SELECT 1;"
            qt = QueryType.TABLE_SCAN

        return sql, qt

    return _next_query
