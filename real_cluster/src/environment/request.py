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

    PING = 0  # SELECT 1; (no-op)
    TABLE_SCAN = auto()  # Simple LIMIT on the whole table
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
    SPECIFIC_VALUE_LOOKUP = auto()
    SELECT_COLUMN_SUBSET = auto()

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


############################################################
##################### QUERY GENERATION #####################
############################################################

from typing import Callable, List, Dict, Tuple
import random
from .request import QueryType


def make_query_generator(
    tables: List[Dict[str, List[str]]],
    *,
    seed: int | None = None,
) -> Callable[[], Tuple[str, QueryType]]:
    """
    Generate SQL queries for use in training.

    Parameters
    ----------
    tables : List of dicts with 'name' and 'columns' keys.
    seed : Optional int to seed random generator.

    Returns
    -------
    A function that yields (SQL string, QueryType).
    """
    rng = random.Random(seed)
    OPS = ["=", "<=", ">="]
    DIRECTIONS = ["ASC", "DESC"]
    AGG_FUNCS = ["AVG", "COUNT", "SUM", "MAX", "MIN"]

    def _next_query() -> Tuple[str, QueryType]:
        qt: QueryType = rng.choice(list(QueryType))

        main = tables[0]
        join = tables[1] if len(tables) > 1 else tables[0]
        main_tbl, join_tbl = main["name"], join["name"]
        main_cols, join_cols = main["columns"], join["columns"]

        # Prefer col1 for joins and filters if available
        join_key = (
            "col1"
            if "col1" in main_cols and "col1" in join_cols
            else rng.choice(main_cols)
        )

        if qt == QueryType.TABLE_SCAN:
            lim = rng.randint(10, 500)
            sql = f"SELECT * FROM {main_tbl} LIMIT {lim};"

        elif qt == QueryType.SPECIFIC_VALUE_LOOKUP:
            col = rng.choice(main_cols)
            val = rng.randint(1, 1000)
            sql = f"SELECT * FROM {main_tbl} WHERE {col} = {val};"

        elif qt == QueryType.WHERE_PREDICATE:
            col = rng.choice(main_cols)
            op = rng.choice(OPS)
            val = rng.randint(0, 1000)
            sql = f"SELECT * FROM {main_tbl} WHERE {col} {op} {val};"

        elif qt == QueryType.RANGE_PREDICATE:
            col = rng.choice(main_cols)
            lo = rng.randint(0, 400)
            hi = lo + rng.randint(50, 400)
            sql = f"SELECT * FROM {main_tbl} WHERE {col} BETWEEN {lo} AND {hi};"

        elif qt == QueryType.SELECT_COLUMN_SUBSET:
            selected = rng.sample(main_cols, rng.randint(1, len(main_cols)))
            lim = rng.randint(10, 200)
            sql = f"SELECT {', '.join(selected)} FROM {main_tbl} LIMIT {lim};"

        elif qt == QueryType.INNER_JOIN:
            lim = rng.randint(10, 200)
            sql = (
                f"SELECT a.{join_key}, b.{join_key} "
                f"FROM {main_tbl} a "
                f"JOIN {join_tbl} b ON a.{join_key} = b.{join_key} "
                f"LIMIT {lim};"
            )

        elif qt == QueryType.LEFT_JOIN:
            lim = rng.randint(10, 200)
            sql = (
                f"SELECT a.{join_key}, b.{join_key} "
                f"FROM {main_tbl} a "
                f"LEFT JOIN {join_tbl} b ON a.{join_key} = b.{join_key} "
                f"LIMIT {lim};"
            )

        elif qt == QueryType.ORDER_LIMIT:
            col = rng.choice(main_cols)
            dir_ = rng.choice(DIRECTIONS)
            lim = rng.randint(10, 200)
            sql = f"SELECT * FROM {main_tbl} ORDER BY {col} {dir_} LIMIT {lim};"

        elif qt == QueryType.GROUP_BY:
            col = rng.choice(main_cols)
            sql = f"SELECT {col}, COUNT(*) FROM {main_tbl} GROUP BY {col};"

        elif qt == QueryType.AGGREGATE:
            func = rng.choice(AGG_FUNCS)
            col = rng.choice(main_cols)
            op = rng.choice(OPS)
            val = rng.randint(0, 1000)
            sql = f"SELECT {func}({col}) FROM {main_tbl} WHERE {col} {op} {val};"

        elif qt == QueryType.SUBQUERY_IN:
            col = rng.choice(main_cols)
            val = rng.randint(0, 1000)
            sql = (
                f"SELECT * FROM {main_tbl} WHERE {col} IN ("
                f"SELECT {col} FROM {join_tbl} WHERE {col} > {val} LIMIT 20);"
            )

        elif qt == QueryType.WINDOW_FUNCTION:
            num = rng.choice(main_cols)
            sql = (
                f"SELECT {num}, SUM({num}) OVER (ORDER BY {num}) AS running_sum "
                f"FROM {main_tbl} LIMIT 100;"
            )

        elif qt == QueryType.MULTI_COLUMN_ORDER:
            col1, col2 = rng.sample(main_cols, 2)
            lim = rng.randint(10, 200)
            sql = (
                f"SELECT * FROM {main_tbl} "
                f"ORDER BY {col1} ASC, {col2} DESC "
                f"LIMIT {lim};"
            )

        elif qt == QueryType.GROUP_BY_HAVING:
            grp = rng.choice(main_cols)
            num = rng.choice(main_cols)
            thr = rng.randint(10, 500)
            sql = (
                f"SELECT {grp}, AVG({num}) AS avg_val "
                f"FROM {main_tbl} "
                f"GROUP BY {grp} "
                f"HAVING AVG({num}) > {thr};"
            )
        else:
            sql = "SELECT 1;"
            qt = QueryType.PING

        return sql, qt

    return _next_query
