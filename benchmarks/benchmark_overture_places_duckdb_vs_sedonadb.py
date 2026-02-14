# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

import argparse
import math
import os
import statistics
import time
from dataclasses import dataclass


DEFAULT_DATASET = (
    "s3://overturemaps-us-west-2/release/2026-01-21.0/theme=places/type=place/"
)


@dataclass
class BenchmarkResult:
    engine: str
    row_count: int
    max_confidence: float | None
    times_sec: list[float]

    @property
    def median_sec(self) -> float:
        return statistics.median(self.times_sec)

    @property
    def mean_sec(self) -> float:
        return statistics.mean(self.times_sec)

    @property
    def min_sec(self) -> float:
        return min(self.times_sec)

    @property
    def max_sec(self) -> float:
        return max(self.times_sec)


def normalize_sedonadb_dataset_path(dataset_path: str) -> str:
    if dataset_path.endswith("/*.parquet"):
        return dataset_path[: -len("*.parquet")]
    if dataset_path.endswith("/*"):
        return dataset_path[:-1]
    return dataset_path


def run_duckdb(
    dataset_path: str,
    region: str,
    warmup: int,
    runs: int,
) -> BenchmarkResult:
    import duckdb

    con = duckdb.connect(database=":memory:")
    con.install_extension("spatial")
    con.load_extension("spatial")
    con.execute(f"SET s3_region = '{region}'")
    duckdb_dataset_path = dataset_path
    if duckdb_dataset_path.endswith("/"):
        duckdb_dataset_path = f"{duckdb_dataset_path}*.parquet"
    query = f"""
        SELECT
            COUNT(*) AS row_count,
            MAX(confidence) AS max_confidence
        FROM read_parquet(
            '{duckdb_dataset_path}',
            filename=true,
            hive_partitioning=1
        )
        WHERE
            categories.primary = 'pizza_restaurant'
            AND bbox.xmin BETWEEN -75 AND -73
            AND bbox.ymin BETWEEN 40 AND 41
    """

    for _ in range(warmup):
        con.execute(query).fetchone()

    times_sec: list[float] = []
    row_count = 0
    max_confidence = None

    for _ in range(runs):
        start = time.perf_counter()
        row = con.execute(query).fetchone()
        elapsed = time.perf_counter() - start
        times_sec.append(elapsed)

        row_count = int(row[0])
        max_confidence = float(row[1]) if row[1] is not None else None

    con.close()
    return BenchmarkResult(
        engine="DuckDB",
        row_count=row_count,
        max_confidence=max_confidence,
        times_sec=times_sec,
    )


def run_sedonadb(
    dataset_path: str,
    region: str,
    warmup: int,
    runs: int,
) -> BenchmarkResult:
    import sedonadb

    os.environ["AWS_SKIP_SIGNATURE"] = "true"
    os.environ["AWS_DEFAULT_REGION"] = region

    con = sedonadb.connect()
    view_name = "overture_places"
    sedonadb_dataset_path = normalize_sedonadb_dataset_path(dataset_path)
    con.read_parquet(sedonadb_dataset_path).to_view(view_name, overwrite=True)
    query = f"""
        SELECT
            COUNT(*) AS row_count,
            MAX(confidence) AS max_confidence
        FROM {view_name}
        WHERE
            categories.primary = 'pizza_restaurant'
            AND ST_Intersects(
                geometry,
                ST_SetSRID(
                    ST_GeomFromText('POLYGON((-75 40,-75 41,-73 41,-73 40,-75 40))'),
                    4326
                )
            )
    """

    for _ in range(warmup):
        con.sql(query).to_arrow_table()

    times_sec: list[float] = []
    row_count = 0
    max_confidence = None

    for _ in range(runs):
        start = time.perf_counter()
        tab = con.sql(query).to_arrow_table()
        elapsed = time.perf_counter() - start
        times_sec.append(elapsed)

        row_count = int(tab.column("row_count")[0].as_py())
        val = tab.column("max_confidence")[0].as_py()
        max_confidence = float(val) if val is not None else None

    return BenchmarkResult(
        engine="SedonaDB",
        row_count=row_count,
        max_confidence=max_confidence,
        times_sec=times_sec,
    )


def ensure_result_parity(results: list[BenchmarkResult]) -> None:
    if len(results) < 2:
        return

    base = results[0]
    for other in results[1:]:
        if base.row_count != other.row_count:
            raise RuntimeError(
                f"Result mismatch: {base.engine}.row_count={base.row_count} "
                f"!= {other.engine}.row_count={other.row_count}"
            )

        if base.max_confidence is None and other.max_confidence is None:
            continue

        if base.max_confidence is None or other.max_confidence is None:
            raise RuntimeError(
                "Result mismatch: one engine returned NULL max_confidence "
                "and the other did not"
            )

        if not math.isclose(base.max_confidence, other.max_confidence, rel_tol=1e-12):
            raise RuntimeError(
                "Result mismatch: "
                f"{base.engine}.max_confidence={base.max_confidence} "
                f"!= {other.engine}.max_confidence={other.max_confidence}"
            )


def print_results(results: list[BenchmarkResult]) -> None:
    print("\nBenchmark results (seconds):")
    print("| Engine | Median | Mean | Min | Max | Runs | row_count | max_confidence |")
    print("|---|---:|---:|---:|---:|---:|---:|---:|")

    for result in results:
        max_conf_text = (
            "NULL" if result.max_confidence is None else f"{result.max_confidence:.12g}"
        )
        print(
            f"| {result.engine} "
            f"| {result.median_sec:.6f} "
            f"| {result.mean_sec:.6f} "
            f"| {result.min_sec:.6f} "
            f"| {result.max_sec:.6f} "
            f"| {len(result.times_sec)} "
            f"| {result.row_count} "
            f"| {max_conf_text} |"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark Overture places aggregate query on DuckDB and SedonaDB "
            "(no write I/O)."
        )
    )
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--region", default="us-west-2")
    parser.add_argument("--warmup", type=int, default=0)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument(
        "--engine",
        choices=["both", "duckdb", "sedonadb"],
        default="both",
        help="Choose which engine(s) to run",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.warmup < 0:
        raise ValueError("--warmup must be >= 0")
    if args.runs <= 0:
        raise ValueError("--runs must be > 0")

    sedonadb_dataset_path = normalize_sedonadb_dataset_path(args.dataset)

    print(f"SedonaDB read_parquet path: {sedonadb_dataset_path}")
    print(
        "\nRunning with "
        f"dataset={args.dataset}, region={args.region}, "
        f"warmup={args.warmup}, runs={args.runs}, engine={args.engine}"
    )

    results: list[BenchmarkResult] = []

    if args.engine in ("both", "duckdb"):
        results.append(
            run_duckdb(
                dataset_path=args.dataset,
                region=args.region,
                warmup=args.warmup,
                runs=args.runs,
            )
        )

    if args.engine in ("both", "sedonadb"):
        results.append(
            run_sedonadb(
                dataset_path=args.dataset,
                region=args.region,
                warmup=args.warmup,
                runs=args.runs,
            )
        )

    ensure_result_parity(results)
    print_results(results)


if __name__ == "__main__":
    main()
