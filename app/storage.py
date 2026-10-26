"""SQLite persistence (predictions, observed outcomes, alerts, job runs).

SQLite is plenty for one location. To move to PostgreSQL later, keep this
class's public methods and swap the implementation.
"""
from __future__ import annotations

import datetime as dt
import json
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

SCHEMA = """
CREATE TABLE IF NOT EXISTS predictions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at TEXT NOT NULL,
    location TEXT NOT NULL,
    feature_date TEXT NOT NULL,
    target_date TEXT NOT NULL,
    probability REAL NOT NULL,
    threshold REAL NOT NULL,
    prediction INTEGER NOT NULL,
    risk_label TEXT NOT NULL,
    data_source TEXT,
    trigger TEXT,
    missing_features TEXT,
    top_factors TEXT,
    warnings TEXT,
    features TEXT
);
CREATE INDEX IF NOT EXISTS idx_predictions_target ON predictions(target_date);
CREATE TABLE IF NOT EXISTS outcomes (
    target_date TEXT PRIMARY KEY,
    observed_max_rainfall_mm REAL NOT NULL,
    extreme INTEGER NOT NULL,
    source TEXT,
    recorded_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS alerts (
    target_date TEXT PRIMARY KEY,
    sent_at TEXT NOT NULL,
    probability REAL NOT NULL,
    detail TEXT
);
CREATE TABLE IF NOT EXISTS runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    started_at TEXT NOT NULL,
    finished_at TEXT,
    ok INTEGER,
    trigger TEXT,
    feature_date TEXT,
    error TEXT
);
"""

_JSON_COLUMNS = ("missing_features", "top_factors", "warnings", "features")


def utcnow() -> str:
    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat()


class Store:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._conn() as c:
            c.execute("PRAGMA journal_mode=WAL")
            c.executescript(SCHEMA)

    @contextmanager
    def _conn(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self.path, timeout=30)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def ping(self) -> bool:
        try:
            with self._conn() as c:
                c.execute("SELECT 1").fetchone()
            return True
        except sqlite3.Error:
            return False

    # ------------------------------------------------------------ predictions
    def save_prediction(self, rec: Dict[str, Any]) -> int:
        with self._conn() as c:
            cur = c.execute(
                """INSERT INTO predictions
                (created_at, location, feature_date, target_date, probability, threshold, prediction,
                 risk_label, data_source, trigger, missing_features, top_factors, warnings, features)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    rec["created_at"], rec["location"], rec["feature_date"], rec["target_date"],
                    rec["predicted_probability"], rec["threshold"], rec["prediction"], rec["risk_label"],
                    rec.get("data_source"), rec.get("trigger"),
                    json.dumps(rec.get("missing_features", [])),
                    json.dumps(rec.get("top_factors", [])),
                    json.dumps(rec.get("warnings", [])),
                    json.dumps(rec.get("features", {})),
                ),
            )
            return int(cur.lastrowid)

    @staticmethod
    def _row(row: sqlite3.Row) -> Dict[str, Any]:
        d = dict(row)
        for col in _JSON_COLUMNS:
            if d.get(col) is not None:
                d[col] = json.loads(d[col])
        d["predicted_probability"] = d.pop("probability")
        return d

    def latest_prediction(self) -> Optional[Dict[str, Any]]:
        with self._conn() as c:
            row = c.execute("SELECT * FROM predictions ORDER BY id DESC LIMIT 1").fetchone()
        return self._row(row) if row else None

    def list_predictions(self, limit: int = 30, start: Optional[str] = None,
                         end: Optional[str] = None, latest_per_day: bool = True) -> List[Dict[str, Any]]:
        sql = "SELECT * FROM predictions p WHERE 1=1"
        args: List[Any] = []
        if start:
            sql += " AND p.target_date >= ?"
            args.append(start)
        if end:
            sql += " AND p.target_date <= ?"
            args.append(end)
        if latest_per_day:
            sql += " AND p.id = (SELECT MAX(id) FROM predictions WHERE target_date = p.target_date)"
        sql += " ORDER BY p.target_date DESC, p.id DESC LIMIT ?"
        args.append(limit)
        with self._conn() as c:
            rows = c.execute(sql, args).fetchall()
        out = []
        for r in rows:
            d = self._row(r)
            d.pop("features", None)   # keep list responses small
            out.append(d)
        return out

    # --------------------------------------------------------------- outcomes
    def record_outcome(self, target_date: str, observed_mm: float, extreme_threshold_mm: float,
                       source: str = "manual") -> Dict[str, Any]:
        extreme = int(observed_mm >= extreme_threshold_mm)
        with self._conn() as c:
            c.execute(
                """INSERT INTO outcomes (target_date, observed_max_rainfall_mm, extreme, source, recorded_at)
                   VALUES (?,?,?,?,?)
                   ON CONFLICT(target_date) DO UPDATE SET
                     observed_max_rainfall_mm=excluded.observed_max_rainfall_mm,
                     extreme=excluded.extreme, source=excluded.source, recorded_at=excluded.recorded_at""",
                (target_date, observed_mm, extreme, source, utcnow()),
            )
        return {"target_date": target_date, "observed_max_rainfall_mm": observed_mm,
                "extreme": bool(extreme), "source": source}

    def live_metrics(self) -> Dict[str, Any]:
        """Confusion matrix of the latest prediction per target date vs recorded outcomes."""
        with self._conn() as c:
            rows = c.execute(
                """SELECT p.prediction AS pred, o.extreme AS actual
                   FROM predictions p JOIN outcomes o ON o.target_date = p.target_date
                   WHERE p.id = (SELECT MAX(id) FROM predictions WHERE target_date = p.target_date)"""
            ).fetchall()
        tp = sum(1 for r in rows if r["pred"] == 1 and r["actual"] == 1)
        fp = sum(1 for r in rows if r["pred"] == 1 and r["actual"] == 0)
        fn = sum(1 for r in rows if r["pred"] == 0 and r["actual"] == 1)
        tn = sum(1 for r in rows if r["pred"] == 0 and r["actual"] == 0)
        precision = tp / (tp + fp) if (tp + fp) else None
        recall = tp / (tp + fn) if (tp + fn) else None
        f1 = (2 * precision * recall / (precision + recall)
              if precision is not None and recall is not None and (precision + recall) else None)
        return {
            "evaluated_days": len(rows), "true_positives": tp, "false_positives": fp,
            "false_negatives": fn, "true_negatives": tn,
            "precision": precision, "recall": recall, "f1": f1,
            "note": "Extreme days are rare; treat these numbers as unreliable until many events are recorded.",
        }

    # ----------------------------------------------------------------- alerts
    def alert_sent(self, target_date: str) -> bool:
        with self._conn() as c:
            return c.execute("SELECT 1 FROM alerts WHERE target_date = ?", (target_date,)).fetchone() is not None

    def mark_alert(self, target_date: str, probability: float, detail: str = "") -> None:
        with self._conn() as c:
            c.execute("INSERT OR IGNORE INTO alerts (target_date, sent_at, probability, detail) VALUES (?,?,?,?)",
                      (target_date, utcnow(), probability, detail))

    # ------------------------------------------------------------------- runs
    def start_run(self, trigger: str) -> int:
        with self._conn() as c:
            return int(c.execute("INSERT INTO runs (started_at, trigger) VALUES (?,?)",
                                 (utcnow(), trigger)).lastrowid)

    def finish_run(self, run_id: int, ok: bool, feature_date: Optional[str] = None,
                   error: Optional[str] = None) -> None:
        with self._conn() as c:
            c.execute("UPDATE runs SET finished_at=?, ok=?, feature_date=?, error=? WHERE id=?",
                      (utcnow(), int(ok), feature_date, error, run_id))

    def last_run(self) -> Optional[Dict[str, Any]]:
        with self._conn() as c:
            row = c.execute("SELECT * FROM runs ORDER BY id DESC LIMIT 1").fetchone()
        return dict(row) if row else None
