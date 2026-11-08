//! The store: an append-only text log that git can sync, and a SQLite cache
//! built from it.
//!
//! ```text
//! lab/data/log/<host>/<utc date>.jsonl   one record per line, only <host> appends
//! lab/data/body/<sha>.json               response bodies, content-addressed
//! lab/data/kernel/<sha>.asc              kernel sources as sent, content-addressed
//! lab/gmmlab.db                          cache; delete it and it is rebuilt
//! ```
//!
//! Every machine appends only to its own directory and every other file is
//! named by its content, so pulling from any number of machines never
//! conflicts. Nothing in `data/` is ever rewritten or deleted.
//!
//! The database is never the truth. Each open ingests whatever log bytes it
//! has not seen yet; a schema change bumps `VERSION`, which throws the cache
//! away and rebuilds it. There are no migrations.
//!
//! Everything derived is a view, evaluated at read time: per-case times,
//! scores, the board, the best time ever seen. The scoring formula exists once,
//! in `score.rs`, and is registered as the `gmm_score` / `gmm_ratio` UDFs so
//! Rust and SQL cannot drift apart.
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

use rusqlite::functions::FunctionFlags;
use rusqlite::{Connection, OpenFlags, TransactionBehavior, params};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};

use crate::err::{Error, Result};
use crate::score;

/// Bump on any change to TABLES or to how records are ingested.
const VERSION: i32 = 2;

const TABLES: &str = r#"
-- How far into each log file this cache has read.
CREATE TABLE IF NOT EXISTS ingested (
    file  TEXT PRIMARY KEY,                 -- path relative to data/log
    bytes INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS blob (
    sha  TEXT PRIMARY KEY,
    text TEXT NOT NULL
);

-- Every HTTP round trip to anything upstream, including the failures.
CREATE TABLE IF NOT EXISTS xlog (
    id       TEXT PRIMARY KEY,              -- time-ordered: sorts as it happened
    host     TEXT NOT NULL,                 -- the machine that made the call
    at       REAL NOT NULL,                 -- request start, unix seconds
    ms       REAL,                          -- round trip, milliseconds
    account  TEXT NOT NULL,
    kind     TEXT NOT NULL,                 -- submit|submission|ranking|user_submissions|oauth_refresh
    method   TEXT NOT NULL,
    url      TEXT NOT NULL,
    req      TEXT,                          -- request body as sent, secrets and source reduced to a sha
    status   INTEGER,                       -- null only on a transport failure
    headers  TEXT,                          -- response headers as a JSON object
    body_sha TEXT,                          -- response body, verbatim, in blob
    error    TEXT,                          -- transport-level failure
    judge_id TEXT,                          -- the judge submission this is about
    sub_id   TEXT                           -- our submission, when there is one
);
CREATE INDEX IF NOT EXISTS xlog_kind  ON xlog(kind, id);
CREATE INDEX IF NOT EXISTS xlog_judge ON xlog(judge_id, id);

CREATE TABLE IF NOT EXISTS kernel (
    sha   TEXT PRIMARY KEY,
    src   TEXT    NOT NULL,
    bytes INTEGER NOT NULL,
    lines INTEGER NOT NULL
);

-- A claimed submission slot. Written before anything is sent.
CREATE TABLE IF NOT EXISTS submission (
    id         TEXT PRIMARY KEY,
    host       TEXT NOT NULL,
    created_at REAL NOT NULL,
    account    TEXT NOT NULL,               -- the label: secrets/accounts/<account>.json
    judge_uid  TEXT,                        -- who that label was, as the judge knows it
    kernel_sha TEXT NOT NULL,
    reason     TEXT
);
CREATE INDEX IF NOT EXISTS submission_account ON submission(account, created_at);

-- What happened to a submission on our side: sent|failed|timeout|imported.
-- The judge's verdict is never written here; it is read out of xlog.
CREATE TABLE IF NOT EXISTS submission_event (
    id       TEXT PRIMARY KEY,
    sub_id   TEXT NOT NULL,
    at       REAL NOT NULL,
    state    TEXT NOT NULL,
    judge_id TEXT,
    error    TEXT
);
CREATE INDEX IF NOT EXISTS submission_event_sub ON submission_event(sub_id, at);

-- Our own device timing of the kernel on every point, given at submit time.
CREATE TABLE IF NOT EXISTS local_time (
    sub_id  TEXT    NOT NULL,
    case_no INTEGER NOT NULL,
    us      REAL    NOT NULL,
    PRIMARY KEY (sub_id, case_no)
);
"#;

/// Views. Dropped and recreated on every connect -- they are free, and a
/// changed definition takes effect without touching the data.
const VIEWS: &str = r#"
DROP VIEW IF EXISTS exchange;
DROP VIEW IF EXISTS verdict;
DROP VIEW IF EXISTS case_raw;
DROP VIEW IF EXISTS case_result;
DROP VIEW IF EXISTS visible;
DROP VIEW IF EXISTS ranking_row;
DROP VIEW IF EXISTS ranking_case;
DROP VIEW IF EXISTS board;
DROP VIEW IF EXISTS testcase;
DROP VIEW IF EXISTS tbest_history;
DROP VIEW IF EXISTS board_case;
DROP VIEW IF EXISTS board_split;
DROP VIEW IF EXISTS board_submission;
DROP VIEW IF EXISTS best_time;
DROP VIEW IF EXISTS submission_row;
DROP VIEW IF EXISTS submission_full;
DROP VIEW IF EXISTS local_vs_judge;
DROP VIEW IF EXISTS poll_history;

-- The log with its bodies attached.
CREATE VIEW exchange AS
SELECT x.*, b.text AS body
FROM xlog x LEFT JOIN blob b ON b.sha = x.body_sha;

-- The newest detail the judge gave us for each submission. Older polls stay in
-- the log; this view is just the last word.
CREATE VIEW verdict AS
SELECT e.judge_id                        AS judge_id,
       e.id                              AS exchange_id,
       e.at                              AS seen_at,
       json_extract(e.body, '$.status')  AS judge_status,
       e.body                            AS raw
FROM exchange e
WHERE e.kind = 'submission' AND e.status = 200 AND e.judge_id IS NOT NULL
  AND e.id = (SELECT MAX(x.id) FROM xlog x
              WHERE x.kind = 'submission' AND x.status = 200
                AND x.judge_id = e.judge_id);

-- One row per test case per verdict, straight out of the stored JSON.
CREATE VIEW case_raw AS
SELECT v.judge_id, v.exchange_id, v.seen_at, v.judge_status,
       r.key + 1                                  AS case_no,
       json_extract(r.value, '$.time')            AS time_us,
       json_extract(r.value, '$.best_time')       AS best_us,
       json_extract(r.value, '$.testcase_status') AS case_status
FROM verdict v, json_each(v.raw, '$.result') r;

CREATE VIEW case_result AS
SELECT judge_id, exchange_id, seen_at, judge_status, case_no, time_us, best_us,
       case_status,
       gmm_score(time_us, best_us)            AS score,
       gmm_ratio(gmm_score(time_us, best_us)) AS ratio
FROM case_raw;

-- The cases the judge showed us a time for, where they actually ran. Which
-- cases those are is the judge's call, not a constant: a hidden case comes back
-- with time null, and a compile error fills the slots with time 0.
CREATE VIEW visible AS
SELECT * FROM case_result
WHERE time_us IS NOT NULL AND time_us > 0
  AND best_us IS NOT NULL AND best_us > 0;

-- Every leaderboard row of every snapshot, from every machine.
CREATE VIEW ranking_row AS
SELECT e.id                                          AS exchange_id,
       e.at                                          AS taken_at,
       json_extract(r.value, '$.rank')               AS rank,
       json_extract(r.value, '$.score')              AS score,
       json_extract(r.value, '$.status')             AS status,
       json_extract(r.value, '$.create_time')        AS created_at,
       json_extract(r.value, '$.submission_id')      AS submission_id,
       json_extract(r.value, '$.submission.ID')      AS submission_seq,
       json_extract(r.value, '$.submission.status')  AS submission_status,
       json_extract(r.value, '$.team._id')           AS team_id,
       json_extract(r.value, '$.team.team_name')     AS team,
       json_extract(r.value, '$.team.ID')            AS team_seq,
       json_extract(r.value, '$.team.team_id')       AS team_code,
       json_extract(r.value, '$.user_id')            AS user_id,
       json_extract(r.value, '$.user.nickname')      AS user,
       json_extract(r.value, '$.submitter._id')      AS submitter_id,
       json_extract(r.value, '$.submitter.nickname') AS submitter,
       json_array_length(r.value, '$.result')        AS cases,
       r.value                                       AS raw
FROM exchange e, json_each(e.body, '$.rows') r
WHERE e.kind = 'ranking' AND e.status = 200;

CREATE VIEW ranking_case AS
SELECT b.exchange_id, b.taken_at, b.rank, b.team, b.team_id, b.submitter,
       b.submission_id,
       b.score                                    AS row_score,
       c.key + 1                                  AS case_no,
       json_extract(c.value, '$.time')            AS time_us,
       json_extract(c.value, '$.best_time')       AS best_us,
       json_extract(c.value, '$.testcase_status') AS case_status,
       json_extract(c.value, '$.precision_ratio') AS precision_ratio,
       json_extract(c.value, '$.testcase_id')     AS case_id
FROM ranking_row b, json_each(b.raw, '$.result') c;

-- Only the most recent snapshot.
CREATE VIEW board AS
SELECT * FROM ranking_row
WHERE exchange_id = (SELECT MAX(id) FROM xlog WHERE kind = 'ranking' AND status = 200);

-- The problem's test cases as the newest ranking reply lists them: how many
-- there are, which are hidden, and the live tbest of each visible one. This is
-- the authority for the case count -- it is not a constant.
CREATE VIEW testcase AS
SELECT t.key + 1                                      AS case_no,
       json_extract(t.value, '$._id')                 AS testcase_id,
       COALESCE(json_extract(t.value, '$.hidden'), 0) AS hidden,
       json_extract(t.value, '$.tbest')               AS tbest_us,
       e.id                                           AS exchange_id,
       e.at                                           AS taken_at
FROM exchange e, json_each(e.body, '$.testcases') t
WHERE e.id = (SELECT MAX(id) FROM xlog WHERE kind = 'ranking' AND status = 200);

-- tbest of every visible case in every ranking snapshot ever taken, from any
-- machine: the competition's tbest over time.
CREATE VIEW tbest_history AS
SELECT e.id AS exchange_id, e.at AS taken_at, e.host,
       t.key + 1                        AS case_no,
       json_extract(t.value, '$.tbest') AS tbest_us
FROM exchange e, json_each(e.body, '$.testcases') t
WHERE e.kind = 'ranking' AND e.status = 200
  AND json_extract(t.value, '$.tbest') IS NOT NULL;

-- The best time we have ever been told about for each case, in our verdicts.
CREATE VIEW best_time AS
SELECT case_no, MIN(best_us) AS best_us, MAX(seen_at) AS seen_at, COUNT(*) AS samples
FROM case_raw
WHERE best_us IS NOT NULL AND best_us > 0
GROUP BY case_no;

-- The newest snapshot's per-case rows, with tbest: the snapshot's own where it
-- gives one, else the best our verdicts have reported.
CREATE VIEW board_case AS
SELECT c.*,
       COALESCE(tc.tbest_us, bt.best_us)                      AS tbest_us,
       CASE WHEN COALESCE(tc.tbest_us, bt.best_us) > 0 AND c.time_us > 0
            THEN c.time_us / COALESCE(tc.tbest_us, bt.best_us) END AS x_tbest,
       CASE WHEN COALESCE(tc.tbest_us, bt.best_us) > 0 AND c.time_us > 0
            THEN gmm_score(c.time_us, COALESCE(tc.tbest_us, bt.best_us)) END AS case_score
FROM ranking_case c
LEFT JOIN testcase  tc ON tc.case_no = c.case_no
LEFT JOIN best_time bt ON bt.case_no = c.case_no
WHERE c.exchange_id = (SELECT MAX(id) FROM xlog WHERE kind = 'ranking' AND status = 200);

-- What each team's own total implies for the cases nobody is shown. Only
-- legitimate because score and visible times come from the SAME board row,
-- and only when that row has a time on every visible case.
CREATE VIEW board_split AS
SELECT b.rank, b.team, b.submitter, b.submission_id, b.score AS total_score,
       b.cases,
       COUNT(c.case_no)                                          AS visible_cases,
       AVG(c.case_score)                                         AS visible_mean,
       CASE WHEN COUNT(c.case_no) = (SELECT COUNT(*) FROM testcase WHERE hidden = 0)
             AND COUNT(c.case_no) < b.cases
            THEN (b.cases * b.score - SUM(c.case_score)) / (b.cases - COUNT(c.case_no))
       END                                                       AS hidden_mean
FROM board b
JOIN board_case c ON c.rank = b.rank AND c.case_score IS NOT NULL
GROUP BY b.rank, b.team, b.submitter, b.submission_id, b.score, b.cases;

-- A submission with its lifecycle folded in.
CREATE VIEW submission_row AS
SELECT s.*,
       (SELECT e.judge_id FROM submission_event e
        WHERE e.sub_id = s.id AND e.judge_id IS NOT NULL ORDER BY e.at LIMIT 1) AS judge_id,
       (SELECT e.at FROM submission_event e
        WHERE e.sub_id = s.id AND e.judge_id IS NOT NULL ORDER BY e.at LIMIT 1) AS sent_at,
       COALESCE((SELECT e.state FROM submission_event e
                 WHERE e.sub_id = s.id ORDER BY e.at DESC LIMIT 1), 'reserved') AS state
FROM submission s;

-- Our submissions with the judge's last word attached.
CREATE VIEW submission_full AS
SELECT s.id, s.host, s.account, s.judge_uid, s.judge_id, s.kernel_sha, s.reason,
       s.created_at, s.sent_at, s.state,
       k.bytes AS kernel_bytes, k.lines AS kernel_lines,
       v.judge_status, v.seen_at AS judged_at,
       (SELECT AVG(score) FROM visible w WHERE w.judge_id = s.judge_id) AS visible_mean,
       (SELECT COUNT(*)   FROM visible w WHERE w.judge_id = s.judge_id) AS visible_cases,
       (SELECT COUNT(*)   FROM local_time l WHERE l.sub_id = s.id)      AS local_cases
FROM submission_row s
LEFT JOIN kernel  k ON k.sha      = s.kernel_sha
LEFT JOIN verdict v ON v.judge_id = s.judge_id;

-- Board rows matched to the submission that earned them, where we hold it.
CREATE VIEW board_submission AS
SELECT b.rank, b.team, b.team_id, b.submitter, b.score, b.submission_id, b.status,
       s.id AS sub_id, s.account AS sub_account, s.reason AS sub_reason,
       s.kernel_sha, b.taken_at
FROM board b
LEFT JOIN submission_row s ON s.judge_id = b.submission_id;

-- Our local timing next to the judge's, per submission and case. judge_us is
-- null on hidden cases: the judge never returns those times.
CREATE VIEW local_vs_judge AS
SELECT s.id AS sub_id, s.account, s.judge_id, l.case_no, l.us AS local_us,
       CASE WHEN c.time_us > 0 THEN c.time_us END AS judge_us,
       c.best_us,
       CASE WHEN c.time_us > 0 THEN c.time_us / l.us END AS judge_over_local
FROM local_time l
JOIN submission_row s ON s.id = l.sub_id
LEFT JOIN case_raw c ON c.judge_id = s.judge_id AND c.case_no = l.case_no;

-- How long judging actually took, from the polls we made.
CREATE VIEW poll_history AS
SELECT judge_id, COUNT(*) AS polls, MIN(at) AS first_poll, MAX(at) AS last_poll,
       MAX(at) - MIN(at) AS span_s
FROM xlog
WHERE kind = 'submission' AND judge_id IS NOT NULL
GROUP BY judge_id;
"#;

pub fn sha256_hex(bytes: &[u8]) -> String {
    let digest = Sha256::digest(bytes);
    let mut out = String::with_capacity(64);
    for byte in digest {
        out.push_str(&format!("{byte:02x}"));
    }
    out
}

/// A record id: 12 hex digits of milliseconds, then 8 of entropy. Ids sort in
/// the order things happened, which is what `MAX(id)` in the views relies on,
/// and two machines never mint the same one.
pub fn new_id(host: &str) -> String {
    static COUNTER: AtomicU64 = AtomicU64::new(0);
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default();
    let seed = format!(
        "{}:{}:{}:{}",
        now.as_nanos(),
        std::process::id(),
        COUNTER.fetch_add(1, Ordering::Relaxed),
        host
    );
    format!("{:012x}{}", now.as_millis(), &sha256_hex(seed.as_bytes())[..8])
}

fn register_udfs(conn: &Connection) -> Result<()> {
    let flags = FunctionFlags::SQLITE_UTF8 | FunctionFlags::SQLITE_DETERMINISTIC;
    conn.create_scalar_function("gmm_score", 2, flags, |ctx| {
        let t: Option<f64> = ctx.get(0)?;
        let best: Option<f64> = ctx.get(1)?;
        Ok(match (t, best) {
            (Some(t), Some(best)) => Some(score::case_score(t, best)),
            _ => None,
        })
    })?;
    conn.create_scalar_function("gmm_ratio", 1, flags, |ctx| {
        let s: Option<f64> = ctx.get(0)?;
        // SQLite has no infinity; null says "unbounded" without faking a number.
        Ok(s.map(score::ratio_from_score).filter(|r| r.is_finite()))
    })?;
    Ok(())
}

/// Write a file once under its content address. A file that is already there
/// has the same content by construction, so it is left alone.
fn write_once(path: &Path, text: &str) -> Result<()> {
    if path.exists() {
        return Ok(());
    }
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let temp = path.with_extension(format!("tmp{}", std::process::id()));
    std::fs::write(&temp, text)?;
    if std::fs::rename(&temp, path).is_err() {
        // Another process got there first; its copy is identical.
        let _ = std::fs::remove_file(&temp);
        if !path.exists() {
            return Err(Error::Io(format!("cannot write {}", path.display())));
        }
    }
    Ok(())
}

fn text(value: &Value) -> Option<String> {
    match value {
        Value::Null => None,
        Value::String(s) => Some(s.clone()),
        other => Some(other.to_string()),
    }
}

pub struct Store {
    pub conn: Connection,
    pub host: String,
    data: PathBuf,
    db: PathBuf,
}

impl Store {
    pub fn open(data: &Path, db: &Path, host: &str) -> Result<Self> {
        std::fs::create_dir_all(data.join("log").join(host))?;
        if let Some(parent) = db.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let mut conn = Self::connect(db)?;
        let version: i32 = conn.query_row("PRAGMA user_version", [], |row| row.get(0))?;
        let fresh: i64 =
            conn.query_row("SELECT COUNT(*) FROM sqlite_master", [], |row| row.get(0))?;
        if fresh > 0 && version != VERSION {
            // The cache is from another schema. It holds nothing the log does
            // not, so it is thrown away rather than migrated.
            drop(conn);
            for suffix in ["", "-wal", "-shm"] {
                let _ = std::fs::remove_file(format!("{}{suffix}", db.display()));
            }
            conn = Self::connect(db)?;
        }
        {
            // Several processes may open at once: publish the whole schema in
            // one transaction so nobody sees half a set of views.
            let tx = conn.transaction_with_behavior(TransactionBehavior::Immediate)?;
            tx.execute_batch(TABLES)?;
            tx.execute_batch(VIEWS)?;
            tx.pragma_update(None, "user_version", VERSION)?;
            tx.commit()?;
        }
        let store = Store {
            conn,
            host: host.to_string(),
            data: data.to_path_buf(),
            db: db.to_path_buf(),
        };
        store.ingest()?;
        Ok(store)
    }

    fn connect(db: &Path) -> Result<Connection> {
        let conn = Connection::open(db)?;
        conn.busy_timeout(std::time::Duration::from_secs(30))?;
        conn.pragma_update(None, "journal_mode", "WAL")?;
        register_udfs(&conn)?;
        Ok(conn)
    }

    /// A second handle that physically cannot write, for `lab sql`.
    pub fn readonly(&self) -> Result<Connection> {
        let conn = Connection::open_with_flags(
            &self.db,
            OpenFlags::SQLITE_OPEN_READ_ONLY | OpenFlags::SQLITE_OPEN_NO_MUTEX,
        )?;
        conn.busy_timeout(std::time::Duration::from_secs(30))?;
        register_udfs(&conn)?;
        Ok(conn)
    }

    // ---- reading the log into the cache ---------------------------------

    /// Read every log byte the cache has not seen. Only whole lines count: a
    /// line still being written is left for the next open.
    fn ingest(&self) -> Result<()> {
        let log = self.data.join("log");
        let mut files = Vec::new();
        for host in std::fs::read_dir(&log)? {
            let host = host?;
            if !host.file_type()?.is_dir() {
                continue;
            }
            for file in std::fs::read_dir(host.path())? {
                let path = file?.path();
                if path.extension().is_some_and(|e| e == "jsonl") {
                    files.push((host.file_name().to_string_lossy().to_string(), path));
                }
            }
        }
        files.sort();

        self.conn.execute_batch("BEGIN IMMEDIATE")?;
        let outcome = (|| -> Result<()> {
            for (host, path) in files {
                let key = format!(
                    "{host}/{}",
                    path.file_name().unwrap_or_default().to_string_lossy()
                );
                let size = std::fs::metadata(&path)?.len();
                let seen: u64 = self
                    .conn
                    .query_row("SELECT bytes FROM ingested WHERE file = ?1", [&key], |r| {
                        r.get::<_, i64>(0)
                    })
                    .map(|b| b as u64)
                    .unwrap_or(0);
                // A file that shrank was replaced; reread it, ids dedupe.
                let start = if seen > size { 0 } else { seen };
                if start == size {
                    continue;
                }
                let mut file = std::fs::File::open(&path)?;
                file.seek(SeekFrom::Start(start))?;
                let mut chunk = Vec::new();
                file.read_to_end(&mut chunk)?;
                let whole = match chunk.iter().rposition(|&b| b == b'\n') {
                    Some(end) => end + 1,
                    None => continue,
                };
                for line in String::from_utf8_lossy(&chunk[..whole]).lines() {
                    let line = line.trim();
                    if line.is_empty() {
                        continue;
                    }
                    let record: Value = serde_json::from_str(line).map_err(|e| {
                        Error::Json(format!("{key}: unreadable record: {e}"))
                    })?;
                    self.apply(&host, &record)?;
                }
                self.conn.execute(
                    "INSERT OR REPLACE INTO ingested (file, bytes) VALUES (?1, ?2)",
                    params![key, (start + whole as u64) as i64],
                )?;
            }
            Ok(())
        })();
        match outcome {
            Ok(()) => self.conn.execute_batch("COMMIT")?,
            Err(_) => self.conn.execute_batch("ROLLBACK")?,
        }
        outcome
    }

    /// Put one log record into the cache. Idempotent: every record has an id.
    fn apply(&self, host: &str, r: &Value) -> Result<()> {
        match r["t"].as_str() {
            Some("exchange") => {
                if let Some(sha) = r["body"].as_str() {
                    self.load_blob(sha)?;
                }
                self.conn.execute(
                    "INSERT OR IGNORE INTO xlog (id, host, at, ms, account, kind, method, \
                     url, req, status, headers, body_sha, error, judge_id, sub_id) \
                     VALUES (?1,?2,?3,?4,?5,?6,?7,?8,?9,?10,?11,?12,?13,?14,?15)",
                    params![
                        r["id"].as_str(),
                        host,
                        r["at"].as_f64(),
                        r["ms"].as_f64(),
                        r["account"].as_str(),
                        r["kind"].as_str(),
                        r["method"].as_str(),
                        r["url"].as_str(),
                        text(&r["req"]),
                        r["status"].as_i64(),
                        text(&r["headers"]),
                        r["body"].as_str(),
                        r["error"].as_str(),
                        r["judge_id"].as_str(),
                        r["sub"].as_str(),
                    ],
                )?;
            }
            Some("submission") => {
                let sha = r["kernel"].as_str().unwrap_or_default();
                self.load_kernel(sha)?;
                self.conn.execute(
                    "INSERT OR IGNORE INTO submission \
                     (id, host, created_at, account, judge_uid, kernel_sha, reason) \
                     VALUES (?1,?2,?3,?4,?5,?6,?7)",
                    params![
                        r["id"].as_str(),
                        host,
                        r["at"].as_f64(),
                        r["account"].as_str(),
                        r["judge_uid"].as_str(),
                        sha,
                        r["reason"].as_str(),
                    ],
                )?;
                if let Some(times) = r["local"].as_array() {
                    for (index, us) in times.iter().enumerate() {
                        self.conn.execute(
                            "INSERT OR IGNORE INTO local_time (sub_id, case_no, us) \
                             VALUES (?1,?2,?3)",
                            params![r["id"].as_str(), index as i64 + 1, us.as_f64()],
                        )?;
                    }
                }
            }
            Some("event") => {
                self.conn.execute(
                    "INSERT OR IGNORE INTO submission_event \
                     (id, sub_id, at, state, judge_id, error) VALUES (?1,?2,?3,?4,?5,?6)",
                    params![
                        r["id"].as_str(),
                        r["sub"].as_str(),
                        r["at"].as_f64(),
                        r["state"].as_str(),
                        r["judge_id"].as_str(),
                        r["error"].as_str(),
                    ],
                )?;
            }
            // A record type from a newer build: skip it, a rebuild picks it up.
            _ => {}
        }
        Ok(())
    }

    fn load_blob(&self, sha: &str) -> Result<()> {
        let path = self.data.join("body").join(format!("{sha}.json"));
        if let Ok(body) = std::fs::read_to_string(path) {
            self.conn.execute(
                "INSERT OR IGNORE INTO blob (sha, text) VALUES (?1, ?2)",
                params![sha, body],
            )?;
        }
        Ok(())
    }

    fn load_kernel(&self, sha: &str) -> Result<()> {
        let path = self.data.join("kernel").join(format!("{sha}.asc"));
        if let Ok(src) = std::fs::read_to_string(path) {
            self.conn.execute(
                "INSERT OR IGNORE INTO kernel (sha, src, bytes, lines) VALUES (?1,?2,?3,?4)",
                params![sha, src, src.len() as i64, src.lines().count() as i64],
            )?;
        }
        Ok(())
    }

    // ---- writing ---------------------------------------------------------

    pub fn new_id(&self) -> String {
        new_id(&self.host)
    }

    /// Append a record to this machine's log, then to the cache. The log is
    /// written first: it is the truth, the cache only follows it.
    pub fn append(&self, record: Value) -> Result<()> {
        let day = chrono::Utc::now().format("%Y-%m-%d").to_string();
        let path = self.data.join("log").join(&self.host).join(format!("{day}.jsonl"));
        let mut line = serde_json::to_string(&record)?;
        line.push('\n');
        let mut file = std::fs::OpenOptions::new().create(true).append(true).open(&path)?;
        file.write_all(line.as_bytes())?;
        file.flush()?;
        self.apply(&self.host, &record)
    }

    pub fn put_body(&self, body: &str) -> Result<String> {
        let sha = sha256_hex(body.as_bytes());
        write_once(&self.data.join("body").join(format!("{sha}.json")), body)?;
        self.conn.execute(
            "INSERT OR IGNORE INTO blob (sha, text) VALUES (?1, ?2)",
            params![sha, body],
        )?;
        Ok(sha)
    }

    /// Store a kernel source by content. Returns its sha; a repeat is a no-op.
    pub fn put_kernel(&self, src: &str) -> Result<String> {
        let sha = sha256_hex(src.as_bytes());
        write_once(&self.data.join("kernel").join(format!("{sha}.asc")), src)?;
        self.load_kernel(&sha)?;
        Ok(sha)
    }

    pub fn event(&self, sub_id: &str, state: &str, judge_id: Option<&str>,
                 error: Option<&str>) -> Result<()> {
        self.append(json!({
            "t": "event", "id": self.new_id(), "at": crate::now(), "sub": sub_id,
            "state": state, "judge_id": judge_id, "error": error,
        }))
    }

    pub fn kernel_src(&self, sha: &str) -> Result<Option<String>> {
        let mut stmt = self.conn.prepare("SELECT src FROM kernel WHERE sha = ?1")?;
        let mut rows = stmt.query(params![sha])?;
        Ok(match rows.next()? {
            Some(row) => Some(row.get(0)?),
            None => None,
        })
    }

    /// A submission named by a prefix of its id or by its judge id.
    pub fn resolve(&self, name: &str) -> Result<String> {
        let mut stmt = self.conn.prepare(
            "SELECT id FROM submission_row WHERE id LIKE ?1 || '%' OR judge_id = ?1 LIMIT 2",
        )?;
        let ids: Vec<String> = stmt
            .query_map([name], |row| row.get(0))?
            .collect::<std::result::Result<_, _>>()?;
        match ids.as_slice() {
            [one] => Ok(one.clone()),
            [] => Err(Error::Other(format!("no submission matches {name:?}"))),
            _ => Err(Error::Other(format!("{name:?} matches more than one submission"))),
        }
    }

    pub fn schema_text(&self) -> Result<String> {
        let mut stmt = self.conn.prepare(
            "SELECT sql FROM sqlite_master \
             WHERE sql IS NOT NULL AND name NOT LIKE 'sqlite_%' \
             ORDER BY CASE type WHEN 'table' THEN 0 WHEN 'index' THEN 1 ELSE 2 END, name",
        )?;
        let mut out = format!(
            "-- log:   {}\n-- cache: {}\n\n",
            self.data.display(),
            self.db.display()
        );
        let mut rows = stmt.query([])?;
        while let Some(row) = rows.next()? {
            let sql: String = row.get(0)?;
            out.push_str(&sql);
            out.push_str(";\n\n");
        }
        Ok(out)
    }
}

impl From<rusqlite::Error> for Error {
    fn from(error: rusqlite::Error) -> Self {
        Error::Db(error.to_string())
    }
}

/// Turn any query into JSON objects keyed by column name. `lab sql` and every
/// `--json` report come out of here, so nothing downstream parses a table.
pub fn query_json(
    conn: &Connection,
    sql: &str,
    params: &[&dyn rusqlite::ToSql],
) -> Result<Vec<Value>> {
    use rusqlite::types::ValueRef;

    let mut stmt = conn.prepare(sql)?;
    let names: Vec<String> = stmt.column_names().map(|n| n.to_string()).collect();
    let mut rows = stmt.query(params)?;
    let mut out = Vec::new();
    while let Some(row) = rows.next()? {
        let mut object = serde_json::Map::new();
        for (index, name) in names.iter().enumerate() {
            let value = match row.get_ref(index)? {
                ValueRef::Null => Value::Null,
                ValueRef::Integer(i) => json!(i),
                ValueRef::Real(f) => json!(f),
                ValueRef::Text(t) => json!(String::from_utf8_lossy(t).to_string()),
                ValueRef::Blob(b) => json!(format!("<{} bytes>", b.len())),
            };
            object.insert(name.clone(), value);
        }
        out.push(Value::Object(object));
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::{Arc, Barrier};

    fn scratch(tag: &str) -> PathBuf {
        let unique = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        std::env::temp_dir().join(format!("lab-{tag}-{}-{unique}", std::process::id()))
    }

    fn submission(store: &Store, kernel: &str) -> String {
        let sha = store.put_kernel(kernel).unwrap();
        let id = store.new_id();
        store
            .append(json!({
                "t": "submission", "id": id, "at": 1.0, "account": "a",
                "judge_uid": "u", "kernel": sha, "reason": "r",
                "local": (1..=15).map(|i| i as f64).collect::<Vec<_>>(),
            }))
            .unwrap();
        id
    }

    #[test]
    fn ids_sort_by_time_and_do_not_collide() {
        let a = new_id("h");
        std::thread::sleep(std::time::Duration::from_millis(2));
        let b = new_id("h");
        assert!(a < b);
        assert_eq!(a.len(), 20);
        let many: std::collections::HashSet<_> = (0..1000).map(|_| new_id("h")).collect();
        assert_eq!(many.len(), 1000);
    }

    #[test]
    fn the_cache_is_rebuilt_from_the_log_alone() {
        let dir = scratch("rebuild");
        let (data, db) = (dir.join("data"), dir.join("cache.db"));
        let id = {
            let store = Store::open(&data, &db, "mac").unwrap();
            let id = submission(&store, "kernel text\n");
            store.event(&id, "sent", Some("J1"), None).unwrap();
            id
        };
        std::fs::remove_file(&db).unwrap();
        let _ = std::fs::remove_file(dir.join("cache.db-wal"));
        let _ = std::fs::remove_file(dir.join("cache.db-shm"));

        let store = Store::open(&data, &db, "win").unwrap();
        let (judge_id, state, host, locals): (String, String, String, i64) = store
            .conn
            .query_row(
                "SELECT judge_id, state, host, local_cases FROM submission_full WHERE id = ?1",
                [&id],
                |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?, r.get(3)?)),
            )
            .unwrap();
        assert_eq!((judge_id.as_str(), state.as_str(), host.as_str(), locals), ("J1", "sent", "mac", 15));
        let sha: String = store
            .conn
            .query_row("SELECT kernel_sha FROM submission WHERE id = ?1", [&id], |r| r.get(0))
            .unwrap();
        assert_eq!(store.kernel_src(&sha).unwrap().as_deref(), Some("kernel text\n"));
        assert_eq!(store.resolve(&id[..14]).unwrap(), id);
        assert_eq!(store.resolve("J1").unwrap(), id);
        drop(store);
        std::fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn another_machines_lines_arrive_on_the_next_open() {
        let dir = scratch("pull");
        let (data, db) = (dir.join("data"), dir.join("cache.db"));
        let store = Store::open(&data, &db, "win").unwrap();
        assert_eq!(
            store.conn.query_row("SELECT COUNT(*) FROM submission", [], |r| r.get::<_, i64>(0)).unwrap(),
            0
        );
        drop(store);

        // What a `git pull` delivers: another host's log, plus a half-written
        // trailing line that must wait for the next open.
        let other = data.join("log").join("mac");
        std::fs::create_dir_all(&other).unwrap();
        let line = json!({"t": "submission", "id": "000000000001aaaaaaaa", "at": 2.0,
                          "account": "b", "kernel": "missing", "local": null});
        std::fs::write(other.join("2026-11-08.jsonl"), format!("{line}\n{{\"t\":\"sub")).unwrap();

        let store = Store::open(&data, &db, "win").unwrap();
        let count = |s: &Store| {
            s.conn.query_row("SELECT COUNT(*) FROM submission", [], |r| r.get::<_, i64>(0)).unwrap()
        };
        assert_eq!(count(&store), 1);
        drop(store);
        // Opening again reads nothing twice.
        let store = Store::open(&data, &db, "win").unwrap();
        assert_eq!(count(&store), 1);
        drop(store);
        std::fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn a_stale_schema_is_discarded_not_migrated() {
        let dir = scratch("version");
        let (data, db) = (dir.join("data"), dir.join("cache.db"));
        {
            let store = Store::open(&data, &db, "h").unwrap();
            submission(&store, "k");
            store.conn.pragma_update(None, "user_version", 0).unwrap();
            store.conn.execute_batch("CREATE TABLE junk (x)").unwrap();
        }
        let store = Store::open(&data, &db, "h").unwrap();
        let junk: i64 = store
            .conn
            .query_row("SELECT COUNT(*) FROM sqlite_master WHERE name='junk'", [], |r| r.get(0))
            .unwrap();
        assert_eq!(junk, 0);
        let subs: i64 =
            store.conn.query_row("SELECT COUNT(*) FROM submission", [], |r| r.get(0)).unwrap();
        assert_eq!(subs, 1);
        drop(store);
        std::fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn concurrent_open_keeps_views_and_rows() {
        let dir = scratch("concurrent");
        let (data, db) = (dir.join("data"), dir.join("cache.db"));
        {
            let store = Store::open(&data, &db, "h").unwrap();
            submission(&store, "sentinel");
        }
        let barrier = Arc::new(Barrier::new(8));
        let workers: Vec<_> = (0..8)
            .map(|_| {
                let (barrier, data, db) = (barrier.clone(), data.clone(), db.clone());
                std::thread::spawn(move || {
                    barrier.wait();
                    for _ in 0..4 {
                        let store = Store::open(&data, &db, "h").unwrap();
                        let views: u32 = store
                            .conn
                            .query_row("SELECT count(*) FROM sqlite_master WHERE type='view'", [], |r| r.get(0))
                            .unwrap();
                        assert_eq!(views, 18);
                        let subs: i64 = store
                            .conn
                            .query_row("SELECT COUNT(*) FROM submission_full", [], |r| r.get(0))
                            .unwrap();
                        assert_eq!(subs, 1);
                    }
                })
            })
            .collect();
        for worker in workers {
            worker.join().unwrap();
        }
        std::fs::remove_dir_all(dir).unwrap();
    }
}
