//! Polling observes the submission currently on the board, not a submission feed.
use std::{collections::HashSet, io::Write, time::Duration};

use serde_json::{Value, json};

use crate::{
    err::{Error, Result},
    judge::{Judge, TEAM_ID},
};

fn submissions(payload: &Value) -> Result<Vec<Value>> {
    let rows = payload
        .get("rows")
        .and_then(Value::as_array)
        .ok_or_else(|| Error::Json("ranking response has no rows array".into()))?;
    // Reject the whole response before advancing the baseline.
    for row in rows {
        if row["submission_id"].as_str().is_none() || row["team"]["_id"].as_str().is_none() {
            return Err(Error::Json(
                "ranking row lacks submission_id or team._id".into(),
            ));
        }
    }
    Ok(rows
        .iter()
        .filter(|r| r["team"]["_id"] != TEAM_ID)
        .cloned()
        .collect())
}

fn emit(event: Value) -> Result<()> {
    let mut out = std::io::stdout().lock();
    writeln!(out, "{event}")?;
    out.flush()?;
    Ok(())
}

/// Seconds between polls, before backoff.
const INTERVAL: u64 = 60;
/// Board rows asked for per poll. The server may return fewer.
const SIZE: usize = 1000;

pub fn run(judge: &Judge<'_>, once: bool) -> Result<()> {
    let (interval, size) = (INTERVAL, SIZE);
    // Reconstruct deduplication from raw historical responses; no second state file.
    // Only accept snapshots this monitor could have consumed successfully.
    let mut seen = HashSet::<String>::new();
    let mut initialized = false;
    {
        let mut stmt = judge
            .store
            .conn
            .prepare("SELECT body FROM exchange WHERE kind='ranking' AND status=200 ORDER BY id")?;
        let bodies = stmt.query_map([], |r| r.get::<_, String>(0))?;
        for body in bodies {
            if let Ok(payload) = serde_json::from_str::<Value>(&body?) {
                if let Ok(rows) = submissions(&payload) {
                    initialized = true;
                    for row in rows {
                        seen.insert(row["submission_id"].as_str().unwrap().to_owned());
                    }
                }
            }
        }
    }
    emit(json!({"event":"watch_started", "at":crate::now(), "interval":interval, "size":size}))?;
    let mut failures: u32 = 0;
    loop {
        let result = judge.fetch_ranking(size).and_then(|p| submissions(&p));
        match result {
            Ok(rows) => {
                failures = 0;
                let baseline = !initialized;
                for row in &rows {
                    let id = row["submission_id"].as_str().unwrap();
                    if !seen.contains(id) {
                        if !baseline {
                            emit(json!({"event":"new_submission", "at":crate::now(), "row":row}))?;
                        }
                        seen.insert(id.to_owned());
                    }
                }
                initialized = true;
                emit(json!({"event":if baseline {"baseline"} else {"poll_ok"},
                    "at":crate::now(), "other_teams":rows.len()}))?;
            }
            Err(error) => {
                failures = failures.saturating_add(1);
                emit(json!({"event":"poll_error", "at":crate::now(),
                    "kind":error.kind(), "error":error.to_string(), "failures":failures}))?;
                if once {
                    return Err(error);
                }
            }
        }
        if once {
            return Ok(());
        }
        let delay = interval
            .saturating_mul(1u64 << failures.min(4))
            .min(interval.max(900));
        std::thread::sleep(Duration::from_secs(delay));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn validates_snapshot_before_filtering_our_team() {
        assert!(submissions(&json!({"error":"expired"})).is_err());
        assert!(submissions(&json!({"rows":[{"submission_id":"x"}]})).is_err());
        let rows = submissions(&json!({"rows":[
            {"submission_id":"ours", "team":{"_id":TEAM_ID}},
            {"submission_id":"theirs", "team":{"_id":"other"}}
        ]}))
        .unwrap();
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0]["submission_id"], "theirs");
        assert!(submissions(&json!({"rows":[]})).unwrap().is_empty());
    }
}
