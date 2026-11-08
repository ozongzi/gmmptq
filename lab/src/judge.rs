//! The judge: an account's submissions, its daily quota and its cooldown.
//!
//! The judge allows 50 submissions a day per account and refuses one inside
//! 180 s of the last, and the day ends at midnight UTC+8 (16:00 UTC). Both
//! limits are checked here before anything is sent, against the judge's own
//! list of the account's submissions -- several machines share an account --
//! plus whatever this machine has claimed that the judge does not list yet.
//!
//! The leaderboard ranks a team by its *latest* submission, not its best, so a
//! broken build sits on the board until something better replaces it.
//!
//! Every report here is a query over the views in `store.rs`, not a decoding
//! of a payload held in memory: the payload went to the log the moment it
//! arrived, and reading it back is the same path whether it landed ten seconds
//! or ten days ago, on this machine or another.
use std::collections::HashSet;
use std::time::Duration;

use serde_json::{Value, json};

use crate::account::Account;
use crate::err::{Error, Result};
use crate::net::{Call, Wire};
use crate::score;
use crate::store::{Store, query_json};

pub const BASE: &str = "https://cannjudge.cn";
pub const PROBLEM_ID: &str = "6ae3fbe81dac5d235b154efa";
/// Our team on the leaderboard. A row is ours by team, not by submitter: any
/// member's submission can be the one the board is showing.
pub const TEAM_ID: &str = "6ae462911dac5d235b71c792";
pub const TEAM_NAME: &str = "你说的队";
pub const COOLDOWN: f64 = 180.0;
pub const DAILY_QUOTA: i64 = 50;
const POLL_TIMEOUT: f64 = 900.0;
/// Board rows fetched with every snapshot.
pub const BOARD_SIZE: usize = 80;
/// Well over a day's quota, so one page reaches past today.
const IMPORT_LIMIT: i64 = 80;
/// The judge's day ends at midnight UTC+8, which is 16:00 UTC.
const DAY_OFFSET: f64 = 8.0 * 3600.0;

pub fn day_start(now: f64) -> f64 {
    let shifted = now + DAY_OFFSET;
    (shifted / 86400.0).floor() * 86400.0 - DAY_OFFSET
}

pub enum Slot {
    /// Our submission id. The slot is ours.
    Granted(String),
    /// Seconds still to wait.
    Cooling(f64),
}

pub struct Judge<'a> {
    pub store: &'a Store,
    pub account: &'a Account,
    wire: Wire<'a>,
}

impl<'a> Judge<'a> {
    pub fn new(store: &'a Store, account: &'a Account) -> Self {
        Judge {
            store,
            account,
            wire: Wire::new(store, account.label.clone(), Duration::from_secs(60)),
        }
    }

    fn headers(&self) -> Result<Vec<(&'static str, String)>> {
        Ok(vec![
            ("Cookie", self.account.judge_cookie()?),
            ("Accept", "application/json".to_string()),
            ("Origin", BASE.to_string()),
        ])
    }

    fn send(&self, mut call: Call) -> Result<Value> {
        for (key, value) in self.headers()? {
            call = call.header(key, value);
        }
        let landed = self.wire.send(call)?;
        if landed.status == 401 || landed.status == 403 {
            return Err(Error::Auth(format!(
                "the judge rejected our cookie (HTTP {}, exchange {}); only a \
                 browser login fixes this -- see git history, tools/onboard.py",
                landed.status, landed.exchange_id
            )));
        }
        if !landed.content_type.contains("json") {
            return Err(Error::Auth(format!(
                "the judge answered with {:?} instead of JSON (exchange {}), which \
                 it does when the session has expired; the page is on disk",
                landed.content_type, landed.exchange_id
            )));
        }
        if landed.status != 200 {
            return Err(Error::Http(format!(
                "HTTP {} (exchange {}, body on disk)",
                landed.status, landed.exchange_id
            )));
        }
        landed.json()
    }

    // ---- quota and cooldown ---------------------------------------------

    /// Everything that decides whether this account may submit right now.
    ///
    /// The judge's own list of this account's submissions is the authority:
    /// several machines submit as the same account, and no single machine's
    /// log has all of them until it pulls. Local rows the judge does not list
    /// yet -- a slot just claimed, a POST that failed -- count on top, so the
    /// answer errs towards waiting.
    pub fn quota(&self) -> Result<Value> {
        let now = crate::now();
        let since = day_start(now);
        let listing = self.fetch_user_submissions(60)?;
        let mut remote_ids = HashSet::new();
        let mut remote_today = 0i64;
        let mut last: Option<f64> = None;
        for row in listing["list"].as_array().cloned().unwrap_or_default() {
            if row["problem"]["_id"].as_str() != Some(PROBLEM_ID) {
                continue;
            }
            let Some(created) = row["create_time"].as_str().and_then(parse_time) else {
                continue;
            };
            if let Some(id) = row["_id"].as_str() {
                remote_ids.insert(id.to_string());
            }
            if created >= since {
                remote_today += 1;
            }
            last = Some(last.map_or(created, |l: f64| l.max(created)));
        }
        let mut local_today = 0i64;
        {
            let mut stmt = self.store.conn.prepare(
                "SELECT judge_id, created_at FROM submission_row                  WHERE account = ?1 AND created_at >= ?2",
            )?;
            let rows = stmt.query_map(rusqlite::params![self.account.label, since], |row| {
                Ok((row.get::<_, Option<String>>(0)?, row.get::<_, f64>(1)?))
            })?;
            for row in rows {
                let (judge_id, _) = row?;
                if judge_id.is_none_or(|id| !remote_ids.contains(&id)) {
                    local_today += 1;
                }
            }
        }
        let local_last = self.local_last()?;
        let last = match (last, local_last) {
            (Some(a), Some(b)) => Some(a.max(b)),
            (a, b) => a.or(b),
        };
        let used = remote_today + local_today;
        let cooling = last.map(|l| (COOLDOWN - (now - l)).max(0.0)).unwrap_or(0.0);
        let left = DAILY_QUOTA - used;
        Ok(json!({
            "account": self.account.label,
            "used_today": used,
            "used_on_judge": remote_today,
            "used_only_here": local_today,
            "left_today": left,
            "quota": DAILY_QUOTA,
            "cooldown_s": cooling,
            "last_submit_at": last,
            "day_resets_at": since + 86400.0,
            "ready": left > 0 && cooling == 0.0,
        }))
    }

    fn local_last(&self) -> Result<Option<f64>> {
        Ok(self.store.conn.query_row(
            "SELECT MAX(created_at) FROM submission WHERE account = ?1",
            rusqlite::params![self.account.label],
            |row| row.get(0),
        )?)
    }

    /// Claim the submission slot. The judge is asked first; then, inside one
    /// transaction, the local cooldown is checked again and the slot written,
    /// so two processes on this machine cannot both believe they had it.
    fn reserve(&self, kernel_sha: &str, reason: &str, local: &[f64]) -> Result<Slot> {
        let state = self.quota()?;
        if state["left_today"].as_i64().unwrap_or(0) <= 0 {
            let left = state["day_resets_at"].as_f64().unwrap_or(0.0) - crate::now();
            return Err(Error::Blocked(format!(
                "{} has used all {DAILY_QUOTA} submissions for today; the allowance                  comes back in {:.1} h",
                self.account.label,
                left / 3600.0
            )));
        }
        let cooling = state["cooldown_s"].as_f64().unwrap_or(0.0);
        if cooling > 0.0 {
            return Ok(Slot::Cooling(cooling));
        }
        let uid = self.account.judge_uid()?;
        self.store.conn.execute_batch("BEGIN IMMEDIATE")?;
        let outcome = (|| -> Result<Slot> {
            if let Some(last) = self.local_last()? {
                let cooling = COOLDOWN - (crate::now() - last);
                if cooling > 0.0 {
                    return Ok(Slot::Cooling(cooling));
                }
            }
            let id = self.store.new_id();
            self.store.append(json!({
                "t": "submission", "id": id, "at": crate::now(),
                "account": self.account.label, "judge_uid": uid,
                "kernel": kernel_sha, "reason": reason, "local": local,
            }))?;
            Ok(Slot::Granted(id))
        })();
        match &outcome {
            Ok(Slot::Granted(_)) => self.store.conn.execute_batch("COMMIT")?,
            _ => self.store.conn.execute_batch("ROLLBACK")?,
        }
        outcome
    }

    // ---- the judge's HTTP surface ---------------------------------------

    fn post_submission(&self, source: &str, sub_id: &str, sha: &str) -> Result<String> {
        let uid = self.account.judge_uid()?;
        let payload = json!({
            "problemId": PROBLEM_ID,
            "files": [{"path": "kernel.asc", "content": source}],
            "tiling_h": "", "tiling_key_h": "", "host_cpp": "", "kernel_cpp": "",
            "userId": uid,
        });
        // The log keeps the shape of what we sent but not a second copy of the
        // source: it is already stored once, addressed by this sha.
        let logged = json!({
            "problemId": PROBLEM_ID,
            "files": [{"path": "kernel.asc", "kernel_sha": sha, "bytes": source.len()}],
            "userId": uid,
        });
        let body = self.send(
            Call::post("submit", format!("{BASE}/api/submissions/submit"))
                .body(payload)
                .log_body(logged)
                .for_submission(sub_id),
        )?;
        body["data"]["submissionId"]
            .as_str()
            .map(|id| id.to_string())
            .ok_or_else(|| {
                Error::Other(format!(
                    "submit rejected: {}",
                    serde_json::to_string(&body).unwrap_or_default()
                ))
            })
    }

    pub fn fetch_submission(&self, judge_id: &str, sub_id: Option<&str>) -> Result<Value> {
        let mut call = Call::get("submission", format!("{BASE}/api/submissions/{judge_id}"))
            .about(judge_id);
        if let Some(id) = sub_id {
            call = call.for_submission(id);
        }
        self.send(call)
    }

    pub fn fetch_ranking(&self, size: usize) -> Result<Value> {
        self.send(
            Call::get("ranking", format!("{BASE}/api/problems/{PROBLEM_ID}/ranking"))
                .query("page", 1)
                .query("size", size),
        )
    }

    /// How many test cases the problem has right now, as the judge says it.
    /// Not a constant: the problem gains cases near the end.
    pub fn case_count(&self) -> Result<usize> {
        let board = self.fetch_ranking(BOARD_SIZE)?;
        board["total_testcase_count"]
            .as_u64()
            .or_else(|| board["testcases"].as_array().map(|cases| cases.len() as u64))
            .filter(|&n| n > 0)
            .map(|n| n as usize)
            .ok_or_else(|| {
                Error::Json("the ranking reply does not say how many cases there are".into())
            })
    }

    fn fetch_user_submissions(&self, limit: i64) -> Result<Value> {
        let uid = self.account.judge_uid()?;
        self.send(
            Call::get("user_submissions", format!("{BASE}/api/submissions/user/{uid}"))
                .query("withCount", 1)
                .query("limit", limit)
                .query("skip", 0)
                .query("order", "desc"),
        )
    }

    // ---- submitting ------------------------------------------------------

    /// Claim a slot (waiting out any cooldown), send, and poll until the first
    /// case has a time. Returns our submission id.
    pub fn submit(
        &self,
        source: &str,
        reason: &str,
        local: &[f64],
        note: &mut dyn FnMut(&str),
    ) -> Result<String> {
        let sha = self.store.put_kernel(source)?;
        let sub_id = loop {
            match self.reserve(&sha, reason, local)? {
                Slot::Granted(id) => break id,
                Slot::Cooling(seconds) => {
                    note(&format!("cooling down, {seconds:.0}s to go"));
                    std::thread::sleep(Duration::from_secs_f64(seconds + 2.0));
                }
            }
        };

        let judge_id = match self.post_submission(source, &sub_id, &sha) {
            Ok(id) => id,
            Err(error) => {
                self.store.event(&sub_id, "failed", None, Some(&error.to_string()))?;
                return Err(error);
            }
        };
        self.store.event(&sub_id, "sent", Some(&judge_id), None)?;
        note(&format!("submitted {sub_id} as {} -> {judge_id}", self.account.label));

        let deadline = crate::now() + POLL_TIMEOUT;
        while crate::now() < deadline {
            std::thread::sleep(Duration::from_secs(12));
            let detail = self.fetch_submission(&judge_id, Some(&sub_id))?;
            let first_ran = detail["result"]
                .get(0)
                .map(|case| !case["time"].is_null())
                .unwrap_or(false);
            if first_ran {
                return Ok(sub_id);
            }
            note(&format!("judging… {}", detail["status"].as_str().unwrap_or("?")));
        }
        self.store.event(&sub_id, "timeout", None, None)?;
        Err(Error::Other(format!(
            "poll timed out; the result will still land, re-check with \
             `lab status {sub_id} --refresh`"
        )))
    }

    /// Backfill this account's submissions that no log of ours holds. They
    /// carry no local timing: that was never measured, so none is invented.
    pub fn import(&self) -> Result<Value> {
        let listing = self.fetch_user_submissions(IMPORT_LIMIT)?;
        let uid = self.account.judge_uid()?;
        let mut added = 0usize;
        let mut skipped = 0usize;
        for row in listing["list"].as_array().cloned().unwrap_or_default() {
            if row["problem"]["_id"].as_str() != Some(PROBLEM_ID) {
                continue;
            }
            let Some(judge_id) = row["_id"].as_str().map(str::to_string) else {
                continue;
            };
            let known: i64 = self.store.conn.query_row(
                "SELECT COUNT(*) FROM submission_row WHERE judge_id = ?1",
                rusqlite::params![judge_id],
                |r| r.get(0),
            )?;
            if known > 0 {
                skipped += 1;
                continue;
            }
            let sub_id = self.store.new_id();
            // Fetching the detail is what puts the results into the log.
            let detail = self.fetch_submission(&judge_id, Some(&sub_id))?;
            let source = detail["files"]
                .as_array()
                .and_then(|files| {
                    files.iter().find(|file| file["path"].as_str() == Some("kernel.asc"))
                })
                .and_then(|file| file["content"].as_str())
                .unwrap_or("")
                .to_string();
            let sha = self.store.put_kernel(&source)?;
            let created = row["create_time"]
                .as_str()
                .and_then(parse_time)
                .unwrap_or_else(crate::now);
            self.store.append(json!({
                "t": "submission", "id": sub_id, "at": created,
                "account": self.account.label, "judge_uid": uid,
                "kernel": sha, "reason": "imported", "local": null,
            }))?;
            self.store.event(&sub_id, "imported", Some(&judge_id), None)?;
            added += 1;
        }
        Ok(json!({"account": self.account.label, "imported": added, "already_known": skipped}))
    }
}

fn parse_time(text: &str) -> Option<f64> {
    chrono::DateTime::parse_from_rfc3339(text)
        .ok()
        .map(|stamp| stamp.timestamp_millis() as f64 / 1000.0)
}

// ---- reading it back ----------------------------------------------------
//
// Reports read only the store: no network, no account. Whoever wants a fresh
// board or verdict fetches it first, and the fetch lands in the log.

/// The board row belonging to the judge user that made this submission.
/// Resolved by uid only: no row is better than someone else's row.
fn board_row_of(store: &Store, uid: &str) -> Result<Option<Value>> {
    if uid.is_empty() {
        return Ok(None);
    }
    let rows = query_json(
        &store.conn,
        "SELECT * FROM board WHERE user_id = ?1 OR submitter_id = ?1 LIMIT 1",
        &[&uid],
    )?;
    Ok(rows.into_iter().next())
}

/// Split a result into the visible cases and what the total implies for the
/// hidden ten -- the free feedback signal every submission carries.
fn analyse(
    store: &Store,
    sub_id: &str,
    judge_id: &str,
    total_score: Option<f64>,
    total_belongs_here: bool,
) -> Result<Value> {
    let cases = query_json(
        &store.conn,
        "SELECT c.case_no, c.time_us, c.best_us, c.score, c.ratio, c.case_status, \
                l.us AS local_us \
         FROM case_result c \
         LEFT JOIN local_time l ON l.sub_id = ?2 AND l.case_no = c.case_no \
         WHERE c.judge_id = ?1 ORDER BY c.case_no",
        &[&judge_id, &sub_id],
    )?;
    let visible = query_json(
        &store.conn,
        "SELECT case_no, time_us, best_us, score, ratio \
         FROM visible WHERE judge_id = ?1 ORDER BY case_no",
        &[&judge_id],
    )?;
    let scores: Vec<f64> =
        visible.iter().map(|row| row["score"].as_f64().unwrap_or(0.0)).collect();
    let mean = if scores.is_empty() {
        0.0
    } else {
        scores.iter().sum::<f64>() / scores.len() as f64
    };
    let mut out = json!({
        "judge_id": judge_id,
        "cases": cases,
        "visible": visible,
        "visible_mean": mean,
        "visible_cases": scores.len(),
        "case_count": cases.len(),
    });
    // The hidden ten can only be inferred when the total belongs to THIS
    // submission. The board shows one row per team, often a different
    // submission; subtracting our visible scores from someone else's total
    // produces a number with no information in it.
    match (total_score, total_belongs_here) {
        (Some(total), true) => {
            if let Some(hidden) = score::hidden_mean(&scores, cases.len(), total) {
                out["total_score"] = json!(total);
                out["hidden_mean"] = json!(hidden);
                out["hidden_cases"] = json!(cases.len() - scores.len());
                let ratio = score::ratio_from_score(hidden);
                out["hidden_ratio"] = if ratio.is_finite() { json!(ratio) } else { Value::Null };
            }
        }
        (Some(total), false) => {
            out["total_score_of_board_row"] = json!(total);
            out["hidden_mean"] = Value::Null;
            out["hidden_unavailable"] = json!(
                "the board row was earned by a different submission, so its total \
                 says nothing about this one's hidden cases"
            );
        }
        (None, _) => {
            out["hidden_mean"] = Value::Null;
            out["hidden_unavailable"] =
                json!("no board row for this account, so there is no total to work from");
        }
    }
    Ok(out)
}

/// Report on one submission against the newest board snapshot we hold.
pub fn report(store: &Store, sub_id: &str) -> Result<Value> {
    let row = query_json(&store.conn, "SELECT * FROM submission_full WHERE id = ?1", &[&sub_id])?
        .into_iter()
        .next()
        .ok_or_else(|| Error::Other(format!("no submission {sub_id}")))?;
    let judge_id = row["judge_id"].as_str().unwrap_or_default().to_string();
    let uid = row["judge_uid"].as_str().unwrap_or_default().to_string();
    let mine = board_row_of(store, &uid)?;
    let total = mine.as_ref().and_then(|r| r["score"].as_f64());
    let same =
        mine.as_ref().and_then(|r| r["submission_id"].as_str()) == Some(judge_id.as_str());
    let mut out = json!({
        "submission": row,
        "board_row": mine.unwrap_or(Value::Null),
        "board_row_is_this_submission": same,
    });
    if !judge_id.is_empty() {
        out["analysis"] = analyse(store, sub_id, &judge_id, total, same)?;
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::day_start;

    #[test]
    fn the_day_turns_at_1600_utc() {
        // 2026-11-01 15:59:59Z is still 2026-11-01 in UTC+8 terms... and
        // 16:00:00Z is the next judging day.
        let before = 1_762_012_799.0; // 2026-11-01T15:59:59Z
        let after = 1_762_012_800.0; // 2026-11-01T16:00:00Z
        assert!(day_start(before) < day_start(after));
        assert_eq!(day_start(after) - day_start(before), 86400.0);
        // A day start is always 16:00 UTC, i.e. 57600 past a UTC midnight.
        assert_eq!(day_start(after).rem_euclid(86400.0), 57600.0);
    }

    #[test]
    fn a_whole_day_shares_one_start() {
        let start = day_start(1_762_012_800.0);
        for step in [1.0, 3600.0, 50_000.0, 86_399.0] {
            assert_eq!(day_start(1_762_012_800.0 + step), start, "step {step}");
        }
    }
}
