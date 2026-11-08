//! lab -- the lab tooling for GroupedMatmulPerTokenQuant.
//!
//! One binary. The truth is a text log under `lab/data/` that git carries
//! between machines; `lab/gmmlab.db` is a SQLite cache rebuilt from it, and
//! every number reported is a view over that cache. `lab sql` hands any of it
//! back as JSON, so nothing needs to parse what the human commands print.
//!
//!     lab quota  -a ozongzi
//!     lab submit kernel.asc -a ozongzi --local 22.7,9.06,... --reason "..."
//!     lab status
//!     lab history
//!     lab sync
//!     lab sql "select case_no, best_us from best_time order by case_no"
mod account;
mod err;
mod judge;
mod minify;
mod net;
mod root;
mod score;
mod store;
mod sync;
mod text;
mod watch;

use std::path::PathBuf;

use chrono::{TimeZone, Utc};
use clap::{Parser, Subcommand};
use serde_json::{Value, json};

use account::Account;
use err::{Error, Result};
use judge::Judge;
use root::Root;
use store::{Store, query_json};

/// Unix seconds as a float, which is what every timestamp in the store is.
pub fn now() -> f64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs_f64())
        .unwrap_or(0.0)
}

fn stamp(seconds: f64, format: &str) -> String {
    Utc.timestamp_opt(seconds as i64, 0)
        .single()
        .map(|t| t.format(format).to_string())
        .unwrap_or_else(|| "?".into())
}

use judge::BOARD_SIZE;
/// Rows the human tables show; everything is in `lab sql`.
const SHOWN: usize = 30;

#[derive(Parser)]
#[command(name = "lab", about = "Submit to CANNJudge and keep every byte it ever said.", version)]
struct Cli {
    /// Emit the structured result instead of the human table.
    #[arg(long, global = true)]
    json: bool,
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand)]
enum Cmd {
    /// Submissions left today and the cooldown, as the judge counts them.
    Quota {
        #[arg(short, long)]
        account: String,
    },
    /// The credential: what we hold and how long it lasts.
    Account {
        #[arg(short, long)]
        account: String,
        /// Rotate the devenv token now instead of waiting for the margin.
        #[arg(long)]
        refresh: bool,
        /// Print a usable devenv access token and nothing else.
        #[arg(long)]
        token: bool,
    },
    /// Send a kernel to the judge, wait out any cooldown, poll for the verdict.
    Submit {
        /// The kernel source; `-` reads stdin, so an assembler can pipe
        /// straight in without a temporary file.
        kernel: PathBuf,
        #[arg(short, long)]
        account: String,
        /// Our own device time of this kernel on every point, in us,
        /// comma-separated, point 1 first. As many as the judge has cases now.
        #[arg(long, value_delimiter = ',', required = true)]
        local: Vec<f64>,
        /// Why this goes to the judge; shows up in history.
        #[arg(long)]
        reason: String,
    },
    /// Report on a submission: an id prefix or judge id, default the latest.
    Status {
        id: Option<String>,
        /// Fetch its verdict and the board again first; never resubmits.
        #[arg(long)]
        refresh: bool,
    },
    /// Our submissions from every machine, newest first.
    History,
    /// Fetch the board, store it, and show it.
    Leaderboard {
        #[arg(short, long)]
        account: String,
    },
    /// Report other teams' newly observed board submissions, as JSON lines.
    Watch {
        #[arg(short, long)]
        account: String,
        /// Fetch once, to check credentials and setup.
        #[arg(long)]
        once: bool,
    },
    /// Backfill an account's submissions that no log holds.
    Import {
        #[arg(short, long)]
        account: String,
    },
    /// Commit this machine's records and credentials, pull, push.
    Sync,
    /// Strip comments and blank lines, exactly as submit does.
    Minify {
        /// Files to strip; `-` or nothing reads stdin.
        files: Vec<PathBuf>,
    },
    /// Print the exact text that was sent to the judge for a submission.
    Source { id: String },
    /// Print the tables and views, so the data model is discoverable.
    Schema,
    /// Run a read-only query and emit JSON rows.
    Sql { query: String },
}

fn main() -> std::process::ExitCode {
    let cli = Cli::parse();
    match run(&cli) {
        Ok(()) => std::process::ExitCode::SUCCESS,
        Err(error) => {
            if cli.json {
                println!("{}", json!({"ok": false, "kind": error.kind(), "error": error.to_string()}));
            } else {
                eprintln!("{error}");
            }
            std::process::ExitCode::FAILURE
        }
    }
}

struct Lab {
    root: Root,
    store: Store,
}

impl Lab {
    fn account(&self, label: &str) -> Result<Account> {
        Account::open(&self.root.accounts(), label)
    }

    /// Sync, but never let a sync failure hide the result of what just ran:
    /// the records are on disk either way, and the next sync carries them.
    fn sync_after(&self) {
        match sync::run(&self.root.dir, &self.store.host) {
            Ok(line) => eprintln!("{line}"),
            Err(error) => eprintln!("not synced ({error}); run `lab sync` later"),
        }
    }
}

fn run(cli: &Cli) -> Result<()> {
    // minify touches nothing else, so it needs no checkout at all.
    if let Cmd::Minify { files } = &cli.cmd {
        return cmd_minify(files);
    }
    let root = Root::find()?;
    if let Cmd::Sync = &cli.cmd {
        println!("{}", sync::run(&root.dir, &root::host())?);
        return Ok(());
    }
    let store = Store::open(&root.data(), &root.db(), &root::host())?;
    let lab = Lab { root, store };
    let store = &lab.store;

    match &cli.cmd {
        Cmd::Quota { account } => cmd_quota(&Judge::new(store, &lab.account(account)?), cli.json),
        Cmd::Account { account, refresh, token } => {
            cmd_account(&lab, &lab.account(account)?, *refresh, *token, cli.json)
        }
        Cmd::Submit { kernel, account, local, reason } => {
            cmd_submit(&lab, &lab.account(account)?, kernel, local, reason, cli.json)
        }
        Cmd::Status { id, refresh } => cmd_status(&lab, id.as_deref(), *refresh, cli.json),
        Cmd::History => cmd_history(store, cli.json),
        Cmd::Leaderboard { account } => {
            cmd_leaderboard(&Judge::new(store, &lab.account(account)?), cli.json)
        }
        Cmd::Watch { account, once } => watch::run(&Judge::new(store, &lab.account(account)?), *once),
        Cmd::Import { account } => {
            let out = Judge::new(store, &lab.account(account)?).import()?;
            if cli.json {
                println!("{out}");
            } else {
                println!(
                    "{}: imported {}, already known {}",
                    out["account"].as_str().unwrap_or("?"),
                    out["imported"],
                    out["already_known"]
                );
            }
            lab.sync_after();
            Ok(())
        }
        Cmd::Source { id } => cmd_source(store, id),
        Cmd::Schema => {
            print!("{}", store.schema_text()?);
            Ok(())
        }
        Cmd::Sql { query } => {
            let rows = query_json(&store.readonly()?, query, &[])?;
            println!("{}", serde_json::to_string_pretty(&Value::Array(rows))?);
            Ok(())
        }
        Cmd::Minify { .. } | Cmd::Sync => unreachable!(),
    }
}

// ---- commands -----------------------------------------------------------

/// A source file, or stdin when the path is `-`.
fn read_source(path: &std::path::Path) -> Result<String> {
    use std::io::Read;
    if path.as_os_str() != "-" {
        return Ok(std::fs::read_to_string(path)?);
    }
    let mut buffer = String::new();
    std::io::stdin().read_to_string(&mut buffer)?;
    Ok(buffer)
}

fn cmd_minify(files: &[PathBuf]) -> Result<()> {
    let targets: Vec<PathBuf> =
        if files.is_empty() { vec![PathBuf::from("-")] } else { files.to_vec() };
    for path in targets {
        print!("{}", minify::strip_source(&read_source(&path)?));
    }
    Ok(())
}

/// The local times, one per case the judge has now, checked before any slot
/// is claimed.
fn local_times(values: &[f64], cases: usize) -> Result<Vec<f64>> {
    if values.len() != cases {
        return Err(Error::Other(format!(
            "--local needs all {cases} points (the judge has {cases} cases now), got {}",
            values.len()
        )));
    }
    if let Some((index, bad)) =
        values.iter().enumerate().find(|(_, us)| !us.is_finite() || **us <= 0.0)
    {
        return Err(Error::Other(format!(
            "--local point {} is {bad}; a device time is a positive number of us",
            index + 1
        )));
    }
    Ok(values.to_vec())
}

fn cmd_quota(judge: &Judge, as_json: bool) -> Result<()> {
    let state = judge.quota()?;
    if as_json {
        println!("{state}");
        return Ok(());
    }
    let cooling = state["cooldown_s"].as_f64().unwrap_or(0.0);
    println!(
        "account    {}  (judge uid {})",
        state["account"].as_str().unwrap_or("?"),
        judge.account.judge_uid().unwrap_or_else(|_| "?".into())
    );
    println!(
        "today      {} used ({} on the judge, {} only here), {} left of {}; resets {} UTC",
        state["used_today"],
        state["used_on_judge"],
        state["used_only_here"],
        state["left_today"],
        judge::DAILY_QUOTA,
        stamp(state["day_resets_at"].as_f64().unwrap_or(0.0), "%Y-%m-%d %H:%M")
    );
    println!(
        "cooldown   {} ({} s between submissions)",
        if cooling > 0.0 { format!("{cooling:.0}s") } else { "ready".into() },
        judge::COOLDOWN
    );
    Ok(())
}

/// The exact bytes that went to the judge, recovered by content address.
fn cmd_source(store: &Store, name: &str) -> Result<()> {
    let id = store.resolve(name)?;
    let sha: String = store.conn.query_row(
        "SELECT kernel_sha FROM submission WHERE id = ?1",
        [&id],
        |row| row.get(0),
    )?;
    let src = store.kernel_src(&sha)?.ok_or_else(|| {
        Error::Db(format!("submission {id} references kernel {sha}, which no log holds"))
    })?;
    print!("{src}");
    Ok(())
}

fn cmd_account(lab: &Lab, one: &Account, refresh: bool, token: bool, as_json: bool) -> Result<()> {
    if token {
        let (token, refreshed) = one.access_token(&lab.store, refresh)?;
        println!("{token}");
        if refreshed {
            lab.sync_after();
        }
        return Ok(());
    }
    if refresh && one.refresh(&lab.store, true)? {
        lab.sync_after();
    }
    let state = one.state();
    if as_json {
        println!("{}", serde_json::to_string_pretty(&state)?);
    } else {
        println!(
            "account    {}  ({})",
            state["account"].as_str().unwrap_or("?"),
            state["path"].as_str().unwrap_or("?")
        );
        match state["ok"].as_bool() {
            Some(true) => println!(
                "token      {} days left, expires {} UTC",
                state["token_days_left"],
                stamp(state["token_expires_at"].as_f64().unwrap_or(0.0), "%Y-%m-%d %H:%M")
            ),
            _ => println!("token      {}", state["error"].as_str().unwrap_or("expired")),
        }
        println!("judge uid  {}", state["judge_uid"].as_str().unwrap_or("?"));
    }
    if state["ok"].as_bool() == Some(true) {
        Ok(())
    } else {
        Err(Error::Expired(
            state["error"].as_str().unwrap_or("the token has expired").to_string(),
        ))
    }
}

fn cmd_submit(
    lab: &Lab,
    one: &Account,
    kernel: &PathBuf,
    local: &[f64],
    reason: &str,
    as_json: bool,
) -> Result<()> {
    // Read the source before anything else: a pipe that produced nothing must
    // not cost a quota slot, and the judge is not asked until it has content.
    let raw = read_source(kernel)?;
    if raw.trim().is_empty() {
        return Err(Error::Other(format!(
            "{} is empty; nothing to submit",
            if kernel.as_os_str() == "-" { "stdin".into() } else { kernel.display().to_string() }
        )));
    }
    // An instrument that computes wrong answers by design must never reach the
    // judge. Its marker lives only in comments, and `strip_source` below removes
    // comments, so what crosses the wire carries no trace of it: the check has to
    // read `raw`, and has to sit here, above `Judge::new`, so a refusal costs no
    // quota slot.
    if let Some(hit) = raw.lines().find(|l| l.contains(probe_marker())) {
        return Err(Error::Other(format!(
            "{} carries {} -- it is a measurement instrument and returns wrong \
             answers by design, so it is not submittable.\n  {}\nRefused before \
             a quota slot was spent.",
            kernel.display(),
            probe_marker(),
            hit.trim()
        )));
    }
    // Spin padding and the empty-kernel shortcut are deliberate tactics, not
    // mistakes, so they are announced and allowed: refusing them would decide
    // something that belongs to whoever is running the campaign. Worth saying
    // out loud because only the *last* submission is reviewed by hand.
    for needle in ["GetSystemCycle", "EmptyKernel"] {
        if raw.contains(needle) {
            eprintln!("note: source contains {needle} -- fine for a probe submission, but only the last submission of the contest is reviewed by hand");
        }
    }
    let judge = Judge::new(&lab.store, one);
    let local = local_times(local, judge.case_count()?)?;
    // What crosses the wire carries no comments and no blank lines. Stored and
    // hashed as sent, so the history and the judge agree on one text.
    let source = minify::strip_source(&raw);
    if source != raw {
        eprintln!(
            "stripped {} lines, {} bytes",
            raw.lines().count() as i64 - source.lines().count() as i64,
            raw.len() as i64 - source.len() as i64
        );
    }
    let mut note = |line: &str| eprintln!("{line}");
    let outcome = judge.submit(&source, reason, &local, &mut note);
    // Whatever happened is in the log now; carry it before reporting.
    let board = judge.fetch_ranking(BOARD_SIZE).err();
    lab.sync_after();
    let sub_id = outcome?;
    if let Some(error) = board {
        eprintln!("(leaderboard unavailable: {error})");
    }
    emit_report(&lab.store, &sub_id, as_json)
}

fn cmd_status(lab: &Lab, name: Option<&str>, refresh: bool, as_json: bool) -> Result<()> {
    let store = &lab.store;
    let sub_id = match name {
        Some(name) => store.resolve(name)?,
        None => store
            .conn
            .query_row(
                if refresh {
                    "SELECT id FROM submission_full WHERE judge_id IS NOT NULL \
                     ORDER BY created_at DESC LIMIT 1"
                } else {
                    "SELECT id FROM submission_full WHERE judge_status IS NOT NULL \
                     ORDER BY created_at DESC LIMIT 1"
                },
                [],
                |row| row.get::<_, String>(0),
            )
            .map_err(|_| Error::Other("no judged submissions in the log yet".into()))?,
    };
    if refresh {
        let (judge_id, label): (Option<String>, String) = store.conn.query_row(
            "SELECT judge_id, account FROM submission_row WHERE id = ?1",
            [&sub_id],
            |row| Ok((row.get(0)?, row.get(1)?)),
        )?;
        let judge_id = judge_id.ok_or_else(|| {
            Error::Other(format!("submission {sub_id} never reached the judge"))
        })?;
        // The account that sent it is the one that may read it back.
        let one = lab.account(&label)?;
        let judge = Judge::new(store, &one);
        judge.fetch_submission(&judge_id, Some(&sub_id))?;
        if let Err(error) = judge.fetch_ranking(BOARD_SIZE) {
            eprintln!("(leaderboard unavailable: {error})");
        }
    }
    emit_report(store, &sub_id, as_json)
}

/// A null ratio means unbounded -- the case scored zero, so "how many times off
/// the best" has no answer. Printing `inf` would read like a measurement.
fn ratio(value: &Value) -> String {
    match value.as_f64() {
        Some(r) if r.is_finite() => format!("{r:.2}"),
        _ => "-".to_string(),
    }
}

fn num(value: &Value) -> String {
    value.as_f64().map(|v| format!("{v:.2}")).unwrap_or_else(|| "-".into())
}

fn emit_report(store: &Store, sub_id: &str, as_json: bool) -> Result<()> {
    let out = judge::report(store, sub_id)?;
    if as_json {
        println!("{}", serde_json::to_string_pretty(&out)?);
        return Ok(());
    }
    let submission = &out["submission"];
    if out["board_row"].is_object() {
        println!(
            "team {} rank {} score {}{}",
            judge::TEAM_NAME,
            out["board_row"]["rank"],
            out["board_row"]["score"],
            if out["board_row_is_this_submission"].as_bool() == Some(true) {
                ""
            } else {
                "  (board shows a different submission)"
            }
        );
    }
    println!(
        "{}  {}  state {}  judge {}  judge_id {}",
        submission["id"].as_str().unwrap_or("?"),
        submission["account"].as_str().unwrap_or("?"),
        submission["state"].as_str().unwrap_or("?"),
        submission["judge_status"].as_str().unwrap_or("-"),
        submission["judge_id"].as_str().unwrap_or("-")
    );
    let analysis = &out["analysis"];
    if !analysis.is_object() {
        return Ok(());
    }
    println!(
        "{:<6}{:>10}{:>10}{:>10}{:>8}{:>8}",
        "case", "local us", "judge us", "best us", "score", "x best"
    );
    for case in analysis["cases"].as_array().cloned().unwrap_or_default() {
        println!(
            "tp{:<4}{:>10}{:>10}{:>10}{:>8}{:>7}x",
            case["case_no"].as_i64().unwrap_or(0),
            num(&case["local_us"]),
            num(&case["time_us"]),
            num(&case["best_us"]),
            if case["time_us"].as_f64().is_some_and(|t| t > 0.0) { num(&case["score"]) } else { "-".into() },
            if case["time_us"].as_f64().is_some_and(|t| t > 0.0) { ratio(&case["ratio"]) } else { "-".into() },
        );
    }
    println!("{:<44}{:>8.2}", "visible mean", analysis["visible_mean"].as_f64().unwrap_or(0.0));
    if let Some(hidden) = analysis["hidden_mean"].as_f64() {
        let label = format!("hidden {} (from total)", analysis["hidden_cases"]);
        println!("{label:<44}{hidden:>8.2}{:>7}x", ratio(&analysis["hidden_ratio"]));
    }
    Ok(())
}

fn cmd_history(store: &Store, as_json: bool) -> Result<()> {
    let rows = query_json(
        &store.conn,
        "SELECT * FROM submission_full ORDER BY created_at DESC LIMIT ?1",
        &[&(SHOWN as i64)],
    )?;
    if as_json {
        println!("{}", serde_json::to_string_pretty(&Value::Array(rows))?);
        return Ok(());
    }
    // The judge's verdicts run to "Time Limit Exceeded", so the columns are
    // padded by display width rather than trusting the values to be short.
    println!(
        "{}  {:<16}{}{}{}{:>7}{:>6}  {}",
        text::pad("id", 20),
        "when (UTC)",
        text::pad("account", 16),
        text::pad("state", 10),
        text::pad("judge", 21),
        "visible",
        "local",
        "reason"
    );
    for row in rows {
        println!(
            "{}  {:<16}{}{}{}{:>7}{:>6}  {}",
            row["id"].as_str().unwrap_or("?"),
            stamp(row["created_at"].as_f64().unwrap_or(0.0), "%m-%d %H:%M:%S"),
            text::pad(row["account"].as_str().unwrap_or("?"), 16),
            text::pad(row["state"].as_str().unwrap_or("?"), 10),
            text::pad(row["judge_status"].as_str().unwrap_or("-"), 21),
            num(&row["visible_mean"]),
            match row["local_cases"].as_i64() {
                Some(n) if n > 0 => n.to_string(),
                _ => "-".into(),
            },
            row["reason"].as_str().unwrap_or("")
        );
    }
    Ok(())
}

fn cmd_leaderboard(judge: &Judge, as_json: bool) -> Result<()> {
    judge.fetch_ranking(BOARD_SIZE)?;
    let conn = &judge.store.conn;
    let uid = judge.account.judge_uid().unwrap_or_default();
    let joined = query_json(conn, "SELECT * FROM board_submission ORDER BY rank", &[])?;
    let cases = query_json(
        conn,
        "SELECT * FROM board_case WHERE time_us IS NOT NULL AND rank <= ?1 ORDER BY rank, case_no",
        &[&(SHOWN as i64)],
    )?;
    if as_json {
        println!(
            "{}",
            serde_json::to_string_pretty(&json!({
                "rows": joined, "cases": cases, "account_uid": uid,
            }))?
        );
        return Ok(());
    }
    println!(
        "{:>5}  {}{:>8}  {:<10} {:<24} {}",
        "rank", text::pad("team", 24), "score", "status", "submission_id", "submitter / our record"
    );
    for row in &joined {
        let ours = row["sub_id"].as_str().is_some();
        let record = match row["sub_id"].as_str() {
            Some(id) => format!(
                "{}  [our {id} {}]",
                row["submitter"].as_str().unwrap_or("?"),
                row["sub_account"].as_str().unwrap_or("?")
            ),
            None => row["submitter"].as_str().unwrap_or("?").to_string(),
        };
        println!(
            "{:>5}  {}{:>8}  {:<10} {:<24} {}{}",
            row["rank"].as_i64().unwrap_or(0),
            text::pad(row["team"].as_str().unwrap_or("?"), 24),
            row["score"],
            row["status"].as_str().unwrap_or("?"),
            row["submission_id"].as_str().unwrap_or("-"),
            record,
            if ours { "  <-- us" } else { "" }
        );
    }
    if !cases.is_empty() {
        println!();
        println!("逐点时间（判题器只公开可见 5 点，隐藏 10 点所有队伍都是 null）");
        println!(
            "{:>5}  {}{:>4} {:>10} {:>10} {:>8}",
            "rank", text::pad("team", 24), "点", "time_us", "best_us", "x best"
        );
        for c in &cases {
            let t = c["time_us"].as_f64().unwrap_or(0.0);
            let b = c["tbest_us"].as_f64().unwrap_or(0.0);
            println!(
                "{:>5}  {}{:>4} {:>10.2} {:>10.2} {:>7}x",
                c["rank"].as_i64().unwrap_or(0),
                text::pad(c["team"].as_str().unwrap_or("?"), 24),
                c["case_no"].as_i64().unwrap_or(0),
                t,
                b,
                if b > 0.0 { format!("{:.2}", t / b) } else { "-".into() }
            );
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::local_times;

    #[test]
    fn local_times_must_cover_every_case_with_positive_numbers() {
        let good: Vec<f64> = (1..=15).map(|i| i as f64 * 1.5).collect();
        assert_eq!(local_times(&good, 15).unwrap(), good);
        assert!(local_times(&good[..14], 15).unwrap_err().to_string().contains("got 14"));
        // The judge grew to 20 cases: 15 is no longer enough.
        assert!(local_times(&good, 20).unwrap_err().to_string().contains("all 20"));
        let twenty: Vec<f64> = (1..=20).map(|i| i as f64).collect();
        assert!(local_times(&twenty, 20).is_ok());
        let mut bad = good.clone();
        bad[8] = 0.0;
        assert!(local_times(&bad, 15).unwrap_err().to_string().contains("point 9"));
        bad[8] = f64::NAN;
        assert!(local_times(&bad, 15).is_err());
    }
}

/// The banner the generator stamps on deliberately-wrong probe kernels.
/// Duplicated from `gen::probe::MARKER` on purpose: `lab` does not depend on
/// `gen`, and a submission guard that could be disabled by a missing crate
/// would be no guard at all.
fn probe_marker() -> &'static str {
    "GMM_WRONG_ON_PURPOSE"
}
