//! An account: `secrets/accounts/<label>.json`, holding a devenv OAuth token
//! and a judge cookie. Which account is always named on the command line;
//! there is no default to fall into by accident.
//!
//! **The credentials are committed.** The repository is private and these are
//! competition accounts; having them travel with the code to every machine is
//! worth more than keeping them out. `lab sync` carries them.
//!
//! The only upkeep is keeping the devenv access token fresh. It lasts ~10 days
//! and every refresh hands back a *new* refresh_token, so a refresh is taken
//! under a file lock and written atomically, and is synced right after: a
//! machine still holding the rotated-away refresh_token cannot refresh until
//! it pulls.
//!
//! There is no onboarding path in this tree. If the refresh token ever dies the
//! browser login has to be redone by hand; the Playwright script that used to
//! do it is in git history at `tools/onboard.py`.
use std::path::{Path, PathBuf};
use std::time::Duration;

use serde_json::{Value, json};

use crate::err::{Error, Result};
use crate::net::{Call, Wire};
use crate::store::Store;

const TOKEN_URL: &str = "https://web-api.gitcode.com/score-proxy/api/v1/oauth/token";
const REFERER: &str = "https://gitcode.com";
/// Refresh this far ahead of expiry so no job straddles the boundary.
const REFRESH_MARGIN: f64 = 36.0 * 3600.0;

/// Every label that has a credential file.
pub fn labels(dir: &Path) -> Vec<String> {
    let mut out: Vec<String> = std::fs::read_dir(dir)
        .map(|entries| {
            entries
                .filter_map(|entry| entry.ok().map(|e| e.path()))
                .filter(|path| path.extension().is_some_and(|e| e == "json"))
                .filter_map(|path| path.file_stem().map(|s| s.to_string_lossy().to_string()))
                .collect()
        })
        .unwrap_or_default();
    out.sort();
    out
}

pub struct Account {
    pub label: String,
    pub path: PathBuf,
    raw: std::cell::RefCell<Option<(f64, Value)>>,
}

impl Account {
    pub fn open(dir: &Path, label: &str) -> Result<Self> {
        let path = dir.join(format!("{label}.json"));
        if !path.is_file() {
            return Err(Error::Other(format!(
                "no account {label:?}; the accounts are: {}",
                labels(dir).join(", ")
            )));
        }
        Ok(Account { label: label.to_string(), path, raw: std::cell::RefCell::new(None) })
    }

    /// Re-read from disk when the file changed, so a refresh taken in another
    /// process (or pulled from another machine) is picked up.
    fn load(&self) -> Result<Value> {
        let mtime = self.mtime()?;
        if let Some((seen, value)) = self.raw.borrow().as_ref() {
            if *seen == mtime {
                return Ok(value.clone());
            }
        }
        let text = std::fs::read_to_string(&self.path)?;
        let value: Value = serde_json::from_str(&text)?;
        *self.raw.borrow_mut() = Some((mtime, value.clone()));
        Ok(value)
    }

    fn mtime(&self) -> Result<f64> {
        let modified = std::fs::metadata(&self.path)?.modified()?;
        Ok(modified
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs_f64())
            .unwrap_or(0.0))
    }

    fn write(&self, raw: &Value) -> Result<()> {
        let temp = self.path.with_extension("json.new");
        std::fs::write(&temp, format!("{}\n", serde_json::to_string_pretty(raw)?))?;
        std::fs::rename(&temp, &self.path)?;
        *self.raw.borrow_mut() = None;
        Ok(())
    }

    // ---- devenv OAuth --------------------------------------------------

    pub fn expires_at(&self) -> Result<f64> {
        let raw = self.load()?;
        let token = &raw["gitcode"];
        let base = match token["created_at"].as_str().and_then(parse_iso8601) {
            Some(seconds) => seconds,
            None => self.mtime()?,
        };
        Ok(base + token["expires_in"].as_f64().unwrap_or(0.0))
    }

    pub fn seconds_left(&self) -> Result<f64> {
        Ok(self.expires_at()? - crate::now())
    }

    /// A usable access token, refreshed first if it is inside the margin.
    /// Returns whether a refresh happened, so the caller knows to sync.
    pub fn access_token(&self, store: &Store, force: bool) -> Result<(String, bool)> {
        let mut refreshed = false;
        if force || self.seconds_left().unwrap_or(0.0) < REFRESH_MARGIN {
            refreshed = self.refresh(store, force)?;
        }
        let token = self.load()?["gitcode"]["access_token"]
            .as_str()
            .unwrap_or_default()
            .to_string();
        Ok((token, refreshed))
    }

    /// Rotate the devenv token. Safe to call from several processes at once.
    /// Returns false when another process had already done it.
    pub fn refresh(&self, store: &Store, force: bool) -> Result<bool> {
        let lock = std::fs::OpenOptions::new()
            .create(true)
            .truncate(false)
            .write(true)
            .open(self.path.with_extension("lock"))?;
        lock.lock()?;
        // Another process may have refreshed while we waited for the lock.
        *self.raw.borrow_mut() = None;
        if !force && self.seconds_left().unwrap_or(0.0) >= REFRESH_MARGIN {
            return Ok(false);
        }
        let mut raw = self.load()?;
        let refresh_token = raw["gitcode"]["refresh_token"]
            .as_str()
            .filter(|t| !t.is_empty())
            .ok_or_else(|| {
                Error::Expired(format!(
                    "{}: no refresh_token; the browser login has to be redone \
                     (see git history, tools/onboard.py)",
                    self.label
                ))
            })?
            .to_string();

        let wire = Wire::new(store, &self.label, Duration::from_secs(30));
        let mut last = String::new();
        for attempt in 0..3u32 {
            let call = Call::post("oauth_refresh", TOKEN_URL)
                .query("grant_type", "refresh_token")
                .query("refresh_token", &refresh_token)
                .header("Referer", REFERER)
                .redacted();
            match wire.send(call) {
                Ok(landed) => {
                    let fresh = landed.json().unwrap_or(json!({}));
                    if landed.status == 200 && fresh["access_token"].is_string() {
                        let mut token = fresh;
                        if !token["created_at"].is_string() {
                            token["created_at"] = json!(chrono::Utc::now().to_rfc3339());
                        }
                        raw["gitcode"] = token;
                        self.write(&raw)?;
                        return Ok(true);
                    }
                    last = format!(
                        "HTTP {} (exchange {}, body on disk)",
                        landed.status, landed.exchange_id
                    );
                }
                Err(error) => last = error.to_string(),
            }
            std::thread::sleep(Duration::from_secs(2 * (attempt as u64 + 1)));
        }
        let left = self.seconds_left().unwrap_or(0.0);
        let hint = "if another machine refreshed it, `lab sync` first";
        if left > 0.0 {
            return Err(Error::Other(format!(
                "{}: refresh failed ({last}); the current token still has {:.1} h; {hint}",
                self.label,
                left / 3600.0
            )));
        }
        Err(Error::Expired(format!(
            "{}: refresh failed ({last}) and the token has expired; {hint}, otherwise \
             the browser login has to be redone (see git history, tools/onboard.py)",
            self.label
        )))
    }

    // ---- judge ---------------------------------------------------------

    pub fn judge_cookie(&self) -> Result<String> {
        Ok(self.load()?["judge"]["cookie"]
            .as_str()
            .ok_or_else(|| Error::Auth(format!("{}: no judge cookie stored", self.label)))?
            .to_string())
    }

    pub fn judge_uid(&self) -> Result<String> {
        Ok(self.load()?["judge"]["uid"]
            .as_str()
            .ok_or_else(|| Error::Auth(format!("{}: no judge uid stored", self.label)))?
            .to_string())
    }

    /// What there is to know about the credential itself.
    pub fn state(&self) -> Value {
        match self.seconds_left() {
            Err(error) => json!({
                "account": self.label,
                "ok": false,
                "error": error.to_string(),
                "path": self.path.display().to_string(),
            }),
            Ok(left) => json!({
                "account": self.label,
                "ok": left > 0.0,
                "path": self.path.display().to_string(),
                "judge_uid": self.judge_uid().unwrap_or_default(),
                "token_days_left": (left / 86400.0 * 100.0).round() / 100.0,
                "token_expires_at": self.expires_at().unwrap_or(0.0),
                "refresh_margin_h": REFRESH_MARGIN / 3600.0,
            }),
        }
    }
}

/// `2026-11-01T02:19:33.123Z` and `…+08:00` both appear in these files.
fn parse_iso8601(text: &str) -> Option<f64> {
    chrono::DateTime::parse_from_rfc3339(text)
        .ok()
        .map(|stamp| stamp.timestamp() as f64 + stamp.timestamp_subsec_nanos() as f64 / 1e9)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn iso8601_with_z_and_with_offset() {
        let z = parse_iso8601("2026-11-01T00:00:00Z").unwrap();
        let offset = parse_iso8601("2026-11-01T08:00:00+08:00").unwrap();
        assert_eq!(z, offset);
    }

    #[test]
    fn junk_is_none() {
        assert!(parse_iso8601("not a date").is_none());
    }

    #[test]
    fn a_missing_account_names_the_ones_that_exist() {
        let dir = std::env::temp_dir().join(format!("lab-accounts-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("b.json"), "{}").unwrap();
        std::fs::write(dir.join("a.json"), "{}").unwrap();
        std::fs::write(dir.join("README.md"), "").unwrap();
        let error = Account::open(&dir, "c").err().unwrap().to_string();
        assert!(error.contains("a, b"), "{error}");
        assert!(Account::open(&dir, "a").is_ok());
        std::fs::remove_dir_all(dir).unwrap();
    }
}
