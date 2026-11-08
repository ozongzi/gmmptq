//! Where everything lives. There is exactly one rule and no override:
//!
//! the project root is the nearest ancestor of the running executable that
//! holds `secrets/accounts/`. `cargo build` puts the binary under
//! `lab/target/<profile>/`, so a build always finds the checkout it came from;
//! a deployed copy finds whatever directory it was installed under, as long as
//! that directory has the same layout.
//!
//! No environment variable, no flag, no fallback to the home directory. The
//! store used to default to `~/gmmlab/gmmlab.db`, outside the repository, and
//! a whole history was lost when the repository moved without it. If the root
//! cannot be found, nothing is created anywhere: the command fails.
use std::path::PathBuf;

use crate::err::{Error, Result};

pub struct Root {
    pub dir: PathBuf,
}

impl Root {
    pub fn find() -> Result<Root> {
        let exe = std::env::current_exe()?;
        for dir in exe.ancestors().skip(1) {
            if dir.join("secrets").join("accounts").is_dir() {
                return Ok(Root { dir: dir.to_path_buf() });
            }
        }
        Err(Error::Other(format!(
            "{} is not inside a project checkout: no ancestor holds secrets/accounts/",
            exe.display()
        )))
    }

    pub fn accounts(&self) -> PathBuf {
        self.dir.join("secrets").join("accounts")
    }

    /// The synced, append-only source of truth. Text, so git can carry it.
    pub fn data(&self) -> PathBuf {
        self.dir.join("lab").join("data")
    }

    /// The local cache rebuilt from `data`. Never committed.
    pub fn db(&self) -> PathBuf {
        self.dir.join("lab").join("gmmlab.db")
    }
}

/// This machine's name, used to give each machine its own log directory so
/// that two machines never append to the same file and a pull never conflicts.
pub fn host() -> String {
    let raw = std::env::var("COMPUTERNAME")
        .ok()
        .or_else(|| std::fs::read_to_string("/etc/hostname").ok())
        .or_else(|| {
            std::process::Command::new("hostname")
                .output()
                .ok()
                .map(|out| String::from_utf8_lossy(&out.stdout).to_string())
        })
        .unwrap_or_default();
    sanitize(&raw)
}

fn sanitize(raw: &str) -> String {
    let name = raw.trim().trim_end_matches(".local").to_lowercase();
    let clean: String = name
        .chars()
        .map(|c| if c.is_ascii_alphanumeric() || c == '-' || c == '_' { c } else { '-' })
        .collect();
    if clean.is_empty() { "unknown".into() } else { clean }
}

#[cfg(test)]
mod tests {
    use super::sanitize;

    #[test]
    fn host_names_become_directory_names() {
        assert_eq!(sanitize("DESKTOP-68DHSTO\r\n"), "desktop-68dhsto");
        assert_eq!(sanitize("Moyus-MacBook-Pro.local\n"), "moyus-macbook-pro");
        assert_eq!(sanitize("a b/c"), "a-b-c");
        assert_eq!(sanitize("  "), "unknown");
    }
}
