//! `lab sync`: carry the log and the credentials between machines with git.
//!
//! Commit only `lab/data` and `secrets/accounts`, rebase onto whatever the
//! other machines pushed, push. The log never conflicts -- each machine appends
//! only to its own directory and everything else is content-addressed -- so a
//! rebase over another machine's sync always applies.
//!
//! Anything else in the working tree is left exactly as it was: the commit is
//! restricted to those two paths, and `--autostash` puts unrelated edits back.
use std::path::Path;
use std::process::Command;

use crate::err::{Error, Result};

const PATHS: [&str; 2] = ["lab/data", "secrets/accounts"];

fn git(root: &Path, args: &[&str]) -> Result<(bool, String)> {
    let out = Command::new("git")
        .arg("-C")
        .arg(root)
        .args(args)
        .output()
        .map_err(|e| Error::Other(format!("cannot run git: {e}")))?;
    let mut text = String::from_utf8_lossy(&out.stdout).to_string();
    text.push_str(&String::from_utf8_lossy(&out.stderr));
    Ok((out.status.success(), text.trim().to_string()))
}

fn must(root: &Path, args: &[&str]) -> Result<String> {
    let (ok, text) = git(root, args)?;
    if ok {
        Ok(text)
    } else {
        Err(Error::Other(format!("git {}: {text}", args.join(" "))))
    }
}

/// Returns a one-line summary of what moved.
pub fn run(root: &Path, host: &str) -> Result<String> {
    let mut add = vec!["add", "--"];
    add.extend(PATHS);
    must(root, &add)?;

    let mut staged = vec!["diff", "--cached", "--quiet", "--"];
    staged.extend(PATHS);
    let (clean, _) = git(root, &staged)?;
    let committed = if clean {
        false
    } else {
        let message = format!("lab: sync from {host}");
        let mut commit = vec!["commit", "-m", &message, "--"];
        commit.extend(PATHS);
        must(root, &commit)?;
        true
    };

    must(root, &["pull", "--rebase", "--autostash"])?;
    if !committed {
        return Ok("synced: nothing new here, pulled".to_string());
    }
    // This pushes the current branch, so any other local commits on it go too.
    must(root, &["push"])?;
    Ok(format!("synced: committed this machine's records as {host}, pulled, pushed"))
}
