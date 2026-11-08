//! The failure modes that callers actually branch on. Everything else collapses
//! into a message, because there is nothing useful to do with it but print it.
use std::fmt;

#[derive(Debug)]
pub enum Error {
    /// The stored judge cookie is no longer accepted. Only a browser login fixes
    /// this, and the browser login no longer lives in this tree -- it is in git
    /// history at `tools/onboard.py`.
    Auth(String),
    /// The credential cannot be renewed from here.
    Expired(String),
    /// We are not allowed to submit right now: quota spent, or still cooling.
    Blocked(String),
    Http(String),
    Db(String),
    Json(String),
    Io(String),
    Other(String),
}

pub type Result<T> = std::result::Result<T, Error>;

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let (tag, body) = match self {
            Error::Auth(m) => ("auth", m),
            Error::Expired(m) => ("expired", m),
            Error::Blocked(m) => ("blocked", m),
            Error::Http(m) => ("http", m),
            Error::Db(m) => ("db", m),
            Error::Json(m) => ("json", m),
            Error::Io(m) => ("io", m),
            Error::Other(m) => ("error", m),
        };
        write!(f, "{tag}: {body}")
    }
}

impl std::error::Error for Error {}

impl Error {
    pub fn kind(&self) -> &'static str {
        match self {
            Error::Auth(_) => "auth",
            Error::Expired(_) => "expired",
            Error::Blocked(_) => "blocked",
            Error::Http(_) => "http",
            Error::Db(_) => "db",
            Error::Json(_) => "json",
            Error::Io(_) => "io",
            Error::Other(_) => "error",
        }
    }
}

impl From<std::io::Error> for Error {
    fn from(error: std::io::Error) -> Self {
        Error::Io(error.to_string())
    }
}

impl From<serde_json::Error> for Error {
    fn from(error: serde_json::Error) -> Self {
        Error::Json(error.to_string())
    }
}
