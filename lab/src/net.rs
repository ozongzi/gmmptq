//! The wire. Every request to anything upstream goes through `Wire::send`, and
//! `Wire::send` writes the exchange to the store before it returns -- including
//! transport failures, 401s, and the HTML login page the judge serves when a
//! session has expired. That is the whole point of this module: there is no
//! path that talks to the judge without leaving a record.
//!
//! What is deliberately *not* recorded: the request's own headers, because the
//! Cookie among them is the credential and it never changes anyway. Token
//! values in an OAuth reply are reduced to a fingerprint, and a kernel source
//! in a submit body is reduced to its sha, because the source itself is already
//! stored once in `kernel`.
use std::time::{Duration, Instant};

use serde_json::{Value, json};
use ureq::Agent;

use crate::err::{Error, Result};
use crate::store::{Store, sha256_hex};

#[derive(Clone, Copy, PartialEq)]
pub enum Method {
    Get,
    Post,
}

impl Method {
    fn as_str(self) -> &'static str {
        match self {
            Method::Get => "GET",
            Method::Post => "POST",
        }
    }
}

pub struct Call {
    pub kind: &'static str,
    pub method: Method,
    pub url: String,
    pub query: Vec<(String, String)>,
    pub headers: Vec<(&'static str, String)>,
    pub body: Option<Value>,
    /// What to write to the log in place of `body`, when the body carries
    /// something that should not be duplicated or should not be stored at all.
    pub log_body: Option<Value>,
    pub judge_id: Option<String>,
    pub sub_id: Option<String>,
    /// Reduce `access_token` / `refresh_token` in the reply to a fingerprint.
    pub redact: bool,
}

impl Call {
    pub fn get(kind: &'static str, url: impl Into<String>) -> Self {
        Call {
            kind,
            method: Method::Get,
            url: url.into(),
            query: Vec::new(),
            headers: Vec::new(),
            body: None,
            log_body: None,
            judge_id: None,
            sub_id: None,
            redact: false,
        }
    }

    pub fn post(kind: &'static str, url: impl Into<String>) -> Self {
        Call { method: Method::Post, ..Call::get(kind, url) }
    }

    pub fn query(mut self, key: &str, value: impl ToString) -> Self {
        self.query.push((key.to_string(), value.to_string()));
        self
    }

    pub fn header(mut self, key: &'static str, value: impl Into<String>) -> Self {
        self.headers.push((key, value.into()));
        self
    }

    pub fn body(mut self, body: Value) -> Self {
        self.body = Some(body);
        self
    }

    pub fn log_body(mut self, body: Value) -> Self {
        self.log_body = Some(body);
        self
    }

    pub fn about(mut self, judge_id: impl Into<String>) -> Self {
        self.judge_id = Some(judge_id.into());
        self
    }

    pub fn for_submission(mut self, sub_id: &str) -> Self {
        self.sub_id = Some(sub_id.to_string());
        self
    }

    pub fn redacted(mut self) -> Self {
        self.redact = true;
        self
    }

    /// The exact URL that goes on the wire, which is also the URL we log.
    fn full_url(&self) -> String {
        if self.query.is_empty() {
            return self.url.clone();
        }
        let mut out = self.url.clone();
        out.push('?');
        for (i, (key, value)) in self.query.iter().enumerate() {
            if i > 0 {
                out.push('&');
            }
            out.push_str(&encode(key));
            out.push('=');
            out.push_str(&encode(value));
        }
        out
    }
}

fn encode(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    for byte in text.bytes() {
        match byte {
            b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'-' | b'_' | b'.' | b'~' => {
                out.push(byte as char)
            }
            _ => out.push_str(&format!("%{byte:02X}")),
        }
    }
    out
}

/// Replace secret values with something that identifies them without being
/// usable. Walks the parsed JSON, so there is no pattern matching on text.
pub fn redact(value: &Value) -> Value {
    const SECRET: [&str; 3] = ["access_token", "refresh_token", "cookie"];
    match value {
        Value::Object(map) => Value::Object(
            map.iter()
                .map(|(key, inner)| {
                    let hidden = SECRET.contains(&key.as_str());
                    match (hidden, inner.as_str()) {
                        (true, Some(text)) => (
                            key.clone(),
                            json!(format!(
                                "<{} chars, sha {}>",
                                text.len(),
                                &sha256_hex(text.as_bytes())[..8]
                            )),
                        ),
                        _ => (key.clone(), redact(inner)),
                    }
                })
                .collect(),
        ),
        Value::Array(items) => Value::Array(items.iter().map(redact).collect()),
        other => other.clone(),
    }
}

pub struct Landed {
    pub exchange_id: String,
    pub status: u16,
    pub content_type: String,
    pub body: String,
}

impl Landed {
    pub fn json(&self) -> Result<Value> {
        serde_json::from_str(&self.body).map_err(|error| {
            Error::Json(format!(
                "exchange {}: {error}; the body is on disk, \
                 `lab sql \"select body from exchange where id='{}'\"`",
                self.exchange_id, self.exchange_id
            ))
        })
    }
}

pub struct Wire<'s> {
    agent: Agent,
    store: &'s Store,
    account: String,
}

impl<'s> Wire<'s> {
    pub fn new(store: &'s Store, account: impl Into<String>, timeout: Duration) -> Self {
        let config = Agent::config_builder()
            .timeout_global(Some(timeout))
            // Without this a 401 arrives as an Err and the body -- which is the
            // only thing that explains the 401 -- never reaches the log.
            .http_status_as_error(false)
            .build();
        Wire { agent: config.into(), store, account: account.into() }
    }

    pub fn send(&self, call: Call) -> Result<Landed> {
        let url = call.full_url();
        let started = Instant::now();
        let at = crate::now();

        let sent = {
            let mut outcome = match call.method {
                Method::Get => {
                    let mut request = self.agent.get(&url);
                    for (key, value) in &call.headers {
                        request = request.header(*key, value);
                    }
                    request.call()
                }
                Method::Post => {
                    let mut request = self.agent.post(&url);
                    for (key, value) in &call.headers {
                        request = request.header(*key, value);
                    }
                    match &call.body {
                        Some(body) => request.send_json(body),
                        None => request.send_empty(),
                    }
                }
            };
            // Read the body here, inside the same step, so a body that fails to
            // read is still an outcome we can record.
            match outcome.as_mut() {
                Ok(response) => {
                    let status = response.status().as_u16();
                    let headers: Value = Value::Object(
                        response
                            .headers()
                            .iter()
                            .map(|(name, value)| {
                                (
                                    name.as_str().to_string(),
                                    json!(value.to_str().unwrap_or("<not utf-8>")),
                                )
                            })
                            .collect(),
                    );
                    let content_type = response
                        .headers()
                        .get("content-type")
                        .and_then(|v| v.to_str().ok())
                        .unwrap_or("")
                        .to_string();
                    let body = response
                        .body_mut()
                        .with_config()
                        .limit(64 * 1024 * 1024)
                        .read_to_string();
                    match body {
                        Ok(text) => Ok((status, headers, content_type, text)),
                        Err(error) => Err(format!("read body: {error}")),
                    }
                }
                Err(error) => Err(error.to_string()),
            }
        };

        let ms = started.elapsed().as_secs_f64() * 1000.0;
        let logged_request = call.log_body.as_ref().or(call.body.as_ref());

        match sent {
            Ok((status, headers, content_type, body)) => {
                let stored = if call.redact {
                    serde_json::from_str::<Value>(&body)
                        .map(|parsed| {
                            serde_json::to_string(&redact(&parsed)).unwrap_or_default()
                        })
                        .unwrap_or(body.clone())
                } else {
                    body.clone()
                };
                let exchange_id = self.record(
                    at,
                    ms,
                    &call,
                    &url,
                    logged_request,
                    Some(status),
                    Some(headers),
                    Some(&stored),
                    None,
                )?;
                Ok(Landed { exchange_id, status, content_type, body })
            }
            Err(error) => {
                let exchange_id = self.record(
                    at,
                    ms,
                    &call,
                    &url,
                    logged_request,
                    None,
                    None,
                    None,
                    Some(&error),
                )?;
                Err(Error::Http(format!("{} {url}: {error} (exchange {exchange_id})",
                                        call.method.as_str())))
            }
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn record(
        &self,
        at: f64,
        ms: f64,
        call: &Call,
        url: &str,
        req: Option<&Value>,
        status: Option<u16>,
        headers: Option<Value>,
        body: Option<&str>,
        error: Option<&str>,
    ) -> Result<String> {
        let body_sha = match body {
            Some(text) => Some(self.store.put_body(text)?),
            None => None,
        };
        let id = self.store.new_id();
        self.store.append(json!({
            "t": "exchange", "id": id, "at": at, "ms": ms, "account": self.account,
            "kind": call.kind, "method": call.method.as_str(), "url": url, "req": req,
            "status": status, "headers": headers, "body": body_sha, "error": error,
            "judge_id": call.judge_id, "sub": call.sub_id,
        }))?;
        Ok(id)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn query_is_encoded_into_the_logged_url() {
        let call = Call::get("ranking", "https://x/y").query("page", 1).query("q", "a b&c");
        assert_eq!(call.full_url(), "https://x/y?page=1&q=a%20b%26c");
    }

    #[test]
    fn redaction_keeps_the_shape_and_loses_the_secret() {
        let before = json!({
            "access_token": "abcdefghij",
            "expires_in": 418705,
            "nested": {"refresh_token": "zzz"},
            "list": [{"access_token": "q"}]
        });
        let after = redact(&before);
        assert_eq!(after["expires_in"], json!(418705));
        let shown = after["access_token"].as_str().unwrap();
        assert!(shown.starts_with("<10 chars, sha "), "{shown}");
        assert!(!shown.contains("abcdefghij"));
        assert!(after["nested"]["refresh_token"].as_str().unwrap().contains("3 chars"));
        assert!(after["list"][0]["access_token"].as_str().unwrap().contains("1 chars"));
    }
}
