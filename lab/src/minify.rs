//! Strip comments and blank lines from a submission on the way to the judge.
//!
//! The repo keeps readable sources; only what crosses the wire is stripped, so
//! this sits in the submit path rather than in the generators. It is a
//! character scanner, not a regex: `#include "acl/acl.h"` and `a / b` must both
//! survive, so string, char and comment states are tracked explicitly. Line
//! structure is preserved -- a preprocessor directive joined to the next line
//! would stop compiling -- and a block comment spanning lines leaves its
//! newlines behind.
//!
//! Stripping an already-stripped source is a no-op, so applying it twice is
//! safe.

#[derive(Clone, Copy, PartialEq)]
enum State {
    Code,
    /// Inside a "…" or '…' literal; carries the delimiter that closes it.
    Str(char),
    LineComment,
    BlockComment,
}

pub fn strip_source(text: &str) -> String {
    let src: Vec<char> = text.chars().collect();
    let mut out = String::with_capacity(text.len());
    let mut state = State::Code;
    let mut i = 0usize;
    while i < src.len() {
        let c = src[i];
        let next = src.get(i + 1).copied();
        match state {
            State::Code => {
                if c == '/' && next == Some('/') {
                    state = State::LineComment;
                    i += 2;
                    continue;
                }
                if c == '/' && next == Some('*') {
                    state = State::BlockComment;
                    i += 2;
                    continue;
                }
                if c == '"' || c == '\'' {
                    state = State::Str(c);
                }
                out.push(c);
            }
            State::Str(delim) => {
                out.push(c);
                if c == '\\' {
                    // An escape takes the next character whole, so a \" or a
                    // trailing \\ cannot be mistaken for the closing quote.
                    if let Some(escaped) = next {
                        out.push(escaped);
                        i += 2;
                        continue;
                    }
                } else if c == delim {
                    state = State::Code;
                }
            }
            State::LineComment => {
                if c == '\n' {
                    state = State::Code;
                    out.push(c);
                }
            }
            State::BlockComment => {
                if c == '\n' {
                    out.push(c); // keep the line structure
                } else if c == '*' && next == Some('/') {
                    state = State::Code;
                    i += 2;
                    continue;
                }
            }
        }
        i += 1;
    }

    let mut kept: Vec<&str> = Vec::new();
    for line in out.split('\n') {
        let line = line.strip_suffix('\r').unwrap_or(line).trim_end();
        if !line.is_empty() {
            kept.push(line);
        }
    }
    let mut result = kept.join("\n");
    result.push('\n');
    result
}

#[cfg(test)]
mod tests {
    use super::strip_source;

    #[test]
    fn division_survives() {
        assert_eq!(strip_source("int x = a / b;"), "int x = a / b;\n");
    }

    #[test]
    fn include_survives() {
        let src = "#include \"acl/acl.h\"\n";
        assert_eq!(strip_source(src), src);
    }

    #[test]
    fn line_comment_goes_but_the_line_break_stays() {
        assert_eq!(strip_source("int a; // note\nint b;\n"), "int a;\nint b;\n");
    }

    #[test]
    fn block_comment_leaves_its_newlines() {
        // Three lines in, the middle one becomes blank and is dropped, but the
        // directive on line 3 must not get glued to line 1.
        let got = strip_source("#define A 1\n/* one\n   two */\n#define B 2\n");
        assert_eq!(got, "#define A 1\n#define B 2\n");
    }

    #[test]
    fn a_comment_opener_inside_a_string_is_text() {
        let src = "const char *s = \"// not a comment /* either\";\n";
        assert_eq!(strip_source(src), src);
    }

    #[test]
    fn escaped_quote_does_not_close_the_string() {
        let src = "const char *s = \"a\\\"// b\";\n";
        assert_eq!(strip_source(src), src);
    }

    #[test]
    fn char_literal_of_a_quote() {
        let src = "char q = '\\'';\nint n = 1;\n";
        assert_eq!(strip_source(src), src);
    }

    #[test]
    fn idempotent() {
        let src = "/* head */\nint a; // x\n\n\nint b;\n";
        let once = strip_source(src);
        assert_eq!(strip_source(&once), once);
        assert_eq!(once, "int a;\nint b;\n");
    }

    #[test]
    fn trailing_whitespace_goes() {
        assert_eq!(strip_source("int a;   \t\n"), "int a;\n");
    }

    #[test]
    fn empty_input() {
        assert_eq!(strip_source(""), "\n");
    }
}
