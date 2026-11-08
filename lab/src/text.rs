//! Display-width padding. The leaderboard is full of CJK team names, and a
//! wide character takes two terminal columns, so `format!("{:<28}")` -- which
//! counts chars -- tears the table apart.
use unicode_width::UnicodeWidthChar;

pub fn width(text: &str) -> usize {
    text.chars().map(|c| c.width().unwrap_or(0)).sum()
}

/// Truncate to `columns` display columns, then pad out to exactly that.
pub fn pad(text: &str, columns: usize) -> String {
    let shown = width(text);
    if shown <= columns {
        return format!("{text}{}", " ".repeat(columns - shown));
    }
    let mut kept = String::new();
    let mut used = 0usize;
    for c in text.chars() {
        let w = c.width().unwrap_or(0);
        if used + w > columns {
            break;
        }
        kept.push(c);
        used += w;
    }
    kept.push_str(&" ".repeat(columns - used));
    kept
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cjk_counts_double() {
        assert_eq!(width("你说的队"), 8);
        assert_eq!(width("abc"), 3);
    }

    #[test]
    fn pad_reaches_exactly_the_column_count() {
        assert_eq!(width(&pad("你说的队", 28)), 28);
        assert_eq!(width(&pad("abc", 10)), 10);
    }

    #[test]
    fn pad_truncates_without_splitting_a_wide_char() {
        // 5 columns cannot hold a third wide character, so it stops at two and
        // pads with one space rather than emitting half a glyph.
        let got = pad("你说的队", 5);
        assert_eq!(got, "你说 ");
        assert_eq!(width(&got), 5);
    }
}
