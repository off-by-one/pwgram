/// Tokenize the input corpus.
///
/// Tokens come in minimal units of graphemes, under the assumption that
/// (diacritic -> character) is not a meaningful transition in most scripts.
use std::collections::HashSet;
use unicode_segmentation::{GraphemeIndices, UnicodeSegmentation};

pub struct PwTokens<'a> {
    source: &'a str,
    ntokens: HashSet<String>,
}

impl<'a> PwTokens<'a> {
    pub fn new(source: &'a str, ntokens: HashSet<String>) -> Self {
        PwTokens { source, ntokens }
    }

    pub fn iter(&'a self) -> PwTokenIter<'a> {
        PwTokenIter {
            pt: self,
            idxs: self.source.grapheme_indices(true),
        }
    }
}

pub struct PwTokenIter<'a> {
    pt: &'a PwTokens<'a>,
    idxs: GraphemeIndices<'a>,
}

impl<'a> Iterator for PwTokenIter<'a> {
    type Item = &'a str;

    fn next(&mut self) -> Option<Self::Item> {
        let (idx, grph) = self.idxs.next()?;
        for token in &self.pt.ntokens {
            if (self.pt.source[idx..]).starts_with(token.as_str()) {
                let mut len = grph.len();
                while len < token.len() {
                    len += self.idxs.next()?.1.len();
                }
                return Some(token.as_str());
            }
        }
        self.pt.source.get(idx..(idx + grph.len()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn it_works() {
        let mut tokens: HashSet<String> = HashSet::new();
        tokens.insert("th".to_string());
        tokens.insert("qu".to_string());
        let text = "that quack";

        let pw_tokens = PwTokens::new(text, tokens);
        let tokenized: Vec<&str> = pw_tokens.iter().collect();

        assert!(tokenized == ["th", "a", "t", " ", "qu", "a", "c", "k"]);
    }
}
