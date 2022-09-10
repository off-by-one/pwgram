use std::collections::HashSet;

pub(crate) struct PwTokens<'a> {
    source: &'a str,
    ntokens: HashSet<String>,
}

impl<'a> PwTokens<'a> {
    pub(crate) fn new(source: &'a str, ntokens: HashSet<String>) -> Self {
        PwTokens { source, ntokens }
    }

    pub(crate) fn iter(&'a self) -> PwTokenIter<'a> {
        PwTokenIter { pt: self, idx: 0 }
    }
}

pub(crate) struct PwTokenIter<'a> {
    pt: &'a PwTokens<'a>,
    idx: usize,
}

impl<'a> Iterator for PwTokenIter<'a> {
    type Item = &'a str;

    fn next(&mut self) -> Option<Self::Item> {
        for token in &self.pt.ntokens {
            if (self.pt.source[self.idx..]).starts_with(token.as_str()) {
                self.idx += token.len();
                return Some(token.as_str());
            }
        }
        self.idx += 1;
        self.pt.source.get((self.idx - 1)..self.idx)
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
