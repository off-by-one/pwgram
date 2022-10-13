use crate::bigram::*;
use std::collections::HashMap;
use std::collections::HashSet;
use std::ops::Not;

/// Train a bigram model from a corpus of tokens.
///
/// Each chunk of delims is ignored, and the following character is treated as
/// an initial character.
///
/// An empty corpus creates an empty BigramModel.
pub fn train<'a>(mut corpus: impl Iterator<Item = &'a str>, delims: &HashSet<&str>) -> BigramModel {
    let mut transitions: HashMap<Option<&str>, HashMap<&str, u64>> = HashMap::new();

    let mut curr = if let Some(curr) = corpus.next() {
        *transitions
            .entry(None)
            .or_default()
            .entry(curr)
            .or_default() += 1;

        curr
    } else {
        return HashMap::<BigramState, BigramTransition>::new().into();
    };

    for next in corpus {
        if !delims.contains(next) {
            *transitions
                .entry(delims.contains(curr).not().then_some(curr))
                .or_default()
                .entry(next)
                .or_default() += 1;
        }

        curr = next;
    }

    transitions
        .into_iter()
        .map(|(c, t)| {
            (
                c.map(|s| s.to_string()).into(),
                from_aggregate(t.into_iter().map(|(s, v)| (s.to_string(), v))).into(),
            )
        })
        .collect::<HashMap<BigramState, BigramTransition>>()
        .into()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn it_works() {
        let corpus = ["a", "b", "a", "b", "c"].to_vec();
        let model = train(corpus.iter().copied(), &HashSet::default());

        let mut pw = String::new();
        for token in model.gen_iter() {
            pw.push_str(&token);
        }

        println!("{}", pw);
    }
}
