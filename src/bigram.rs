use ngrams::Ngram;
use rand::thread_rng;
use rand::Rng;
use std::collections::BTreeMap;
use std::collections::HashMap;
use std::collections::HashSet;

pub(crate) type BigramToken<'a> = &'a str;

/// Current state of a bigram model - either a token, or None at the start of
/// generation.
pub(crate) type BigramState<'a> = Option<BigramToken<'a>>;

/// A single bigram transition
pub(crate) type BigramTransitionMap<'a> = BTreeMap<u8, BigramToken<'a>>;
pub(crate) type BigramModelMap<'a> = HashMap<BigramState<'a>, BigramTransitionMap<'a>>;

pub(crate) trait BigramTransition<T, U> {
    fn rand(&self) -> U;
    fn from_aggregate(transitions: HashMap<U, u64>) -> Self;
}

pub(crate) trait BigramModel {
    fn gen_iter(&self) -> BigramGeneratorIter;
}

impl<T: Clone> BigramTransition<u8, T> for BTreeMap<u8, T> {
    fn rand(&self) -> T {
        let rand: u8 = thread_rng().gen();
        self.range(rand..).next().map(|(_, s)| s).unwrap().clone()
    }

    fn from_aggregate(transitions: HashMap<T, u64>) -> Self {
        let sum: u64 = transitions.iter().map(|(_, n)| n).sum();
        let mut transitions: Vec<(T, u64)> = transitions.into_iter().collect();
        transitions.sort_by_key(|(_, n)| *n);
        transitions
            .into_iter()
            .fold(
                (BTreeMap::<u8, T>::new(), 0u64),
                |(mut map, cdf), (next, count)| {
                    let ncdf = cdf + count;
                    let key = (ncdf * (u8::MAX as u64)) / sum;
                    map.insert(key as u8, next);
                    (map, ncdf)
                },
            )
            .0
    }
}

pub struct BigramGeneratorIter<'a> {
    model: &'a BigramModelMap<'a>,
    state: BigramState<'a>,
}

impl<'a> Iterator for BigramGeneratorIter<'a> {
    type Item = BigramToken<'a>;

    fn next(&mut self) -> Option<Self::Item> {
        self.state = Some(self.model.get(&self.state)?.rand());
        self.state.as_ref().cloned()
    }
}

impl<'a> BigramModel for HashMap<BigramState<'a>, BigramTransitionMap<'a>> {
    fn gen_iter(&self) -> BigramGeneratorIter {
        BigramGeneratorIter {
            model: self,
            state: None,
        }
    }
}

pub(crate) fn train<'a>(
    corpus: impl Iterator<Item = &'a str>,
    delims: &HashSet<&str>,
) -> BigramModelMap<'a> {
    let mut bigrams = corpus.ngrams(2).peekable();

    let mut transitions: HashMap<BigramState, HashMap<BigramToken, u64>> = HashMap::new();

    if let Some(bigram) = bigrams.peek() {
        let fst = bigram[0];

        *transitions.entry(None).or_default().entry(fst).or_default() += 1;
    }

    for bigram in bigrams {
        let (curr, next) = (bigram[0], bigram[1]);

        if delims.contains(next) {
            continue;
        }

        *transitions
            .entry(Some(curr))
            .or_default()
            .entry(next)
            .or_default() += 1
    }

    transitions
        .into_iter()
        .map(|(c, t)| (c, BigramTransitionMap::from_aggregate(t)))
        .collect()
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
