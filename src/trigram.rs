/// This file is not used. It took too much training data to make it generate a
/// high entropy password. However, it was similar quality to the bigram model
/// without manually entering a list of digraphs, so I might try to improve it.
///
use ngrams::Ngram;
use rand::thread_rng;
use rand::Rng;
use std::borrow::Cow;
use std::collections::BTreeMap;
use std::collections::HashMap;
use std::collections::HashSet;

type TrigramToken = Cow<'static, String>;
type TrigramState = (Option<TrigramToken>, Option<TrigramToken>);

pub struct TrigramTransition {
    pub(crate) transitions: BTreeMap<u8, TrigramToken>,
    pub(crate) entropy: f64,
}

impl TrigramTransition {
    fn rand(&self) -> TrigramToken {
        let rand: u8 = thread_rng().gen();

        self.transitions
            .range(rand..)
            .next()
            .map(|(_, s)| s)
            .unwrap()
            .clone()
    }
}

impl<'a> From<HashMap<TrigramToken, u64>> for TrigramTransition {
    fn from(transitions: HashMap<TrigramToken, u64>) -> Self {
        let sum: u64 = transitions.iter().map(|(_, n)| n).sum();
        let mut transitions: Vec<(TrigramToken, u64)> = transitions.into_iter().collect();
        transitions.sort_by_key(|(_, n)| *n);
        let transitions = transitions
            .into_iter()
            .fold(
                (BTreeMap::<u8, TrigramToken>::new(), 0u64),
                |(mut map, cdf), (next, count)| {
                    let ncdf = cdf + count;
                    let key = (ncdf * (u8::MAX as u64)) / sum;
                    map.insert(key as u8, next);
                    (map, ncdf)
                },
            )
            .0;

        let entropy = transitions
            .iter()
            .fold((0.0f64, 0u8), |(entropy, pcdf), (&cdf, _)| {
                let p = ((cdf - pcdf) as f64) / u8::MAX as f64;
                let mut e = -p * p.log2();
                e = if e.is_normal() { e } else { 0.0f64 };
                (entropy + e, cdf)
            })
            .0;

        TrigramTransition {
            transitions,
            entropy,
        }
    }
}

pub struct TrigramModel {
    pub(crate) tokens: HashSet<TrigramToken>,

    pub(crate) transitions: HashMap<TrigramState, TrigramTransition>,
}

pub struct TrigramGeneratorIter<'a> {
    pub(crate) model: &'a TrigramModel,

    pub(crate) state: TrigramState,
}

impl<'a> Iterator for TrigramGeneratorIter<'a> {
    type Item = TrigramToken;

    fn next(&mut self) -> Option<Self::Item> {
        let transition = self.model.transitions.get(&self.state)?;

        let next = transition.rand();
        self.state = (self.state.1.clone(), Some(next.clone()));

        Some(next)
    }
}

impl TrigramModel {
    pub fn train<'a, 'b>(
        corpus: impl Iterator<Item = &'a str>,
        delims: &HashSet<&str>,
    ) -> TrigramModel {
        let mut trigrams = corpus.ngrams(3).peekable();

        let mut tokens = HashMap::<&str, TrigramToken>::new();
        let mut transitions = HashMap::<TrigramState, HashMap<TrigramToken, u64>>::new();

        if let Some(trigram) = trigrams.peek() {
            let (fst, snd) = (trigram[0], trigram[1]);
            let fst = tokens
                .entry(fst)
                .or_insert_with_key(|s| Cow::Owned(s.to_string()))
                .clone();
            let snd = tokens
                .entry(snd)
                .or_insert_with_key(|s| Cow::Owned(s.to_string()))
                .clone();

            *transitions
                .entry((None, None))
                .or_default()
                .entry(fst.clone())
                .or_default() += 1;

            *transitions
                .entry((None, Some(fst)))
                .or_default()
                .entry(snd)
                .or_default() += 1;
        }

        for trigram in trigrams {
            let next = trigram[2];

            if delims.contains(next) {
                continue;
            }

            let next: TrigramToken = tokens
                .entry(next)
                .or_insert_with_key(|s| Cow::Owned(s.to_string()))
                .clone();

            let (prev, curr) = (trigram[0], trigram[1]);

            let (prev, curr): TrigramState = match (delims.contains(prev), delims.contains(curr)) {
                (_, true) => (None, None),
                (true, false) => (
                    None,
                    Some(
                        tokens
                            .entry(curr)
                            .or_insert_with_key(|s| Cow::Owned(s.to_string()))
                            .clone(),
                    ),
                ),
                (false, false) => (
                    Some(
                        tokens
                            .entry(prev)
                            .or_insert_with_key(|s| Cow::Owned(s.to_string()))
                            .clone(),
                    ),
                    Some(
                        tokens
                            .entry(curr)
                            .or_insert_with_key(|s| Cow::Owned(s.to_string()))
                            .clone(),
                    ),
                ),
            };

            *transitions
                .entry((prev, curr))
                .or_default()
                .entry(next)
                .or_default() += 1
        }

        let transitions: HashMap<TrigramState, TrigramTransition> = transitions
            .into_iter()
            .map(|(c, t)| (c, TrigramTransition::from(t)))
            .collect();

        TrigramModel {
            tokens: tokens.into_values().into_iter().collect(),
            transitions,
        }
    }

    pub fn gen_iter(&self) -> TrigramGeneratorIter {
        TrigramGeneratorIter {
            model: &self,
            state: (None, None),
        }
    }
}

fn main() {
    let training_path = Path::new(TRAINING_DATA);

    let mut training_file = match File::open(&training_path) {
        Err(why) => panic!("couldn't open {}: {}", TRAINING_DATA, why),
        Ok(file) => file,
    };

    // Read the file contents into a string, returns `io::Result<usize>`
    let mut s = String::new();
    match training_file.read_to_string(&mut s) {
        Err(why) => panic!("couldn't read {}: {}", TRAINING_DATA, why),
        Ok(_) => (), //print!("read {}\n", TRAINING_DATA),
    }

    let tokens: HashSet<String> = HashSet::new();
    let tokens = tokenize::PwTokens::new(s.as_str(), tokens);

    let mut delims: HashSet<&str> = HashSet::new();
    delims.insert("\n");
    let model = TrigramModel::train(tokens.iter(), &delims);

    let mut entropy = 0.0f64;
    let mut pw = String::new();

    loop {
        for token in model.gen_iter() {
            pw.push_str(&token);
            entropy += 1.0;
            if entropy >= MAX_ENTROPY {
                break;
            }
        }
        if entropy >= MAX_ENTROPY {
            break;
        }
    }

    println!("{}\n", pw);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn it_works() {
        let corpus = ["a", "b", "a", "b", "c"].to_vec();
        let model = TrigramModel::train(corpus.iter().copied(), &HashSet::default());

        let mut target = 10usize;
        let mut pw = String::new();
        for token in model.gen_iter() {
            pw.push_str(&token);
            target -= 1;
            if target == 0 {
                break;
            }
        }
        println!("{}", pw);
    }
}
