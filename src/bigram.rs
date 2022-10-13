/// Core data model for working with bigrams. Invariants are not enforced in the
/// type system because everything is type aliased, but if you use these
/// functions to create and access bigrams they will be respected.
///
/// The core model is BTreeMaps from u8 to tokens. The 256-value space of a u8
/// is the underlying sample space, and each entry (u8, token) covers the
/// interval from this entry until the next one in the map with the token.
///
/// As a result, a random sample can be taken by generating a random u8, and
/// finding the largest key less than or equal to it.
use rand::thread_rng;
use rand::Rng;
use serde::Deserialize;
use serde::Serialize;
use std::collections::BTreeMap;
use std::collections::HashMap;

pub type Ω = u8;

/// Current state of a bigram model - either a token, or None at the start of
/// generation.
#[derive(Clone, Debug, PartialEq, Hash, Eq, Deserialize, Serialize)]
pub struct BigramState(Option<String>);

impl From<Option<String>> for BigramState {
    fn from(bs: Option<String>) -> Self {
        BigramState(bs)
    }
}

/// A single bigram transition.
///
/// Encapsulate a map from the probability space to tokens.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct BigramTransition(pub BTreeMap<Ω, String>);

impl BigramTransition {
    /// Grab a random token, proportionally to its frequency.
    fn rand(&self) -> String {
        let rand: Ω = thread_rng().gen();
        self.0.range(rand..).next().map(|(_, s)| s).unwrap().clone()
    }
}

impl From<BTreeMap<Ω, String>> for BigramTransition {
    fn from(source: BTreeMap<Ω, String>) -> Self {
        Self(source)
    }
}

/// Convert from item counts to the sample-able probability space.
///
/// A little bit out of place generic just to save on code reuse. This happens
/// to also be useful for the entropy estimator, which operates on events that
/// are bigram states instead of tokens.
pub fn from_aggregate<T>(transitions: impl Iterator<Item = (T, u64)>) -> BTreeMap<Ω, T> {
    let mut transitions: Vec<(T, u64)> = transitions.collect();
    let sum: u64 = transitions.iter().map(|(_, n)| n).sum();
    transitions.sort_by_key(|(_, n)| *n);
    transitions
        .into_iter()
        .fold(
            (BTreeMap::<Ω, T>::new(), 0u64),
            |(mut map, cdf), (next, count)| {
                let ncdf = cdf + count;
                let key = (ncdf * (Ω::MAX as u64)) / sum;
                map.insert(key as Ω, next);
                (map, ncdf)
            },
        )
        .0
        .into()
}

/// The core bigram model.
///
/// This contains a map from the current bigram state (None for initial symbol,
/// otherwise a token) to a bigram transition.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct BigramModel(pub HashMap<BigramState, BigramTransition>);

// Allowing dead code because each function is only used in one binary, and is
// marked as unused in the other
#[allow(dead_code)]
impl BigramModel {
    pub fn gen_iter(&self) -> BigramGeneratorIter {
        BigramGeneratorIter {
            model: self,
            state: None.into(),
        }
    }

    pub fn serialize(&self) -> Result<String, serde_lexpr::Error> {
        serde_lexpr::to_string(self)
    }

    pub fn deserialize(serialized: &str) -> Result<Self, serde_lexpr::Error> {
        serde_lexpr::from_str::<BigramModel>(serialized)
    }
}

impl From<HashMap<BigramState, BigramTransition>> for BigramModel {
    fn from(source: HashMap<BigramState, BigramTransition>) -> Self {
        Self(source)
    }
}

/// Iterator that generates a random string using this bigram model. This is
/// an iterator to offer more flexibility in terms of password metrics and
/// generation - the consumer can control when to start or stop based on the
/// type and security level of password desired.
pub struct BigramGeneratorIter<'a> {
    model: &'a BigramModel,
    state: BigramState,
}

impl<'a> Iterator for BigramGeneratorIter<'a> {
    type Item = String;

    fn next(&mut self) -> Option<Self::Item> {
        self.state = Some(self.model.0.get(&self.state)?.rand()).into();
        self.state.0.as_ref().cloned()
    }
}
