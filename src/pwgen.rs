use std::{
    collections::{btree_map, BTreeMap, HashMap},
    iter::zip,
};

use crate::bigram::{
    BigramGeneratorIter, BigramModel, BigramModelMap, BigramState, BigramToken, BigramTransition,
};

struct StateTransitionIter<'a, T: Clone> {
    iter: btree_map::Iter<'a, u8, T>,
    cdf: u8,
}

impl<'a, T: Clone> StateTransitionIter<'a, T> {
    fn iter(states: &'a BTreeMap<u8, T>) -> Self {
        StateTransitionIter {
            iter: states.iter(),
            cdf: 0,
        }
    }
}

impl<'a, T: Clone> Iterator for StateTransitionIter<'a, T> {
    type Item = (u8, T);

    fn next(&mut self) -> Option<Self::Item> {
        let (ncdf, token) = self.iter.next()?;
        let item = (ncdf - self.cdf, token.clone());
        self.cdf = *ncdf;
        Some(item)
    }
}

struct BigramEntropyEstimator<'a> {
    model: &'a BigramModelMap,
    states: BTreeMap<u8, BigramState>,
}

impl<'a> BigramEntropyEstimator<'a> {
    fn new(model: &'a BigramModelMap) -> Self {
        let mut initial = BTreeMap::<u8, BigramState>::new();
        initial.insert(u8::MAX, None);
        BigramEntropyEstimator {
            model,
            states: initial,
        }
    }
}

impl<'a> Iterator for BigramEntropyEstimator<'a> {
    type Item = f64;

    fn next(&mut self) -> Option<Self::Item> {
        let mut nexts = HashMap::<BigramState, u64>::new();

        for (p_state, token) in StateTransitionIter::iter(&self.states) {
            if let Some(transitions) = self.model.get(&token) {
                for (n_state, token) in StateTransitionIter::iter(transitions) {
                    *nexts.entry(Some(token)).or_default() += (p_state as u64) * (n_state as u64);
                }
            } else {
                *nexts.entry(None).or_default() += (p_state as u64) * (u8::MAX as u64);
            }
        }

        self.states = BTreeMap::<u8, BigramState>::from_aggregate(nexts);

        Some(
            self.states
                .iter()
                .fold((0.0f64, 0u8), |(entropy, pcdf), (&cdf, _)| {
                    let p = ((cdf - pcdf) as f64) / u8::MAX as f64;
                    let mut e = -p * p.log2();
                    e = if e.is_normal() { e } else { 0.0f64 };
                    (entropy + e, cdf)
                })
                .0,
        )
    }
}

struct BigramPwgenerator<'a> {
    model: &'a BigramModelMap,
    iterator: BigramGeneratorIter<'a>,
}

impl<'a> BigramPwgenerator<'a> {
    fn new(model: &'a BigramModelMap) -> Self {
        BigramPwgenerator {
            model,
            iterator: model.gen_iter(),
        }
    }
}

impl<'a> Iterator for BigramPwgenerator<'a> {
    type Item = BigramToken;

    fn next(&mut self) -> Option<Self::Item> {
        self.iterator.next().or_else(|| {
            self.iterator = self.model.gen_iter();
            self.iterator.next()
        })
    }
}

pub(crate) fn pwgen(model: BigramModelMap, minent: f64) -> String {
    let mut entropy: f64 = 0.0;
    let mut pw: String = String::new();

    for (token, e) in zip(
        BigramPwgenerator::new(&model),
        BigramEntropyEstimator::new(&model),
    ) {
        pw.push_str(&token);
        entropy += e;

        if entropy >= minent {
            break;
        }
    }

    pw
}
