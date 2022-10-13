use std::{
    collections::{btree_map, BTreeMap, HashMap},
    iter::zip,
};

use crate::bigram::{from_aggregate, BigramGeneratorIter, BigramModel, BigramState, Ω};

/// Iterator which converts the sample-able form of the Ω sample space to the
/// actual interval lengths for each item. i.e. (0, t1), (100, t2) would be
/// converted to (100, t1), (155, t2), which is the proportion of each item
/// when sampled.
///
/// The generic allows code reuse between tokens and states.
struct SampleSpaceiter<'a, T: Clone> {
    iter: btree_map::Iter<'a, Ω, T>,
    cdf: Ω,
}

impl<'a, T: Clone> SampleSpaceiter<'a, T> {
    fn iter(states: &'a BTreeMap<Ω, T>) -> Self {
        SampleSpaceiter {
            iter: states.iter(),
            cdf: 0,
        }
    }
}

impl<'a, T: Clone> Iterator for SampleSpaceiter<'a, T> {
    type Item = (Ω, T);

    fn next(&mut self) -> Option<Self::Item> {
        let (ncdf, token) = self.iter.next()?;
        let item = (ncdf - self.cdf, token.clone());
        self.cdf = *ncdf;
        Some(item)
    }
}

/// Iterator, output the added entropy per token.
///
/// Inteded to be run in parallel to BigramGeneratorIter.
struct BigramEntropyEstimator<'a> {
    model: &'a BigramModel,
    states: BTreeMap<Ω, BigramState>,
}

impl<'a> BigramEntropyEstimator<'a> {
    fn new(model: &'a BigramModel) -> Self {
        let mut initial = BTreeMap::<Ω, BigramState>::new();
        initial.insert(Ω::MAX, None.into());
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

        for (p_state, state) in SampleSpaceiter::iter(&self.states) {
            if let Some(transitions) = self.model.0.get(&state) {
                for (p_transition, token) in SampleSpaceiter::iter(&transitions.0) {
                    *nexts.entry(Some(token).into()).or_default() +=
                        (p_state as u64) * (p_transition as u64);
                }
            } else {
                *nexts.entry(None.into()).or_default() += (p_state as u64) * (Ω::MAX as u64);
            }
        }

        self.states = from_aggregate(nexts.into_iter());

        Some(
            self.states
                .iter()
                .fold((0.0f64, Ω::MIN), |(entropy, pcdf), (&cdf, _)| {
                    let p = ((cdf - pcdf) as f64) / Ω::MAX as f64;
                    let mut e = -p * p.log2();
                    e = if e.is_normal() { e } else { 0.0f64 };
                    (entropy + e, cdf)
                })
                .0,
        )
    }
}

/// The core password generator.
///
/// Make use of the BigramModel::gen_iter, but restart it when it stops to allow
/// passwords of arbitrary length.
struct BigramPwgenerator<'a> {
    model: &'a BigramModel,
    iterator: BigramGeneratorIter<'a>,
}

impl<'a> BigramPwgenerator<'a> {
    fn new(model: &'a BigramModel) -> Self {
        BigramPwgenerator {
            model,
            iterator: model.gen_iter(),
        }
    }
}

impl<'a> Iterator for BigramPwgenerator<'a> {
    type Item = String;

    fn next(&mut self) -> Option<Self::Item> {
        self.iterator.next().or_else(|| {
            self.iterator = self.model.gen_iter();
            self.iterator.next()
        })
    }
}

/// Generate password with at least minent amount of entropy.
pub fn pwgen(model: BigramModel, minent: f64) -> String {
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
