//! Implementation of Hidden Markov Models.

use std::cmp::Ordering;

use ndarray::prelude::*;
use num_traits::Zero;
use ordered_float::OrderedFloat;

use super::probs::{LogProb, Prob};

/// Errors
quick_error! {
    #[derive(Debug, PartialEq)]
    pub enum HMMError {
        InvalidDimension(an0: usize, an1: usize, bn: usize, bm: usize, pin: usize) {
            description("invalid dimensions on construction")
            display(
                "inferred from A: N_0={}, N_1={} (must be equal), from B: N={}, M={}, from \
                pi: N={}", an0, an1, bn, bm, pin)
        }
    }
}

/// A simple Hidden Markov Model with `N` states emitting discrete symbols (alpha bet size `M`) as
/// described in Rabiner's tutorial with state transition matrix `A`, observation symbol
/// probability distribution `B`, and initial state distribution `pi`.
#[derive(Debug, PartialEq)]
pub struct HMM {
    /// The state transition matrix (size `NxN`), `A` in Rabiner's tutorial.
    transition: Array2<LogProb>,
    /// The observation symbol probability distribution (size `NxM`), `B` in Rabiner's tutorial.
    observation: Array2<LogProb>,
    /// The initial state distribution (size `M`), `pi` in Rabiner's tutorial.
    initial: Array1<LogProb>,
}

impl HMM {
    /// Construct new Hidden MarkovModel with the given transition, observation, and initial
    /// state matrices and vectors already in log-probability space.
    pub fn new(
        transition: Array2<LogProb>,
        observation: Array2<LogProb>,
        initial: Array1<LogProb>,
    ) -> Result<Self, HMMError> {
        let (an0, an1) = transition.dim();
        let (bn, bm) = observation.dim();
        let pin = initial.dim();

        if an0 != an1 || an0 != bn || an0 != pin {
            Err(HMMError::InvalidDimension(an0, an1, bn, bm, pin))
        } else {
            Ok(Self {
                transition,
                observation,
                initial,
            })
        }
    }

    /// Construct new Hidden MarkovModel with the given transition, observation, and initial
    /// state matrices and vectors already as `Prob` values.
    pub fn with_prob(
        transition: &Array2<Prob>,
        observation: &Array2<Prob>,
        initial: &Array1<Prob>,
    ) -> Result<Self, HMMError> {
        Self::new(
            transition.map(|x| LogProb::from(*x)),
            observation.map(|x| LogProb::from(*x)),
            initial.map(|x| LogProb::from(*x)),
        )
    }

    /// Construct new Hidden MarkovModel with the given transition, observation, and initial
    /// state matrices and vectors with probabilities as `f64` values.
    pub fn with_float(
        transition: &Array2<f64>,
        observation: &Array2<f64>,
        initial: &Array1<f64>,
    ) -> Result<Self, HMMError> {
        Self::new(
            transition.map(|x| LogProb::from(Prob(*x))),
            observation.map(|x| LogProb::from(Prob(*x))),
            initial.map(|x| LogProb::from(Prob(*x))),
        )
    }

    /// Return number of states.
    pub fn num_symbols(&self) -> usize {
        self.initial.len()
    }

    /// Return number of states.
    pub fn num_states(&self) -> usize {
        self.transition.dim().0
    }

    /// Compute most likely sequence of states given a list of observations using the Viterbi
    /// algorithm (maximum a posteriori/MAP).
    pub fn viterbi(&self, observations: Vec<usize>) -> (Vec<usize>, LogProb) {
        // The matrix with probabilities.
        let mut vals = Array2::<LogProb>::zeros((observations.len(), self.num_states()));
        // For each cell in `vals`, a pointer to the row in the previous column (for the traceback).
        let mut from = Array2::<usize>::zeros((observations.len(), self.num_states()));

        // Compute matrix.
        for (i, o) in observations.iter().enumerate() {
            if i == 0 {
                // Initial column.
                for (j, p) in self.initial.iter().enumerate() {
                    vals[[0, j]] = p + self.observation[[j, *o]];
                    from[[0, j]] = j;
                }
            } else {
                // TODO: need to store pointers back...
                // Subsequent columns.
                for j in 0..self.num_states() {
                    let x = vals.subview(Axis(0), i - 1)
                        .iter()
                        .enumerate()
                        .max_by(|(a, x), (b, y)| {
                            if x.is_zero() && y.is_zero() {
                                Ordering::Equal
                            } else if x.is_zero() {
                                Ordering::Less
                            } else if y.is_zero() {
                                Ordering::Greater
                            } else {
                                (*x + self.transition[[*a, j]])
                                    .partial_cmp(&(*y + self.transition[[*b, j]]))
                                    .unwrap()
                            }
                        })
                        .map(|(x, y)| (x, *y))
                        .unwrap();
                    vals[[i, j]] = x.1 + self.transition[[x.0, j]] + self.observation[[j, *o]];
                    from[[i, j]] = x.0;
                }
            }
        }

        // Traceback through matrix.
        let n = observations.len();
        let mut result: Vec<usize> = Vec::new();
        let mut curr = 0;
        let mut res_prob = LogProb::ln_zero();
        for (i, col) in vals.axis_iter(Axis(0)).rev().enumerate() {
            if i == 0 {
                let tmp = col.iter()
                    .enumerate()
                    .max_by_key(|&(_, item)| OrderedFloat(**item))
                    .unwrap();
                curr = tmp.0;
                res_prob = *tmp.1;
            } else {
                curr = from[[n - i, curr]];
            }
            result.push(curr);
        }
        result.reverse();

        (result, res_prob)
    }

    // Compute the probability of a series of observations using the forward algorithm.
    pub fn forward(&self, observations: Vec<usize>) -> LogProb {
        // The matrix with probabilities.
        let mut vals = Array2::<LogProb>::zeros((observations.len(), self.num_states()));

        // Compute matrix.
        for (i, o) in observations.iter().enumerate() {
            if i == 0 {
                // Initial column.
                for (j, p) in self.initial.iter().enumerate() {
                    vals[[0, j]] = p + self.observation[[j, *o]];
                }
            } else {
                // Subsequent columns.
                for j in 0..self.num_states() {
                    let xs = (0..self.num_states())
                        .map(|k| {
                            vals[[i - 1, k]] + self.transition[[k, j]] + self.observation[[j, *o]]
                        })
                        .collect::<Vec<LogProb>>();
                    vals[[i, j]] = LogProb::ln_sum_exp(&xs);
                }
            }
        }

        // Compute final probability.
        LogProb::ln_sum_exp(vals.row(observations.len() - 1).into_slice().unwrap())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_viterbi_toy_example() {
        // We construct the toy example from Borodovsky & Ekisheva (2006), pp. 80.
        //
        // http://cecas.clemson.edu/~ahoover/ece854/refs/Gonze-ViterbiAlgorithm.pdf
        //
        // States: 0=High GC content, 1=Low GC content
        // Symbols: 0=A, 1=C, 2=G, 3=T
        let transition = array![[0.5, 0.5], [0.4, 0.6]];
        let observation = array![[0.2, 0.3, 0.3, 0.2], [0.3, 0.2, 0.2, 0.3]];
        let initial = array![0.5, 0.5];

        let hmm = HMM::with_float(&transition, &observation, &initial)
            .expect("Dimensions should be consistent");
        let (path, log_prob) = hmm.viterbi(vec![2, 2, 1, 0, 1, 3, 2, 0, 0]);
        let prob = Prob::from(log_prob);

        assert_eq!(vec![0, 0, 0, 1, 1, 1, 1, 1, 1], path);
        assert_relative_eq!(4.25e-8_f64, *prob, epsilon = 1e-9_f64);
    }

    #[test]
    fn test_forward_toy_example() {
        // Same toy example as above.
        let transition = array![[0.5, 0.5], [0.4, 0.6]];
        let observation = array![[0.2, 0.3, 0.3, 0.2], [0.3, 0.2, 0.2, 0.3]];
        let initial = array![0.5, 0.5];

        let hmm = HMM::with_float(&transition, &observation, &initial)
            .expect("Dimensions should be consistent");
        let log_prob = hmm.forward(vec![2, 2, 1, 0]);
        let prob = Prob::from(log_prob);

        assert_relative_eq!(0.0038432_f64, *prob, epsilon = 0.0001);
    }
}
