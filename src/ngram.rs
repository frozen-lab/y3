use serde::Deserialize;
use std::collections::{HashMap, HashSet};
use std::fs::File;
use std::io::BufReader;

type Index = HashMap<String, Vec<String>>;

#[derive(Debug, Deserialize)]
pub struct NgramIndex {
    index: Index,
}

impl NgramIndex {
    pub fn load_index(file_path: &str) -> Result<Self, serde_json::Error> {
        let file = File::open(file_path).expect("Unable to open index");
        let reader = BufReader::new(file);

        let index: Index = serde_json::from_reader(reader)?;

        Ok(Self { index })
    }

    pub fn query_candidates(&self, word: &str, n: usize) -> Option<Vec<String>> {
        let ngrams = Self::generate_ngrams(word, n);
        let mut candidates = HashSet::new();

        for ngram in ngrams {
            if let Some(words) = self.index.get(&ngram) {
                candidates.extend(words.iter().cloned());
            }
        }

        if candidates.contains(word) {
            None
        } else {
            // Return the candidate set for further processing
            Some(candidates.into_iter().collect())
        }
    }

    fn generate_ngrams(word: &str, n: usize) -> Vec<String> {
        word.chars()
            .collect::<Vec<char>>()
            .windows(n)
            .map(|window| window.iter().collect())
            .collect()
    }
}
