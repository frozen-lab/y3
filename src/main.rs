use std::io;
use y3::{gpu::Gpu, ngram::NgramIndex, reader::Reader, tokenizer::Tokenizer};

fn main() -> io::Result<()> {
    let args: Vec<String> = std::env::args().collect();

    if args.len() < 2 {
        print_help();
        return Ok(());
    }

    let reader = Reader::new(&args[1])?;
    let mut tokenizer = Tokenizer::new();
    let ngram = NgramIndex::load_index("./dict/n_gram.json").expect("Failed to load index");

    tokenizer.tokenize(reader.path())?;

    // Retrieve the list of tokens from the tokenizer
    let tokens = tokenizer.tokens();
    let gpu = Gpu::new().expect("Failed to load GPU module");

    for token in tokens {
        if let Some(candidates) = ngram.query_candidates(&token.word(), 2) {
            println!("Word: {}", token.word());

            // Calculate edit distances using the GPU
            let distances = gpu
                .calculate_edit_distances(&token.word(), &candidates)
                .expect("Unable to perform edit distance calculations");

            // Sort candidates by distance
            let mut candidate_pairs: Vec<(&String, i32)> = candidates
                .iter()
                .zip(distances.iter())
                .map(|(candidate, &distance)| (candidate, distance))
                .collect();

            candidate_pairs.sort_by_key(|&(_, distance)| distance);

            // Retrieve the closest 3–5 candidates
            let closest_candidates: Vec<&String> = candidate_pairs
                .into_iter()
                .take(5)
                .map(|(candidate, _)| candidate)
                .collect();

            println!("Closest candidates: {:?}", closest_candidates);
        } else {
            println!("Word: {} is already in the dictionary.", token.word());
        }
    }

    Ok(())
}

fn print_help() {
    const TEXT: &str = r#"
    Usage:
        y3 <file_path>

    Description:

    Y3 reads a file, extracts words, and checks for spelling errors.

    Example:
    
    y3 dummy_text.txt

    "#;

    println!("{TEXT}");
}
