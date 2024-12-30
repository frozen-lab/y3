use std::io;
use y3::{
    gpu::Gpu,
    ngram::NgramIndex,
    reader::Reader,
    tokenizer::{Token, Tokenizer},
};

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

    let tokens = tokenizer.tokens();
    let gpu = Gpu::new().expect("Failed to load GPU module");

    for token in tokens {
        if let Some(candidates) = ngram.query_candidates(&token.word(), 2) {
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

            // Get the closest candidates
            let closest_candidates: Vec<&String> = candidate_pairs
                .into_iter()
                .take(5)
                .map(|(candidate, _)| candidate)
                .collect();

            print_warning(reader.path(), token, &closest_candidates);
        }
    }

    Ok(())
}

fn print_warning(file_path: &str, token: &Token, closest_candidates: &[&String]) {
    println!(
        "--> {file_path}:{}.{}\n   |\n{}  | {}\n   | {}\n   |\n   = suggestions: {:?}",
        token.position().line_no(),
        token.position().start(),
        token.position().line_no(),
        format!(
            "{:width$}{}",
            "",
            &token.word(),
            width = token.position().start()
        ),
        format!("{:width$}^", "", width = token.position().start()),
        closest_candidates,
    );
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
