use std::io;
use y3::{ngram::NgramIndex, reader::Reader, tokenizer::Tokenizer};

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

    let candidates = ngram.query_candidates(tokens[0].word(), 2);

    println!("Candidates for '{}': {:?}", tokens[0].word(), candidates);

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
