use std::io;

use y3::reader::Reader;

fn main() -> io::Result<()> {
    let args: Vec<String> = std::env::args().collect();

    if args.len() < 2 {
        print_help();

        return Ok(());
    }

    let file_path = &args[1];

    let reader = Reader::new(file_path)?;

    println!("Path: {:?}", reader.path());

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
