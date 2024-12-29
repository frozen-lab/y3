//!
//! # Tokenizer
//!
//! It provides a tokenizer for input files to generate [Token]'s for spell checking
//!
//! ## Working
//!
//! It follows a multi-stage tokenization approach:
//!
//! ### Preprocessing
//!
//! - Read the input_file line by line
//! - Extract chunks by splitting on whitespaces
//! - Remove special characters or non-alphabetical characters from the edges (both start
//!   & end)
//! - Ignore patterns like URLs, FilePath's, etc.
//!
//! ### Token Extraction
//!
//! - Deconstructs camelCase and PascalCase
//! - Maintains contextual special characters (e.g. "sh🤬t" -> ["sh🤬t"])
//! - Eliminates non-meaningful tokens (e.g. single letters, emojis, trailing or starting
//!   symbols)
//!
//! ## Ignored Patterns
//!
//! List of patterns which are ignored while tokenization
//!
//! - URL and file path
//! - Regex pattern
//! - Numeric content like phone numbers
//! - Punctuation and symbol's like emojis
//!
//! ## Considerations
//!
//! - Single-letter tokens are discarded
//! - Standalone numeric strings are ignored
//! - Case sensitivity is preserved during token generation
//!

use regex::Regex;
use std::{
    fs::File,
    io::{self, BufRead, BufReader},
};

///
/// Struct to represent the position of the [Token] in the input file
///
#[derive(Debug)]
pub struct Position {
    ///
    /// Byte offset where the token starts in the input file
    ///
    start: usize,

    ///
    /// Byte offset where the token ends in the input file
    ///
    end: usize,

    ///
    /// 1-based line number of the token in the input file
    ///
    line_no: usize,
}

impl Position {
    ///
    /// Getter to read the [start] position offset for the token
    ///
    /// # Example
    ///
    /// ```rust
    /// use y3::tokenizer::{Token};
    ///
    /// let token = Token::new("word", 0, 3, 1);
    ///
    /// assert_eq!(token.position().start(), 0);
    /// ```
    ///
    pub fn start(&self) -> usize {
        self.start
    }

    ///
    /// Getter to read the [end] position offset for the token
    ///
    /// # Example
    ///
    /// ```rust
    /// use y3::tokenizer::{Token};
    ///
    /// let token = Token::new("word", 0, 3, 1);
    ///
    /// assert_eq!(token.position().end(), 3);
    /// ```
    ///
    pub fn end(&self) -> usize {
        self.end
    }

    ///
    /// Getter to read the [line_no] of the token
    ///
    /// # Example
    ///
    /// ```rust
    /// use y3::tokenizer::{Token};
    ///
    /// let token = Token::new("word", 0, 3, 1);
    ///
    /// assert_eq!(token.position().line_no(), 1);
    /// ```
    ///
    pub fn line_no(&self) -> usize {
        self.line_no
    }
}

///
/// Struct representing word parsed from input file to be spell checked
///
#[derive(Debug)]
pub struct Token {
    ///
    /// Parsed word from the input file
    ///
    word: String,

    ///
    /// Position offset of the token in the input file.
    ///
    /// It's used to show position of the misspelled word to the user
    ///
    position: Position,
}

impl Token {
    ///
    /// Create a new instance of [Token]
    ///
    /// # Arguments
    ///
    /// * `word` - A string slice representing content of the token.
    /// * `start` - The starting byte index of the token.
    /// * `end` - The ending byte index of the token.
    /// * `line_no` - 1-based line number representing where the token is located.
    ///
    /// # Returns
    ///
    /// * `Token` - A new [`Token`] instance with the specified word and position metadata.
    ///
    /// # Example
    ///
    /// ```rust
    /// use y3::tokenizer::{Token};
    ///
    /// let token = Token::new("word", 0, 3, 1);
    ///
    /// assert_eq!(token.word(), "word");
    /// assert_eq!(token.position().start(), 0);
    /// assert_eq!(token.position().end(), 3);
    /// assert_eq!(token.position().line_no(), 1);
    /// ```
    ///
    pub fn new(word: &str, start: usize, end: usize, line_no: usize) -> Self {
        Self {
            word: word.to_string(),
            position: Position {
                start,
                end,
                line_no,
            },
        }
    }

    ///
    /// Getter to read the parsed `word`
    ///
    /// # Example
    ///
    /// ```rust
    /// use y3::tokenizer::{Token};
    ///
    /// let token = Token::new("word", 0, 3, 1);
    ///
    /// assert_eq!(token.word(), "word");
    /// ```
    ///
    pub fn word(&self) -> &str {
        &self.word
    }

    ///
    /// Getter to read the parsed [Position]
    ///
    /// # Example
    ///
    /// ```rust
    /// use y3::tokenizer::{Token};
    ///
    /// let token = Token::new("word", 0, 3, 1);
    ///
    /// assert_eq!(token.position().start(), 0);
    /// assert_eq!(token.position().end(), 3);
    /// assert_eq!(token.position().line_no(), 1);
    /// ```
    ///
    pub fn position(&self) -> &Position {
        &self.position
    }
}

///
/// A structure holding [Regex] patterns to be used while parsing
///
#[derive(Debug)]
struct Patterns {
    ///
    /// List of [Regex] patterns to be ignored while parsing.
    ///
    /// Fallowing types are ignored,
    /// - Link's, URL's,
    /// - File Paths
    /// - Direct numbers like "1234"
    /// - Email like patterns
    /// - Regular Expressions
    ///
    ignore_patterns: Vec<Regex>,

    ///
    /// A [Regex] pattern to match against potential tokens
    ///
    word_pattern: Regex,

    ///
    /// A [Regex] pattern to split words to form tokens
    ///
    /// Useful while separating words like,
    /// - `snake_case` to ["snake", "case"]
    /// - `Get-Item` to ["Get", "Item"]
    /// - `run—but` to ["run", "but"]
    ///
    split_pattern: Regex,
}

impl Patterns {
    ///
    /// Creates a new instance of `Patterns` with predefined [Regex] patterns.
    ///
    fn new() -> Self {
        Self {
            ignore_patterns: vec![
                Regex::new(r"https?://\S+").unwrap(),           // URLs
                Regex::new(r"[\w\-\.]+(/[\w\-\.]+)+").unwrap(), // File paths
                Regex::new(r"\b\d+\b").unwrap(),                // Pure numbers
                Regex::new(r"\\[a-zA-Z]+[{[^()]+}]*").unwrap(), // Regex patterns
                Regex::new(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b").unwrap(), // Email-like patterns
            ],
            word_pattern: Regex::new(r"[a-zA-Z]+[0-9]*[a-zA-Z]*").unwrap(), // potential tokens
            split_pattern: Regex::new(r"[ _\-—]").unwrap(), // split formats like -, _, etc.
        }
    }
}

///
/// A custom tokenizer which reads through the input file and parses words
/// to be spell checked as [Token]'s
///
#[derive(Debug)]
pub struct Tokenizer {
    ///
    /// List of parsed tokens from the input file
    ///
    tokens: Vec<Token>,

    ///
    /// Set of [Regex] patterns used for parsing tokens
    ///
    patterns: Patterns,
}

impl Tokenizer {
    ///
    /// Getter to read the list of parsed [Token]'s
    ///
    pub fn tokens(&self) -> &[Token] {
        &self.tokens
    }

    ///
    /// Create an instance of [Tokenizer]
    ///
    /// # Example
    ///
    /// ```rust
    /// use y3::tokenizer::Tokenizer;
    ///
    /// let mut tokenizer = Tokenizer::new();
    /// assert_eq!(tokenizer.tokens().len(), 0);
    /// ```
    ///
    pub fn new() -> Self {
        Self {
            tokens: Vec::new(),
            patterns: Patterns::new(),
        }
    }

    ///
    /// Clear the list of parsed [Token]'s
    ///
    /// ## NOTE
    ///
    /// It has no effect on the allocated memory for the [Vec].
    ///
    /// This saves the overhead of reallocating the memory again, it
    /// simply uses pre-allocated memory for upcoming tokens.
    ///
    pub fn clear_tokens(&mut self) {
        self.tokens.clear();
    }

    ///
    /// Parse [Token]'s from the [file_path]
    ///
    pub fn tokenize(&mut self, file_path: &str) -> io::Result<()> {
        let file = File::open(file_path)?;
        let reader = BufReader::new(file);

        for (line_no, line) in reader.lines().enumerate() {
            let line = line?;
            let mut offset = 0;

            // Step 1: Split by spaces
            let chunks: Vec<&str> = line.split_whitespace().collect();

            for chunk in chunks {
                let mut chunk = chunk.trim();

                // Step 2: Remove symbols and brackets at start or end
                chunk = chunk
                    .trim_start_matches(|c: char| !c.is_alphanumeric() && c != '\'')
                    .trim_end_matches(|c: char| !c.is_alphanumeric() && c != '\'');

                if chunk.is_empty() {
                    continue;
                }

                // Step 3: Eliminate using [ignore_patterns]
                if self
                    .patterns
                    .ignore_patterns
                    .iter()
                    .any(|p| p.is_match(chunk))
                {
                    continue;
                }

                // Step 4: Split joined words using [split_patterns]
                let sub_chunks: Vec<&str> = self.patterns.split_pattern.split(chunk).collect();

                for sub_chunk in sub_chunks {
                    if sub_chunk.is_empty() {
                        continue;
                    }

                    // Step 5: Extract tokens using [word_pattern]
                    for mat in self.patterns.word_pattern.find_iter(sub_chunk) {
                        let word = mat.as_str().to_string();

                        // Ignore single letters
                        if word.len() == 1 {
                            continue;
                        }

                        // Step 6: Preprocess tokens (e.g., split camelCase, convert TITLEcase)
                        let split_words = Self::split_word_cases(&word);

                        for split_word in split_words {
                            let start = offset + mat.start();
                            let end = offset + mat.end();

                            self.tokens.push(Token {
                                word: split_word,
                                position: Position {
                                    start,
                                    end: end - 1,
                                    line_no: line_no + 1,
                                },
                            });
                        }
                    }
                }

                // Update offset by the length of the original chunk plus one (for the space)
                // Adjust to account for spaces
                offset += chunk.len() + 1;
            }
        }

        Ok(())
    }

    ///
    /// Splits a given word into smaller words based on their case transitions.
    ///
    /// It is useful for tokenizing _camelCase_ and _PascalCase_ words into their
    /// component parts.
    ///
    /// # Arguments
    ///
    /// * `word` - A string slice representing the word to be split.
    ///
    /// # Returns
    ///
    /// * `Vec<String>` - A vector of `String` containing the individual word components split based
    /// on case transitions.
    ///
    /// e.g. "camelCaseExample", outputs -> `["camel", "Case", "Example"]`
    ///
    /// # Notes
    ///
    /// - Consecutive uppercase letters (e.g., "TITLECase") are kept together
    /// - Words without case transitions (e.g., "simple") are returned as a
    /// single-element vector.
    ///
    fn split_word_cases(word: &str) -> Vec<String> {
        let mut result = Vec::new();
        let mut start = 0;

        for (i, c) in word.char_indices() {
            if i > 0 && c.is_uppercase() && !word[start..i].chars().all(char::is_uppercase) {
                result.push(word[start..i].to_string());
                start = i;
            }
        }

        // Append remaining part of the word
        result.push(word[start..].to_string());

        result
    }
}
