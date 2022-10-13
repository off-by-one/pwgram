#![feature(hash_set_entry)]

mod bigram;
mod tokenize;

#[path = "bigram-train.rs"]
mod bigram_train;

use std::{
    collections::HashSet,
    error::Error,
    fs::File,
    io::{self, Read},
    path::Path,
};

use clap::Parser;

use crate::bigram_train::train;

#[derive(Parser)]
#[clap(
    author = "Vlad (off-by-one@github)",
    version = "v2.0.0",
    about = "Train a bigram model on a wordlist.",
    long_about = "\
Train a bigram model on a wordlist.

The wordlist may be given as a filename, or piped into stdin. The output is
printed to stdout, and should be saved to an appropriately labeled .pwgram file.
"
)]
struct Cli {
    /// File containing training data for the bigram model.
    corpus: Option<String>,

    /// Optional list of comma-separated multigraphs in your script. Helps
    /// pronounceability of the result.
    #[clap(short, long, value_parser)]
    multigraphs: Vec<String>,

    /// Additional list of delimeters. Newline will always be a delimeter to
    /// avoid multiline passwords.
    #[clap(short, long, value_parser)]
    delimeters: Vec<String>,
}

fn main() -> Result<(), Box<dyn Error>> {
    let cli = Cli::parse();

    let mut input: Box<dyn Read> = if let Some(path) = cli.corpus {
        let path = Path::new(&path);
        let file = File::open(path)?;
        Box::new(file)
    } else {
        Box::new(io::stdin())
    };

    let mut corpus = String::new();
    input.read_to_string(&mut corpus)?;

    let tokens = tokenize::PwTokens::new(corpus.as_str(), cli.multigraphs.into_iter().collect());

    let delimeters: HashSet<&str> = ["\n"]
        .iter()
        .cloned()
        .chain(cli.delimeters.iter().map(|s| s.as_str()))
        .collect();

    let model = train(tokens.iter(), &delimeters);

    println!("{}\n", model.serialize()?);

    Ok(())
}
