#![feature(hash_set_entry)]

use std::{collections::HashSet, fs::File, io::Read, path::Path};

use clap::Parser;

use crate::pwgen::pwgen;

mod bigram;
mod pwgen;
mod tokenize;

#[derive(Parser)]
#[clap(author = "Vlad (off-by-one@github)", version = "v1.0.0", about, long_about = None)]
struct Cli {
    /// File containing training data for the bigram model.
    #[clap(short, long, value_parser)]
    data: String,

    /// Optional list of multigraphs in your script. Helps pronounceability of the result.
    #[clap(short, long, value_parser)]
    tokens: Vec<String>,

    /// Optional list of delimeters. Newline will always be a delimeter.
    #[clap(short, long, value_parser)]
    separators: Vec<String>,

    /// Desired minimum password entropy.
    #[clap(short, long, value_parser)]
    entropy: Option<f64>,
}

fn main() {
    let cli = Cli::parse();

    let path = Path::new(&cli.data);

    let mut file = match File::open(path) {
        Err(why) => panic!("couldn't open training data: {}", why),
        Ok(file) => file,
    };

    let mut data = String::new();

    if let Err(why) = file.read_to_string(&mut data) {
        panic!("couldn't read file: {}", why);
    }

    let tokens = tokenize::PwTokens::new(data.as_str(), cli.tokens.iter().cloned().collect());

    let separators: HashSet<&str> = ["\n"]
        .iter()
        .cloned()
        .chain(cli.separators.iter().map(|s| s.as_str()))
        .collect();
    let model = bigram::train(tokens.iter(), &separators);

    let pw = pwgen(model, cli.entropy.unwrap_or(90.0));

    println!("{}\n", pw);
}
