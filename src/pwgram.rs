#![feature(hash_set_entry)]

mod bigram;

#[path = "bigram-gen.rs"]
mod bigram_gen;

use std::{
    error::Error,
    fs::File,
    io::{self, Read},
    path::Path,
};

use clap::Parser;

use crate::{bigram::BigramModel, bigram_gen::pwgen};

#[derive(Parser)]
#[clap(
    author = "Vlad (off-by-one@github)",
    version = "v2.0.0",
    about = "Generate a password using a pwgram bigram model.",
    long_about = "\
Generate a password using a pwgram bigram model.

The model may be given as a filename, or piped into stdin. The output is printed
to stdout.
"
)]
struct Cli {
    /// File containing training data for the bigram model.
    model: Option<String>,

    /// Desired minimum password entropy.
    #[clap(short, long, value_parser, default_value_t = 90.0)]
    entropy: f64,
}

fn main() -> Result<(), Box<dyn Error>> {
    let cli = Cli::parse();

    let mut input: Box<dyn Read> = if let Some(path) = cli.model {
        let path = Path::new(&path);
        let file = File::open(path)?;
        Box::new(file)
    } else {
        Box::new(io::stdin())
    };

    let mut data = String::new();
    input.read_to_string(&mut data)?;

    let model: BigramModel = BigramModel::deserialize(data.as_str())?;
    let pw = pwgen(model, cli.entropy);
    println!("{}\n", pw);

    Ok(())
}
