# Y3

Yeet your typo's into the shadow realm before it makes it to production!

Y3 is a GPU-accelerated spell checker designed specifically for coding projects. 

## How It Works

Y3 scans your source file for potential typos, identifies the location of the issue, and offers a 
list of suggestions:

```shell
--> src/reader.rs:11.4
   |
11 |     ngram
   |     ^
   |
   = suggestions: ["gram", "ingram", "engram", "cram", "grav"]
```

## Installation

1. Ensure you have [Rust](https://www.rust-lang.org/) installed.
2. Clone the repository:

   ```bash
   git clone https://github.com/frozen-beak/y3.git
   ```

3. Build the project:

   ```bash
   cargo build --release
   ```

4. Run Y3:

   ```bash
   ./target/release/y3 [file path]
   ```
   
---

**Why Y3?**

Because typos are best left in the shadow realm.

