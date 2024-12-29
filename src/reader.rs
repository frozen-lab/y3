use std::{fs, io};

pub struct Reader {
    path: String,
}

impl Reader {
    pub fn new(file_path: &str) -> io::Result<Self> {
        let metadata = fs::metadata(file_path)?;

        if metadata.is_file() {
            return Ok(Self {
                path: String::from(file_path),
            });
        }

        // Provided path is not a valid file
        Err(io::Error::new(
            io::ErrorKind::NotFound,
            "File not found at the provided path",
        ))
    }

    pub fn path(&self) -> &str {
        &self.path
    }
}
