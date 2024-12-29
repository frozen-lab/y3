use std::{fs, io, path::PathBuf};

pub struct Reader {
    path: PathBuf,
}

impl Reader {
    pub fn new(file_path: &str) -> io::Result<Self> {
        let metadata = fs::metadata(file_path)?;

        if metadata.is_file() {
            return Ok(Self {
                path: PathBuf::from(file_path),
            });
        }

        // Provided path is not a valid file
        Err(io::Error::new(
            io::ErrorKind::NotFound,
            "File not found at the provided path",
        ))
    }

    pub fn path(&self) -> &PathBuf {
        &self.path
    }
}
