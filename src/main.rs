use rustacuda::{launch, prelude::*};
use std::error::Error;
use std::ffi::CString;

const MAX_WORD_LENGTH: usize = 128;

pub struct EditDistanceCalculator {
    module: Module,
    stream: Stream,
    _context: Context,
}

impl EditDistanceCalculator {
    pub fn new() -> Result<Self, Box<dyn Error>> {
        // Initialize CUDA
        rustacuda::init(CudaFlags::empty())?;
        let device = Device::get_device(0)?;
        let context =
            Context::create_and_push(ContextFlags::MAP_HOST | ContextFlags::SCHED_AUTO, device)?;

        // Load the PTX module
        let ptx = CString::new(include_str!("../gpu/distance.ptx"))?;
        let module = Module::load_from_string(&ptx)?;
        let stream = Stream::new(StreamFlags::NON_BLOCKING, None)?;

        Ok(Self {
            module,
            stream,
            _context: context,
        })
    }

    pub fn calculate_distances(
        &self,
        target_word: &str,
        word_list: &[String],
    ) -> Result<Vec<i32>, Box<dyn Error>> {
        let num_words = word_list.len();

        // Prepare host data
        let mut h_word_list = vec![0u8; num_words * MAX_WORD_LENGTH];
        let mut h_word_lengths = vec![0i32; num_words];

        // Copy words into contiguous array and record lengths
        for (i, word) in word_list.iter().enumerate() {
            let start = i * MAX_WORD_LENGTH;
            let word_bytes = word.as_bytes();
            h_word_list[start..start + word_bytes.len()].copy_from_slice(word_bytes);
            h_word_lengths[i] = word_bytes.len() as i32;
        }

        // Allocate device memory
        let mut d_word = DeviceBuffer::from_slice(target_word.as_bytes())?;
        let mut d_word_list = DeviceBuffer::from_slice(&h_word_list)?;
        let mut d_word_lengths = DeviceBuffer::from_slice(&h_word_lengths)?;
        let mut d_distances = DeviceBuffer::from_slice(&vec![0i32; num_words])?;

        // Launch kernel
        let block_size = 256;
        let grid_size = (num_words as u32 + block_size - 1) / block_size;

        unsafe {
            // Get the function from the module
            let function_name = CString::new("compute_edit_distance")?;
            let function = self.module.get_function(&function_name)?;

            // Create a reference to the stream for the macro
            let stream = &self.stream;

            let result = launch!(function<<<grid_size, block_size, 0, stream>>>(
                d_word.as_device_ptr(),
                target_word.len() as i32,
                d_word_list.as_device_ptr(),
                d_word_lengths.as_device_ptr(),
                d_distances.as_device_ptr(),
                num_words as i32
            ));
            result?;
        }

        // Synchronize and get results
        self.stream.synchronize()?;
        let mut distances = vec![0i32; num_words];
        d_distances.copy_to(&mut distances)?;

        Ok(distances)
    }
}

// Example usage
fn main() -> Result<(), Box<dyn Error>> {
    let calculator = EditDistanceCalculator::new()?;

    let target_word = "example";
    let word_list = vec![
        "sample".to_string(),
        "example".to_string(),
        "exemplar".to_string(),
        "simple".to_string(),
        "examples".to_string(),
    ];

    let distances = calculator.calculate_distances(target_word, &word_list)?;

    // Print results
    for (word, distance) in word_list.iter().zip(distances.iter()) {
        println!(
            "Edit distance between '{}' and '{}': {}",
            target_word, word, distance
        );
    }

    Ok(())
}
