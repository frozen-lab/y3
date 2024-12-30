
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
