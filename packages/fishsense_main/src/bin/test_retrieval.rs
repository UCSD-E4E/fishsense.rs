use fish_logic::db::{get_database_stats, get_all_fish_labels, search_similar_fish};
use std::error::Error;
use rand::Rng;

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error>> {
    get_database_stats().await?; // this gets the database stats first
    let labels = get_all_fish_labels().await?;
    println!("{} unique fish species", labels.len());
    
    let max = std::cmp::min(labels.len(), 20); // Display first 20 labels
    for i in 0..max {
        let label = &labels[i];
        println!("   - {}", label);
    } 
    let mut rng = rand::thread_rng();  // vector search with a random embedding
    let mut random_vector: Vec<f32> = Vec::with_capacity(128);
    
    for _ in 0..128 {
        let value = rng.gen_range(-1.0..1.0);
        random_vector.push(value);
    };
    
    match search_similar_fish(random_vector, 5).await {
        Ok(similar_fish) => {
            println!("Top 5 most similar fish to the random embedding");
            let mut i = 0;
            for fish in similar_fish.iter() {
                let label = &fish.0;
                let distance = fish.1;
                let id = fish.2;
                println!("{}. {} (distance: {:.4}, id: {})", i + 1, label, distance, id);
                i += 1;
            }
        }
        Err(e) => {
            println!("Error running{}", e);
        }
    }
    println!("\ntests completed");
    
    Ok(())
}