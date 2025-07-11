// Data ingestion binary - currently disabled
// Uncomment this code to re-enable data ingestion

/*
use fish_logic::db::ingest_data;

#[tokio::main]
async fn main() {
    if let Err(e) = ingest_data().await {
        eprintln!("Error during data ingestion: {}", e);
    }
}
*/
// Stub main function for when ingestion is disabled
fn main() {
    println!("To re-enable ingestion, uncomment the code in this file and in db.rs");
}