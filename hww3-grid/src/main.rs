use anyhow::{bail, Result};
use hww3_grid::config::Config;
use hww3_grid::pipeline;
use std::path::Path;

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() != 2 {
        bail!("usage: hww3-grid <config.json>");
    }
    let cfg = Config::load(Path::new(&args[1]))?;
    let (grid, summary) = pipeline::build(&cfg)?;
    pipeline::write(&cfg, &grid, &summary)?;
    println!("{}", serde_json::to_string_pretty(&summary)?);
    if !summary.mesh.is_valid() {
        bail!("mesh validation failed");
    }
    Ok(())
}
