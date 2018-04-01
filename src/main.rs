extern crate rand;

mod layer;
mod cost;
mod brain;

use brain::Brain;
use layer::Linear;
use cost::Quadratic;

fn main() {
    let mut brain = Brain::<Quadratic, Linear>::new(2);
    brain.add_layer(26*26);
    brain.add_layer(200);
    brain.add_layer(150);
    brain.add_layer(10);
}
