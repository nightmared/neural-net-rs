extern crate rand;

mod layer;
mod mnist;
mod cost;
mod brain;

use brain::Brain;
use layer::Linear;
use cost::Quadratic;
use mnist::Mnist;

use rand::{thread_rng, Rng};

fn main() {
    let mut brain = Brain::<Quadratic, Linear>::new(28*28);
    brain.add_layer(200);
    brain.add_layer(150);
    brain.add_layer(10);

    let train_mnist = Mnist::new("train-images.idx3-ubyte", "train-labels.idx1-ubyte").unwrap();
    let test_mnist = Mnist::new("t10k-images.idx3-ubyte", "t10k-labels.idx1-ubyte").unwrap();

    println!("Data loaded: {} training images & {} test images !", train_mnist.number, test_mnist.number);
    println!("Images are {}x{} pixels wide.", train_mnist.columns, train_mnist.rows);

    println!("Error before training on train data: {}", train_mnist.measure_error(&mut brain));
    println!("Error before training on test data: {}", test_mnist.measure_error(&mut brain));
    let mut rng = thread_rng();
    for i in 0..600 {
        brain.backpropagation(&train_mnist.images[i*100..i*100+100], &train_mnist.results[i*100..i*100+100]).unwrap();
    }
    println!("Error after training on train data: {}", train_mnist.measure_error(&mut brain));
    println!("Error after training on test data: {}", test_mnist.measure_error(&mut brain));
}
