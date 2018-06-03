#![feature(iterator_flatten)]
extern crate rand;

mod layer;
mod mnist;
mod cost;
mod brain;

use std::fs::File;
use brain::Brain;
use mnist::Mnist;

use rand::{thread_rng, Rng};

fn main() {
    let mut brain = Brain::new(28*28);
    //brain.add_layer(100);
    brain.add_layer(10);

    let train_mnist = Mnist::new("train-images.idx3-ubyte", "train-labels.idx1-ubyte").unwrap();
    let test_mnist = Mnist::new("t10k-images.idx3-ubyte", "t10k-labels.idx1-ubyte").unwrap();

    println!("Data loaded: {} training images & {} test images !", train_mnist.number, test_mnist.number);
    println!("Images are {}x{} pixels wide.", train_mnist.columns, train_mnist.rows);

    println!("Error before training on train data: {}", train_mnist.measure_error(&mut brain));
    println!("Error before training on test data: {}", test_mnist.measure_error(&mut brain));
    let mut rng = thread_rng();
    let iters = 500;
    let batch_size = 100;
    let eps = 0.03;
    for i in 0..iters {
        // Select 100 random elements
        let idx = rng.gen::<usize>()%60;
        brain.backpropagation(&train_mnist.images[idx*batch_size..(idx+1)*batch_size], &train_mnist.results[idx*batch_size..(idx+1)*batch_size]).unwrap();
        let test_res = test_mnist.measure_error(&mut brain);
        println!("{}/{} batches processed, error is {}", i+1, iters, test_res);
        if test_res < eps {
            break;
        }
    }
    println!("Error after training on train data: {}", train_mnist.measure_error(&mut brain));
    println!("Error after training on test data: {}", test_mnist.measure_error(&mut brain));
    brain.save(File::create("brain").unwrap()).unwrap();
    let mut brain = Brain::load_from_file(File::open("brain").unwrap()).unwrap();
    println!("Error after training on train data: {}", train_mnist.measure_error(&mut brain));
    println!("Error after training on test data: {}", test_mnist.measure_error(&mut brain));
}
