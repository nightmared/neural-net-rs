#![feature(iterator_flatten, extern_prelude)]
extern crate rand;
extern crate glutin;
extern crate gl;

mod layer;
mod mnist;
mod city;
mod cost;
mod brain;
mod gui;
mod tools;

use std::fs::File;
use brain::Brain;
//use mnist::Mnist;
use gui::show_gui;
use city::City;
use glutin::GlContext;
use rand::{thread_rng, Rng};

const TEST_SIZE: usize = 150;
const ITERS: usize = 2000;
const BATCH_SIZE: usize = 25;
const EPS: f64 = 0.01;

fn main() {
    //let train_mnist = Mnist::new("train-images.idx3-ubyte", "train-labels.idx1-ubyte").unwrap();
    //let test_mnist = Mnist::new("t10k-images.idx3-ubyte", "t10k-labels.idx1-ubyte").unwrap();
    //println!("Data loaded: {} training images & {} test images !", train_mnist.number, test_mnist.number);
    //println!("Images are {}x{} pixels wide.", train_mnist.columns, train_mnist.rows);
    let mut train = City::new("rond64_bin").unwrap();
	// create a test dataset of TEST_SIZE images
	let mut test = City {
		number: TEST_SIZE,
		img_len: train.img_len,
		out_len: train.out_len,
		images: vec![],
		results: vec![]
	};
	for i in 0..TEST_SIZE {
		test.images.push(train.images.pop().unwrap());
		test.results.push(train.results.pop().unwrap());
	}
	train.number -= TEST_SIZE;
    let mut rng = thread_rng();

    let mut brain = Brain::load_from_file(File::open("brain").unwrap()).unwrap();
    //let mut brain = Brain::new(train.img_len*train.img_len);
    //brain.add_layer(1500);
    //brain.add_layer(250);
    //brain.add_layer(25);
    //brain.add_layer(train.out_len);

    //show_gui(&mut brain, &train, &test, &mut rng);

    println!("Error before training on train data: {}", train.measure_error(&mut brain, &mut rng));
    println!("Error before training on test data: {}", test.measure_error(&mut brain, &mut rng));
    for i in 0..ITERS {
        // Select BATCH_SIZE random elements
        let idx = rng.gen::<usize>()%(train.number-BATCH_SIZE);
        brain.backpropagation(&train.images[idx..idx+BATCH_SIZE], &train.results[idx..idx+BATCH_SIZE]).unwrap();
        if i % 10 == 0 {
            let test_res = train.measure_error(&mut brain, &mut rng);
            println!("{}/{} batches processed, error is {}", i+1, ITERS, test_res);
            if test_res < EPS {
                break;
            }
        }
    }
    println!("Error after training on train data: {}", train.measure_error(&mut brain, &mut rng));
    println!("Error after training on test data: {}", test.measure_error(&mut brain, &mut rng));
    brain.save(File::create("brain").unwrap()).unwrap();
}
