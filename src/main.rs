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

fn main() {
    //let train_mnist = Mnist::new("train-images.idx3-ubyte", "train-labels.idx1-ubyte").unwrap();
    //let test_mnist = Mnist::new("t10k-images.idx3-ubyte", "t10k-labels.idx1-ubyte").unwrap();
    //println!("Data loaded: {} training images & {} test images !", train_mnist.number, test_mnist.number);
    //println!("Images are {}x{} pixels wide.", train_mnist.columns, train_mnist.rows);
    let mut train = City::new("db-bin").unwrap();
	// create a test dataset of 50 images
	let mut test = City {
		number: 50,
		img_len: train.img_len,
		out_len: train.out_len,
		images: vec![],
		results: vec![]
	};
	for i in 0..50 {
		test.images.push(train.images.pop().unwrap());
		test.results.push(train.results.pop().unwrap());
	}
	train.number -= 50;
    let mut rng = thread_rng();

    let mut brain = Brain::new(train.img_len*train.img_len);
    brain.add_layer(1000);
    brain.add_layer(100);
    brain.add_layer(train.out_len);

    //show_gui(&mut brain, &train, &test, &mut rng);

    println!("Error before training on train data: {}", train.measure_error(&mut brain, &mut rng));
    println!("Error before training on test data: {}", test.measure_error(&mut brain, &mut rng));
    let iters = 250;
    let batch_size = 100;
    let eps = 0.03;
    for i in 0..iters {
        // Select batch_size random elements
        let idx = rng.gen::<usize>()%(train.number-batch_size);
        brain.backpropagation(&train.images[idx..idx+batch_size], &train.results[idx..idx+batch_size]).unwrap();
        let test_res = train.measure_error(&mut brain, &mut rng);
        println!("{}/{} batches processed, error is {}", i+1, iters, test_res);
        if test_res < eps {
            break;
        }
    }
    println!("Error after training on train data: {}", train.measure_error(&mut brain, &mut rng));
    println!("Error after training on test data: {}", test.measure_error(&mut brain, &mut rng));
    brain.save(File::create("brain").unwrap()).unwrap();
}
