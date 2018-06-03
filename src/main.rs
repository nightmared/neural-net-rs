#![feature(iterator_flatten, extern_prelude)]
extern crate rand;
extern crate glutin;
extern crate gl;

mod layer;
mod mnist;
mod cost;
mod brain;
mod gui;

use std::fs::File;
use std::{thread, time};
use brain::Brain;
use mnist::Mnist;
use glutin::GlContext;
use rand::{thread_rng, Rng};

fn main() {
    let mut brain = Brain::new(28*28);
    //brain.add_layer(100);
    brain.add_layer(10);

    let train_mnist = Mnist::new("train-images.idx3-ubyte", "train-labels.idx1-ubyte").unwrap();
    let test_mnist = Mnist::new("t10k-images.idx3-ubyte", "t10k-labels.idx1-ubyte").unwrap();
    println!("Data loaded: {} training images & {} test images !", train_mnist.number, test_mnist.number);
    println!("Images are {}x{} pixels wide.", train_mnist.columns, train_mnist.rows);
    let mut rng = thread_rng();

    let mut events_loop = glutin::EventsLoop::new();
    let window = glutin::WindowBuilder::new().with_title("TIPE");
    let context = glutin::ContextBuilder::new();
    let gl_window = glutin::GlWindow::new(window, context, &events_loop).unwrap();
    let _ = unsafe { gl_window.make_current() };
	let idx = rng.gen::<usize>()%60000;
	let mut gui = gui::Gui::new(&gl_window, train_mnist.images[idx].as_slice());
    events_loop.run_forever(|event| {
        match event {
            glutin::Event::WindowEvent { event, .. } => match event {
                glutin::WindowEvent::CloseRequested => return glutin::ControlFlow::Break,
                glutin::WindowEvent::Resized(w, h) => gl_window.resize(w, h),
                _ => ()
            },
            _ => (),
        }
		gui.redraw(train_mnist.images[idx].as_slice());
        let _ = gl_window.swap_buffers();
        glutin::ControlFlow::Continue
    });

    println!("Error before training on train data: {}", train_mnist.measure_error(&mut brain));
    println!("Error before training on test data: {}", test_mnist.measure_error(&mut brain));
    let iters = 500;
    let batch_size = 100;
    let eps = 0.03;
    for i in 0..iters {
        // Select batch_size random elements
        let idx = rng.gen::<usize>()%(60000/batch_size);
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
}
