extern crate rand;

mod neuron;
mod layer;
mod brain;

use brain::{Brain, CostFunction};
use layer::LayerKind;

fn main() {
    let data : [[f64; 2]; 4] = [[0., 0.], [0., 1.], [1., 0.], [1., 1.]];
    let res : [[f64; 1]; 4] = [[0.], [0.], [0.], [1.]];
    let mut brain = Brain::new(CostFunction::Quadratic, 2);
    brain.add_layer(LayerKind::Linear, 2);
    brain.add_layer(LayerKind::Linear, 1);
    for i in 0..100 {
        let mut train: [&[f64]; 25] = [&[0.; 2]; 25];
        let mut train_res: [&[f64]; 25] = [&[0.; 1]; 25];
        for k in 0..train.len() {
            let index = rand::random::<usize>() % 4;
            train[k] = &data[index];
            train_res[k] = &res[index];
        }
            let index = rand::random::<usize>() % 4;
        println!("{:?} - got: {} - expected: {:?}", &data[index], brain.run(&data[index]).unwrap()[0], res[index]);
//        println!("cost : {:?}", brain.cost(train[rand::random::<usize>() % 25], train_res[rand::random::<usize>() % 25]));
        //println!("{:?}", brain.layers[1]);
        brain.backpropagation(&train, &train_res);
    }
}
