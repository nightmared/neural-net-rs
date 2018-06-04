use std::fs::File;
use std::io::Write;
use std::{io, mem};
use layer::Linear;
use tools::*;

use layer;
use cost;

pub struct Brain {
    input_size: usize,
    pub layers: Vec<layer::Linear>,
    // variables used for backpropagation (everything is already allocated, so it ought to be
    // faster, right ?)
    pub tmp_layers: Vec<layer::Linear>,
    pub deltas: Vec<Vec<f64>>,
    pub learn_rate: f64,
}

impl Brain {
    pub fn new(input_size: usize) -> Self {
        Brain {
            input_size,
            layers: Vec::new(),
            tmp_layers: Vec::new(),
            deltas: Vec::new(),
            learn_rate: 0.5
        }
    }

    pub fn set_learn_rate(&mut self, rate: f64) {
        self.learn_rate = rate;
    }

    fn get_last_layer_input_size(&self) -> usize {
        if self.layers.is_empty() {
            self.input_size
        } else {
            self.layers[self.layers.len()-1].size()
        }
    }

    pub fn add_layer(&mut self, output_size: usize) {
        let input_size = self.get_last_layer_input_size();
        let new_layer = layer::Linear::new(input_size, output_size);
        self.tmp_layers.push(new_layer.clone());

        self.layers.push(new_layer);
        self.deltas.push(vec![0.; output_size]);
    }

    pub fn add_existing_layer(&mut self, layer: layer::Linear) -> Result<(), ()> {
        let input_size = self.get_last_layer_input_size();
        if layer.input_size == input_size {
            self.tmp_layers.push(layer.clone());
            self.deltas.push(vec![0.; layer.length]);
            self.layers.push(layer);
            Ok(())
        } else {
            Err(())
        }
    }

    pub fn forward(&mut self, data: &[f64]) -> Result<(), &'static str> {
        if data.len() != self.input_size {
            return Err("Wrong input data size !");
        }
        self.layers.iter_mut()
            .fold(data, |out, layer| layer.forward(out));
        Ok(())
    }

    pub fn backpropagation(&mut self, data_arr: &[Vec<f64>], expected_result_arr: &[Vec<f64>]) -> Result<(), &str> {
        if data_arr.len() != expected_result_arr.len() {
            return Err("Wrong data size");
        }
        // -----------INIT TEMPORARY VALUES TO 0----------------
        self.tmp_layers.iter_mut().fold(self.input_size,
            |input_size, ref mut layer| {
                for i in 0..layer.size() {
                    layer.set_bias(i, 0.);
                    for k in 0..input_size {
                        layer.set_weight(k, i, 0.);
                    }
                }
                layer.size()
        });

        let batch_size = data_arr.len();
        let batch_learn_rate = -self.learn_rate/(batch_size as f64);
        for idx in 0..batch_size {
            let data = &data_arr[idx];
            let expected_result = &expected_result_arr[idx];

            self.forward(data)?;

            let mut index = self.layers.len() - 1;
            let cur_layer = &self.layers[index];

            // -------------COMPUTE DELTAS-----------------
            // last layer
            for neuron in 0..cur_layer.size() {
                self.deltas[index][neuron] =
                    cost::cost_derivative(cur_layer.get_outputs(), expected_result, neuron)
                    * layer::act_fun_derivative(cur_layer.get_potentials()[neuron]);
            }

            while index > 0 {
                index -= 1;
                let cur_layer = &self.layers[index];
                let next_layer = &self.layers[index+1];
                for neuron in 0..cur_layer.size() {
                    let mut sum = 0.;
                    for next in 0..next_layer.size() {
                        sum += next_layer.get_weight(neuron, next) * self.deltas[index+1][next];
                    }
                    self.deltas[index][neuron] =
                        sum * layer::act_fun_derivative(cur_layer.get_potentials()[neuron]);
                }
            }

            // ------------GENERATE TEMPORARY WEIGHTS AND BIASES-----------------
            for layer in (0..self.tmp_layers.len()).rev() {
                let mut new_layer = &mut self.tmp_layers[layer];
                for neuron in 0..new_layer.size() {
                    let previous_layer_outputs =
                        if layer == 0 {
                            data
                        } else {
                            self.layers[layer-1].get_outputs()
                        };
                    new_layer.add_bias(neuron, self.deltas[layer][neuron]);
                    for i in 0..previous_layer_outputs.len() {
                        new_layer.add_weight(i, neuron, self.deltas[layer][neuron]*previous_layer_outputs[i]);
                    }
                }
            }
        }

        // ------------UPDATE WEIGHTS AND BIASES-----------------
        self.layers.iter_mut()
            .zip(self.tmp_layers.iter())
            .fold(self.input_size, |input_size, (ref mut layer, ref tmp_layer)| {
                for i in 0..layer.size() {
                    layer.add_bias(i, batch_learn_rate * tmp_layer.get_bias(i));
                    for k in 0..input_size {
                        layer.add_weight(k, i, batch_learn_rate * tmp_layer.get_weight(k, i));
                    }
                }
                layer.size()
        });

        // Wow ! Everything went fine ;) (or not !)
        Ok(())
    }
    pub fn cost(&mut self, inputs: &[f64], expected_results: &[f64]) -> Result<f64, &str> {
        self.forward(inputs)?;
        Ok(cost::cost(self.layers[self.layers.len()-1].get_outputs(), expected_results))
    }
    pub fn get_outputs(&self) -> &[f64] {
        &self.layers[self.layers.len()-1].get_outputs()
    }
    pub fn save(&self, mut dir: File) -> Result<(), io::Error> {
        write_dq(&mut dir, self.layers.len())?;
        write_dq(&mut dir, unsafe{mem::transmute(self.learn_rate)})?;
        for i in 0..self.layers.len() {
            write_dq(&mut dir, self.layers[i].input_size)?;
            write_dq(&mut dir, self.layers[i].length)?;
            write_arr(&mut dir, &self.layers[i].weights)?;
            write_arr(&mut dir, &self.layers[i].bias)?;
        }
        dir.flush()?;
        Ok(())
    }
    pub fn load_from_file(mut dir: File) -> Result<Self, io::Error> {
        let layers_num = read_dq(&mut dir)?;
        let learn_rate = read_dq(&mut dir)?;
        let mut b = Brain {
            input_size: 0,
            layers: Vec::new(),
            tmp_layers: Vec::new(),
            deltas: Vec::new(),
            learn_rate: unsafe{mem::transmute(learn_rate)}
        };
        for i in 0..layers_num {
            let in_size = read_dq(&mut dir)?;
            if i == 0 {
                b.input_size = in_size;
            }
            let out_size = read_dq(&mut dir)?;
            let mut weights = read_arr(&mut dir, in_size*out_size)?;
            let mut bias = read_arr(&mut dir, out_size)?;
            let mut layer = Linear::new(in_size, out_size);
            layer.weights = weights;
            layer.bias = bias;
            b.add_existing_layer(layer).unwrap();
        }
        Ok(b)
    }
}
