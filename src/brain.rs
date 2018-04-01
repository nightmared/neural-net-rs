use std::marker;
use layer::Layer;
use cost::CostFunction;

pub struct Brain<C: CostFunction, L: Layer + Sized> {
    input_size: usize,
    pub layers: Vec<L>,
    // variables used for backpropagation (everything is already allocated, so it ought to be
    // faster, right ?)
    pub tmp_layers: Vec<L>,
    pub deltas: Vec<Vec<f64>>,
    pub learn_rate: f64,
    // I really really *enjoy* the feeling of using dark magic !
    _marker_cost: marker::PhantomData<C>
}

impl<C: CostFunction, L: Layer + Sized + Clone> Brain<C, L> {
    pub fn new(input_size: usize) -> Self {
        Brain {
            input_size,
            layers: Vec::new(),
            tmp_layers: Vec::new(),
            deltas: Vec::new(),
            learn_rate: 2.,
            _marker_cost: marker::PhantomData
        }
    }
    
    pub fn add_layer(&mut self, output_size: usize) {
        let input_size =
            if self.layers.is_empty() {
                self.input_size
            } else {
                self.layers[self.layers.len()-1].size()
            };
        let new_layer = L::new(input_size, output_size);
        self.tmp_layers.push(new_layer.clone());

        self.layers.push(new_layer);
        self.deltas.push(vec![0.; output_size]);
    }

    pub fn forward(&mut self, data: &[f64]) -> Result<(), &'static str> {
        if data.len() != self.input_size {
            return Err("Wrong input data size !");
        }
        self.layers.iter_mut()
            .fold(data, |out, layer| layer.forward(out));
        Ok(())
    }

    pub fn backpropagation(&mut self, data_arr: &[&[f64]], expected_result_arr: &[&[f64]]) -> Result<(), &str> {
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
                    C::cost_derivative(cur_layer.get_outputs(), expected_result, neuron)
                    * L::act_fun_derivative(cur_layer.get_potentials()[neuron]);
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
                        sum * L::act_fun_derivative(cur_layer.get_potentials()[neuron]);
                }
            }

            // ------------GENERATE TEMPORARY WEIGHTS AND BIASES-----------------
            for layer in (0..self.tmp_layers.len()).rev() {
                for neuron in 0..self.tmp_layers[layer].size() {
                    let previous_layer_outputs =
                        if layer == 0 {
                            data
                        } else {
                            self.layers[layer-1].get_outputs()
                        };
                    let mut new_layer = &mut self.tmp_layers[layer];
                    new_layer.add_bias(neuron, -self.learn_rate * self.deltas[layer][neuron]);
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
                    layer.set_bias(i, tmp_layer.get_bias(i));
                    for k in 0..input_size {
                        layer.set_weight(k, i, tmp_layer.get_weight(k, i));
                    }
                }
                layer.size()
        });

        // Wow ! Everything went fine ;)
        Ok(())
    }
}
