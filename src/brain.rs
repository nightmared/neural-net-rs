use layer::Layer;
use layer::LayerKind;

// a single cost for the whole network, maybe move this to a layer basis
#[derive(Debug)]
pub enum CostFunction {
    Quadratic
}

#[derive(Debug)]
pub struct Brain {
    cost: CostFunction,
    pub layers: Vec<Layer>,
    pub learn_rate: f64
}

impl Brain {
    pub fn new(cost: CostFunction, input_size: usize) -> Brain {
        Brain {
            cost,
            layers: vec![Layer::new(LayerKind::Identity, input_size, input_size)],
            learn_rate: 2.
        }
    }
    
    pub fn add_layer(&mut self, kind: LayerKind, output_size: usize) {
        let input_size = self.layers[self.layers.len()-1].size();
        self.layers.push(Layer::new(kind, input_size, output_size));
    }

    pub fn run(&mut self, data: &[f64]) -> Result<&[f64], &str> {
        if data.len() != self.layers[0].size() {
            return Err("Wrong input data size !");
        }
        self.layers[0].layer_results.copy_from_slice(data);
        // Per the current implementation, we do not require the first layer to hold any value,
        // it's thus useless right now
        self.layers.iter_mut().skip(1).fold(data, |acc, layer| {
            layer.run(acc)
        });
        Ok(&self.layers[self.layers.len()-1].layer_results)
    }

    pub fn cost(&mut self, data: &[f64], expected_result: &[f64]) -> f64 {
        self.run(data);
        let mut cost: f64 = 0.;
        let res = &self.layers[self.layers.len()-1].layer_results;
        for i in 0..res.len() {
            cost += (res[i] - expected_result[i]).powi(2);
        }
        1./2. * cost
    }

    pub fn backpropagation(&mut self, data: &[&[f64]], expected_result: &[&[f64]]) {
        let dataset_size = data.len();
        assert_eq!(dataset_size, expected_result.len());

        let mut delta: Vec<Vec<f64>> = Vec::with_capacity(self.layers.len());
        for layer in (1..self.layers.len()).rev() {
            delta.push(Vec::with_capacity(self.layers[layer].neurons.len()));
        }

        for e in 0..dataset_size {
            delta.clear();
            self.run(data[e]);
            for layer in (1..self.layers.len()).rev() {
                let mut delta_layer = Vec::with_capacity(self.layers[layer].neurons.len());
                for neuron in 0..self.layers[layer].neurons.len() {
                    delta_layer.push(
                        if layer == self.layers.len()-1 {
                            // compute ΔC
                            match self.cost {
                                CostFunction::Quadratic => {
                                   self.layers[layer].act_fun_derivative(self.layers[layer].neurons_results[neuron])*(self.layers[layer].layer_results[neuron]-expected_result[e][neuron])
                                }
                            }
                        } else {
                            let mut tmp = 0.;
                            for next_neuron in 0..self.layers[layer+1].neurons.len() {
                                tmp += delta[layer+1][next_neuron] * self.layers[layer+1].neurons[next_neuron].weights[neuron];
                            }
                            tmp * self.layers[layer].act_fun_derivative(self.layers[layer].neurons_results[neuron])
                    });
                }
                // TODO: get rid of this mess
                delta.push(delta_layer);
            }
        }
        for layer in (1..self.layers.len()).rev() {
            for neuron in 0..self.layers[layer].neurons.len() {
                for wk in 0..self.layers[layer].neurons[neuron].weights.len() {
                    self.layers[layer].neurons[neuron].weights[wk] -= self.learn_rate/(dataset_size as f64) * delta[layer][neuron] * self.layers[layer-1].layer_results[wk];
                }
                self.layers[layer].neurons[neuron].bias -= self.learn_rate/(dataset_size as f64) * delta[layer][neuron];
        }
    }
}
}
