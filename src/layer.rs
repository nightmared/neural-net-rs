use rand::{thread_rng, Rng};

pub trait Layer {
    fn new(input_size: usize, output_size: usize) -> Self;
    fn size(&self) -> usize;
    fn act_fun(f64) -> f64;
    fn act_fun_derivative(f64) -> f64;
    fn forward(&mut self, &[f64]) -> &[f64];
    fn get_outputs(&self) -> &[f64];
    fn get_potentials(&self) -> &[f64];
    fn get_weight(&self, from: usize, to: usize) -> f64;
    fn get_bias(&self, neuron: usize) -> f64;
    fn set_weight(&mut self, from: usize, to: usize, val: f64);
    fn set_bias(&mut self, neuron: usize, val: f64);
    fn add_weight(&mut self, from: usize, to: usize, val: f64);
    fn add_bias(&mut self, neuron: usize, val: f64);
}

#[derive(Debug, Clone)]
pub struct Linear {
    pub input_size: usize,
    pub length: usize,
    // Matrix of all the weights
    // wij = weights[i*self.input_size+j]
    pub weights: Vec<f64>,
    pub bias: Vec<f64>,
    // temporary storage for results of the sum Σw*x+b
    pub potentials: Vec<f64>,
    // after activation
    pub outputs: Vec<f64>
}

impl Layer for Linear {
    fn new(input_size: usize, output_size: usize) -> Linear {
        let mut rng = thread_rng();
        let mut weights = vec![0.; output_size*input_size];
        let mut bias = vec![0.; output_size];
        let potentials = vec![0.; output_size];
        let outputs = vec![0.; output_size];
        for i in 0..weights.len() {
            weights[i] = rng.gen::<f64>() - 0.5;
        }
        for i in 0..bias.len() {
            bias[i] = rng.gen::<f64>() - 0.5;
        }
        Linear {
            input_size,
            length: output_size,
            weights,
            bias,
            potentials,
            outputs
        }
    }
    fn size(&self) -> usize {
        self.length
    }
    fn act_fun(x: f64) -> f64 {
        // sigmoid
        1./(1.+f64::exp(-x))
    }
    fn act_fun_derivative(x: f64) -> f64 {
        let exp = f64::exp(-x);
        exp/(1. + exp)/(1. + exp)
    }
    fn forward(&mut self, previous_layer: &[f64]) -> &[f64] {
        for i in 0..self.length {
            self.potentials[i] = vector_dot(previous_layer, &self.weights[i*self.input_size..(i+1)*self.input_size], self.bias[i]);
            self.outputs[i] = Self::act_fun(self.potentials[i]);
        }
        &self.outputs
    }
    #[inline]
    fn get_outputs(&self) -> &[f64] {
        &self.outputs
    }
    #[inline]
    fn get_potentials(&self) -> &[f64] {
        &self.potentials
    }
    #[inline]
    fn get_weight(&self, from: usize, to: usize) -> f64 {
       self.weights[to*self.input_size+from] 
    }
    #[inline]
    fn get_bias(&self, neuron: usize) -> f64 {
        self.bias[neuron]
    }
    #[inline]
    fn set_weight(&mut self, from: usize, to: usize, val: f64) {
       self.weights[to*self.input_size+from] = val;
    }
    #[inline]
    fn set_bias(&mut self, neuron: usize, val: f64) {
        self.bias[neuron] = val;
    }
    #[inline]
    fn add_weight(&mut self, from: usize, to: usize, val: f64) {
       self.weights[to*self.input_size+from] += val;
    }
    #[inline]
    fn add_bias(&mut self, neuron: usize, val: f64) {
        self.bias[neuron] += val;
    }
}


#[inline]
fn vector_dot(previous_layer: &[f64], weights: &[f64], bias: f64) -> f64 {
    let mut sum = 0.;
    for i in 0..previous_layer.len() {
        sum += previous_layer[i] * weights[i];
    }
    sum + bias
}
