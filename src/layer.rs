use neuron::Neuron;
use rand;

#[derive(Debug)]
pub enum LayerKind {
    // Input neurons
    Identity,
    // sigmoid activation function
    Linear
}

#[derive(Debug)]
pub struct Layer {
    kind: LayerKind,
	pub neurons: Vec<Neuron>,
    // temporary storage for results of the sum Σw*x+b
	pub neurons_results: Vec<f64>,
    // results : layer_results = act_fun(neurons_results) 
    pub layer_results: Vec<f64>
}

fn sigmoid(x: f64) -> f64 {
    1./(1.+f64::exp(-x))
}

fn vector_sum(previous_layer: &[f64], weights: &[f64], bias: f64) -> f64 {
    previous_layer.iter()
        .zip(weights.iter())
        .fold(bias, |acc, (&v, &w)| acc + v*w)
}

impl Layer {
    pub fn new(kind: LayerKind, input_size: usize, output_size: usize) -> Layer {
        let mut neurons = Vec::with_capacity(output_size);
        let mut layer_results = Vec::with_capacity(output_size);
        let mut neurons_results = Vec::with_capacity(output_size);
        for _ in 0..output_size {
            neurons.push(Neuron::new(input_size));
            layer_results.push(rand::random::<f64>());
            neurons_results.push(0.);
        }

        Layer {
            kind,
            neurons,
            neurons_results,
            layer_results
        }
    }

    pub fn size(&self) -> usize {
        self.neurons.len()
    }

    pub fn act_fun(&self, x: f64) -> f64 {
        match self.kind {
            LayerKind::Linear => { sigmoid(x) },
            LayerKind::Identity => { unreachable!() }
        }
    }

    pub fn act_fun_derivative(&self, x: f64) -> f64 {
        match self.kind {
            LayerKind::Linear => { sigmoid(x)*(1.-sigmoid(x)) },
            LayerKind::Identity => { unreachable!() }
        }
    }

    pub fn run(&mut self, previous_results: &[f64]) -> &[f64] {
        for i in 0..self.neurons.len() {
            self.neurons_results[i] = vector_sum(previous_results, &self.neurons[i].weights, self.neurons[i].bias);
            self.layer_results[i] = self.act_fun(self.neurons_results[i]);
        }
        &self.layer_results
    }
}
