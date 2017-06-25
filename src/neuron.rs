fn sigmoid(x: f64) -> f64 {
	1./(1.+f64::exp(-x))
}

fn vector_sum(previous_layer: &[f64], weights: &[f64]) -> f64 {
    previous_layer.iter()
        .zip(weights.iter())
        .fold(0., |acc, (&v, &w)| acc + v*w)
}

pub struct Neuron {
	pub output_size: usize,
    pub w: Vec<f64>,
	pub bias: f64,
    pub sum: f64,
	pub adder: fn(previous_layer: &[f64], weights: &[f64]) -> f64,
	pub aggregation: fn(sum: f64) -> f64
}

impl Neuron {
    pub fn new(input_size: usize, output_size: usize) -> Neuron {
        let mut n = Neuron {
            output_size,
            w: Vec::with_capacity(input_size),
            // TODO: randomize
            bias: 0.,
            sum: 0.,
            adder: vector_sum,
            aggregation: sigmoid
        };
        for i in 0..input_size {
            // TODO: randomize
            n.w.push(0.);
        }
        n
    }
    pub fn compute(&mut self, previous_results: &[f64]) -> f64 {
        self.sum = (self.adder)(previous_results, &self.w);
        (self.aggregation)(self.sum)
    }
}
