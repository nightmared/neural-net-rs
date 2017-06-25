use neuron::Neuron;

pub struct Layer {
	pub neurons: Vec<Neuron>,
	pub neurons_results: Vec<f64>
}

impl Layer {
    pub fn new(input_size: usize, output_size: usize) -> Layer {
        let mut l = Layer {
            neurons: Vec::with_capacity(output_size),
            neurons_results: Vec::with_capacity(output_size)
        };
        for i in 0..output_size {
            l.neurons.push(Neuron::new(input_size, output_size));
            l.neurons_results.push(0.);
        }
        l
    }
    pub fn run(&mut self, previous_results: &[f64]) {
        self.neurons_results.clear();
        for n in self.neurons.iter_mut() {
            self.neurons_results.push(n.compute(previous_results));
        }
    }
}
