use rand;

#[derive(Debug)]
pub struct Neuron {
    pub weights: Vec<f64>,
	pub bias: f64,
}

impl Neuron {
    pub fn new(input_size: usize) -> Neuron {
        let mut weights =  Vec::with_capacity(input_size);
        for _ in 0..input_size {
            weights.push(rand::random::<f64>());
        }

        Neuron {
            weights,
            bias: rand::random::<f64>()
        }
    }
}
