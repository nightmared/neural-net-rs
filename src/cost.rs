pub trait CostFunction {
    fn cost(outputs: &[f64], expected_result: &[f64]) -> f64;
	// derivative of the error relative to some neuron output
	fn cost_derivative(outputs: &[f64], expected_result: &[f64], neuron_index: usize) -> f64;
}

pub struct Quadratic {}

impl CostFunction for Quadratic {
	fn cost(outputs: &[f64], expected_result: &[f64]) -> f64 {
		let mut cost: f64 = 0.;
        for i in 0..outputs.len() {
            cost += (outputs[i] - expected_result[i]).powi(2);
        }
        1./2. * cost
	}
	#[inline]
	fn cost_derivative(outputs: &[f64], expected_result: &[f64], neuron_index: usize) -> f64 {
		outputs[neuron_index] - expected_result[neuron_index]
	}
}
