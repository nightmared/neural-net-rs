pub fn cost(outputs: &[f64], expected_result: &[f64]) -> f64 {
    let mut cost: f64 = 0.;
    for i in 0..outputs.len() {
        cost += (outputs[i] - expected_result[i]).powi(2);
    }
    1./2. * cost
}
pub fn cost_derivative(outputs: &[f64], expected_result: &[f64], neuron_index: usize) -> f64 {
    outputs[neuron_index] - expected_result[neuron_index]
}
