use layer::Layer;
use std::fs::File;
use std::io::BufReader;
use std::io::prelude::*;
use std::cmp::max;

pub struct Brain {
	pub layer: Vec<Layer>,
    pub input_size: usize,
	pub inner_size: usize,
	pub output_size: usize,
	pub layer_size: usize,
    pub learn_rate: f64
}

impl Brain {
    pub fn new(input_size: usize, inner_size: usize, output_size: usize, layer_size: usize) -> Brain {
        let mut b = Brain {
            layer: Vec::with_capacity(layer_size),
            input_size,
            inner_size,
            output_size,
            layer_size,
            learn_rate: 0.5
        };
        b.layer.push(Layer::new(input_size, inner_size)); 
        for i in 1..layer_size-1 {
           b.layer.push(Layer::new(inner_size, inner_size)); 
        }
        b.layer.push(Layer::new(inner_size, output_size));
        b
    }

    pub fn load_from_file(dir: &str) -> Brain {
        let file = File::open(dir).unwrap();
        let mut buf_reader = BufReader::new(file);
        let mut text = String::new();
        buf_reader.read_to_string(&mut text).unwrap();
	    let lines: Vec<Vec<&str>> = text.lines().map(|x| x.split('/').collect()).collect();

        let mut b = Brain::new(lines[0][0].parse().unwrap(), lines[0][1].parse().unwrap(), lines[0][2].parse().unwrap(), lines[0][3].parse().unwrap());

        for i in 0..b.inner_size {
            for j in 0..b.input_size {
                b.layer[0].neurons[i].w[j] = lines[i+1][j].parse().unwrap();	//weigth
            }
            b.layer[0].neurons[i].bias = lines[i+1][b.inner_size].parse().unwrap();	//bias
        }

        for l in 1..b.layer_size-1 {	//set all innerNeuron weights biases
            for i in 0..b.inner_size {
                let line: &Vec<&str> = &lines[1+b.input_size+i+(l-1)*b.inner_size];
                for j in 0..b.inner_size {
                    b.layer[l].neurons[i].w[j] = line[j].parse().unwrap();	//weigth
                }
                b.layer[l].neurons[i].bias = line[b.inner_size].parse().unwrap();	//bias
            }
        }
        for i in 0..b.inner_size {	//set all outputNeuron weights biases
            let line: &Vec<&str> = &lines[1+b.input_size+b.inner_size*(b.layer_size-1)+i];
            for j in 0..b.output_size {
                b.layer[b.layer_size-1].neurons[i].w[j] = line[j].parse().unwrap();	//weight
            }
            b.layer[b.layer_size-1].neurons[i].bias = line[b.output_size].parse().unwrap();	//bias
        }
        b
    }

    fn set_learning_rate(&mut self, rate: f64) {
       self.learn_rate = rate; 
    }

    pub fn save(&self, dir: &str) {
        let len = 1+self.input_size+self.inner_size*(self.layer_size-2)+self.output_size;
        let mut file = File::create(dir).unwrap();

        write!(file, "{}/{}/{}/{}\n", self.input_size, self.inner_size, self.output_size, self.layer_size).unwrap();

        for i in 0..self.inner_size { // set all inputNeuron weights biases
            for j in 0..self.input_size {
                write!(file, "{}/", self.layer[0].neurons[i].w[j]).unwrap(); // weight
            }
            write!(file, "{}/\n", self.layer[0].neurons[i].bias).unwrap(); // bias
        }
        for l in 1..self.layer_size {	//set all innerNeuron weights biases
            for i in 0..self.inner_size {
                for j in 0..self.inner_size {
                    write!(file, "{}/", self.layer[l].neurons[i].w[j]).unwrap(); // weight
                }
                write!(file, "{}/\n", self.layer[l].neurons[i].bias).unwrap(); // bias
            }
        }
        for i in 0..self.output_size {	//set all outputNeuron weights
            for j in 0..self.inner_size {
                write!(file, "{}/", self.layer[self.layer_size-2].neurons[i].w[j]).unwrap(); // weight
            }
            write!(file, "{}/\n", self.layer[self.layer_size-2].neurons[i].bias).unwrap(); // bias
        }
    }

    pub fn run(&mut self, inputs: &[f64]) {
        let mut tmp = inputs;
        for l in self.layer.iter_mut() {
    		l.run(tmp);
            tmp = &l.neurons_results;
        }
    }

    pub fn backPropagation(&mut self, inputs: &[f64], target: &[f64]) {
        self.run(inputs);
        let mut error: Vec<f64> = Vec::with_capacity(self.output_size);
        let mut current_deriv = Vec::with_capacity(max(self.input_size, max(self.output_size, self.inner_size)));
        let mut previous_deriv = Vec::with_capacity(max(self.input_size, max(self.output_size, self.inner_size)));

        // derivative of the output layer
        for i in 0..self.output_size {
            let Si = self.layer[self.layer_size-1].neurons_results[i];
            let exp_sum = f64::exp(self.layer[self.layer_size-1].neurons[i].sum);
            // f = sigmoid'
            let fi = -exp_sum/((1.+exp_sum)*(1.+exp_sum));
            previous_deriv[i] = 2. * (Si - target[i]) * fi;
        }

        // apply it to the weights
        for cur_layer in (0..self.layer_size-1).rev() {
            for neuron in 0..self.layer[cur_layer].neurons.len() {
                let previous_layer = if cur_layer == 0 { inputs } else { &self.layer[cur_layer-1].neurons_results };
                current_deriv.clear();
                for previous_neuron in 0..previous_layer.len() {
                    current_deriv.push(self.learn_rate * previous_deriv[neuron] * previous_layer[previous_neuron]);
                    self.layer[cur_layer].neurons[neuron].w[previous_neuron] -= current_deriv[neuron];
                }
            }
            let tmp = current_deriv;
            current_deriv = previous_deriv;
            previous_deriv = tmp;
        }
        // TODO: handle biases
    /*
        // first iteration weights
        int l = b.layer_size-2; // before last layer
        for (int i=0; i<b.layer[l].inLength; i++) {
            for (int j=0; j<b.layer[l].outLength; j++) {
                double neuron_out = b.layer[l].outNeuron[j].output;
                struct neuron* ni = b.layer[l].neurons[i];
                b.error += abs(target[j] - neuron_out);
                ni.wVar[j] = (neuron_out - target[j])*neuron_out*(1-neuron_out)*ni.output;
                ni.wVarTotal[j] += ni.wVar[j];
            }
        }
        // first iteration biases
        l = b.layer_size-1;
        for (int i=0; i<b.layer[l].inLength; i++) {
            struct neuron* ni = b.layer[l].neurons[i];
            ni.bVar = (ni.output - target[i])*ni.output*(1-ni.output);
            ni.bVarTotal += ni.bVar;
        }
        // recurring iteration
        for (l=b.layer_size-3; l>=0; l--) {
            layer_backPropW(b.layer[l]);
        }
        for (l=b.layer_size-2; l>=0; l--) {
            layer_backPropB(b.layer[l]);
        }
    */
    }
}
/*
void brain_learn(struct brain* b, char* dir) {
	brain_loadData(b, dir);
	// learn

	struct timespec start;
	clock_gettime(CLOCK_MONOTONIC, &start);

	double eps = (double)(b.outputSize*b.inDataLength)/(2.*data_out_length);
	double lastError = eps+1;
	int batchNumber = b.input_size/(batchSize*1.0)+1;
	printf("batch number: %i\n", batchNumber);
	b.error = eps+1;
	printf("Start learning with a learning rate of: %f\nDataSet size: %i\nEPS = %f\n", learnRate, b.inDataLength, eps);
	int c = 0;
	while (b.error > eps) {	// learn loop
		if (c%30000== 1) {
			printf("Error: %f - learnRate: %f\n", b.error, learnRate);
		}
		b.error = 0;
		for (int i=0; i<batchNumber; i++) {	// batch itr
			int s;
			int d = batchSize*(i%batchNumber);
			for (s=d; s<b.inDataLength && s<d+batchSize; s++) {
				brain_run(b, b.inData[s]);
				brain_backPropagation(b, b.outData[s]);
			}
			if (s != d) {
				// weight and biases changes
				for (int l=0; l<b.layer_size-1; l++) {
					layer_varChange(b.layer[l], s-d);
				}
			}
		}
		if (c%1000 == 1) {
			if (lastError == b.error) {
				// the error stagnates, time to break the loop
				break;
			}
			lastError = b.error;
		}
		c++;
	}

	struct timespec stop;
	clock_gettime(CLOCK_MONOTONIC, &stop);

	printf("Final Error: %f\n", b.error);
	printf("It took locally %f ms\n", (double)(stop.tv_sec-start.tv_sec)*.001/c);
	printf("It took %f s and %i dataset iterations\n", (double)(stop.tv_sec-start.tv_sec), (int)c);
	int test = 0;
	for (int s=0; s<b.inDataLength; s++) {
		brain_run(b, b.inData[s]);
		if (brain_outToInt(b) == brain_outDataToInt(b, s)) {
			test++;
		}
	}
	printf("%f%% de reussite\n", test*100./b.inDataLength);
}


void brain_backPropagation(struct brain* b, double* target) {
	double* error = malloc(sizeof(double)*b->output_size);
	double* current_derivative = malloc(sizeof(double)*MAX(b->output_size, b->inner_size));
	double* previous_derivative = malloc(sizeof(double)*MAX(b->output_size, b->inner_size));
	if (error == NULL || current_derivative == NULL || previous_derivative == NULL)
		exit(1);

	// derivative of the output layer
	for (int i = 0; i < b->output_size; i++) {
		double Si = b->layer[b->layer_size-1].neurons_results[i];
		double exp_sum = exp(b->layer[b->layer_size-1].neurons[i].sum);
		// f = sigmoid'
		double fi = -exp_sum/((1+exp_sum)*(1+exp_sum));
		previous_derivative[i] = 2 * (Si - target[i]) * fi;
	}

	// apply it to the weights
	for (int cur_layer = b->layer_size-1; cur_layer > -1; cur_layer--) {
		for (int neuron = 0; neuron < b->layer[cur_layer].length; neuron++) {
			struct layer* previous_layer = b->layer[cur_layer].previous_layer;
			for (int previous_neuron = 0; previous_neuron < previous_layer->length; previous_neuron++) {
				current_derivative[neuron] = learnRate * previous_derivative[neuron] * b->layer[cur_layer].previous_layer->neurons_results[previous_neuron];
				b->layer[cur_layer].neurons[neuron].w[previous_neuron] -= current_derivative[neuron];
			}
		}
		double* tmp = current_derivative;
		current_derivative = previous_derivative;
		previous_derivative = tmp;
	}
	// TODO: handle biases
/*
	// first iteration weights
	int l = b.layer_size-2; // before last layer
	for (int i=0; i<b.layer[l].inLength; i++) {
		for (int j=0; j<b.layer[l].outLength; j++) {
			double neuron_out = b.layer[l].outNeuron[j].output;
			struct neuron* ni = b.layer[l].neurons[i];
			b.error += abs(target[j] - neuron_out);
			ni.wVar[j] = (neuron_out - target[j])*neuron_out*(1-neuron_out)*ni.output;
			ni.wVarTotal[j] += ni.wVar[j];
		}
	}
	// first iteration biases
	l = b.layer_size-1;
	for (int i=0; i<b.layer[l].inLength; i++) {
		struct neuron* ni = b.layer[l].neurons[i];
		ni.bVar = (ni.output - target[i])*ni.output*(1-ni.output);
		ni.bVarTotal += ni.bVar;
	}
	// recurring iteration
	for (l=b.layer_size-3; l>=0; l--) {
		layer_backPropW(b.layer[l]);
	}
	for (l=b.layer_size-2; l>=0; l--) {
		layer_backPropB(b.layer[l]);
	}
*/
}

void brain_randomize(struct brain* b) {
	srand(time(NULL));
	for (int l=0; l<b->layer_size; l++) {
		for (int i=0; i<b->layer[l].input_length; i++) {
			for (int j=0; j<b->layer[l].length; j++) {
				b->layer[l].neurons[i].w[j] = RANDOM_WEIGHT();
			}
			b->layer[l].neurons[i].bias = RANDOM_WEIGHT();
		}
	}
}




*/
