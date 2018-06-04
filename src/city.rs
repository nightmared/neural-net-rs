use std::io;
use std::io::prelude::*;
use std::fs::{Metadata, File, metadata};
use brain::Brain;
use tools::*;

use rand::{thread_rng, Rng};

#[derive(Clone, Debug)]
pub struct City {
    pub number: usize,
    pub images: Vec<Vec<f64>>,
    pub results: Vec<Vec<f64>>
}

impl City {
    pub fn new(data: &str) -> Result<City, io::Error> {
        let mut fd = File::open(data)?;
        let header_length = read_byte(&mut fd)? as usize;
        let in_len = read_4b(&mut fd)?;
        let out_len = read_2b(&mut fd)?;
        let length = metadata(data)?.len() as usize;
        let mut number_tot = (length - header_length)/(in_len + out_len);

        // finish reading the header
        let _ = read_4b(&mut fd)?;
        let _ = read_4b(&mut fd)?;

        let mut images = Vec::with_capacity(number_tot);
        let mut results = Vec::with_capacity(number_tot);
        let mut number = 0;
        for k in 0..number_tot {
            let mut v = Vec::with_capacity(in_len/3);
            let buffer = read_arr_u8(&mut fd, in_len)?;
            for i in 0..(in_len/3) {
                v.push((0.6*buffer[3*i] as f64 + 0.3*buffer[3*i+1] as f64 + 0.1*buffer[3*i+2] as f64)/256.);
            }
            let out = read_arr_u8(&mut fd, out_len)?;
            if out[2] == 127 {
                images.push(v);
                results.push(vec![out[0] as f64/256., out[1] as f64/256.]);
                number += 1;
            }
        }

        Ok(City {
            number,
            images,
            results
        })
    }

    pub fn measure_error(&self, network: &mut Brain) -> f64 {
        let mut rng = thread_rng();
        let mut sum = 0.;
        for _ in 0..20 {
            let index: usize = rng.gen::<usize>()%self.number;
            network.forward(&self.images[index]).unwrap();
            let mut val = 0;
            let outputs = network.get_outputs();
            for i in 1..outputs.len() {
                sum += ((outputs[i]*256. - self.results[index][i]*256.)*(outputs[i]*256. - self.results[index][i]*256.))/2.;
            }
        }
        sum / 20.
    }
}
