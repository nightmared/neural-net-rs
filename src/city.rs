use std::io;
use std::io::prelude::*;
use std::fs::{Metadata, File, metadata};
use brain::Brain;
use tools::*;
use rand::{ThreadRng, Rng};

#[derive(Clone, Debug)]
pub struct City {
    pub number: usize,
    pub img_len: usize,
    pub out_len: usize,
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
        let img_len = (in_len as f64).sqrt() as usize;

        // finish reading the header
        let _ = read_4b(&mut fd)?;
        let _ = read_4b(&mut fd)?;

        let mut images = Vec::with_capacity(number_tot);
        let mut results = Vec::with_capacity(number_tot);
        let mut number = 0;
        for _ in 0..number_tot {
            let buffer = read_arr_u8(&mut fd, img_len*img_len)?;
            let out = read_arr_u8(&mut fd, out_len)?;
            if out[2] as usize == img_len - 1 {
                images.push(buffer.iter().map(|&x| x as f64/256.).collect());
                results.push(vec![out[0] as f64/256., out[1] as f64/256.]);
                number += 1;
            }
        }

        println!("New city with in_len={},out_len={},num_images={},img_size={}", in_len, out_len, number_tot, img_len);

        Ok(City {
            number,
            img_len,
            out_len: out_len - 1,
            images,
            results
        })
    }

    pub fn measure_error(&self, network: &mut Brain, rng: &mut ThreadRng) -> f64 {
        let mut sum = 0.;
        for _ in 0..50 {
            let index: usize = rng.gen::<usize>()%self.number;
            network.forward(&self.images[index]).unwrap();
            let outputs = network.get_outputs();
            for i in 0..outputs.len() {
                sum += ((outputs[i] - self.results[index][i]).powi(2))/2.;
            }
        }
        sum / 50.
    }
}
