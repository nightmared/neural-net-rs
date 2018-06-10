use std::{io, mem};
use std::io::{Write, Read};
use std::fs::File;

// Ugh, let's not talk to anyone about all this, right ?
// Also: do not ever, ever, EVER call this function with a value whose size is not 8 bytes (beware
// of 32 bit platforms !)...
pub fn write_dq(fd: &mut File, val: usize) -> Result<(), io::Error> {
    fd.write_all(unsafe { mem::transmute::<_, &[u8; 8]>(&[val])})?;
    Ok(())
}

pub fn write_arr(fd: &mut File, arr: &[f64]) -> Result<(), io::Error> {
    let mut new_arr = Vec::with_capacity(arr.len()*8);
    for e in arr {
        new_arr.extend_from_slice(unsafe { mem::transmute::<&f64, &[u8; 8]>(e)});
    }
    fd.write_all(new_arr.as_slice())?;
    Ok(())
}

pub fn read_byte(fd: &mut File) -> Result<u8, io::Error> {
    let mut buffer = [0];
    fd.read_exact(&mut buffer)?;
    Ok(buffer[0])
}

pub fn read_dq(fd: &mut File) -> Result<usize, io::Error> {
    let mut buffer = [0; 8];
    fd.read_exact(&mut buffer)?;
    Ok(unsafe { mem::transmute::<_, usize>(buffer) })
}

pub fn read_arr(fd: &mut File, size: usize) -> Result<Vec<f64>, io::Error> {
    let mut data: Vec<u8> = vec![0; size*8];
    fd.read_exact(data.as_mut_slice())?;
    let mut vec = Vec::with_capacity(size);
    for i in 0..size {
        vec.push(unsafe { mem::transmute::<&[u8], &[f64]>(&data[i*8..(i+1)*8])}[0]);
    }
    Ok(vec)
}

pub fn read_arr_u8(fd: &mut File, size: usize) -> Result<Vec<u8>, io::Error> {
    let mut data: Vec<u8> = vec![0; size];
    fd.read_exact(data.as_mut_slice())?;
    Ok(data)
}

pub fn read_2b(fd: &mut File) -> Result<usize, io::Error> {
    let a = read_byte(fd)? as usize;
    let b = read_byte(fd)? as usize;
    Ok((a<<8)+b)
}

pub fn read_4b(fd: &mut File) -> Result<usize, io::Error> {
    let a = read_byte(fd)? as usize;
    let b = read_byte(fd)? as usize;
    let c = read_byte(fd)? as usize;
    let d = read_byte(fd)? as usize;
    Ok((a<<24)+(b<<16)+(c<<8)+d)
}
