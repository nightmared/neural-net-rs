extern crate glutin;
use gl::types::*;
use std::fs::File;
use std::io::Read;
use std::ffi::CString;
use std::os::raw::c_void;
use std::ptr;
use std::mem;
use glutin::GlContext;
use city::City;
use brain::Brain;
use rand::{Rng, ThreadRng};


fn shader_cc(fname: &str, shader_type: GLenum) -> Result<GLuint, &'static str> {
	// read shader
    let mut f = File::open(fname).unwrap();
	let mut buffer = String::new();
    f.read_to_string(&mut buffer).unwrap();

	// Compile it
	let mut test: GLint = gl::FALSE as i32;
	let shader: GLuint;
    unsafe {
	    shader = gl::CreateShader(shader_type);
	    gl::ShaderSource(shader, 1, &CString::new(buffer).unwrap().as_ptr(), ptr::null());
	    gl::CompileShader(shader);
		// has compilation failed ?
    	gl::GetShaderiv(shader, gl::COMPILE_STATUS, &mut test);
    }

	if test != gl::TRUE as i32 {
		return Err("Shader compilation failed");
	}

    Ok(shader)
}

fn new_prog(path_vs: &str, path_fs: &str) -> GLuint {
	let vs: GLuint = shader_cc(path_vs, gl::VERTEX_SHADER).expect("vertex shader should compile");
	let fs: GLuint = shader_cc(path_fs, gl::FRAGMENT_SHADER).expect("fragment shader should compile");

	let program: GLuint;
	unsafe {
		program = gl::CreateProgram();

		gl::AttachShader(program, vs);
		gl::AttachShader(program, fs);

		//gl::DeleteShader(vs);
		//gl::DeleteShader(fs);

		gl::LinkProgram(program);
		gl::UseProgram(program);
	}
	program
}

pub struct Gui {
	prog: GLuint,
	tex: GLuint,
    img0: Vec<u32>,
    img1: Vec<u32>,
    img_len: i32
}

static VERTEX_DATA: [f32; 32] = [
	0., 1., 1.0, 0.0,
	0., -1., 1.0, 1.0,
	-1., -1., 0.0, 1.0,
	-1., 1., 0.0, 0.0,
	1., 1., 1.0, 0.0,
	1., -1., 1.0, 1.0,
	0., -1., 0.0, 1.0,
	0., 1., 0.0, 0.0
];

static elements: [GLuint; 12] = [
    0, 1, 2, 2, 3, 0, 4, 5, 6, 6, 7, 4
];

impl Gui {
    pub fn new(gl_window: &glutin::GlContext,  img_len: i32) -> Gui {
        let tmp = vec![255 as u32; (img_len*img_len) as usize];
		gl::load_with(|ptr| gl_window.get_proc_address(ptr) as *const _);
        let shader_prog = new_prog("./src/shaders/vertex.shader", "./src/shaders/fragment.shader");
		let tex = unsafe {
            gl::BindFragDataLocation(shader_prog, 0, b"outColor\0".as_ptr() as *const _);
			let mut vao = mem::uninitialized();
            gl::GenVertexArrays(1, &mut vao);
            gl::BindVertexArray(vao);

			let mut vbo = mem::uninitialized();
			gl::GenBuffers(1, &mut vbo);
			gl::BindBuffer(gl::ARRAY_BUFFER, vbo);
			gl::BufferData(gl::ARRAY_BUFFER, (VERTEX_DATA.len() * mem::size_of::<f32>()) as gl::types::GLsizeiptr,
				VERTEX_DATA.as_ptr() as *const _, gl::STATIC_DRAW);

            let position_vertex = gl::GetAttribLocation(shader_prog, b"position\0".as_ptr() as *const _) as u32;
            gl::EnableVertexAttribArray(position_vertex);
            gl::VertexAttribPointer(position_vertex, 2, gl::FLOAT, gl::FALSE, 4 * mem::size_of::<f32>() as i32, ptr::null());

			let position_tex = gl::GetAttribLocation(shader_prog, b"texcoord\0".as_ptr() as *const _) as u32;
			gl::EnableVertexAttribArray(position_tex);
			gl::VertexAttribPointer(position_tex, 2, gl::FLOAT, gl::FALSE, 4 * mem::size_of::<f32>() as i32, (2 * mem::size_of::<f32>()) as *const _);

			let mut ebo = mem::uninitialized();
            gl::GenBuffers(1, &mut ebo);
            gl::BindBuffer(gl::ELEMENT_ARRAY_BUFFER, ebo);
            gl::BufferData(gl::ELEMENT_ARRAY_BUFFER, (elements.len() * mem::size_of::<f32>()) as gl::types::GLsizeiptr,
                elements.as_ptr() as *const _, gl::STATIC_DRAW);

			let mut tex: GLuint = mem::uninitialized();
			gl::GenTextures(1, &mut tex as *mut u32);

			gl::BindTexture(gl::TEXTURE_2D, tex);
			gl::TexParameteri(gl::TEXTURE_2D, gl::TEXTURE_MIN_FILTER, gl::NEAREST as i32);
			gl::TexParameteri(gl::TEXTURE_2D, gl::TEXTURE_MAG_FILTER, gl::NEAREST as i32);
            gl::TexParameteri(gl::TEXTURE_2D, gl::TEXTURE_WRAP_S, gl::CLAMP_TO_EDGE as i32);
            gl::TexParameteri(gl::TEXTURE_2D, gl::TEXTURE_WRAP_T, gl::CLAMP_TO_EDGE as i32);
			gl::TexImage2D(gl::TEXTURE_2D, 0, gl::RGBA as i32, img_len, img_len, 0, gl::RGBA, gl::UNSIGNED_INT_8_8_8_8, tmp.as_ptr() as *const c_void);
            gl::Uniform1i(gl::GetUniformLocation(shader_prog, b"tex\0".as_ptr() as *const _), 0);
			tex
		};
        Gui { tex, prog: shader_prog, img0: tmp.clone(), img1: tmp, img_len }
    }

    pub fn update_img(&mut self, img: Vec<u32>, pos: usize) {
		if pos == 0 {
			self.img0 = img;
		} else {
			self.img1 = img;
		}
    }

	pub fn draw_rectangle(&mut self, pos: usize, x: f64, y: f64, color: u32) {
		let (mut x, mut y) = ((x*self.img_len as f64).floor() as usize, (y*self.img_len as f64).floor() as usize);
		let mut img = if pos == 0 {
			&mut self.img0
		} else {
			&mut self.img1
		};
        if (x+1)*self.img_len as usize+y+1 >= (self.img_len*self.img_len) as usize {
            x = self.img_len as usize-2;
            y = self.img_len as usize-2;
        }
		img[x*self.img_len as usize+y] = (color<<8)+255;
		img[(x+1)*self.img_len as usize+y] = (color<<8)+255;
		img[x*self.img_len as usize+y+1] = (color<<8)+255;
		img[(x+1)*self.img_len as usize+y+1] = (color<<8)+255;
	}

    pub fn redraw(&mut self) {
        unsafe {
            gl::ClearColor(0.0, 0.0, 1.0, 1.0);
			gl::Clear(gl::COLOR_BUFFER_BIT);
			gl::TexImage2D(gl::TEXTURE_2D, 0, gl::RGBA as i32, self.img_len, self.img_len, 0, gl::RGBA, gl::UNSIGNED_INT_8_8_8_8, self.img0.as_ptr() as *const c_void);
			gl::DrawElements(gl::TRIANGLES, 6, gl::UNSIGNED_INT, 0 as *const _);
			gl::TexImage2D(gl::TEXTURE_2D, 0, gl::RGBA as i32, self.img_len, self.img_len, 0, gl::RGBA, gl::UNSIGNED_INT_8_8_8_8, self.img1.as_ptr() as *const c_void);
			gl::DrawElements(gl::TRIANGLES, 6, gl::UNSIGNED_INT, 24 as *const _);
        }
    }
}

fn display_img(brain: &mut Brain, gui: &mut Gui, dataset: &City, pos: usize, idx: usize) {
    let mut img: Vec<u32> = dataset.images[idx].iter().map(|&x| { let y = (x*256.) as u32; (y<<24)+(y<<16)+(y<<8)+255}).collect();
    gui.update_img(img.clone(), pos);
    gui.draw_rectangle(pos, dataset.results[idx][0], dataset.results[idx][1], 255<<8);
    brain.forward(&dataset.images[idx]);
    let out = brain.get_outputs();
    gui.draw_rectangle(pos, out[0], out[1], 255<<16);
}

pub fn show_gui(mut brain: &mut Brain, train: &City, test: &City, rng: &mut ThreadRng) {
    let mut events_loop = glutin::EventsLoop::new();
    let window = glutin::WindowBuilder::new()
        .with_title("TIPE")
        .with_always_on_top(true)
        .with_dimensions(1000, 500);
    let context = glutin::ContextBuilder::new();
    let gl_window = glutin::GlWindow::new(window, context, &events_loop).unwrap();
    let _ = unsafe { gl_window.make_current() };
	let mut gui = Gui::new(&gl_window, 128);
	let mut running = true;
	while running {
		events_loop.poll_events(|event| {
			match event {
				glutin::Event::WindowEvent { event, .. } => match event {
					glutin::WindowEvent::CloseRequested => running = false,
					glutin::WindowEvent::Resized(w, h) => gl_window.resize(w, h),
					glutin::WindowEvent::KeyboardInput { device_id, input } => {
                        if input.state == glutin::ElementState::Pressed {
                            if let Some(glutin::VirtualKeyCode::Numpad0) = input.virtual_keycode {
                                let mut idx = rng.gen::<usize>()%train.number;
                                display_img(&mut brain, &mut gui, &train, 0, idx);
                            } else if let Some(glutin::VirtualKeyCode::Numpad1) = input.virtual_keycode {
                                let mut idx = rng.gen::<usize>()%test.number;
                                display_img(&mut brain, &mut gui, &test, 1, idx);
                            }
                       }
                    },
					_ => ()
				},
				_ => (),
			}
		});
		gui.redraw();
        let _ = gl_window.swap_buffers();
    }
}
