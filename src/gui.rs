extern crate glutin;
use gl::types::*;
use std::fs::File;
use std::io::Read;
use std::ffi::CString;
use std::os::raw::c_void;
use std::ptr;
use std::mem;


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
	tex: GLuint
}

static VERTEX_DATA: [f32; 16] = [
	-0.5, -0.5, 0.0, 0.0,
	0.0, 0.5, 1.0, 0.0,
	0.5, -0.5, 1.0, 1.0,
	-0.5, -0.5, 0.0, 1.0
];

static elements: [GLuint; 6] = [
    0, 1, 2, 2, 3, 0
];

impl Gui {
    pub fn new(gl_window: &glutin::GlContext, img: &[f64]) -> Gui {
		gl::load_with(|ptr| gl_window.get_proc_address(ptr) as *const _);
        let shader_prog = new_prog("./src/shaders/vertex.shader", "./src/shaders/fragment.shader");
		let tex = unsafe {
			let mut vbo = mem::uninitialized();
			gl::GenBuffers(1, &mut vbo);
			gl::BindBuffer(gl::ARRAY_BUFFER, vbo);
			gl::BufferData(gl::ARRAY_BUFFER, (VERTEX_DATA.len() * mem::size_of::<f32>()) as gl::types::GLsizeiptr,
				VERTEX_DATA.as_ptr() as *const _, gl::STATIC_DRAW);
			let mut vao = mem::uninitialized();
            gl::GenVertexArrays(1, &mut vao);
            gl::BindVertexArray(vao);

            let position_vertex = gl::GetAttribLocation(shader_prog, b"position\0".as_ptr() as *const _) as u32;
            gl::EnableVertexAttribArray(position_vertex);
            gl::VertexAttribPointer(position_vertex, 2, gl::FLOAT, gl::FALSE, 5 * mem::size_of::<f32>() as i32, ptr::null());

			let position_tex = gl::GetAttribLocation(shader_prog, b"texcoord\0".as_ptr() as *const _) as u32;
			gl::EnableVertexAttribArray(position_tex);
			gl::VertexAttribPointer(position_tex, 2, gl::FLOAT, gl::FALSE, 5 * mem::size_of::<f32>() as i32, (2 * mem::size_of::<f32>()) as *const _);

			let mut ebo = mem::uninitialized();
            gl::GenBuffers(1, &mut ebo);
            gl::BindBuffer(gl::ELEMENT_ARRAY_BUFFER, ebo);
            gl::BufferData(gl::ELEMENT_ARRAY_BUFFER, (elements.len() * mem::size_of::<f32>()) as gl::types::GLsizeiptr,
                elements.as_ptr() as *const _, gl::STATIC_DRAW);

			let mut tex = mem::uninitialized();
			gl::GenTextures(1, &mut tex);
            gl::ActiveTexture(gl::TEXTURE0);
			gl::BindTexture(gl::TEXTURE_2D, tex);
			gl::TexParameteri(gl::TEXTURE_2D, gl::TEXTURE_MAG_FILTER, gl::NEAREST as i32);
			gl::TexImage2D(gl::TEXTURE_2D, 0, gl::RED as i32, 28, 28, 0, gl::RED_INTEGER, gl::UNSIGNED_BYTE, img as *const [f64] as *const c_void);
			tex
		};
        Gui { tex, prog: shader_prog }
    }

    pub fn redraw(&mut self, img: &[f64]) {
        unsafe {
            gl::ClearColor(0.0, 0.0, 1.0, 1.0);
			gl::Clear(gl::COLOR_BUFFER_BIT);
			gl::DrawElements(gl::TRIANGLES, 6, gl::UNSIGNED_INT, 0 as *const _);
        }
    }
}
