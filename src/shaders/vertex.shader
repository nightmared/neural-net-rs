#version 150 core

attribute vec2 position;
attribute vec2 texcoord;
out vec2 Texcoord;

void main() {
    gl_Position = vec4(position, 0.0, 1.0);
    Texcoord = texcoord;
}
