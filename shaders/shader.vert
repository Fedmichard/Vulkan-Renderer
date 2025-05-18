#version 450

// we're going to taking in a vec2 for our vertex position and a vec3 for our rgb values
// inPosition and inColor 
// assigns indices to these inputs so we can reference them later
layout(location = 0) in vec2 inPosition;
layout(location = 1) in vec3 inColor;

// The vertex shader will output a fragment color to the next step of our graphics pipeline
layout(location = 0) out vec3 fragColor;


void main() {
    gl_Position = vec4(inPosition, 0.0, 1.0);
    fragColor = inColor;
}