#version 450

// binding describes where in a descriptor set the uniform buffer will be in
// uniform means constant, they will remain constant for all shaders during a single draw call
// this is just a struct that holds all the uniform values that will be used throughout the entire rendering process
layout(binding = 0) uniform UniformBufferObject {
    mat4 model;
    mat4 view;
    mat4 proj;
} ubo;

// we're going to taking in a vec2 for our vertex position and a vec3 for our rgb values
// inPosition and inColor 
// assigns indices to these inputs so we can reference them later
layout(location = 0) in vec2 inPosition;
layout(location = 1) in vec3 inColor;

// The vertex shader will output a fragment color to the next step of our graphics pipeline
layout(location = 0) out vec3 fragColor;


void main() {
    gl_Position = ubo.proj * ubo.view * ubo.model * vec4(inPosition, 0.0, 1.0);
    fragColor = inColor;
}