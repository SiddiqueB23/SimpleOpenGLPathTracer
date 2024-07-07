#version 430 core
out vec4 FragColor;

in vec2 TexCoords;

uniform sampler2D tex;
uniform float t;

void main()
{             
    vec3 texCol = texture(tex, TexCoords).rgb;      
    // FragColor = vec4(sqrt(texCol/t), 1.0);
    // FragColor = vec4(texCol/t, 1.0);
    FragColor = vec4(texCol, 1.0);
}

