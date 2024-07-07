#version 430 core

layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;

layout(rgba32f, binding = 0) uniform image2D imgOutput;

uniform float frameCount;
uniform float vfov;
uniform float defocus_angle;
uniform vec3 lookfrom; // Point camera is looking from
uniform vec3 lookat; // Point camera is looking at

float focal_length = 1.0;
vec3 vup = vec3(0, 1, 0); // Camera-relative "up" direction

struct Sphere {
    vec3 pos;
    float r;
    int materialIndex;
};
layout(std430, binding = 3) buffer ssbo1
{
    Sphere S[];
};
uniform int sphereCount;

struct Material {
    vec4 albedo;
    vec4 emissive_colour;
    float emission_strength;
    int type;
    float smoothness;
    float ri;
};
layout(std430, binding = 4) buffer ssbo2
{
    Material M[];
};
// Material M[5];

struct Ray {
    vec3 origin;
    vec3 dir;
};

const int maxBounces = 10;

uint rng_state;
uint wang_hash(uint seed)
{
    seed = (seed ^ 61) ^ (seed >> 16);
    seed *= 9;
    seed = seed ^ (seed >> 4);
    seed *= 0x27d4eb2d;
    seed = seed ^ (seed >> 15);
    return seed;
}
uint rand_xorshift() {
    // Xorshift algorithm from George Marsaglia's paper
    rng_state ^= (rng_state << 13);
    rng_state ^= (rng_state >> 17);
    rng_state ^= (rng_state << 5);
    return rng_state;
}
float random_float() {
    return float(rand_xorshift()) * (1.0 / 4294967296.0);
}
vec3 random_vec3() {
    float costheta = float(rand_xorshift()) * (1.0 / 4294967296.0) * 2.0 - 1.0;
    float theta = acos(costheta);
    float phi = float(rand_xorshift()) * (1.0 / 4294967296.0) * 6.28318531;
    return vec3(sin(theta) * cos(phi), sin(theta) * sin(phi), cos(theta));
}
vec2 random_in_disk() {
    vec2 v;
    for (int i = 0; i < 10; i++) {
        v = vec2(random_float(), random_float()) * 2.0 - 1.0;
        if (dot(v, v) < 1.0) {
            return v;
        }
    }
    return vec2(0.0, 0.0);
}

Ray getStartRay() {
    // unit vectors in camera space
    vec3 u, v, w;
    w = normalize(lookat - lookfrom);
    u = normalize(cross(w, vup));
    v = cross(u, w);

    //focal plane lies at look from and is perpendicular to vec3(lookat - lookfrom)
    focal_length = length(lookat - lookfrom);
    float defocus_radius = focal_length * tan(radians(defocus_angle / 2.0));

    //getting texture space coordinates and transforming to points on focal plane
    ivec2 texelCoord = ivec2(gl_GlobalInvocationID.xy);
    vec2 p = vec2(
            (gl_GlobalInvocationID.x - 640.0),
            (gl_GlobalInvocationID.y - 360.0)
        );
    p *= tan(radians(vfov / 2.0)) * focal_length / 360.0;
    vec2 rand_disk = random_in_disk() * defocus_radius;
    vec3 ray_o = lookfrom + rand_disk.x * u + rand_disk.y * v;
    vec3 ray_d = lookfrom + p.x * u + p.y * v + w * focal_length - ray_o;
    return Ray(ray_o, normalize(ray_d));
}

float raySphereIntersection(Ray ray, int index) {
    vec3 sp_c = S[index].pos;
    float radius = S[index].r;
    vec3 L = sp_c - ray.origin;
    float tca = dot(L, ray.dir);
    float d2 = dot(L, L) - tca * tca;
    float t0, t1;
    if (d2 <= radius * radius) {
        float thc = sqrt(radius * radius - d2);
        t0 = tca - thc;
        t1 = tca + thc;
        if (t0 > t1) {
            float temp = t0;
            t0 = t1;
            t1 = temp;
        }
        if (t0 < 0) {
            t0 = t1;
        }
        if (t0 >= 0) {
            return t0;
        }
    }
    return -1;
}

void getNearestIntersection(Ray ray, inout float t, inout int hit_index) {
    t = -1;
    hit_index = -1;
    float t0;
    for (int j = 0; j < sphereCount; j++) {
        t0 = raySphereIntersection(ray, j);
        if (t0 >= 0) {
            if (t == -1 || t0 < t) {
                t = t0;
                hit_index = j;
            }
        }
    }
}

void main() {
    if (frameCount == 0.0) {
        ivec2 texelCoord = ivec2(gl_GlobalInvocationID.xy);
        imageStore(imgOutput, texelCoord, vec4(0.0, 0.0, 0.0, 0.0));
        return;
    }
    rng_state = wang_hash((gl_GlobalInvocationID.x + 720 * gl_GlobalInvocationID.y) + 720 * 1280 * uint(frameCount));

    // M[0] = Material(vec4(0.5, 0.5, 0.5, 1.0), vec4(0.0, 0.0, 0.0, 0.0), 0, 0, 0.0, 0.0);
    // M[1] = Material(vec4(1.0, 1.0, 1.0, 1.0), vec4(0.0, 0.0, 0.0, 0.0), 0, 0, 0.3, 0.1);
    // M[2] = Material(vec4(1.0, 1.0, 1.0, 1.0), vec4(0.0, 0.0, 0.0, 0.0), 0, 1, 0.0, 1.5);
    // M[3] = Material(vec4(1.0, 1.0, 1.0, 1.0), vec4(0.0, 0.0, 0.0, 0.0), 0, 1, 0.0, 1.0 / 1.5);
    // M[4] = Material(vec4(0.0, 0.0, 0.0, 1.0), vec4(1.0, 1.0, 1.0, 1.0), 1, 0, 0.0, 0.0);

    // S[0] = Sphere(vec3(0.0, 0.0, 3.0), 1.0, 0);
    // S[1] = Sphere(vec3(2.1, 0.0, 3.0), 1.0, 1);
    // S[2] = Sphere(vec3(-2.1, 0.0, 3.0), 1.0, 2);
    // S[3] = Sphere(vec3(0.0, -101.0, 3.0), 100.0, 0);
    // S[4] = Sphere(vec3(-2.1, 0.0, 3.0), 0.8, 3);

    Ray ray = getStartRay();
    vec4 color = vec4(1.0, 1.0, 1.0, 1.0);
    vec4 light = vec4(0.0, 0.0, 0.0, 0.0);
    vec3 sp_c;
    float radius;
    float t = 1;
    int hit_index = -1;
    for (int i = 0; i < maxBounces; i++) {
        getNearestIntersection(ray, t, hit_index);
        if (hit_index == -1) {
            float a = (ray.dir.y + 1) * 0.5;
            color *= 1.0 * ((1.0 - a) * vec4(1.0, 1.0, 1.0, 1.0) + a * vec4(0.5, 0.7, 1.0, 1.0));
            light += color;
            break;
        }
        sp_c = S[hit_index].pos;
        radius = S[hit_index].r;
        vec3 hitpoint = ray.origin + ray.dir * t;
        vec3 normal = (hitpoint - sp_c) / radius;
        Material material = M[S[hit_index].materialIndex];
        if (material.type == 0) {
            vec3 unitsphere = random_vec3();
            ray.origin = hitpoint + normal * 0.01;
            float smoothness = material.smoothness;
            float ri = material.ri;
            vec4 albedo = material.albedo;
            vec3 diffuse_dir = normalize(normal + 0.99 * unitsphere);
            vec3 specular_dir = normalize(ray.dir - 2 * dot(ray.dir, normal) * normal);
            float is_specular_bounce = float(random_float() <= ri);
            smoothness *= is_specular_bounce;
            ray.dir = diffuse_dir * (1 - smoothness) + specular_dir * smoothness;
            light += color * material.emission_strength * material.emissive_colour;
            color *= albedo * (1 - is_specular_bounce) + is_specular_bounce * vec4(1.0, 1.0, 1.0, 1.0);
            // ray.dir = normalize(ray.dir - 2 * dot(ray.dir, normal) * normal + material.smoothness * unitsphere);
        } else if (material.type == 1) {
            color *= material.albedo;
            float ri = material.ri;
            float etai_over_etat = 1.0 / ri;
            if (dot(ray.dir, normal) > 0) {
                normal = -normal;
                etai_over_etat = 1 / etai_over_etat;
            }
            float cos_theta = min(dot(-ray.dir, normal), 1.0);
            float sin_theta = sqrt(1.0 - cos_theta * cos_theta);
            float r0 = (1 - ri) / (1 + ri);
            r0 = r0 * r0;
            r0 = r0 + (1 - r0) * pow((1 - cos_theta), 5);
            if (etai_over_etat * sin_theta > 1.0 || r0 > random_float()) {
                vec3 unitsphere = random_vec3();
                ray.dir = normalize(ray.dir - 2 * dot(ray.dir, normal) * normal + material.smoothness * unitsphere);
                ray.origin = hitpoint + ray.dir * 0.01;
            } else {
                vec3 r_out_perp = etai_over_etat * (ray.dir + cos_theta * normal);
                vec3 r_out_parallel = -sqrt(abs(1.0 - dot(r_out_perp, r_out_perp))) * normal;
                ray.dir = normalize(r_out_perp + r_out_parallel);
                ray.origin = hitpoint + ray.dir * 0.01;
            }
        }
    }

    ivec2 texelCoord = ivec2(gl_GlobalInvocationID.xy);
    vec4 prev = imageLoad(imgOutput, texelCoord);
    float frameWeight = 1 / frameCount;
    light = prev * (1 - frameWeight) + light * frameWeight;
    imageStore(imgOutput, texelCoord, light);
}
