#version 430 core

layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;

layout(rgba32f, binding = 0) uniform image2D imgOutput;

uniform float frameCount;
uniform float vfov;
uniform float defocus_angle;
uniform vec3 lookfrom; // Point camera is looking from
uniform vec3 lookat; // Point camera is looking at

float focal_length = 1.0;
vec3 vup = vec3(0, 1, 0); // Camera-relative "up" direction

layout(std430, binding = 2) buffer VB
{
    float V[];
};
struct index_t {
    int vertex_index;
    int normal_index;
    int texcoord_index;
};
struct triangle_index {
    index_t v0, v1, v2;
};
layout(std430, binding = 3) buffer VI
{
    triangle_index I[50000];
};
uniform int triCount;

struct AABB {
    vec3 min_v;
    vec3 max_v;
};
struct BVHNode {
    AABB aabb;
    int start, end;
    int lc, rc;
};
layout(std430, binding = 5) buffer BVHbuffer
{
    BVHNode BVH[];
};

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

float rayTriangleIntersection(Ray ray, int index) {
    int i0 = I[index].v0.vertex_index, i1 = I[index].v1.vertex_index, i2 = I[index].v2.vertex_index;
    vec3 v0 = vec3(V[i0 * 3], V[i0 * 3 + 1], V[i0 * 3 + 2]);
    vec3 v1 = vec3(V[i1 * 3], V[i1 * 3 + 1], V[i1 * 3 + 2]);
    vec3 v2 = vec3(V[i2 * 3], V[i2 * 3 + 1], V[i2 * 3 + 2]);
    vec3 orig = ray.origin, dir = ray.dir;
    vec3 v0v1 = v1 - v0;
    vec3 v0v2 = v2 - v0;
    vec3 pvec = cross(dir, v0v2);
    float det = dot(pvec, v0v1);
    if (abs(det) < 0.000001) return -1;
    float invDet = 1 / det;
    vec3 tvec = orig - v0;
    float u = dot(tvec, pvec) * invDet;
    if (u < 0 || u > 1) return -1;

    vec3 qvec = cross(tvec, v0v1);
    float v = dot(qvec, dir) * invDet;
    if (v < 0 || u + v > 1) return -1;

    return dot(v0v2, qvec) * invDet;
}

bool rayAABBIntersection(Ray ray, AABB aabb, float t) {
    // vec3 tMin = (aabb.min_v - ray.origin) / ray.dir;
    // vec3 tMax = (aabb.max_v - ray.origin) / ray.dir;
    // vec3 t1 = min(tMin, tMax);
    // vec3 t2 = max(tMin, tMax);
    // float tNear = max(max(t1.x, t1.y), t1.z);
    // float tFar = min(min(t2.x, t2.y), t2.z);
    // if (tNear > tFar) return -1;
    // if (tNear < 0) tNear = tFar;
    // if (tNear < 0) return -1;
    // return tNear;
    vec3 ray_inv = vec3(1.0, 1.0, 1.0) / ray.dir;
    float tx1 = (aabb.min_v.x - ray.origin.x) * ray_inv.x;
    float tx2 = (aabb.max_v.x - ray.origin.x) * ray_inv.x;
    float tmin = min(tx1, tx2);
    float tmax = max(tx1, tx2);
    float ty1 = (aabb.min_v.y - ray.origin.y) * ray_inv.y;
    float ty2 = (aabb.max_v.y - ray.origin.y) * ray_inv.y;
    tmin = max(tmin, min(ty1, ty2));
    tmax = min(tmax, max(ty1, ty2));
    float tz1 = (aabb.min_v.z - ray.origin.z) * ray_inv.z;
    float tz2 = (aabb.max_v.z - ray.origin.z) * ray_inv.z;
    tmin = max(tmin, min(tz1, tz2));
    tmax = min(tmax, max(tz1, tz2));
    return tmax >= max(0.0, tmin) && tmin < t;
}
/*
void getNearestIntersection(Ray ray, inout float t, inout int hit_index, inout vec3 normal,inout vec3 hitpoint) {
    t = -1;
    hit_index = -1;
    float t0;
    for (int j = 0; j < triCount; j++) {
        t0 = rayTriangleIntersection(ray, j);
        if (t0 >= 0) {
            if (t == -1 || t0 < t) {
                t = t0;
                hit_index = j;
            }
        }
    }
    if(hit_index!=-1){
        int i0 = I[hit_index].v0.vertex_index,i1 = I[hit_index].v1.vertex_index,i2 = I[hit_index].v2.vertex_index;
        vec3 v0 = vec3(V[i0*3],V[i0*3+1],V[i0*3+2]);
        vec3 v1 = vec3(V[i1*3],V[i1*3+1],V[i1*3+2]);
        vec3 v2 = vec3(V[i2*3],V[i2*3+1],V[i2*3+2]);
        normal = cross(v1-v0,v2-v0);
        normal *= -sign(dot(normal,ray.dir));
        hitpoint = ray.origin + ray.dir*t;
    }
}*/

int stk[16];
int sp = 0;

void getNearestIntersection(Ray ray, inout float t, inout int hit_index, inout vec3 normal, inout vec3 hitpoint) {
    t = -1;
    hit_index = -1;
    sp = 0;
    stk[sp] = 0;
    sp++;
    while (sp > 0) {
        sp--;
        BVHNode bvhnode = BVH[stk[sp]];
        float t0 = 100000.0;
        if (rayAABBIntersection(ray, bvhnode.aabb, t0)) {
            if (bvhnode.lc == -1) {
                for (int j = bvhnode.start; j <= bvhnode.end; j++) {
                    t0 = rayTriangleIntersection(ray, j);
                    if (t0 >= 0) {
                        if (t == -1 || t0 < t) {
                            t = t0;
                            hit_index = j;
                        }
                    }
                }
            } else {
                stk[sp] = bvhnode.lc;
                sp++;
                stk[sp] = bvhnode.rc;
                sp++;
            }
        }
    }
    if (hit_index != -1) {
        int i0 = I[hit_index].v0.vertex_index, i1 = I[hit_index].v1.vertex_index, i2 = I[hit_index].v2.vertex_index;
        vec3 v0 = vec3(V[i0 * 3], V[i0 * 3 + 1], V[i0 * 3 + 2]);
        vec3 v1 = vec3(V[i1 * 3], V[i1 * 3 + 1], V[i1 * 3 + 2]);
        vec3 v2 = vec3(V[i2 * 3], V[i2 * 3 + 1], V[i2 * 3 + 2]);
        normal = cross(v1 - v0, v2 - v0);
        normal *= -sign(dot(normal, ray.dir));
        hitpoint = ray.origin + ray.dir * t;
    }
}

void main() {
    if (frameCount == 0.0) {
        ivec2 texelCoord = ivec2(gl_GlobalInvocationID.xy);
        imageStore(imgOutput, texelCoord, vec4(0.0, 0.0, 0.0, 0.0));
        return;
    }
    rng_state = wang_hash((gl_GlobalInvocationID.x + 720 * gl_GlobalInvocationID.y) + 720 * 1280 * uint(frameCount));

    Ray ray = getStartRay();
    vec4 color = vec4(1.0, 1.0, 1.0, 1.0);
    vec4 light = vec4(0.0, 0.0, 0.0, 0.0);
    vec3 sp_c, hitpoint, normal;
    float radius;
    float t = 1;
    int hit_index = -1;
    Material material = M[0];
    //vec3(-0.032745, -0.969753, -0.883882) vec3(1.373505, 0.998997, 0.819243)
    // if(distance(BVH[2].aabb.max_v,vec3(1.373505, 0.998997, 0.819243))<0.01) {
    //     light = vec4(1.0, 1.0, 0.0, 0.0);
    // }
    for (int i = 0; i < maxBounces; i++) {
        // rayTriangleIntersection(ray,hitpoint,normal,hit_index,t);
        getNearestIntersection(ray, t, hit_index, normal, hitpoint);
        // for(int i=11;i<15;i++){
        //     if (BVH[i].lc==-1 && rayAABBIntersection(ray, BVH[i].aabb, 1000.0)) light.r = (i+1)*0.05;
        // }
        // for(int i=5;i<9;i++){
        //     if (BVH[i].lc==-1 && rayAABBIntersection(ray, BVH[i].aabb, 1000.0)) light.g = (i+1)*0.05;
        // }
        // break;
        // if (rayAABBIntersection(ray, BVH[1].aabb, 1000.0)) light.g = 1.0;
        // if (rayAABBIntersection(ray, BVH[2].aabb, 1000.0)) light.b = 1.0;
        // break;
        if (hit_index == -1) {
            float a = (ray.dir.y + 1) * 0.5;
            color *= 1.0 * ((1.0 - a) * vec4(1.0, 1.0, 1.0, 1.0) + a * vec4(0.5, 0.7, 1.0, 1.0));
            light += color;
        }

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
    light = (prev * (1 - frameWeight) + light * frameWeight);
    imageStore(imgOutput, texelCoord, light);
}
