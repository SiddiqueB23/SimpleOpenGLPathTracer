A simple path tracer using OpenGL Compute Shaders.  
Uses simple BVHs for optimisation.  
Supports only triangles OR spheres.  
Supports specular-diffuse(plastic and metallic),refracitve(only with spheres), emissive materials.  

Stanford dragon with tris reduced to 5%(44K)
![Stanford Dragon05](main/dragon_05.png?raw=true "Stanford Dragon05")

TODO:
- Add credits and links for thirdparty sources in README
- Optimise generation of BVH
- Add camera control
- Experimment with heuristics for BVH Node splitting
- Refactor and optimise shaders
- Refactor main.cpp
- Rework refractive materials to work with triangles
- Add support for multiple meshes
