struct Uniforms {
    threshold: f32,
}

struct VsOut {
    @builtin(position) pos: vec4f,
    @location(0) uv: vec2f,
};

@vertex
fn vs_main(@builtin(vertex_index) vid: u32) -> VsOut {
    // Fullscreen triangle
    var pos = array<vec2f, 3>(
        vec2f(-1.0, -1.0),
        vec2f(3.0, -1.0),
        vec2f(-1.0, 3.0),
    );

    // UVs (can go outside 0..1; we clamp later)
    var uv = array<vec2f, 3>(
        vec2f(0.0, 1.0),
        vec2f(2.0, 1.0),
        vec2f(0.0, -1.0),
    );

    var out: VsOut;
    out.pos = vec4f(pos[vid], 0.0, 1.0);
    out.uv = uv[vid];
    return out;
}


@group(0) @binding(0) var<uniform> uniforms: Uniforms;
@group(0) @binding(1) var blurred_tex_a: texture_2d<f32>;
@group(0) @binding(2) var blurred_tex_b: texture_2d<f32>;


@fragment
fn cs_main(in: VsOut) -> @location(0) vec4f {
    // Clamp UV to avoid sampling outside (because our fullscreen triangle uses UV beyond 0..1)
    let uv = clamp(in.uv, vec2f(0.0), vec2f(1.0));

    let dims = textureDimensions(blurred_tex_a);

    let texel = vec2u(
        min(u32(uv.x * f32(dims.x)), dims.x - 1u),
        min(u32(uv.y * f32(dims.y)), dims.y - 1u)
    );

    let pixel_a = textureLoad(blurred_tex_a, texel, 0).xyz;
    let pixel_b = textureLoad(blurred_tex_b, texel, 0).xyz;

    // TODO: switch to using distance squared for optimization
    let delta = pixel_a - pixel_b;
    let distance = length(delta);

    // store how many neighbor pixels have higher delta distances
    var num_greater_neighbors = 0u;

    for (var dx = -1; dx <= 1; dx++) {
        for (var dy = -1; dy <= 1; dy++) {
            // dont compare the pixel to itsself
            if (dx == 0 && dy == 0) {continue;}

            let diff = vec2i(dx, dy);

            let neighbor_a = textureLoad(blurred_tex_a, vec2i(texel) + diff, 0).xyz;
            let neighbor_b = textureLoad(blurred_tex_b, vec2i(texel) + diff, 0).xyz;

            let neighbor_delta = neighbor_a - neighbor_b;
            let neighbor_distance = length(neighbor_delta);

            if (neighbor_distance > distance) {
                num_greater_neighbors++;
            }
        }
    }

    if (num_greater_neighbors <= 2 && distance >= uniforms.threshold) {
        return vec4f(1.0, 0.0, 0.0, 1.0); // white
    } else {
        return vec4f(0.0, 0.0, 0.0, 1.0); // black
    }
}
