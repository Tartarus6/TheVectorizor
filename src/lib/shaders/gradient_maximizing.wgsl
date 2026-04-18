/*
This shader culls the gaussian gradient texture to be just the maximums.

The purpose in TheVectorizer, is for this shader to be given the gaussian gradient
texture. For each pixel in the gradient texture, this shader looks at the neighboring 
pixels along the gradient normal to see if that pixel is the maximum or not.

This basically just filters the texture to be only the edge pixels

TODO: update above description for when this shader actually does sub-pixel stuff
*/

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


/*
grad_tex: texture_2d<f32>
    x -> theta    (direction of gradient)
    y -> grad_mag (magnitude of gradient)
    z -> 0        (unused)
    a -> 0        (unused)
*/
@group(0) @binding(0) var grad_tex: texture_2d<f32>;
@group(0) @binding(1) var grad_sampler: sampler;


@fragment
fn cs_main(in: VsOut) -> @location(0) vec4f {
    // Clamp UV to avoid sampling outside (because our fullscreen triangle uses UV beyond 0..1)
    let uv = clamp(in.uv, vec2f(0.0), vec2f(1.0));

    let dims = textureDimensions(grad_tex);

    // Sample with linear filtering to allow sub-pixel neighborhood checks.
    let grad_pixel = textureSampleLevel(grad_tex, grad_sampler, uv, 0.0);
    let theta = grad_pixel.x;
    let grad_mag = grad_pixel.y;

    let normal = vec2f(cos(theta), sin(theta));

    let texel_size = 1.0 / vec2f(f32(dims.x), f32(dims.y));
    let offset_uv = normal * texel_size;

    let neighbor_a_mag = textureSampleLevel(
        grad_tex,
        grad_sampler,
        clamp(uv + offset_uv, vec2f(0.0), vec2f(1.0)),
        0.0
    ).y;
    let neighbor_b_mag = textureSampleLevel(
        grad_tex,
        grad_sampler,
        clamp(uv - offset_uv, vec2f(0.0), vec2f(1.0)),
        0.0
    ).y;


    if (grad_mag >= neighbor_a_mag && grad_mag >= neighbor_b_mag) {
        return vec4f(1, 0.5 * cos(2 * theta), 0.5 * sin(2 * theta), grad_mag);
    } else {
        return vec4f(0, 0, 0, 0);
    }
}
