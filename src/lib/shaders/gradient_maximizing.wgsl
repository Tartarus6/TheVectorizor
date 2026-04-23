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
    w -> 0        (unused)

output:
    x -> edge flag
    y -> grad_mag
    z -> theta
    w -> above LOW flag
*/
@group(0) @binding(0) var grad_tex: texture_2d<f32>;
@group(0) @binding(1) var grad_sampler: sampler;


// TODO: move this LOW value to an external variable passed through uniforms
const LOW: f32 = 0.05;


@fragment
fn cs_main(in: VsOut) -> @location(0) vec4f {
    // Clamp UV to avoid sampling outside (because our fullscreen triangle uses UV beyond 0..1)
    let uv = clamp(in.uv, vec2f(0.0), vec2f(1.0));

    let dims = textureDimensions(grad_tex);

    let texel = vec2u(
        min(u32(uv.x * f32(dims.x)), dims.x - 1u),
        min(u32(uv.y * f32(dims.y)), dims.y - 1u)
    );

    // Sample with linear filtering to allow sub-pixel neighborhood checks.
    let grad_pixel = textureSampleLevel(grad_tex, grad_sampler, uv, 0.0);
    let theta = grad_pixel.x;
    let grad_mag = grad_pixel.y;

    // let normal = vec2f(cos(theta), sin(theta));

    // let texel_size = 1.0 / vec2f(f32(dims.x), f32(dims.y));
    // let offset_uv = normal * texel_size;

    // let neighbor_a_mag = textureSampleLevel(
    //     grad_tex,
    //     grad_sampler,
    //     clamp(uv + offset_uv, vec2f(0.0), vec2f(1.0)),
    //     0.0
    // ).y;
    // let neighbor_b_mag = textureSampleLevel(
    //     grad_tex,
    //     grad_sampler,
    //     clamp(uv - offset_uv, vec2f(0.0), vec2f(1.0)),
    //     0.0
    // ).y;

    let right = textureLoad(grad_tex, clamp(vec2i(texel) + vec2i(1, 0), vec2i(0, 0), vec2i(dims) - vec2i(1, 1)), 0);
    let left = textureLoad(grad_tex, clamp(vec2i(texel) + vec2i(-1, 0), vec2i(0, 0), vec2i(dims) - vec2i(1, 1)), 0);
    let down = textureLoad(grad_tex, clamp(vec2i(texel) + vec2i(0, 1), vec2i(0, 0), vec2i(dims) - vec2i(1, 1)), 0);
    let up = textureLoad(grad_tex, clamp(vec2i(texel) + vec2i(0, -1), vec2i(0, 0), vec2i(dims) - vec2i(1, 1)), 0);

    let down_right = textureLoad(grad_tex, clamp(vec2i(texel) + vec2i(1, 1), vec2i(0, 0), vec2i(dims) - vec2i(1, 1)), 0);
    let up_left = textureLoad(grad_tex, clamp(vec2i(texel) + vec2i(-1, -1), vec2i(0, 0), vec2i(dims) - vec2i(1, 1)), 0);
    let up_right = textureLoad(grad_tex, clamp(vec2i(texel) + vec2i(1, -1), vec2i(0, 0), vec2i(dims) - vec2i(1, 1)), 0);
    let down_left = textureLoad(grad_tex, clamp(vec2i(texel) + vec2i(-1, 1), vec2i(0, 0), vec2i(dims) - vec2i(1, 1)), 0);


    // TODO: refine the node position to have the maximum not just be the center of the pixel (check neighbors to find actual maximum)

    if (grad_mag < LOW) {
        return vec4f(0, 0, 0, 0);
    }

    if (grad_mag > right.y && grad_mag > left.y && grad_mag > down.y && grad_mag > up.y && grad_mag > down_right.y && grad_mag > up_left.y && grad_mag > up_right.y && grad_mag > down_left.y) {
        // return vec4f(1, 0, 0, 1);
        // return vec4f(1, 0.5 * cos(2 * theta), 0.5 * sin(2 * theta), grad_mag);
        return vec4f(1, theta, grad_mag, 1);
    }

    // return vec4f(0, 0.5 * cos(2 * theta), 0.5 * sin(2 * theta), grad_mag);
    // return vec4f(0, 0, 0, 0);
    return vec4f(0, theta, grad_mag, 0.2);


    // // if (grad_mag >= neighbor_a_mag && grad_mag >= neighbor_b_mag && grad_mag > 0.001) {
    // if (grad_mag >= neighbor_a_mag && grad_mag >= neighbor_b_mag && grad_mag > 0.1) {
    //     // return vec4f(1, grad_mag, theta, 1);
    //     return vec4f(1, 0.5 * cos(2 * theta), 0.5 * sin(2 * theta), 1);
    // } else {
    //     return vec4f(0, 0, 0, 0);
    // }
}
