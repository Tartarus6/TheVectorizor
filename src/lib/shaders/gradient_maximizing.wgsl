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

    // TODO: is there a way to cast the pixels as some struct, so that it's extra obvious that what the xyzw values are?

    // TODO: refine the node position to have the maximum not just be the center of the pixel (check neighbors to find actual maximum)

    if (grad_mag < LOW) {
        return vec4f(0, 0, 0, 0);
    }

    var greatest = true; // store whether neighbor of greater magnitude has been found (set to false if neighbor is found)
    for (var dx = -1; dx <= 1; dx ++) {
        for (var dy = -1; dy <= 1; dy ++) {
            if (dx == 0 && dy == 0) { continue; }

            let neighbor = textureLoad(grad_tex, clamp(vec2i(texel) + vec2i(dx, dy), vec2i(0), vec2i(dims) - vec2i(1)), 0);

            if (grad_mag <= neighbor.y) {
                greatest = false;
            }
        }
    }

    if (greatest) {
        return vec4f(1, grad_mag, theta, 1);
    }

    return vec4f(0, grad_mag, theta, 1);
}
