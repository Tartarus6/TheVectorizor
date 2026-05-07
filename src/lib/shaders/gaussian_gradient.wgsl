/*
This shader calculates the gradient (the derivative) of the given texture.

The purpose in TheVectorizer, is for this shader to be given a gaussian blurred texture,
then this shader calculates the direction and magnitude of color change at each pixel in
the x and y directions.
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


@group(0) @binding(0) var in_tex: texture_2d<f32>;

/*
output: texture_2d<f32>
	x -> grad_mag (magnitude of gradient)
	y -> theta    (direction of gradient)
    z -> 0        (unused)
    w -> 0        (unused)
*/


@fragment
fn cs_main(in: VsOut) -> @location(0) vec4f {
    // Clamp UV to avoid sampling outside (because our fullscreen triangle uses UV beyond 0..1)
    let uv = clamp(in.uv, vec2f(0.0), vec2f(1.0));

    let dims = textureDimensions(in_tex);

    let texel = vec2u(
        min(u32(uv.x * f32(dims.x)), dims.x - 1u),
        min(u32(uv.y * f32(dims.y)), dims.y - 1u)
    );

    // TODO: probably want to take a better approximation of the derivative. this should work good enough for now, though
    let right = textureLoad(in_tex, clamp(vec2i(texel) + vec2i(1, 0), vec2i(0, 0), vec2i(dims) - vec2i(1, 1)), 0);
    let left = textureLoad(in_tex, clamp(vec2i(texel) + vec2i(-1, 0), vec2i(0, 0), vec2i(dims) - vec2i(1, 1)), 0);
    let down = textureLoad(in_tex, clamp(vec2i(texel) + vec2i(0, 1), vec2i(0, 0), vec2i(dims) - vec2i(1, 1)), 0);
    let up = textureLoad(in_tex, clamp(vec2i(texel) + vec2i(0, -1), vec2i(0, 0), vec2i(dims) - vec2i(1, 1)), 0);

    let down_right = textureLoad(in_tex, clamp(vec2i(texel) + vec2i(1, 1), vec2i(0, 0), vec2i(dims) - vec2i(1, 1)), 0);
    let up_left = textureLoad(in_tex, clamp(vec2i(texel) + vec2i(-1, -1), vec2i(0, 0), vec2i(dims) - vec2i(1, 1)), 0);
    let up_right = textureLoad(in_tex, clamp(vec2i(texel) + vec2i(1, -1), vec2i(0, 0), vec2i(dims) - vec2i(1, 1)), 0);
    let down_left = textureLoad(in_tex, clamp(vec2i(texel) + vec2i(-1, 1), vec2i(0, 0), vec2i(dims) - vec2i(1, 1)), 0);

    // sobel filter
    let gx = 2 * (right - left) + (down_right - up_left) + (up_right - down_left); // dColor / dx
    let gy = 2 * (down - up) + (down_right - up_left) + (down_left - up_right);    // dColor / dy

    // --- DiZenzo-style multi-channel gradient tensor ---
    let A = dot(gx, gx);
    let B = dot(gx, gy);
    let C = dot(gy, gy);

    let theta = 0.5 * atan2(2f * B, A - C);

    let grad_mag = sqrt(0.5 * ((A + C) + sqrt((A - C)*(A - C) + (4f * B * B))));

    // TODO: switch texture format to match the one used for gradient maximizing
    return vec4f(grad_mag, theta, 0, 0);
}
