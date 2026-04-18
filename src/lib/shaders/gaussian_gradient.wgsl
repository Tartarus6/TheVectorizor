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

    let gx = right.xyz - left.xyz; // dColor / dx
    let gy = down.xyz - up.xyz;    // dColor / dy

    // --- DiZenzo-style multi-channel gradient tensor ---
    let A = dot(gx, gx);
    let B = dot(gx, gy);
    let C = dot(gy, gy);

    let theta = 0.5 * atan2(2f * B, A - C);
    let grad_dir = vec2f(cos(theta), sin(theta));  // TODO: is this needed?

    let grad_mag = sqrt(0.5 * ((A + C) + sqrt((A - C)*(A - C) + (4f * B * B))));

    return vec4f(theta, grad_mag, 0, 0);

    // TODO: below stuff is just for testing visualization
    // let combined = vec3f(abs(gx.x * cos(theta)) + abs(gy.x * sin(theta)), gx.yz * abs(cos(theta)) + gy.yz * abs(sin(theta))); // shows the color gradient
    // let combined = vec3f(grad_mag, grad_dir * 0.5 * grad_mag); // shows the direction of the gradient

    // return vec4f(combined, 1);
}
