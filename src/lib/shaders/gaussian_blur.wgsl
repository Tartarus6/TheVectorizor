struct Uniforms {
    radius: u32,
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
@group(0) @binding(1) var input_tex: texture_2d<f32>;
@group(0) @binding(2) var<storage, read> gaussian_weights: array<f32>;

@fragment
fn blur_horizontal(in: VsOut) -> @location(0) vec4f {
    // Clamp UV to avoid sampling outside (because our fullscreen triangle uses UV beyond 0..1)
    let uv = clamp(in.uv, vec2f(0.0), vec2f(1.0));

    let dims = textureDimensions(input_tex);

    let texel = vec2u(
        min(u32(uv.x * f32(dims.x)), dims.x - 1u),
        min(u32(uv.y * f32(dims.y)), dims.y - 1u)
    );

    // TODO: prevent the need for separate texture load for transparency
    let transparency = textureLoad(input_tex, texel, 0).a;

    var sum = vec3<f32>(0.0);
    let r = i32(uniforms.radius);

    for (var i: i32 = -r; i <= r; i = i + 1) {
        let x = clamp(i32(texel.x) + i, 0, i32(dims.x) - 1);
        let sample = textureLoad(input_tex, vec2i(x, i32(texel.y)), 0).xyz;
        let idx = u32(abs(i));
        sum = sum + sample * gaussian_weights[idx];
    }

    return vec4f(sum, transparency);
}

@fragment
fn blur_vertical(in: VsOut) -> @location(0) vec4f {
    // Clamp UV to avoid sampling outside (because our fullscreen triangle uses UV beyond 0..1)
    let uv = clamp(in.uv, vec2f(0.0), vec2f(1.0));

    let dims = textureDimensions(input_tex);

    let texel = vec2u(
        min(u32(uv.x * f32(dims.x)), dims.x - 1u),
        min(u32(uv.y * f32(dims.y)), dims.y - 1u)
    );

    // TODO: prevent the need for separate texture load for transparency
    let transparency = textureLoad(input_tex, texel, 0).a;

    var sum = vec3f(0.0);
    let r = i32(uniforms.radius);

    for (var i: i32 = -r; i <= r; i = i + 1) {
        let y = clamp(i32(texel.y) + i, 0, i32(dims.y) - 1);
        let sample = textureLoad(input_tex, vec2i(i32(texel.x), y), 0).xyz;
        let idx = u32(abs(i));
        sum = sum + sample * gaussian_weights[idx];
    }

    return vec4f(sum, transparency);
}
