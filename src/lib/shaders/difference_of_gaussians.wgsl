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
@group(0) @binding(1) var input_texA: texture_2d<f32>;
@group(0) @binding(2) var input_texB: texture_2d<f32>;


@fragment
fn cs_main(in: VsOut) -> @location(0) vec4f {
    // Clamp UV to avoid sampling outside (because our fullscreen triangle uses UV beyond 0..1)
    let uv = clamp(in.uv, vec2f(0.0), vec2f(1.0));

    let dims = textureDimensions(input_texA);

    let texel = vec2u(
        min(u32(uv.x * f32(dims.x)), dims.x - 1u),
        min(u32(uv.y * f32(dims.y)), dims.y - 1u)
    );

    let pixelA = textureLoad(input_texA, texel, 0).rgb;
    let pixelB = textureLoad(input_texB, texel, 0).rgb;

    let delta = pixelA - pixelB;
    let d = length(delta);

    if (d <= uniforms.threshold) {
        return vec4f(1.0, 0.0, 0.0, 1.0);
    } else {
        return vec4f(0.0, 0.0, 0.0, 1.0);
    }
}
