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

@group(0) @binding(0) var oklab_texture: texture_2d<f32>;
struct DebugUniforms {
    show_edge_pixels: u32,
};

@group(0) @binding(1) var<uniform> debug_uniforms: DebugUniforms;

@fragment
fn fs_main(in: VsOut) -> @location(0) vec4f {
    // Clamp UV to avoid sampling outside (because our fullscreen triangle uses UV beyond 0..1)
    let uv = clamp(in.uv, vec2f(0.0), vec2f(1.0));


    let dims = textureDimensions(oklab_texture, 0);
    let texel = vec2u(
        min(u32(uv.x * f32(dims.x)), dims.x - 1u),
        min(u32(uv.y * f32(dims.y)), dims.y - 1u)
    );
    var oklab = textureLoad(oklab_texture, texel, 0);
    if (debug_uniforms.show_edge_pixels != 0u) {
        let pix = oklab;
        oklab = vec4f(pix.x, 0.5 * cos(pix.z * 2.0), 0.5 * sin(pix.z * 2.0), pix.w);
    }

    let linear = oklab_to_linear(oklab.rgb);
    let srgb = linear_to_srgb(linear);

    // return the resulting color
    return vec4f(srgb, oklab.a);
}

//Convert linear RGB to sRGB
fn linear_to_srgb(lin: vec3f) -> vec3f {
    return pow(lin, vec3(1.0/2.2));
}

fn oklab_to_linear(oklab: vec3<f32>) -> vec3<f32> {
    var l = oklab.x + oklab.y * 0.3963377774 + oklab.z * 0.2158037573;
    var m = oklab.x + oklab.y * -0.1055613458 + oklab.z * -0.0638541728;
    var s = oklab.x + oklab.y * -0.0894841775 + oklab.z * -1.2914855480;
    l = l * l * l; m = m * m * m; s = s * s * s;
    var r = l * 4.0767416621 + m * -3.3077115913 + s * 0.2309699292;
    var g = l * -1.2684380046 + m * 2.6097574011 + s * -0.3413193965;
    var b = l * -0.0041960863 + m * -0.7034186147 + s * 1.7076147010;
    r = clamp(r, 0.0, 1.0); g = clamp(g, 0.0, 1.0); b = clamp(b, 0.0, 1.0);
    return vec3(r, g, b);
}