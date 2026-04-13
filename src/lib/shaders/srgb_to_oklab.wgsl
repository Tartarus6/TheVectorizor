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

@group(0) @binding(0) var srgb_texture: texture_2d<f32>;
@group(0) @binding(1) var<storage,read_write> oklab_texture: array<f32>;

@fragment
fn fs_main(in: VsOut) -> @location(0) vec4f {
    // Clamp UV to avoid sampling outside (because our fullscreen triangle uses UV beyond 0..1)
    let uv = clamp(in.uv, vec2f(0.0), vec2f(1.0));

    let dims = textureDimensions(srgb_texture, 0);
    let texel = vec2u(
        min(u32(uv.x * f32(dims.x)), dims.x - 1u),
        min(u32(uv.y * f32(dims.y)), dims.y - 1u)
    );
    let srgb = textureLoad(srgb_texture, texel, 0);

    let linear = srgb_to_linear(srgb.rgb);

    let oklab = linear_to_oklab(linear);

    // return the resulting color
    return vec4f(oklab, srgb.a);
}

//Convert sRGB to linear RGB
fn srgb_to_linear(rgb: vec3f) -> vec3f {
    return pow(rgb, vec3(2.2));
}

fn linear_to_oklab(linear: vec3f) -> vec3f {
    let im1 = mat3x3f(0.4121656120, 0.2118591070, 0.0883097947,
                          0.5362752080, 0.6807189584, 0.2818474174,
                          0.0514575653, 0.1074065790, 0.6302613616);
                       
    let im2 = mat3x3f(0.2104542553, 1.9779984951, 0.0259040371,
                          0.7936177850, -2.4285922050, 0.7827717662,
                          -0.0040720468, 0.4505937099, -0.8086757660);
                       
    let lms: vec3f = im1 * linear;
            
    return im2 * (sign(lms) * pow(abs(lms), vec3(1.0/3.0)));
}
