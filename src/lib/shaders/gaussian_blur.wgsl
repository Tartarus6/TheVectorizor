struct Uniforms {
    radius: u32,
}


@group(0) @binding(0) var<uniform> uniforms: Uniforms;
@group(0) @binding(1) var input_tex: texture_2d<f32>;
@group(0) @binding(2) var output_tex: texture_storage_2d<rgba16float, write>;
@group(0) @binding(3) var<storage, read> gaussian_weights: array<f32>;

@compute @workgroup_size(16, 16, 1)
fn blur_horizontal(@builtin(global_invocation_id) id: vec3<u32>) {
    let dims = textureDimensions(input_tex);
    if (id.x >= u32(dims.x) || id.y >= u32(dims.y)) { return; }

    let coord_i = vec2<i32>(i32(id.x), i32(id.y));
    let transparency = textureLoad(input_tex, coord_i, 0).a;

    var sum = vec3<f32>(0.0);
    let r = i32(uniforms.radius);

    for (var i: i32 = -r; i <= r; i = i + 1) {
        let x = clamp(i32(id.x) + i, 0, i32(dims.x) - 1);
        let sample = textureLoad(input_tex, vec2<i32>(x, i32(id.y)), 0).rgb;
        let idx = u32(abs(i));
        sum = sum + sample * gaussian_weights[idx];
    }

    textureStore(output_tex, coord_i, vec4<f32>(sum, transparency));
}

@compute @workgroup_size(16, 16, 1)
fn blur_vertical(@builtin(global_invocation_id) id: vec3<u32>) {
    let dims = textureDimensions(input_tex);
    if (id.x >= u32(dims.x) || id.y >= u32(dims.y)) { return; }

    let coord_i = vec2<i32>(i32(id.x), i32(id.y));
    let transparency = textureLoad(input_tex, coord_i, 0).a;
    let r = i32(uniforms.radius);

    var sum = vec3<f32>(0.0);
    for (var i: i32 = -r; i <= r; i = i + 1) {
        let y = clamp(i32(id.y) + i, 0, i32(dims.y) - 1);
        let sample = textureLoad(input_tex, vec2<i32>(i32(id.x), y), 0).rgb;
        let idx = u32(abs(i));
        sum = sum + sample * gaussian_weights[idx];
    }

    textureStore(output_tex, coord_i, vec4<f32>(sum, transparency));
}
