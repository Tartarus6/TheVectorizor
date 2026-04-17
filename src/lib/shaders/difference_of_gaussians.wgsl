struct Uniforms {
    threshold: f32,
}

@group(0) @binding(0) var<uniform> uniforms: Uniforms;
@group(0) @binding(1) var input_texA: texture_2d<f32>;
@group(0) @binding(2) var input_texB: texture_2d<f32>;
@group(0) @binding(3) var output_tex: texture_storage_2d<rgba16float, write>;

@compute @workgroup_size(16, 16, 1)
fn cs_main(@builtin(global_invocation_id) id: vec3<u32>) {
    let dims = textureDimensions(input_texA);
    if (id.x >= u32(dims.x) || id.y >= u32(dims.y)) { return; }

    let coord: vec2<i32> = vec2<i32>(i32(id.x), i32(id.y));

    let pixelA = textureLoad(input_texA, coord, 0).rgb;
    let pixelB = textureLoad(input_texB, coord, 0).rgb;

    let delta = pixelA - pixelB;
    let d = length(delta);

    if (d >= uniforms.threshold) {
        textureStore(output_tex, coord, vec4<f32>(1.0, 1.0, 1.0, 1.0));
    } else {
        textureStore(output_tex, coord, vec4<f32>(0.0, 0.0, 0.0, 1.0));
    }
}
