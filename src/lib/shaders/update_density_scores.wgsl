struct Uniforms {
    base_bandwidth: f32,
}

@group(0) @binding(0) var<uniform> uniforms: Uniforms;
@group(0) @binding(1) var input_colors: texture_2d<f32>;
@group(0) @binding(2) var<storage,read_write> output_density_scores: array<vec4<f32>>

@compute @workgroup_size(64)
fn cs_main(@builtin(global_invocation_id) id: vec3<u32>) {
    let BANDWIDTH_SQUARED = uniforms.base_bandwidth * uniforms.base_bandwidth;
    let idx = id.x;
    let dims = textureDimensions(input_colors);
    let coord = vec2<u32>(idx % dims.x, idx / dims.x);

    if idx >= dims.x * dims.y { return; }

    let color = textureLoad(input_colors, coord).rgb;

    var density_score: f32 = 0f;

    for (var i = 0u; i < dims.y; i++) {
        for (var j = 0u; j < dims.x; j++) {
            if (i == idx) {continue;}

            let other = textureLoad(input_colors, vec2<u32>(j, i)).rgb;
            let delta = color - other;
            let dist_squared = dot(delta, delta);

            density_score += exp(-dist_squared / (2.0 * BANDWIDTH_SQUARED));
        }
    }

    output_density_scores[idx] = density_score;
}
