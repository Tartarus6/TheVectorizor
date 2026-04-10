struct Uniforms {
    base_bandwidth: f32,
}

@group(0) @binding(0) var<uniform> uniforms: Uniforms;
@group(0) @binding(1) var<storage, read> input_colors: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read_write> output_density_scores: array<f32>;

@compute @workgroup_size(64)
fn cs_main(@builtin(global_invocation_id) id: vec3<u32>) {
    let BANDWIDTH_SQUARED = uniforms.base_bandwidth * uniforms.base_bandwidth;
    let idx = id.x;
    if idx >= arrayLength(&input_colors) { return; }

    let color = input_colors[idx].rgb;

    var density_score: f32 = 0f;

    for (var i = 0u; i < arrayLength(&input_colors); i++) {
        if (i == idx) {continue;}
        
        let other = input_colors[i].rgb;
        let delta = color - other;
        let dist_squared = dot(delta, delta);

        density_score += exp(-dist_squared / (2.0 * BANDWIDTH_SQUARED));
    }

    output_density_scores[idx] = density_score;
}
