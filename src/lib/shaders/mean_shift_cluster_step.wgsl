struct Uniforms {
    base_bandwidth: f32,
    median_density_score: f32,
}


@group(0) @binding(0) var<uniform> uniforms: Uniforms;
@group(0) @binding(1) var<storage, read> input_colors: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read> input_density_scores: array<f32>;
@group(0) @binding(3) var<storage, read_write> output_colors: array<vec4<f32>>;


@compute @workgroup_size(64)
fn cs_main(@builtin(global_invocation_id) id: vec3<u32>) {
    let idx = id.x;
    if idx >= arrayLength(&input_colors) { return; }

    let color = input_colors[idx].rgb;

    var cluster_sum = vec3<f32>(0.0);
    var cluster_count = 0u;

    let bandwidth = get_bandwidth(input_density_scores[idx]);
    let bandwidth_squared = bandwidth * bandwidth;
    
    for (var i = 0u; i < arrayLength(&input_colors); i++) {
        let other = input_colors[i].rgb;
        let delta = color - other;
        let dist_squared = dot(delta, delta);

        if dist_squared < bandwidth_squared {
            cluster_sum += other;
            cluster_count += 1u;
        }

    }

    let new_color = cluster_sum / f32(cluster_count);
    output_colors[idx] = vec4<f32>(new_color, 1.0);
}

// per-color bandwidth calculation
fn get_bandwidth(density_score: f32) -> f32 {
    const alpha = 0.35; // controls how strongly the density matters
    const epsilon = 0.000001; // prevents divide by zero
    const min_mult = 0.7;
    const max_mult = 1.8;

    return (
        uniforms.base_bandwidth *
        clamp(pow(uniforms.median_density_score / (density_score + epsilon), alpha), min_mult, max_mult)
    );
}