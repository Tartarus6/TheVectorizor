struct Uniforms {
    base_bandwidth: f32,
    median_density_score: f32,
}


@group(0) @binding(0) var<uniform> uniforms: Uniforms;
@group(0) @binding(1) var input_colors: texture_2d<f32>;
@group(0) @binding(2) var<storage,read_write> input_density_scores: array<f32>;
@group(0) @binding(3) var output_colors: texture_storage_2d<rgba8unorm, write>;


@compute @workgroup_size(16, 16)
fn cs_main(@builtin(global_invocation_id) id: vec3u) {
    let dims = textureDimensions(input_colors);

    if id.x >= dims.x || id.y >= dims.y {return;}

    let index = id.x + dims.x * id.y;
    let color = textureLoad(input_colors, id.xy, 0).rgb;
    var cluster_sum = vec3f(0.0);
    var cluster_count = 0u;

    let bandwidth = get_bandwidth(input_density_scores[index]);
    let bandwidth_squared = bandwidth * bandwidth;

    for (var i = 0u; i < dims.x; i++) {
        for (var j = 0u; j < dims.y; j++) {
            let other = textureLoad(input_colors, vec2u(i,j), 0).rgb;
            let delta = color - other;
            let dist_squared = dot(delta, delta);

            if dist_squared < bandwidth_squared {
                cluster_sum += other;
                cluster_count += 1u;
            }
        }
    }

    let new_color = cluster_sum / f32(cluster_count);
    textureStore(output_colors, id.xy, vec4f(new_color, 1.0));
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
