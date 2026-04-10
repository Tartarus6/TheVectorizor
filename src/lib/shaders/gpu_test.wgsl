struct Uniforms {
    num_colors: f32,
    bandwidth: f32,
}


@group(0) @binding(0) var<uniform> uniforms: Uniforms;
@group(0) @binding(1) var<storage, read> input_colors: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read_write> output_colors: array<vec4<f32>>;



@compute @workgroup_size(64)
fn cs_main(@builtin(global_invocation_id) id: vec3<u32>) {
    let BANDWIDTH_SQUARED = uniforms.bandwidth * uniforms.bandwidth;
    let idx = id.x;
    if idx >= u32(uniforms.num_colors) { return; }

    let color = input_colors[idx].rgb;

    var cluster_sum = vec3<f32>(0.0);
    var cluster_count = 0u;

    for (var i = 0u; i < u32(uniforms.num_colors); i++) {
        let other = input_colors[i].rgb;
        let delta = color - other;
        let dist = dot(delta, delta);

        if dist < BANDWIDTH_SQUARED {
            cluster_sum += other;
            cluster_count += 1u;

        }

    }

    let new_color = cluster_sum / f32(cluster_count);
    output_colors[idx] = vec4<f32>(new_color, 1.0);
}