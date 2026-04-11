struct Uniforms {
    base_bandwidth: f32,
    cluster_check_radius: f32, /// how many a square of double this size, in the texture, around the pixel is the are checked for creating the cluster
}

@group(0) @binding(0) var<uniform> uniforms: Uniforms;
@group(0) @binding(1) var input_colors: texture_2d<f32>;
@group(0) @binding(2) var<storage,read_write> output_density_scores: array<f32>;


@compute @workgroup_size(16, 16)
fn cs_main(@builtin(global_invocation_id) id: vec3u) {
    let dims = textureDimensions(input_colors);

    if id.x >= dims.x || id.y >= dims.y {return;}

    let index = id.x + dims.x * id.y;
    let color = textureLoad(input_colors, id.xy, 0).rgb;
    var density_score: f32 = 0f;

    let BANDWIDTH_SQUARED = uniforms.base_bandwidth * uniforms.base_bandwidth;
    let x0 = select(0u, id.x - u32(uniforms.cluster_check_radius), id.x >= u32(uniforms.cluster_check_radius));
    let y0 = select(0u, id.y - u32(uniforms.cluster_check_radius), id.y >= u32(uniforms.cluster_check_radius));
    let x1 = min(dims.x, id.x + u32(uniforms.cluster_check_radius) + 1u);
    let y1 = min(dims.y, id.y + u32(uniforms.cluster_check_radius) + 1u);

    for (var i = x0; i < x1; i++) {
        for (var j = y0; j < y1; j++) {
            if (i == id.x && j == id.y) {continue;}

            let other = textureLoad(input_colors, vec2u(i, j), 0).rgb;
            let delta = color - other;
            let dist_squared = dot(delta, delta);

            density_score += exp(-dist_squared / (2.0 * BANDWIDTH_SQUARED));
        }
    }

    output_density_scores[index] = density_score;
}
