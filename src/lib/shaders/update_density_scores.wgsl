struct Uniforms {
    base_bandwidth: f32,
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
    for (var i = 0u; i < dims.x; i++) {
        for (var j = 0u; j < dims.y; j++) {
            if (i == id.x && j == id.y) {continue;}

            let other = textureLoad(input_colors, vec2u(i, j), 0).rgb;
            let delta = color - other;
            let dist_squared = dot(delta, delta);

            density_score += exp(-dist_squared / (2.0 * BANDWIDTH_SQUARED));
        }
    }

    output_density_scores[index] = density_score;
}
