struct FloatUniforms {
    base_bandwidth: f32,
}

struct UintUniforms {
    cluster_check_radius: u32, /// how many a square of double this size, in the texture, around the pixel is the are checked for creating the cluster
    tile_x: u32,    /// the low x value of the current tile (basically the x-offset for this shader pass)
    tile_y: u32,    /// the low y value of the current tile (basically the y-offset for this shader pass)
    tile_size: u32, /// the size of each tile (the range of x and y for this shader pass)
}

@group(0) @binding(0) var<uniform> float_uniforms: FloatUniforms;
@group(0) @binding(1) var<uniform> uint_uniforms: UintUniforms;
@group(0) @binding(2) var input_colors: texture_2d<f32>;
@group(0) @binding(3) var<storage,read_write> output_density_scores: array<f32>;


// TODO: handle transparent pixels (need to somehow count how many non-transparent pixels there are, so that we divide by the right number to get the mean)
@compute @workgroup_size(16, 16)
fn cs_main(@builtin(global_invocation_id) id: vec3u) {
    let dims = textureDimensions(input_colors);

    // apply tile offsets to get this thread's pixel's position
    let pos = vec2u(id.x + uint_uniforms.tile_x, id.y + uint_uniforms.tile_y);

    if pos.x >= dims.x || pos.y >= dims.y {return;}

    let index = pos.x + dims.x * pos.y;
    let color = textureLoad(input_colors, pos, 0);
    var density_score: f32 = 0f;

    let BANDWIDTH_SQUARED = float_uniforms.base_bandwidth * float_uniforms.base_bandwidth;
    let x0 = select(0u, pos.x - uint_uniforms.cluster_check_radius, pos.x >= uint_uniforms.cluster_check_radius);
    let y0 = select(0u, pos.y - uint_uniforms.cluster_check_radius, pos.y >= uint_uniforms.cluster_check_radius);
    let x1 = min(dims.x, pos.x + uint_uniforms.cluster_check_radius + 1u);
    let y1 = min(dims.y, pos.y + uint_uniforms.cluster_check_radius + 1u);

    for (var i = x0; i < x1; i++) {
        for (var j = y0; j < y1; j++) {
            // dont compare the pixel to itsself
            if (i == pos.x && j == pos.y) {continue;}

            let other = textureLoad(input_colors, vec2u(i, j), 0);

            let delta = color.xyz - other.xyz;
            let dist_squared = dot(delta, delta);

            density_score += exp(-dist_squared / (2.0 * BANDWIDTH_SQUARED));
        }
    }

    output_density_scores[index] = density_score;
}
