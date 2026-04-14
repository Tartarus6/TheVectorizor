struct FloatUniforms {
    base_bandwidth: f32,
    mean_density_score: f32,
    alpha: f32, /// controls how strongly the density matters
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
@group(0) @binding(3) var<storage,read_write> input_density_scores: array<f32>;
@group(0) @binding(4) var output_colors: texture_storage_2d<rgba16float, write>;


// TODO: handle transparent pixels
@compute @workgroup_size(16, 16)
fn cs_main(@builtin(global_invocation_id) id: vec3u) {
    let dims = textureDimensions(input_colors);

    // apply tile offsets to get this thread's pixel's position
    let pos = vec2u(id.x + uint_uniforms.tile_x, id.y + uint_uniforms.tile_y);

    // if pixel not within texture, return
    if pos.x >= dims.x || pos.y >= dims.y {return;}

    let index = pos.x + dims.x * pos.y;
    let color = textureLoad(input_colors, pos, 0);
    var cluster_sum = vec4f(0.0);

    let bandwidth = get_bandwidth(input_density_scores[index]);
    let bandwidth_squared = bandwidth * bandwidth;

    // count the total weights to divide by later
    var weights_sum = 0f;

    let x0 = select(0u, pos.x - uint_uniforms.cluster_check_radius, pos.x >= uint_uniforms.cluster_check_radius);
    let y0 = select(0u, pos.y - uint_uniforms.cluster_check_radius, pos.y >= uint_uniforms.cluster_check_radius);
    let x1 = min(dims.x, pos.x + uint_uniforms.cluster_check_radius + 1u);
    let y1 = min(dims.y, pos.y + uint_uniforms.cluster_check_radius + 1u);
    for (var i = x0; i < x1; i++) {
        for (var j = y0; j < y1; j++) {
            // dont compare the pixel to itsself
            if (i == pos.x && j == pos.y) {continue;}

            let other = textureLoad(input_colors, vec2u(i,j), 0);

            let color_delta = color.xyz - other.xyz;
            let image_delta = vec2f(pos) - vec2f(f32(i), f32(j));
            let color_dist_squared = dot(color_delta, color_delta);
            let image_dist_squared = dot(image_delta, image_delta);

            let weight = exp(-(color_dist_squared * image_dist_squared) / (2 * bandwidth_squared));
            cluster_sum += other * weight;
            weights_sum += weight;
        }
    }

    // TODO: do we want to include the alpha in the average, or should we instead set it to always be 1? or what?
    let new_color = cluster_sum / weights_sum;
    textureStore(output_colors, pos, vec4f(new_color));
}


// per-color bandwidth calculation
fn get_bandwidth(density_score: f32) -> f32 {
    const epsilon = 0.000001; // prevents divide by zero
    const min_mult = 0.1f;
    const max_mult = 2f;

    // let mult = clamp(pow(float_uniforms.mean_density_score / (density_score + epsilon), alpha), min_mult, max_mult);
    let mult = clamp(float_uniforms.mean_density_score / (density_score + epsilon), min_mult, max_mult);

    return float_uniforms.base_bandwidth * mult;
}
