struct FloatUniforms {
    base_bandwidth: f32,
}

struct UintUniforms {
    tile_x: u32,    /// the low x value of the current tile (basically the x-offset for this shader pass)
    tile_y: u32,    /// the low y value of the current tile (basically the y-offset for this shader pass)
    tile_size: u32, /// the size of each tile (the range of x and y for this shader pass)
}


@group(0) @binding(0) var<uniform> float_uniforms: FloatUniforms;
@group(0) @binding(1) var<storage,read> input_mean_density_sum: array<f32>;
@group(0) @binding(2) var<uniform> uint_uniforms: UintUniforms;
@group(0) @binding(3) var input_colors: texture_2d<f32>;
@group(0) @binding(4) var<storage,read_write> input_density_scores: array<f32>;
@group(0) @binding(5) var output_colors: texture_storage_2d<rgba16float, write>;


const PI = 3.1415926535;


// TODO: handle transparent pixels
@compute @workgroup_size(16, 16)
fn cs_main(@builtin(global_invocation_id) id: vec3u) {
    let dims = textureDimensions(input_colors);
    let mean_density_score = input_mean_density_sum[0] / f32(dims.x * dims.y);

    // apply tile offsets to get this thread's pixel's position
    let pos = vec2u(id.x + uint_uniforms.tile_x, id.y + uint_uniforms.tile_y);

    // if pixel not within texture, return
    if pos.x >= dims.x || pos.y >= dims.y {return;}

    let index = pos.x + dims.x * pos.y;
    let color = textureLoad(input_colors, pos, 0);
    var cluster_sum = vec4f(0.0);

    let bandwidth = get_bandwidth(input_density_scores[index], mean_density_score);
    let bandwidth_squared = bandwidth * bandwidth;

    // count the total weights to divide by later
    var weights_sum = 0f;

    var checks_per_ring = 12u;
    var prev_radius = 0f;
    for (var radius = 1f; radius < f32(max(dims.x, dims.y)); radius *= 1.25) {
        // get multiplier based on step size, each pixel checked counts for the square of pixels around it
        let multiplier = PI * (f32(radius) * f32(radius) - f32(prev_radius) * f32(prev_radius)) / f32(checks_per_ring);

        for (var i=0u; i<checks_per_ring; i++) {
            let angle = (2 * PI) * (f32(i) / f32(checks_per_ring));
            let offset = vec2f(cos(angle), sin(angle)) * f32(radius);

            // TODO: could probably do some mins and select stuff to make other_pos be a vec2u
            let other_pos = vec2i(pos) + vec2i(offset);

            // TODO: this if statement can probably be cleaned or improved
            // if pixel is outside of texture, skip
            if (other_pos.x < 0 || other_pos.x >= i32(dims.x) || other_pos.y < 0 || other_pos.y >= i32(dims.y)) {
                continue;
            }

            let other = textureLoad(input_colors, other_pos, 0);

            let color_delta = color.xyz - other.xyz;
            let image_delta = vec2f(pos) - vec2f(other_pos);
            let color_dist_squared = dot(color_delta, color_delta);
            let image_dist_squared = dot(image_delta, image_delta);

            let weight = exp2(-(color_dist_squared * image_dist_squared) / (2 * bandwidth_squared)) * f32(multiplier);
            cluster_sum += other * weight;
            weights_sum += weight;
        }

        // update prev_radius
        prev_radius = radius;
    }

    // TODO: do we want to include the alpha in the average, or should we instead set it to always be 1? or what?
    let new_color = cluster_sum / weights_sum;
    textureStore(output_colors, pos, vec4f(new_color));
}


// per-color bandwidth calculation
fn get_bandwidth(density_score: f32, mean_density_score: f32) -> f32 {
    const epsilon = 0.000001; // prevents divide by zero
    const min_mult = 0.1f;
    const max_mult = 2f;

    // let mult = clamp(pow(mean_density_score / (density_score + epsilon), alpha), min_mult, max_mult);
    let mult = clamp(mean_density_score / (density_score + epsilon), min_mult, max_mult);

    return float_uniforms.base_bandwidth * mult;
}
