struct FloatUniforms {
    base_bandwidth: f32,
}

struct UintUniforms {
    tile_x: u32,    /// the low x value of the current tile (basically the x-offset for this shader pass)
    tile_y: u32,    /// the low y value of the current tile (basically the y-offset for this shader pass)
    tile_size: u32, /// the size of each tile (the range of x and y for this shader pass)
}

@group(0) @binding(0) var<uniform> float_uniforms: FloatUniforms;
@group(0) @binding(1) var<uniform> uint_uniforms: UintUniforms;
@group(0) @binding(2) var input_colors: texture_2d<f32>;
@group(0) @binding(3) var<storage, read_write> output_density_scores: array<f32>;
@group(0) @binding(4) var output_density_scores_texture: texture_storage_2d<rgba16float, write>;


const PI = 3.1415926535;


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

    let bandwidth_squared = float_uniforms.base_bandwidth * float_uniforms.base_bandwidth;

    let checks_per_ring = 12u;
    var prev_radius = 0f;
    for (var radius = 1f; radius < f32(max(dims.x, dims.y)); radius *= 1.25) {
        // get multiplier based on step size, each pixel checked counts for the square of pixels around it
        let multiplier = PI * (f32(radius) * f32(radius) - f32(prev_radius) * f32(prev_radius)) / f32(checks_per_ring);

        for (var i=0u; i<checks_per_ring; i++) {
            // TODO: precompute offsets, since sin() and cos() are kinda expensive
            let angle = (2 * PI) * f32(i / checks_per_ring);
            let offset = vec2f(cos(angle), sin(angle)) * f32(radius);

            // TODO: could probably do some mins and select stuff to make other_pos be a vec2u
            let other_pos = vec2i(pos) + vec2i(offset);

            // TODO: this if statement can probably be cleaned or improved
            // if pixel is outside of texture, skip
            if (other_pos.x < 0 || other_pos.x >= i32(dims.x) || other_pos.y < 0 || other_pos.y >= i32(dims.y)) {
                continue;
            }

            let other = textureLoad(input_colors, other_pos, 0);

            let delta = color.xyz - other.xyz;
            let dist_squared = dot(delta, delta);

            density_score += exp2(-dist_squared / (2.0 * bandwidth_squared)) * f32(multiplier);
        }

        // update prev_radius
        prev_radius = radius;
    }

    output_density_scores[index] = density_score;
    textureStore(output_density_scores_texture, pos, vec4f(density_score, 0, 0, 1));
}
