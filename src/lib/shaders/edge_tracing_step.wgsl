/*
This shader culls the gaussian gradient texture to be just the maximums.

The purpose in TheVectorizer, is for this shader to be given the gaussian gradient
texture. For each pixel in the gradient texture, this shader looks at the neighboring 
pixels along the gradient normal to see if that pixel is the maximum or not.

This basically just filters the texture to be only the edge pixels

TODO: update above description for when this shader actually does sub-pixel stuff
*/

struct VsOut {
    @builtin(position) pos: vec4f,
    @location(0) uv: vec2f,
};

@vertex
fn vs_main(@builtin(vertex_index) vid: u32) -> VsOut {
    // Fullscreen triangle
    var pos = array<vec2f, 3>(
        vec2f(-1.0, -1.0),
        vec2f(3.0, -1.0),
        vec2f(-1.0, 3.0),
    );

    // UVs (can go outside 0..1; we clamp later)
    var uv = array<vec2f, 3>(
        vec2f(0.0, 1.0),
        vec2f(2.0, 1.0),
        vec2f(0.0, -1.0),
    );

    var out: VsOut;
    out.pos = vec4f(pos[vid], 0.0, 1.0);
    out.uv = uv[vid];
    return out;
}


/*
input_tex:
    x -> edge flag
    y -> grad_mag
    z -> theta
    w -> 1

output_tex:
    x -> edge flag
    y -> grad_mag
    z -> theta
    w -> 1
*/
@group(0) @binding(0) var input_tex: texture_2d<f32>;
@group(0) @binding(1) var grad_sampler: sampler;


const PI = 3.1415926535;


@fragment
fn cs_main(in: VsOut) -> @location(0) vec4f {
    // Clamp UV to avoid sampling outside (because our fullscreen triangle uses UV beyond 0..1)
    let uv = clamp(in.uv, vec2f(0.0), vec2f(1.0));

    let dims = textureDimensions(input_tex);

    let texel = vec2u(
        min(u32(uv.x * f32(dims.x)), dims.x - 1u),
        min(u32(uv.y * f32(dims.y)), dims.y - 1u)
    );

    // Sample with linear filtering to allow sub-pixel neighborhood checks.
    let grad_pixel = textureSampleLevel(input_tex, grad_sampler, uv, 0.0);

    if (grad_pixel.w == 0) {
        return grad_pixel;
    }

    for (var dx = -1; dx<=1; dx++) {
        for (var dy = -1; dy<=1; dy++) {
            // dont compare pixel with itsself
            if (dx == 0 && dy == 0) { continue; }

            let neighbor = textureLoad(input_tex, clamp(vec2i(texel) + vec2i(dx, dy), vec2i(0, 0), vec2i(dims) - vec2i(1, 1)), 0);

            // skip neighbor if it's not an edge
            if (neighbor.x != 1) { continue; }

            let offset = choose_edge_neighbor(neighbor);

            if ((offset.x == dx && offset.y == dy) || (offset.x == -dx && offset.y == -dy)) {
                return vec4f(1, grad_pixel.yz, 1);
            }
            
        }
    }

    return grad_pixel;
}


/*
    pix_value:
        x -> edge flag
        y -> grad_mag
        z -> theta
        a -> 1

    This function takes in the value of a pixel, and returns the offset of the edge it correlates to
*/
fn choose_edge_neighbor(pix_value: vec4f) -> vec2i {
    let theta = pix_value.z;
    
    let section = u32(abs(theta + (PI / 8.0)) / (2 * PI) * 8.0);

    // TODO: there must be a less horrible way of doing this
    switch section {
        case 0, 8: {
            return vec2i(1, 0);
        }
        case 1: {
            return vec2i(1, 1);
        }
        case 2: {
            return vec2i(1, 0);
        }
        case 3: {
            return vec2i(-1, 1);
        }
        case 4: {
            return vec2i(-1, 0);
        }
        case 5: {
            return vec2i(-1, -1);
        }
        case 6: {
            return vec2i(0, -1);
        }
        case 7: {
            return vec2i(1, -1);
        }
        default: {
            return vec2i(0, 0);
        }
    }
}
