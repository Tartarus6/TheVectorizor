/*
This shader culls the gaussian gradient texture to be just the maximums.

The purpose in TheVectorizer, is for this shader to be given the gaussian gradient
texture. For each pixel in the gradient texture, this shader looks at the neighboring
pixels along the gradient normal to see if that pixel is the maximum or not.

This basically just filters the texture to be only the edge pixels

TODO: update above description for when this shader actually does sub-pixel stuff
*/

/*
input_tex:
    x -> edge flag
    y -> grad_mag
    z -> theta
    w -> above LOW flag

output_tex:
    x -> edge flag
    y -> grad_mag
    z -> theta
    w -> above LOW flag
*/
@group(0) @binding(0) var input_tex: texture_2d<f32>;
@group(0) @binding(1) var grad_sampler: sampler;
@group(0) @binding(2) var output_tex: texture_storage_2d<rgba16float, write>;


const PI = 3.1415926535;


@compute @workgroup_size(16, 16)
fn cs_main(@builtin(global_invocation_id) gid: vec3u) {
    let dims = textureDimensions(input_tex);

    if (gid.x >= dims.x || gid.y >= dims.y) {
        return;
    }

    let texel = gid.xy;
    let uv = (vec2f(texel) + vec2f(0.5, 0.5)) / vec2f(dims);

    var out_pixel = textureSampleLevel(input_tex, grad_sampler, uv, 0.0);
    let in_pixel = textureLoad(input_tex, clamp(vec2i(texel), vec2i(0, 0), vec2i(dims) - vec2i(1, 1)), 0);
    let edge_flag = in_pixel.x;
    let grad_mag = in_pixel.y;
    let theta = in_pixel.z;
    let above_low_flag = in_pixel.w;


    // TODO: might be able to save these texture stores with some initial copying or something
    textureStore(output_tex, texel, in_pixel);


    if (above_low_flag != 1f || edge_flag != 1f) {
        return;
    }


    let right = textureLoad(input_tex, clamp(vec2i(texel) + vec2i(1, 0), vec2i(0), vec2i(dims) - vec2i(1)), 0);
    let left = textureLoad(input_tex, clamp(vec2i(texel) + vec2i(-1, 0), vec2i(0), vec2i(dims) - vec2i(1)), 0);
    let down = textureLoad(input_tex, clamp(vec2i(texel) + vec2i(0, 1), vec2i(0), vec2i(dims) - vec2i(1)), 0);
    let up = textureLoad(input_tex, clamp(vec2i(texel) + vec2i(0, -1), vec2i(0), vec2i(dims) - vec2i(1)), 0);

    let down_right = textureLoad(input_tex, clamp(vec2i(texel) + vec2i(1, 1), vec2i(0), vec2i(dims) - vec2i(1)), 0);
    let up_left = textureLoad(input_tex, clamp(vec2i(texel) + vec2i(-1, -1), vec2i(0), vec2i(dims) - vec2i(1)), 0);
    let up_right = textureLoad(input_tex, clamp(vec2i(texel) + vec2i(1, -1), vec2i(0), vec2i(dims) - vec2i(1)), 0);
    let down_left = textureLoad(input_tex, clamp(vec2i(texel) + vec2i(-1, 1), vec2i(0), vec2i(dims) - vec2i(1)), 0);

    let section = u32((theta + (PI / 4.0) + (PI / 2.0)) / (2 * PI) * 4.0) % 4;

    let dirs = array<vec2i, 4>(
        vec2i(1, 0),   // right
        vec2i(0, -1),  // up
        vec2i(-1, 0),  // left
        vec2i(0, 1)    // down
    );

    let dir = dirs[section];
    let perp = vec2i(-dir.y, dir.x);

    var candidates = array<vec2i, 3>(
        dir + perp,
        dir,
        dir - perp,
    );

    var best_pix = vec4f(0.0);  // default, should be overwritten
    var best_pos = vec2u(0, 0); // default, should be overwritten

    for (var i = 0u; i < 3u; i = i + 1u) {
        let pos = vec2u(clamp(vec2i(texel) + candidates[i], vec2i(0), vec2i(dims) - vec2i(1)));
        let cand_pix = textureLoad(input_tex, pos, 0);
        
        // if candidate is already part of an edge, just pick that one immediately
        if (cand_pix.x != 0) {
            best_pix = cand_pix;
            best_pos = pos;
            break;
        }

        // if candidate is new best, update best
        if (cand_pix.y > best_pix.y) {
            best_pix = cand_pix;
            best_pos = pos;
        }
    }
    textureStore(output_tex, best_pos, vec4f(1, best_pix.yz, 1));

    best_pix = vec4f(0.0);  // default, should be overwritten
    best_pos = vec2u(0, 0); // default, should be overwritten

    // TODO: replace for loop below to use arrayLength(candidates) or somethign else to remove the hardcoded loop count
    for (var i = 0u; i < 3; i = i + 1u) {
        let pos = vec2u(clamp(vec2i(texel) - candidates[i], vec2i(0), vec2i(dims) - vec2i(1)));
        let cand_pix = textureLoad(input_tex, pos, 0);

        // if candidate is already part of an edge, just pick that one immediately
        if (cand_pix.x != 0) {
            best_pix = cand_pix;
            best_pos = pos;
            break;
        }

        // if candidate is new best, update best
        if (cand_pix.y > best_pix.y) {
            best_pix = cand_pix;
            best_pos = pos;
        }
    }
    textureStore(output_tex, best_pos, vec4f(1, best_pix.yz, 1));


    // for (var dx = -1; dx<=1; dx++) {
    //     for (var dy = -1; dy<=1; dy++) {
    //         // dont compare pixel with itsself
    //         if (dx == 0 && dy == 0) { continue; }

    //         let neighbor = textureLoad(input_tex, clamp(vec2i(texel) + vec2i(dx, dy), vec2i(0, 0), vec2i(dims) - vec2i(1, 1)), 0);

    //         // skip neighbor if it's not an edge
    //         if (neighbor.x != 1) { continue; }

    //         let offset = choose_edge_neighbor(neighbor);

    //         if ((offset.x == dx && offset.y == dy) || (offset.x == -dx && offset.y == -dy)) {
    //             out_pixel = vec4f(1, out_pixel.yz, 1);
    //             textureStore(output_tex, vec2i(texel), out_pixel);
    //             return;
    //         }

    //     }
    // }



}

/*
    pix_value:
        x -> edge flag
        y -> grad_mag
        z -> theta
        a -> 1

    This function takes in the value of a pixel, and returns the offset of the edge it correlates to
*/
// fn choose_edge_neighbor(pix_value: vec4f) -> vec2i {
//     let theta = pix_value.z;

//     let section = u32((theta + (PI / 8.0) + (PI / 2.0)) / (2 * PI) * 8.0) % 8;

//     // TODO: there must be a less horrible way of doing this
//     switch section {
//         case 0, 8: {
//             return vec2i(1, 0);
//         }
//         case 1: {
//             return vec2i(1, 1);
//         }
//         case 2: {
//             return vec2i(0, 1);
//         }
//         case 3: {
//             return vec2i(-1, 1);
//         }
//         case 4: {
//             return vec2i(-1, 0);
//         }
//         case 5: {
//             return vec2i(-1, -1);
//         }
//         case 6: {
//             return vec2i(0, -1);
//         }
//         case 7: {
//             return vec2i(1, -1);
//         }
//         default: {
//             return vec2i(0, 0);
//         }
//     }
// }
