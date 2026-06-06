/*
This shader culls the gaussian gradient texture to be just the maximums.

The purpose in TheVectorizer, is for this shader to be given the gaussian gradient
texture. For each pixel in the gradient texture, this shader looks at the neighboring
pixels along the gradient normal to see if that pixel is the maximum or not.

This basically just filters the texture to be only the edge pixels

This shader also calculates a subpixel offset amount for potential edge pixels by
checking neighboring gradient values. The purpose is to help produce a smoother edge
in the final result.

This shader sets the following values:
	grad_tex
		- subpixel_offset
	edge_tex
		- edge flag
*/



/*
grad_tex/out_grad_tex: texture_2d<f32>
	x → grad_mag        (magnitude of gradient)
	y → theta           (direction of gradient)
	z → subpixel_offset (in the direction of the gradient)
	w → 0               (unused)

out_edge_tex (rgba16uint):
    x → edge flag        (whether this pixel is part of an edge)
    y → packed neighbors (bitmask to say which of the 8 neighbor pixels are connected edge pixels)
    z → 0                (unused)
    w → 0                (unused)
*/
@group(0) @binding(0) var grad_tex: texture_2d<f32>;
@group(0) @binding(1) var grad_sampler: sampler;
@group(0) @binding(2) var out_grad_tex: texture_storage_2d<rgba16float, write>;
@group(0) @binding(3) var out_edge_tex: texture_storage_2d<rgba16uint, write>;


// TODO: move this LOW value to an external variable passed through uniforms
const LOW: f32 = 0.1;
const MAX_NEIGHBORS: u32 = 8u; // maximum packed neighbors (one per 8-connected direction)


@compute @workgroup_size(16, 16)
fn cs_main(@builtin(global_invocation_id) gid: vec3u) {
    let dims = textureDimensions(grad_tex);

    if (gid.x >= dims.x || gid.y >= dims.y) {
        return;
    }

    let texel = gid.xy;
    let uv = (vec2f(texel) + vec2f(0.5, 0.5)) / vec2f(dims);

    // TODO: might want to switch below to just use textureLoad
    // Sample with linear filtering to allow sub-pixel neighborhood checks.
    let grad_pixel = textureSampleLevel(grad_tex, grad_sampler, uv, 0.0);
    let grad_mag = grad_pixel.x;
    let theta = grad_pixel.y;

    // TODO: (maybe) add check to this edge thing to make sure that the edge pixel isn't transparent
    // if pixel is on the outside of image, then mark it as an edge so that backgrounds and edges that hit the edge are properly handled
    if (texel.x == 0 || texel.x == dims.x - 1 || texel.y == 0 || texel.y == dims.y - 1) {
        // get the correct neighbors
        var neighbor_offsets = array<vec2i, MAX_NEIGHBORS>(
            vec2i(0), vec2i(0), vec2i(0), vec2i(0),
            vec2i(0), vec2i(0), vec2i(0), vec2i(0)
            );
        if (texel.x == 0 && texel.y == 0) {
            // top left
            neighbor_offsets[0] = vec2i(1, 0); // east
            neighbor_offsets[1] = vec2i(0, 1); // south
        } else if (texel.x == dims.x - 1 && texel.y == 0) {
            // top right
            neighbor_offsets[0] = vec2i(-1, 0); // west
            neighbor_offsets[1] = vec2i(0, 1);  // south
        } else if (texel.x == 0 && texel.y == dims.y - 1) {
            // bottom left
            neighbor_offsets[0] = vec2i(1, 0);  // east
            neighbor_offsets[1] = vec2i(0, -1); // north
        } else if (texel.x == dims.x - 1 && texel.y == dims.y - 1) {
            // bottom right
            neighbor_offsets[0] = vec2i(-1, 0); // west
            neighbor_offsets[1] = vec2i(0, -1); // north
        } else if (texel.x == 0 || texel.x == dims.x - 1) {
            // left/right
            neighbor_offsets[0] = vec2i(0, 1);  // south
            neighbor_offsets[1] = vec2i(0, -1); // north
        } else if (texel.y == 0 || texel.y == dims.y - 1) {
            // top/bottom
            neighbor_offsets[0] = vec2i(1, 0);  // east
            neighbor_offsets[1] = vec2i(-1, 0); // west
        }

        let packed: u32 = pack_neighbors(neighbor_offsets, 2);

   	    textureStore(out_grad_tex, texel, grad_pixel);
        textureStore(out_edge_tex, texel, vec4u(1u, packed, 0u, 0u));
        return;
    }

    // TODO: is there a way to cast the pixels as some struct, so that it's extra obvious that what the xyzw values are?

    // TODO: refine the node position to have the maximum not just be the center of the pixel (check neighbors to find actual maximum)

    if (grad_mag < LOW) {
        textureStore(out_grad_tex, texel, grad_pixel);
        textureStore(out_edge_tex, texel, vec4u(0u, 0u, 0u, 0u));
        return;
    }

    // --- Finding Edge Seed Pixels ---
    var greatest = true; // store whether neighbor of greater magnitude has been found (set to false if neighbor is found)
    // for each neighboring pixel
    for (var dx = -1; dx <= 1; dx ++) {
        for (var dy = -1; dy <= 1; dy ++) {
            // dont compare the pixel to itsself
            if (dx == 0 && dy == 0) { continue; }

            let neighbor_pix = textureLoad(grad_tex, clamp(vec2i(texel) + vec2i(dx, dy), vec2i(0), vec2i(dims) - vec2i(1)), 0);
            let neighbor_grad_mag = neighbor_pix.x;

            // if the neighbor has a greater magnitude, then our pixel isn't the greatest
            if (grad_mag <= neighbor_grad_mag) {
                greatest = false;
            }
        }
    }

    // TODO: might be worth it to move subpixel shifting to the edge tracing steps to improve performance
    // save subpixel offset in grad_tex
    let subpixel_offset = get_subpixel_offset(grad_pixel, texel, dims);
    textureStore(out_grad_tex, texel, vec4f(grad_pixel.xy, subpixel_offset, grad_pixel.w));
    // let subpixel_offset = 0f;

    // TODO: rename greatest to prevent confusion with greatest self vs greatest neighbor
    // --- Marking Edge Seeds ---
    if (greatest) {
        // mark this pixel as part of an edge
        textureStore(out_edge_tex, texel, vec4u(1u, 0u, 0u, 0u));
        return;
    }

    // these pixels might later be part of an edge, so their magnitude and theta need to be stored
    textureStore(out_edge_tex, texel, vec4u(0u));
}

fn get_subpixel_offset(grad_pixel: vec4f, texel: vec2u, dims: vec2u) -> f32 {
	let neighbor_checks: u32 = 2; // number of neighbor checks in each direction (so value of 2 would mean checking 4 neighbors + 1 for the pixel itsself)
	let neighbor_check_distance: f32 = 1.5;
    let theta = grad_pixel.y;


    // ?NOTE: this commented out section is an older (maybe worse idk) method of calculating the subpixel offset
    // // pick the neighbor with the greatest gradient magnitude
    // var greatest_neighbor_pix = vec4f(0); // temporary, should be overwritten
    // var greatest_neighbor_flip = 0;        // temporary, should be overwritten
    // for (var flip = -1; flip <= 1; flip += 2) {
    //     let neighbor_offset = f32(flip) * vec2f(cos(theta), sin(theta)) * neighbor_check_distance;
    //     let neighbor_pos_uv = (vec2f(texel) + neighbor_offset) / vec2f(dims);

    //     let neighbor_pix = textureSampleLevel(grad_tex, grad_sampler, neighbor_pos_uv, 0.0);

    //     if (neighbor_pix.x > greatest_neighbor_pix.x) {
    //         greatest_neighbor_pix = neighbor_pix;
    //         greatest_neighbor_flip = flip;
    //     }
    // }

    // // calculating subpixel_offset based off of greatest gradient neighbor
    // let subpixel_offset = (greatest_neighbor_pix.x * f32(greatest_neighbor_flip) * neighbor_check_distance) / (grad_pixel.x + greatest_neighbor_pix.x);

    var weighted_offset_sum = 0f; // sum of pixel offsets in direction of gradient weighted by that pixel's gradient magnitude
    var weights_sum = 0f;         // sum of the gradient magnitudes (for normalizing)

    for (var mult: i32 = -i32(neighbor_checks); mult <= i32(neighbor_checks); mult += 1) {
    	let offset_magnitude = f32(mult) * neighbor_check_distance / f32(neighbor_checks);
        let neighbor_offset = vec2f(cos(theta), sin(theta)) * offset_magnitude;
        let neighbor_pos_uv = (vec2f(texel) + neighbor_offset) / vec2f(dims);

        let neighbor_pix = textureSampleLevel(grad_tex, grad_sampler, neighbor_pos_uv, 0.0);
        let neighbor_grad_mag = neighbor_pix.x;

        // TODO: figure out how to weight it so that neighbors who's gradient alligns better are weighted higher than missaligned neighbors
        let weight = neighbor_grad_mag;

        weighted_offset_sum += offset_magnitude * weight;
        weights_sum += weight;
    }

    let subpixel_offset = weighted_offset_sum / weights_sum;

    // return 0;
    return subpixel_offset;
}

// TODO: remove this code duplication if possible. maybe there's a way to shader this const and functions between shaders, idk.
// Used for packing and unpacking neighbor offsets
// Direction order: E, NE, N, NW, W, SW, S, SE
const DIRS: array<vec2i, 8> = array<vec2i, 8>(
    vec2i(1, 0),   // 0: E
    vec2i(1, -1),  // 1: NE
    vec2i(0, -1),  // 2: N
    vec2i(-1, -1), // 3: NW
    vec2i(-1, 0),  // 4: W
    vec2i(-1, 1),  // 5: SW
    vec2i(0, 1),   // 6: S
    vec2i(1, 1)    // 7: SE
);

/*
Function that packs the position deltas of two neighbors into one uint for storing in the texture

Note: the order of the neighbors does not matter.

Input: (dx A, dy A, dx B, dy B)
Example:
	□ · ·
	· ■ ·   →   Input is [(-1, -1), (1, 1)]   →   Output is 2³ + 2⁷   →   10001000 (binary)   →   136 (decimal)
	· · □

	· □ ·
	· ■ □   →   Input is [(0, -1), (1, 0), (-1, 1)]   →   Output is 2² + 2⁰ + 2⁵   →   00100101 (binary)   →   37 (decimal)
	□ · ·

*/
fn pack_neighbors(neighbors: array<vec2i, MAX_NEIGHBORS>, neighbor_count: u32) -> u32 {
    // pack up to MAX_NEIGHBORS offsets into a direction bitmask
    var packed: u32 = 0u;
    let count = min(neighbor_count, MAX_NEIGHBORS);

    for (var i: u32 = 0u; i < count; i = i + 1u) {
        let dir: u32 = offset_to_direction(neighbors[i]);
        packed = packed | (1u << dir);
    }

    return packed;
}

struct UnpackedNeighbors {
    neighbors: array<vec2i, MAX_NEIGHBORS>,
    count: u32,
};

fn unpack_neighbors(packed_neighbors: u32) -> UnpackedNeighbors {
    var result = UnpackedNeighbors(
        array<vec2i, MAX_NEIGHBORS>(
            vec2i(0), vec2i(0), vec2i(0), vec2i(0),
            vec2i(0), vec2i(0), vec2i(0), vec2i(0)
        ),
        0u
    );

    for (var i: u32 = 0u; i < 8u; i = i + 1u) {
        if ((packed_neighbors & (1u << i)) != 0u) {
            if (result.count < MAX_NEIGHBORS) {
                result.neighbors[result.count] = DIRS[i];
                result.count = result.count + 1u;
            }
        }
    }

    return result;
}

/*
Takes in an 8-connected offset and turns it into an 8-connected direction from 0 to 7

Input: (dx, dy)
	-1 <= dx, dy <= 1
	dx and dy can't both be 0
Output:
	Output follows this pattern

	3 2 1
	4 ■ 0
	5 6 7

Example:
	□ · ·
	· ■ ·   →   Input is (-1, -1)   →   Output is 3
	· · ·

	· · ·
	· ■ □   →   Input is (1, 0)   →   Output is 0
	· · ·
*/
fn offset_to_direction(offset: vec2i) -> u32{
	// TODO: validate input maybe?

    for (var i: u32 = 0; i < 8; i = i + 1) {
        if (all(offset == DIRS[i])) {
            return i;
        }
    }
    return 0u; // invalid inputs just default to 0
}
