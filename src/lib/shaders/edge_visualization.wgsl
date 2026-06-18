/*
This shader is just for visualizing the workings of the edge tracing.
It is not needed in any way for creating the final output.
*/

struct VsOut {
    @builtin(position) pos: vec4f,
    @location(0) uv: vec2f,
};

/*
edge_tex (rgba16uint):
	x → edge flag        (whether this pixel is part of an edge)
	y → packed neighbors (bitmask to say which of the 8 neighbor pixels are connected edge pixels)
	z → edge_id          (unique edge id, corresponds to the starting index of the pixel's connections)
	w → 0                (unused)
*/

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

struct EdgeData {
    next_edge_idx: u32,
    jump_next_idx: u32,
    face_id: u32,
    pos_idx: u32,
    color: vec4f,
};

@group(0) @binding(0) var edge_tex: texture_2d<u32>;
@group(0) @binding(1) var<storage, read> edge_data: array<EdgeData>;

@fragment
fn fs_main(in: VsOut) -> @location(0) vec4f {
    let uv = clamp(in.uv, vec2f(0.0), vec2f(1.0));

    let dims = textureDimensions(edge_tex, 0);

    let texel = vec2u(
        min(u32(uv.x * f32(dims.x)), dims.x - 1u),
        min(u32(uv.y * f32(dims.y)), dims.y - 1u)
    );

    let edge_pix = textureLoad(edge_tex, texel, 0);
    let edge_flag = edge_pix.x;
    let packed_connections = edge_pix.y;
    let edge_id = edge_pix.z;

    if (edge_flag == 0u) {
        return vec4f(0.0, 0.0, 0.0, 1.0);
    }

    let num_connections = countOneBits(packed_connections);

    var color_sum = vec3f(0);
    for (var i = 0u; i < num_connections; i++) {
	   	let face_id = edge_data[edge_id + i].face_id;
	    let color = face_color(face_id);

		color_sum += color;
    }

    let avg_color = color_sum / f32(num_connections);

    return vec4f(avg_color, 1.0);
}

// hash function
fn hash_u32(x: u32) -> f32 {
    var h = x;
    h ^= h >> 16u;
    h *= 0x7feb352du;
    h ^= h >> 15u;
    h *= 0x846ca68bu;
    h ^= h >> 16u;

    return f32(h) / 4294967295.0;
}

// deterministic random color from face_id
fn face_color(face_id: u32) -> vec3f {
    return vec3f(
        hash_u32(face_id),
        hash_u32(face_id + 1u),
        hash_u32(face_id + 2u)
    );
}
