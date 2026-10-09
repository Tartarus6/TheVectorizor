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
	z → edge_id low      (low 16 bits of the unique edge id / starting connection index)
	w → edge_id high     (high 16 bits of the edge id; full id = z | (w << 16))
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

@group(0) @binding(0) var edge_tex: texture_2d<u32>;

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

    if (edge_flag == 0u) {
        return vec4f(0.0, 0.0, 0.0, 1.0);
    }

    return vec4f(1.0);
}
