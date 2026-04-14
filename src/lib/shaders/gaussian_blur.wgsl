struct Uniforms {
    radius: u32,
}


@group(0) binding(0) var<uniform> uniforms:Uniforms;
@group(0) binding(1) var input_tex :texture_2d<rgba8unorm>;
@group(0) binding(2) var output_tex : texture_storage_2d<rgba8unorm, write>;

@compute @workgroup_size(256)
fn blur_horizontal(@builtin(global_invocation_id) id: vec3u) {

    let dims = textureDimensions(input_colors);

    if id.x >= dims.x || id.y >= dims.y {return;}
    let transparency = textureLoad(input_tex,vec2<u32>(id.x,id.y),0).a;

    var sum = vec3<f32>(0.0);


    for (var i = -radius; i<= radius; i++){

        let x = clamp(i32(id.x)+i,0 ,i32(dims.x)-1);

        let sample = textureLoad(input_tex,vec2<i32>(x,i32(id.y)),0).rgb;
        sum += sample * gaussian_weights[abs(i)];

    }

    textureStore(output_tex,vec2<u32>(id.x,id.y),vec4<f32>(sum,transparency));
}

@compute @workgroup_size(256)
fn blur_vertical(@builtin(global_invocation_id) id:vec3u){
    let dims = textureDimensions(input_colors);

    if id.x >= dims.x || id.y >= dims.y {return;}
    let transparency = textureLoad(input_tex,vec2<u32>(id.x,id.y),0).a;

    var sum = vec3<f32>(0.0);


    for (var i = -radius; i<= radius; i++){


        let y = clamp(i32(id.y)+i,0 ,i32(dims.y)-1);

        let sample = textureLoad(input_tex,vec2<i32>(i32(id.x),y),0).rgb;
        sum += sample * gaussian_weights[abs(i)];

    }

    textureStore(output_tex,vec2<u32>(id.x,id.y),vec4<f32>(sum,transparency));
}
