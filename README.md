# The Vectorizor

This is a webapp that can be used to turn simple raster (pixel) images into vector images.


## Goals/Limitations
The focus is on simple images. That means limited color palettes, no gradients, etc. Some examples of the targets would be simple logos or the art from Slime Rancher 2 (except for the few that have gradients).


## Stages
The following are the basic stages of the vectorization process. Certain small things are omitted.

0. Srgb → Oklab
    - *Srgb input colors are transformed to Oklab for uniform color math*
    - The input RGB values are also multiplied by the alpha values (converted to premultiplied).
1. Mean Shift Cluster
    - *Colors are clustered to remove noise in the original image*
    - Colors that are close to each other (in terms of pixel distance and also color distance) are made to be closer to the same color.
2. Gradient Calculation
    - *The gradient (derivative) of the image is taken*
    - Each pixel compares to those near it to find the magnitude and direction of color change that this pixel causes.
3. Gradient Maximizing
    - *Pixels with high gradient magnitude are marked as part of an edge (serving as "seeds" to build the edges from)*
    - At an edge, the colors of the image change suddenly. So pixels with high magnitude should be part of an edge.
    - Subpixel Offset:
        - This stage also calculates the "subpixel offset" of each pixel.
        - By comparing to the neighboring pixels in the gradient direction, a more precise subpixel position for the edge can be found.
4. Edge Tracing
    - *The edges are "grown" out from the "seeds" that were found in the Gradient Maximizing stage*
    - This step actually runs many many times, with each pass extending some edge on both sides.
    - For each pixel marked as an edge, look at the pixels "in front of" and "behind" it, pick the "best" ones, then mark those as edges
        - "in front of" and "behind" are based on the gradient direction
        - "best" pretty much means highest gradient magnitude
5. Face Tracing
    - *Follow connections between edge pixels to build faces*
    - This step also runs many times, with each step doubling the length explored by each edge pixel.
    - This uses a "pointer jumping" algorithm
    - While building the faces, the colors of the face is also averaged out
6. SVG Creation
    - *Build an SVG based on the faces*
    - This is the only step that isn't a shader
    - For each face, add all edges of that face to a polygon
    - Apply subpixel offsets (for smoother edges)
    - Color the polygons based on the colors identified in face tracing
