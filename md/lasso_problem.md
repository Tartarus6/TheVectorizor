# Lasso Problem

On more complex and larger images, the edge tracing does not work well.
The top half of the svg will be created, but the bottom half-ish is just left blank.

I believe (at least part of) the problem is that more complex loops are not handled well.

Here's an example:
    ╭─╮     ╭─╮
    ╰─┴─────┴─╯

When a shape like the one above is made in the edge mask, the middle won't be properly handled.

## There's 2 components to this issue

### *1*: These shapes are often created on larger images
This is definetly a problem because, once a loop is made, the edge tracing stops.

The pattern is so common because of how the edge tracing steps prioritize extending to existing edges.

Maybe face tracing can somehow be done thoughout the edge tracing steps, so it can know whether a choice would lead to a bad shape such as a lasso.


### *2*: These shapes aren't properly handled by the face tracing
These shapes were not considered when the code was being written, so it's untested behaviour.

I'm not sure exactly what is happening when complex shapes such as these are going through the face tracing.
