// Source: this code is from https://dev.to/neophen/zoom-pan-clamped-to-a-container-2lp

type State = {
	container: HTMLElement
	element: HTMLElement
	minScale: number
	maxScale: number
	scaleSensitivity: number
	accumulatedDeltaScale: number
	transform: {
		originOffset: boolean
		originX: number
		originY: number
		translateX: number
		translateY: number
		scale: number
	}
}

const DEFAULT_TRANSFORMATION = {
	originOffset: false,
	originX: 0,
	originY: 0,
	translateX: 0,
	translateY: 0,
	scale: 1
}

const hasPositionChanged = ({ pos, prevPos }: { pos: number; prevPos: number }) => pos !== prevPos

const valueInRange = ({ minScale, maxScale, transform: { scale } }: State) => scale <= maxScale && scale >= minScale

const getTranslate =
	(state: State) =>
	({ pos, axis }: { pos: number; axis: 'x' | 'y' }) => {
		const { originX, originY, translateX, translateY, scale } = state.transform
		const axisIsX = axis === 'x'
		const prevPos = axisIsX ? originX : originY
		const translate = axisIsX ? translateX : translateY

		return valueInRange(state) && hasPositionChanged({ pos, prevPos })
			? translate + (pos - prevPos * scale) * (1 - 1 / scale)
			: translate
	}

const getMatrix = ({ scale, translateX, translateY }: { scale: number; translateX: number; translateY: number }) =>
	`matrix(${scale}, 0, 0, ${scale}, ${translateX}, ${translateY})`

const clamp = (value: number, min: number, max: number) => Math.max(Math.min(value, max), min)

const getNewScale = (deltaScale: number, { transform: { scale }, minScale, maxScale, scaleSensitivity }: State) => {
	const newScale = scale + deltaScale / (scaleSensitivity / scale)
	return clamp(newScale, minScale, maxScale)
}

const clampedTranslate = ({ axis, translate, state }: { axis: 'x' | 'y'; translate: number; state: State }) => {
	const { scale, originX, originY } = state.transform
	const axisIsX = axis === 'x'
	const origin = axisIsX ? originX : originY
	const axisKey = axisIsX ? 'offsetWidth' : 'offsetHeight'

	const containerSize = state.container[axisKey]
	const imageSize = state.element[axisKey]
	const bounds = state.element.getBoundingClientRect()

	const imageScaledSize = axisIsX ? bounds.width : bounds.height

	const defaultOrigin = imageSize / 2
	const originOffset = (origin - defaultOrigin) * (scale - 1)

	const range = Math.max(0, Math.round(imageScaledSize) - containerSize)

	const max = Math.round(range / 2)
	const min = 0 - max

	return clamp(translate, min + originOffset, max + originOffset)
}

const renderClamped = ({ state, translateX, translateY }: { state: State; translateX: number; translateY: number }) => {
	const { originX, originY, scale } = state.transform
	state.transform.translateX = clampedTranslate({ axis: 'x', translate: translateX, state })
	state.transform.translateY = clampedTranslate({ axis: 'y', translate: translateY, state })

	requestAnimationFrame(() => {
		if (state.transform.originOffset) {
			state.element.style.transformOrigin = `${originX}px ${originY}px`
		}
		state.element.style.transform = getMatrix({
			scale,
			translateX: state.transform.translateX,
			translateY: state.transform.translateY
		})
	})
}

const pan = (state: State, { originX, originY }: { originX: number; originY: number }) => {
	renderClamped({
		state,
		translateX: state.transform.translateX + originX,
		translateY: state.transform.translateY + originY
	})
}

const canPan = (state: State) => ({
	panBy: (origin: { originX: number; originY: number }) => pan(state, origin),
	panTo: ({ originX, originY, scale }: { originX: number; originY: number; scale: number }) => {
		state.transform.scale = clamp(scale, state.minScale, state.maxScale)

		pan(state, {
			originX: originX - state.transform.translateX,
			originY: originY - state.transform.translateY
		})
	}
})

const canZoom = (state: State) => ({
	zoomPan: ({
		scale: scaleValue,
		x,
		y,
		deltaX,
		deltaY
	}: {
		scale: number
		x: number
		y: number
		deltaX: number
		deltaY: number
	}) => {
		const {
			minScale,
			maxScale,
			transform: { scale }
		} = state
		const newScale = clamp(scaleValue, minScale, maxScale)
		const { left, top } = state.element.getBoundingClientRect()
		const originX = x - left
		const originY = y - top
		const newOriginX = originX / scale
		const newOriginY = originY / scale
		const translate = getTranslate(state)
		const translateX = translate({ pos: originX, axis: 'x' })
		const translateY = translate({ pos: originY, axis: 'y' })

		state.transform = {
			originOffset: true,
			originX: newOriginX,
			originY: newOriginY,
			translateX,
			translateY,
			scale: newScale
		}

		pan(state, { originX: deltaX, originY: deltaY })
	},
	zoom: ({ x, y, deltaScale }: { x: number; y: number; deltaScale: number }) => {
		const {
			element,
			transform: { scale }
		} = state
		const { left, top } = element.getBoundingClientRect()
		const newScale = getNewScale(deltaScale, state)
		const originX = x - left
		const originY = y - top
		const newOriginX = originX / scale
		const newOriginY = originY / scale

		const translate = getTranslate(state)
		const translateX = translate({ pos: originX, axis: 'x' })
		const translateY = translate({ pos: originY, axis: 'y' })

		state.transform = {
			...state.transform,
			originOffset: true,
			originX: newOriginX,
			originY: newOriginY,
			scale: newScale
		}

		renderClamped({ state, translateX, translateY })
	},
	zoomTo: ({ newScale, x, y }: { newScale: number; x: number; y: number }) => {
		const {
			element,
			transform: { scale }
		} = state

		const { left, top } = element.getBoundingClientRect()
		const originX = x - left
		const originY = y - top
		const newOriginX = originX / scale
		const newOriginY = originY / scale

		const translate = getTranslate(state)
		const translateX = translate({ pos: originX, axis: 'x' })
		const translateY = translate({ pos: originY, axis: 'y' })

		state.transform = {
			originOffset: true,
			originX: newOriginX,
			originY: newOriginY,
			scale: newScale,
			translateX,
			translateY
		}

		requestAnimationFrame(() => {
			state.element.style.transformOrigin = `${newOriginX}px ${newOriginY}px`
			state.element.style.transform = getMatrix({
				scale: newScale,
				translateX,
				translateY
			})
		})
	}
})

const canInspect = (state: State) => ({
	getScale: () => state.transform.scale,
	reset: () => {
		state.transform.scale = state.minScale
		pan(state, { originX: 0, originY: 0 })
		state.transform = DEFAULT_TRANSFORMATION
	},
	getState: () => state
})

export const renderer = ({
	minScale,
	maxScale,
	element,
	container,
	scaleSensitivity = 10
}: {
	container: HTMLElement
	element: HTMLElement
	minScale: number
	maxScale: number
	scaleSensitivity?: number
}) => {
	const state: State = {
		container,
		element,
		minScale,
		maxScale,
		scaleSensitivity,
		accumulatedDeltaScale: 0,
		transform: DEFAULT_TRANSFORMATION
	}

	return {
		...canZoom(state),
		...canPan(state),
		...canInspect(state)
	}
}

type InstanceState = 'idle' | 'singleGesture' | 'multiGesture' | 'mouse'

const MIN_SCALE = 1
const MAX_SCALE = 8
// TODO: isnt there a native way to handle double taps?
const DOUBLE_TAP_TIME = 185 // milliseconds

const stateIs = (state: InstanceState, ...states: InstanceState[]) => states.includes(state)

const getPinchDistance = (event: TouchEvent): number =>
	Math.hypot(event.touches[0].pageX - event.touches[1].pageX, event.touches[0].pageY - event.touches[1].pageY)

const getMidPoint = (event: TouchEvent): { x: number; y: number } => ({
	x: (event.touches[0].pageX + event.touches[1].pageX) / 2,
	y: (event.touches[0].pageY + event.touches[1].pageY) / 2
})

const onDoubleTap = ({
	instance,
	scale,
	x,
	y
}: {
	instance: ReturnType<typeof renderer>
	scale: number
	x: number
	y: number
}): number => {
	if (scale < MAX_SCALE) {
		instance.zoomTo({ newScale: MAX_SCALE, x, y })
		return MAX_SCALE
	} else {
		instance.reset()
		return MIN_SCALE
	}
}

export const addZoomPan = ({ container, image }: { container: HTMLElement; image: HTMLImageElement }) => {
	let state: InstanceState = 'idle'

	let scaleValue = 1;
	const currentScale = () => scaleValue
	const setCurrentScale = (value: number) => {
		scaleValue = value
		container.style.cursor = value === MIN_SCALE ? 'zoom-in' : 'move'
	}

	let lastTapTime = 0
	let deviceHasTouch = false
	let wheelTimeout: ReturnType<typeof setTimeout> | undefined

	const start: { x: number; y: number; distance: number; touches: Touch[] } = {
		x: 0,
		y: 0,
		distance: 0,
		touches: []
	}

	const instance = renderer({
		container,
		minScale: MIN_SCALE,
		maxScale: MAX_SCALE,
		element: image,
		scaleSensitivity: 20
	})

	const onStart = (event: TouchEvent) => {
		deviceHasTouch = true

		if (stateIs(state, 'multiGesture')) return

		const touchCount = event.touches.length

		if (touchCount === 2 && stateIs(state, 'idle', 'singleGesture')) {
			const { x, y } = getMidPoint(event)

			start.x = x
			start.y = y
			start.distance = getPinchDistance(event) / currentScale()
			start.touches = [event.touches[0], event.touches[1]]

			lastTapTime = 0 // Reset to prevent misinterpretation as a double tap
			state = 'multiGesture'
			return
		}

		if (touchCount !== 1) {
			state = 'idle'
			return
		}

		state = 'singleGesture'

		const [touch] = event.touches

		start.x = touch.pageX
		start.y = touch.pageY
		start.distance = 0
		start.touches = [touch]
	}

	const onMove = (event: TouchEvent) => {
		if (stateIs(state, 'idle')) return

		const touchCount = event.touches.length

		if (stateIs(state, 'multiGesture') && touchCount === 2) {
			event.preventDefault()
			const scale = getPinchDistance(event) / start.distance

			const { x, y } = getMidPoint(event)

			instance.zoomPan({ scale, x, y, deltaX: x - start.x, deltaY: y - start.y })

			start.x = x
			start.y = y
			return
		}

		if (
			currentScale() === MIN_SCALE ||
			!stateIs(state, 'singleGesture') ||
			touchCount !== 1 ||
			event.touches[0]?.identifier !== start.touches[0]?.identifier
		) {
			return
		}
		event.preventDefault()

		const [touch] = event.touches

		const deltaX = touch.pageX - start.x
		const deltaY = touch.pageY - start.y

		instance.panBy({ originX: deltaX, originY: deltaY })

		start.x = touch.pageX
		start.y = touch.pageY
	}

	const onEndTouch = (event: TouchEvent) => {
		if (stateIs(state, 'idle') || event.touches.length !== 0) {
			return
		}

		const currentTime = new Date().getTime()
		const tapLength = currentTime - lastTapTime

		if (tapLength < DOUBLE_TAP_TIME && tapLength > 0) {
			event.preventDefault()
			const [touch] = event.changedTouches
			if (!touch) return
			setCurrentScale(onDoubleTap({ instance, scale: currentScale(), x: touch.clientX, y: touch.clientY }))
		}

		lastTapTime = currentTime
		setCurrentScale(instance.getScale())
		state = 'idle'
	}

	const onWheel = (event: WheelEvent) => {
		if (deviceHasTouch) return
		event.preventDefault()
		instance.zoom({
			deltaScale: Math.sign(event.deltaY) > 0 ? -1 : 1,
			x: event.pageX,
			y: event.pageY
		})

		clearTimeout(wheelTimeout)
		wheelTimeout = setTimeout(() => {
			setCurrentScale(instance.getScale())
		}, 100)
	}

	const onMouseMove = (event: MouseEvent) => {
		if (deviceHasTouch) return
		if (event.buttons !== 1 || currentScale() === MIN_SCALE) {
			return
		}
		event.preventDefault()

		if (event.movementX === 0 && event.movementY === 0) {
			return
		}

		state = 'mouse'

		instance.panBy({ originX: event.movementX, originY: event.movementY })
	}

	const onMouseEnd = () => {
		if (deviceHasTouch) return
		state = 'idle'
		setCurrentScale(instance.getScale())
	}

	const onMouseUp = (event: MouseEvent) => {
		if (deviceHasTouch) return
		if (!stateIs(state, 'mouse')) {
			setCurrentScale(onDoubleTap({ instance, scale: currentScale(), x: event.pageX, y: event.pageY }))
		}
		onMouseEnd()
	}

	container.addEventListener('touchstart', onStart, { passive: false })
	container.addEventListener('touchmove', onMove, { passive: false })
	container.addEventListener('touchend', onEndTouch, { passive: false })
	container.addEventListener('touchcancel', onEndTouch, { passive: false })

	container.addEventListener('mousemove', onMouseMove, { passive: false })
	container.addEventListener('mouseup', onMouseUp, { passive: false })
	container.addEventListener('mouseleave', onMouseEnd, { passive: false })
	container.addEventListener('mouseout', onMouseEnd, { passive: false })
	container.addEventListener('wheel', onWheel, { passive: false })

	const reset = () => {
		state = 'idle'
		setCurrentScale(1)
		lastTapTime = 0

		start.x = 0
		start.y = 0
		start.distance = 0
		start.touches = []

		instance.reset()
	}

	const destroy = () => {
		container.removeEventListener('touchstart', onStart)
		container.removeEventListener('touchmove', onMove)
		container.removeEventListener('touchend', onEndTouch)
		container.removeEventListener('touchcancel', onEndTouch)

		container.removeEventListener('mousemove', onMouseMove)
		container.removeEventListener('mouseup', onMouseUp)
		container.removeEventListener('mouseleave', onMouseEnd)
		container.removeEventListener('mouseout', onMouseEnd)
		container.removeEventListener('wheel', onWheel)
	}

	return {
		reset,
		destroy
	}
}
