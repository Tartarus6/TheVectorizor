// Zoom & pan for an image inside a clipping container.
//
// The image is transformed with `transform-origin: 0 0` and `translate(tx, ty) scale(s)`, so a point
// at image-space coordinate u lands at (layoutOrigin + t + u * s). Zooming about a cursor position
// therefore only needs t' = p - (p - t) * (s' / s), where p is the cursor relative to the image's
// untransformed layout origin. That keeps the point under the cursor fixed.

const MIN_SCALE = 1
const MAX_SCALE = 8
const DOUBLE_TAP_TIME = 185 // milliseconds
const SETTLE_SCALE = 1.3 // below this scale, zooming out eases the image back to its resting position
const WHEEL_ZOOM_SPEED = 0.0015 // exponent per pixel of wheel delta

type InstanceState = 'idle' | 'singleGesture' | 'multiGesture' | 'mouse'

const clamp = (value: number, min: number, max: number) => Math.max(min, Math.min(value, max))

const getPinchDistance = (event: TouchEvent): number =>
	Math.hypot(
		event.touches[0].clientX - event.touches[1].clientX,
		event.touches[0].clientY - event.touches[1].clientY
	)

const getMidPoint = (event: TouchEvent): { x: number; y: number } => ({
	x: (event.touches[0].clientX + event.touches[1].clientX) / 2,
	y: (event.touches[0].clientY + event.touches[1].clientY) / 2
})

export const addZoomPan = ({ container, image }: { container: HTMLElement; image: HTMLImageElement }) => {
	let scale = MIN_SCALE
	let tx = 0
	let ty = 0

	let state: InstanceState = 'idle'
	let deviceHasTouch = false
	let lastTapTime = 0
	let lastPinchDistance = 0
	let lastX = 0
	let lastY = 0

	image.style.transformOrigin = '0 0'

	const updateCursor = () => {
		container.style.cursor = scale === MIN_SCALE ? 'zoom-in' : 'move'
	}

	// Keep the scaled image covering the container. An axis where the image is smaller than the container
	// is left free while zooming (so it anchors on the cursor); when panning it just stays inside the container.
	const clampTranslate = (baseLeft: number, baseTop: number, panning: boolean) => {
		const c = container.getBoundingClientRect()
		const w = image.offsetWidth * scale
		const h = image.offsetHeight * scale

		const fit = (base: number, t: number, size: number, cStart: number, cSize: number) => {
			const lo = cStart + cSize - size - base
			const hi = cStart - base
			if (size >= cSize) return clamp(t, lo, hi)
			return panning ? clamp(t, hi, lo) : t
		}

		tx = fit(baseLeft, tx, w, c.left, c.width)
		ty = fit(baseTop, ty, h, c.top, c.height)
	}

	const render = () => {
		image.style.transform = `translate(${tx}px, ${ty}px) scale(${scale})`
		updateCursor()
	}

	// Zoom to newScale keeping the image point under (clientX, clientY) stationary.
	const zoomAt = (clientX: number, clientY: number, newScale: number) => {
		const rect = image.getBoundingClientRect()
		const baseLeft = rect.left - tx
		const baseTop = rect.top - ty
		const px = clientX - baseLeft
		const py = clientY - baseTop

		newScale = clamp(newScale, MIN_SCALE, MAX_SCALE)
		const k = newScale / scale
		tx = px - (px - tx) * k
		ty = py - (py - ty) * k

		// Zooming out, ease the offset to the origin over the last stretch so reaching 1x doesn't jump.
		if (newScale < scale) {
			const from = Math.min(scale, SETTLE_SCALE) - 1
			const to = Math.min(newScale, SETTLE_SCALE) - 1
			const f = from > 0 ? to / from : 0
			tx *= f
			ty *= f
		}
		scale = newScale

		clampTranslate(baseLeft, baseTop, false)
		render()
	}

	const panBy = (dx: number, dy: number) => {
		const rect = image.getBoundingClientRect()
		const baseLeft = rect.left - tx
		const baseTop = rect.top - ty
		tx += dx
		ty += dy
		clampTranslate(baseLeft, baseTop, true)
		render()
	}

	const reset = () => {
		scale = MIN_SCALE
		tx = 0
		ty = 0
		state = 'idle'
		lastTapTime = 0
		render()
	}

	const toggleZoom = (x: number, y: number) => {
		if (scale < MAX_SCALE) {
			zoomAt(x, y, MAX_SCALE)
		} else {
			reset()
		}
	}

	const onStart = (event: TouchEvent) => {
		deviceHasTouch = true
		if (state === 'multiGesture') return

		if (event.touches.length === 2) {
			const { x, y } = getMidPoint(event)
			lastX = x
			lastY = y
			lastPinchDistance = getPinchDistance(event)
			lastTapTime = 0 // prevent misinterpreting as a double tap
			state = 'multiGesture'
			return
		}

		if (event.touches.length !== 1) {
			state = 'idle'
			return
		}

		state = 'singleGesture'
		lastX = event.touches[0].clientX
		lastY = event.touches[0].clientY
	}

	const onMove = (event: TouchEvent) => {
		if (state === 'idle') return

		if (state === 'multiGesture' && event.touches.length === 2) {
			event.preventDefault()
			const { x, y } = getMidPoint(event)
			const distance = getPinchDistance(event)

			zoomAt(x, y, scale * (distance / lastPinchDistance))
			panBy(x - lastX, y - lastY)

			lastPinchDistance = distance
			lastX = x
			lastY = y
			return
		}

		if (scale === MIN_SCALE || state !== 'singleGesture' || event.touches.length !== 1) return
		event.preventDefault()

		const [touch] = event.touches
		panBy(touch.clientX - lastX, touch.clientY - lastY)
		lastX = touch.clientX
		lastY = touch.clientY
	}

	const onEndTouch = (event: TouchEvent) => {
		if (state === 'idle' || event.touches.length !== 0) return

		const now = Date.now()
		const tapLength = now - lastTapTime

		if (tapLength < DOUBLE_TAP_TIME && tapLength > 0) {
			event.preventDefault()
			const [touch] = event.changedTouches
			if (touch) toggleZoom(touch.clientX, touch.clientY)
		}

		lastTapTime = now
		state = 'idle'
	}

	const onWheel = (event: WheelEvent) => {
		if (deviceHasTouch) return
		event.preventDefault()

		const delta = event.deltaMode === 1 ? event.deltaY * 16 : event.deltaY
		zoomAt(event.clientX, event.clientY, scale * Math.exp(-clamp(delta, -200, 200) * WHEEL_ZOOM_SPEED))
	}

	const onMouseMove = (event: MouseEvent) => {
		if (deviceHasTouch) return
		if (event.buttons !== 1 || scale === MIN_SCALE) return
		event.preventDefault()

		if (event.movementX === 0 && event.movementY === 0) return

		state = 'mouse'
		panBy(event.movementX, event.movementY)
	}

	const onMouseEnd = () => {
		if (deviceHasTouch) return
		state = 'idle'
	}

	const onMouseUp = (event: MouseEvent) => {
		if (deviceHasTouch) return
		if (state !== 'mouse') toggleZoom(event.clientX, event.clientY)
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

	updateCursor()

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

	return { reset, destroy }
}
