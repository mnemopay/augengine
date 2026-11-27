/**
 * Renderer-side mirror of `@mnemopay/augengine-audio2face`'s ARKit shape
 * count. We don't import from augengine-audio2face directly because that
 * would invert the dependency arrow (the engine depends on the audio
 * package, not vice versa).
 *
 * Keep this in sync with `augengine-audio2face/src/arkit-shapes.ts:ARKIT_SHAPE_COUNT`.
 * The pair is asserted equal in `blendshapes.test.ts` when the audio2face
 * package is available, so a drift fails CI.
 */

export const ARKIT_BLENDSHAPE_COUNT = 52;
