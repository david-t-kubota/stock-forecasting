import {AbsoluteFill, useCurrentFrame, useVideoConfig} from 'remotion';

// Light blue background
const BACKGROUND = '#B8E4F9';

// Each bar: color, visible width (px), gap after it (px)
// Varying thicknesses give the "different widths" look
const BARS = [
  {color: '#FF6B9D', width: 90,  gap: 160},  // pink      — wide
  {color: '#C084FC', width: 55,  gap: 240},  // purple    — thin
  {color: '#FCD34D', width: 130, gap: 100},  // yellow    — thickest
  {color: '#4ADE80', width: 75,  gap: 190},  // green     — medium
  {color: '#FB923C', width: 100, gap: 130},  // orange    — wide
  {color: '#F87171', width: 55,  gap: 210},  // coral     — thin
  {color: '#38BDF8', width: 85,  gap: 145},  // sky blue  — medium
  {color: '#2DD4BF', width: 65,  gap: 175},  // teal      — medium-thin
] as const;

// Total width of one repeating pattern tile
// 250 + 295 + 230 + 265 + 230 + 265 + 230 + 240 = 2005 px
const TILE_WIDTH = BARS.reduce((sum, b) => sum + b.width + b.gap, 0);

// Pre-calculate each bar's left edge within one tile
const BAR_X = BARS.map((_, i) =>
  BARS.slice(0, i).reduce((sum, b) => sum + b.width + b.gap, 0),
);

// How many tile copies to render side-by-side so the container is always
// wider than the visible diagonal (≈ 2203 px) even at maximum scroll offset.
// 6 copies = 12 030 px — far more than enough.
const NUM_COPIES = 6;

const ANGLE_DEG = -45;

export const DiagonalBars = () => {
  const frame = useCurrentFrame();
  const {durationInFrames, width, height} = useVideoConfig();

  // Enough height to cover the full screen regardless of rotation angle
  const containerH = Math.sqrt(width * width + height * height) * 1.5;

  // The container slides by exactly one TILE_WIDTH over the full 30 s,
  // so frame 0 and frame 900 (loop point) are visually identical → seamless.
  const offset = (frame / durationInFrames) * TILE_WIDTH;

  return (
    <AbsoluteFill style={{backgroundColor: BACKGROUND, overflow: 'hidden'}}>
      {/*
        The container is:
          1. centred on the screen  (translate -50%, -50%)
          2. rotated -45°           (rotate)
          3. scrolled by `offset`   (translateX — moves along the rotated axis)

        This makes the stripes appear to glide diagonally across the screen.
      */}
      <div
        style={{
          position: 'absolute',
          top: '50%',
          left: '50%',
          width: NUM_COPIES * TILE_WIDTH,
          height: containerH,
          transform: `translate(-50%, -50%) rotate(${ANGLE_DEG}deg) translateX(${-offset}px)`,
        }}
      >
        {Array.from({length: NUM_COPIES}).flatMap((_, tileIdx) =>
          BARS.map((bar, barIdx) => (
            <div
              key={`${tileIdx}-${barIdx}`}
              style={{
                position: 'absolute',
                left: tileIdx * TILE_WIDTH + BAR_X[barIdx],
                top: 0,
                width: bar.width,
                height: '100%',
                backgroundColor: bar.color,
                opacity: 0.88,
              }}
            />
          )),
        )}
      </div>
    </AbsoluteFill>
  );
};
