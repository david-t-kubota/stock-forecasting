---
name: remotion
description: Build, develop, and render programmatic videos with Remotion. Use when the user wants to create video animations with React code, set up a Remotion project, create video compositions, add animations, or render video files using the Remotion framework.
---

# Remotion Skill

Create programmatic videos with React and Remotion. This skill covers project setup, composition creation, animation patterns, previewing in Remotion Studio, and rendering.

## Project Setup

### New Remotion Project

```bash
npx create-video@latest
```

Bootstrap with a specific template:
```bash
npx create-video@latest --template blank
npx create-video@latest --template hello-world
npx create-video@latest --template react-three-fiber
```

### Add Remotion to an Existing React Project

```bash
npm install remotion @remotion/cli
```

Create the entry point `src/index.ts`:
```ts
import {registerRoot} from 'remotion';
import {Root} from './Root';

registerRoot(Root);
```

## Core Concepts

### Composition

A composition defines video dimensions, FPS, and duration:

```tsx
// src/Root.tsx
import {Composition} from 'remotion';
import {MyVideo} from './MyVideo';

export const Root = () => {
  return (
    <Composition
      id="MyVideo"
      component={MyVideo}
      durationInFrames={150}
      fps={30}
      width={1920}
      height={1080}
    />
  );
};
```

### Animation Hooks

```tsx
import {useCurrentFrame, useVideoConfig, interpolate, spring, Easing} from 'remotion';

const MyVideo = () => {
  const frame = useCurrentFrame();
  const {fps, durationInFrames, width, height} = useVideoConfig();

  // Linear interpolation: opacity 0→1 over first 30 frames
  const opacity = interpolate(frame, [0, 30], [0, 1], {
    extrapolateLeft: 'clamp',
    extrapolateRight: 'clamp',
  });

  // Spring animation
  const scale = spring({
    frame,
    fps,
    config: {damping: 10, stiffness: 100, mass: 1},
  });

  return (
    <div style={{opacity, transform: `scale(${scale})`}}>
      Hello world!
    </div>
  );
};
```

### Sequences

Arrange components at specific time positions:

```tsx
import {Sequence, AbsoluteFill} from 'remotion';

const Timeline = () => (
  <AbsoluteFill>
    <Sequence from={0} durationInFrames={60}>
      <Title />
    </Sequence>
    <Sequence from={60} durationInFrames={90}>
      <Content />
    </Sequence>
  </AbsoluteFill>
);
```

### Media Components

```tsx
import {Audio, Video, Img, AbsoluteFill, staticFile} from 'remotion';

<AbsoluteFill>
  <Video src={staticFile('video.mp4')} />
  <Audio src={staticFile('audio.mp3')} />
  <Img src={staticFile('image.png')} />
</AbsoluteFill>
```

Place static assets in the `public/` directory; reference them with `staticFile('filename')`.

## Workflow

Make a todo list and work through each step one at a time.

### 1. Understand Requirements

Clarify what video needs to be created:
- **Dimensions**: 1920×1080 (Full HD), 1080×1080 (social square), 1080×1920 (vertical/Reels)
- **Duration**: seconds × FPS = `durationInFrames` (e.g., 5 seconds at 30 fps = 150 frames)
- **Content**: animated text, images, charts, video clips, audio
- **Output format**: MP4 (default), WebM, GIF, etc.

### 2. Set Up Project

If Remotion is not installed:
- **New standalone video project**: `npx create-video@latest`
- **Add to existing React project**: `npm install remotion @remotion/cli`

Verify the setup launches correctly:
```bash
npx remotion studio
```

### 3. Create Composition

Define the composition in `src/Root.tsx` and create the component file (e.g., `src/MyVideo.tsx`). Set the correct `fps`, `width`, `height`, and `durationInFrames`.

### 4. Implement Animations

Use these Remotion primitives:
- `useCurrentFrame()` — current frame index (starts at 0)
- `useVideoConfig()` — fps, width, height, durationInFrames
- `interpolate(frame, inputRange, outputRange, options)` — map frame values to CSS/style values
- `spring({frame, fps, config})` — physics-based easing
- `<Sequence from={N}>` — offset a component's timeline
- `<AbsoluteFill>` — layer components that fill the composition

### 5. Preview in Studio

```bash
npx remotion studio
```

Open `http://localhost:3000`. Use the timeline scrubber to inspect every frame and verify animations behave correctly end-to-end.

### 6. Render Video

```bash
# Render the default composition
npx remotion render

# Render a specific composition to a named output
npx remotion render src/index.ts MyVideo output.mp4

# Common render options
npx remotion render src/index.ts MyVideo output.mp4 \
  --codec=h264 \
  --crf=18 \
  --concurrency=4
```

Render option reference:
| Option | Values | Notes |
|---|---|---|
| `--codec` | `h264`, `h265`, `vp8`, `vp9`, `prores`, `gif` | Default: `h264` |
| `--crf` | `0–51` | Lower = better quality; 18 is visually lossless for H.264 |
| `--concurrency` | integer | Parallel rendering threads |
| `--frames` | `0-150` | Render only a frame range |
| `--image-format` | `jpeg`, `png` | Per-frame format |

### 7. Validate Output

- Open the rendered file and confirm it plays correctly
- Verify duration, resolution, and frame rate match the composition settings
- Test on the target platform (web embed, social upload, etc.)

## Configuration

### remotion.config.ts

```ts
import {Config} from '@remotion/cli/config';

Config.setVideoImageFormat('jpeg');
Config.setOverwriteOutput(true);
Config.setConcurrency(4);
```

### package.json Scripts

```json
{
  "scripts": {
    "studio": "remotion studio",
    "build": "remotion render",
    "render": "remotion render src/index.ts MyVideo out/video.mp4"
  }
}
```

## Common Patterns

### Fade In / Fade Out

```tsx
const opacity = interpolate(
  frame,
  [0, 20, durationInFrames - 20, durationInFrames],
  [0, 1, 1, 0],
  {extrapolateLeft: 'clamp', extrapolateRight: 'clamp'},
);
```

### Slide In from Left

```tsx
import {Easing} from 'remotion';

const x = interpolate(frame, [0, 30], [-width, 0], {
  extrapolateLeft: 'clamp',
  extrapolateRight: 'clamp',
  easing: Easing.out(Easing.quad),
});

<div style={{transform: `translateX(${x}px)`}}>...</div>
```

### Spring Pop-in

```tsx
const scale = spring({frame, fps, config: {damping: 12, stiffness: 200}});

<div style={{transform: `scale(${scale})`}}>...</div>
```

### Typewriter Text Effect

```tsx
const chars = Math.floor(
  interpolate(frame, [0, 60], [0, text.length], {extrapolateRight: 'clamp'}),
);
const visibleText = text.slice(0, chars);
```

### Staggered List Items

```tsx
{items.map((item, i) => {
  const delay = i * 5; // 5-frame stagger between items
  const opacity = interpolate(frame, [delay, delay + 20], [0, 1], {
    extrapolateLeft: 'clamp',
    extrapolateRight: 'clamp',
  });
  return <div key={i} style={{opacity}}>{item}</div>;
})}
```

## Wrap Up

After completing the implementation, provide the user a summary in this format:

* **Compositions created**: list each composition ID, resolution, FPS, and duration
* **Validation results**:
  1. ✅/‼️ Studio preview (`npx remotion studio` — scrubber works through all frames)
  2. ✅/‼️ Render completed (output file path, file size, duration)
* **Render command used** (with full flags)
* **Next steps** (optional): deploying to Remotion Lambda for cloud rendering, or Remotion Player for embedding in a React app
