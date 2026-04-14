import {Composition} from 'remotion';
import {DiagonalBars} from './DiagonalBars';

export const Root = () => (
  <Composition
    id="DiagonalBars"
    component={DiagonalBars}
    durationInFrames={900} // 30 seconds at 30 fps
    fps={30}
    width={1920}
    height={1080}
  />
);
