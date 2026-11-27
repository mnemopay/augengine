/** Bounded fixed-step scheduling. Overload is reported, never hidden in dt. */
export class FixedStepClock {
  private accumulator = 0;
  elapsed = 0;
  droppedSeconds = 0;
  constructor(
    readonly dt = 1 / 120,
    readonly maxSteps = 32,
  ) {
    if (
      !Number.isFinite(dt) ||
      dt <= 0 ||
      !Number.isSafeInteger(maxSteps) ||
      maxSteps < 1
    )
      throw new Error("Invalid fixed-step settings");
  }
  advance(seconds: number, tick: (dt: number) => void): number {
    if (!Number.isFinite(seconds) || seconds < 0)
      throw new Error("Simulation delta must be finite and nonnegative");
    const accepted = Math.min(seconds, this.dt * this.maxSteps);
    this.droppedSeconds += seconds - accepted;
    this.accumulator += accepted;
    let steps = 0;
    while (
      this.accumulator + this.dt * 1e-9 >= this.dt &&
      steps < this.maxSteps
    ) {
      tick(this.dt);
      this.accumulator = Math.max(0, this.accumulator - this.dt);
      this.elapsed += this.dt;
      steps++;
    }
    return steps;
  }
  get alpha(): number {
    return this.accumulator / this.dt;
  }
  reset(): void {
    this.accumulator = 0;
    this.elapsed = 0;
    this.droppedSeconds = 0;
  }
  snapshot(): {
    dt: number;
    maxSteps: number;
    accumulator: number;
    elapsed: number;
    droppedSeconds: number;
  } {
    return {
      dt: this.dt,
      maxSteps: this.maxSteps,
      accumulator: this.accumulator,
      elapsed: this.elapsed,
      droppedSeconds: this.droppedSeconds,
    };
  }
  restore(s: ReturnType<FixedStepClock["snapshot"]>): void {
    if (
      s.dt !== this.dt ||
      s.maxSteps !== this.maxSteps ||
      ![s.accumulator, s.elapsed, s.droppedSeconds].every(Number.isFinite) ||
      s.accumulator < 0 ||
      s.accumulator >= this.dt ||
      s.elapsed < 0 ||
      s.droppedSeconds < 0
    )
      throw new Error("Invalid fixed-clock snapshot");
    this.accumulator = s.accumulator;
    this.elapsed = s.elapsed;
    this.droppedSeconds = s.droppedSeconds;
  }
}
