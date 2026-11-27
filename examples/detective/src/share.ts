/**
 * "Share your verdict": a spoiler-free line of text and a 1200×630 picture
 * card drawn on a canvas, both made on the device. No tracking, no upload,
 * no third-party share widgets — the player copies or saves them.
 */
export type ShareVerdict = "solved" | "weak" | "wrong";

export interface ShareStats {
  turns: number;
  cluesFound: number;
  cluesTotal: number;
  liesCracked: number;
  liesTotal: number;
}

export function verdictLabel(v: ShareVerdict): string {
  return v === "solved" ? "Case closed" : v === "weak" ? "Right suspect, weak case" : "The killer walked free";
}

/** Never names the killer, method or motive, so it is safe to post. */
export function shareText(o: { title: string; verdict: ShareVerdict; stats: ShareStats; url: string }): string {
  const s = o.stats;
  return (
    `${o.title} (an AugEngine detective case): ${verdictLabel(o.verdict).toLowerCase()}. ` +
    `${s.cluesFound}/${s.cluesTotal} clues, ${s.liesCracked}/${s.liesTotal} lies cracked, ${s.turns} moves. ` +
    `Can you do better? ${o.url}`
  );
}

function loadImage(url: string): Promise<HTMLImageElement | null> {
  return new Promise((resolve) => {
    const img = new Image();
    img.onload = () => resolve(img);
    img.onerror = () => resolve(null);
    img.src = url;
  });
}

function wrap(ctx: CanvasRenderingContext2D, text: string, maxWidth: number): string[] {
  const words = text.split(/\s+/);
  const lines: string[] = [];
  let line = "";
  for (const w of words) {
    const next = line ? `${line} ${w}` : w;
    if (ctx.measureText(next).width > maxWidth && line) {
      lines.push(line);
      line = w;
    } else line = next;
  }
  if (line) lines.push(line);
  return lines;
}

export async function drawShareCard(o: {
  title: string;
  tagline: string;
  verdict: ShareVerdict;
  stats: ShareStats;
  portraitUrl: string;
  footer: string;
}): Promise<Blob> {
  const W = 1200;
  const H = 630;
  const c = document.createElement("canvas");
  c.width = W;
  c.height = H;
  const ctx = c.getContext("2d");
  if (!ctx) throw new Error("no 2D canvas");
  const bg = ctx.createLinearGradient(0, 0, W, H);
  bg.addColorStop(0, "#0d0f16");
  bg.addColorStop(1, "#1b1510");
  ctx.fillStyle = bg;
  ctx.fillRect(0, 0, W, H);
  // Portrait (the victim), same origin so the canvas stays exportable.
  const img = await loadImage(o.portraitUrl);
  const px = 60;
  const py = 70;
  const pw = 368;
  const ph = 490;
  if (img) {
    ctx.save();
    ctx.beginPath();
    ctx.rect(px, py, pw, ph);
    ctx.clip();
    const scale = Math.max(pw / img.width, ph / img.height);
    ctx.drawImage(img, px + (pw - img.width * scale) / 2, py + (ph - img.height * scale) / 2, img.width * scale, img.height * scale);
    ctx.restore();
  }
  ctx.strokeStyle = "#6b5a3e";
  ctx.lineWidth = 4;
  ctx.strokeRect(px, py, pw, ph);
  const x = 480;
  const gold = "#d9a441";
  ctx.fillStyle = "#a59e90";
  ctx.font = "26px Georgia, 'Times New Roman', serif";
  ctx.fillText("AN AUGENGINE DETECTIVE CASE", x, 110);
  ctx.fillStyle = gold;
  ctx.font = "bold 64px Georgia, 'Times New Roman', serif";
  ctx.fillText(o.title, x, 185);
  ctx.fillStyle = "#cfc6b5";
  ctx.font = "italic 26px Georgia, 'Times New Roman', serif";
  wrap(ctx, o.tagline, W - x - 60).slice(0, 2).forEach((l, i) => ctx.fillText(l, x, 232 + i * 34));
  ctx.fillStyle = o.verdict === "solved" ? "#6fbf73" : "#d0564c";
  ctx.font = "bold 46px Georgia, 'Times New Roman', serif";
  ctx.fillText(verdictLabel(o.verdict), x, 350);
  ctx.fillStyle = "#ece6da";
  ctx.font = "30px Georgia, 'Times New Roman', serif";
  const s = o.stats;
  ctx.fillText(`${s.cluesFound} of ${s.cluesTotal} clues found`, x, 412);
  ctx.fillText(`${s.liesCracked} of ${s.liesTotal} lies cracked`, x, 456);
  ctx.fillText(`${s.turns} moves`, x, 500);
  ctx.fillStyle = "#a59e90";
  ctx.font = "24px Georgia, 'Times New Roman', serif";
  ctx.fillText(o.footer, x, 562);
  return new Promise((resolve, reject) => c.toBlob((b) => (b ? resolve(b) : reject(new Error("toBlob failed"))), "image/png"));
}
