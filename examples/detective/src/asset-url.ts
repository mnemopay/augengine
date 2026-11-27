/// <reference types="vite/client" />
/**
 * Prefix a root-relative public path ("/portraits/…", "/busts/…") with Vite's base, so the
 * game works both at "/" and under a sub-path such as augengine.com/detective/.
 */
export function assetUrl(path: string, base: string = import.meta.env.BASE_URL): string {
  const b = base.endsWith("/") ? base : `${base}/`;
  return b + path.replace(/^\/+/, "");
}
