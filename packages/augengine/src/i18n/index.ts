/**
 * i18n — Phase 12. ICU MessageFormat-lite localization.
 *
 * Per-locale string bundles. Fallback chain: requested → en → key. Plural
 * + select via ICU subset. No external dep.
 */

export type Locale = string; // BCP 47 — "en-US", "ja-JP", etc.

export interface MessageBundle {
  locale: Locale;
  /** Map of message id → ICU template string. */
  messages: Record<string, string>;
}

const ICU_SELECT_RE = /\{(\w+),\s*(plural|select),\s*([\s\S]*?)\}/;

export class I18n {
  private bundles = new Map<Locale, Record<string, string>>();
  private fallback: Locale = "en";
  private current: Locale = "en";

  register(bundle: MessageBundle): void {
    this.bundles.set(bundle.locale, bundle.messages);
  }

  setLocale(locale: Locale): void { this.current = locale; }

  /** Render a localized message. Unknown keys return the key itself. */
  t(id: string, vars: Record<string, string | number> = {}): string {
    const tries = [this.current, this.shortLocale(this.current), this.fallback];
    for (const loc of tries) {
      const bundle = this.bundles.get(loc);
      if (bundle && id in bundle) {
        return this.render(bundle[id]!, vars);
      }
    }
    return id;
  }

  private shortLocale(loc: Locale): Locale {
    const dash = loc.indexOf("-");
    return dash >= 0 ? loc.slice(0, dash) : loc;
  }

  private render(template: string, vars: Record<string, string | number>): string {
    let out = template;
    // ICU plural/select first.
    const match = ICU_SELECT_RE.exec(out);
    if (match) {
      const [whole, varName, kind, body] = match;
      const value = vars[varName!];
      const branches = parseBranches(body!);
      const branchKey = kind === "plural"
        ? choosePluralBranch(branches, Number(value))
        : (branches[String(value)] ?? branches.other);
      const expanded = (branchKey ?? "").replace(/#/g, String(value ?? ""));
      out = out.replace(whole!, expanded);
    }
    // Simple {var} substitutions.
    return out.replace(/\{(\w+)\}/g, (_, k) => {
      const v = vars[k];
      return v === undefined ? `{${k}}` : String(v);
    });
  }
}

function parseBranches(body: string): Record<string, string> {
  // Branch syntax: `=N {text}` or `keyword {text}`. We greedy-match {...} bodies.
  const out: Record<string, string> = {};
  const re = /(=?\w+)\s*\{((?:[^{}]|\{[^{}]*\})*)\}/g;
  let m: RegExpExecArray | null;
  while ((m = re.exec(body)) !== null) {
    out[m[1]!] = m[2]!;
  }
  return out;
}

function choosePluralBranch(branches: Record<string, string>, n: number): string | undefined {
  if (`=${n}` in branches) return branches[`=${n}`];
  if (n === 0 && "zero" in branches) return branches.zero;
  if (n === 1 && "one" in branches) return branches.one;
  if (n === 2 && "two" in branches) return branches.two;
  if (n >= 3 && n <= 10 && "few" in branches) return branches.few;
  if (n >= 11 && "many" in branches) return branches.many;
  return branches.other;
}
