import type { Scene } from "../../scene/scene.js";
import { type Entity, type EntityId } from "../../scene/types.js";
import {
  GameTag,
  getGameComponent,
  type UIComponent,
  MISSION_ENTITY_ID,
} from "./components.js";
import type { GameSystem, SystemContext } from "./system.js";

export class UISystem implements GameSystem {
  readonly id = "ui";
  private readonly containerId: string;
  private rootElement: HTMLElement | null = null;
  private elementCache = new Map<EntityId, HTMLElement>();

  /**
   * Stored rendered HTML strings per entity. Exposing this allows headless/Node
   * tests to assert on compiled output without requiring JSDOM/document globals.
   */
  readonly renderedOutput = new Map<EntityId, string>();

  constructor(containerId = "hud-overlay") {
    this.containerId = containerId;
  }

  private getRootElement(): HTMLElement | null {
    if (typeof document === "undefined") return null;
    if (this.rootElement) return this.rootElement;

    let el = document.getElementById(this.containerId);
    if (!el) {
      el = document.createElement("div");
      el.id = this.containerId;
      // Inject baseline styles to keep UI layer clean above canvas
      el.style.position = "absolute";
      el.style.top = "0";
      el.style.left = "0";
      el.style.width = "100%";
      el.style.height = "100%";
      el.style.pointerEvents = "none";
      document.body.appendChild(el);
    }
    this.rootElement = el;
    return el;
  }

  /**
   * Helper to interpolate a single template using component bindings.
   */
  private renderTemplate(scene: Scene, entity: Entity, ui: UIComponent): string {
    let result = ui.template;

    for (const [key, binding] of Object.entries(ui.bindings)) {
      let val: any = undefined;

      // 1. Resolve variable from current entity
      const comp = getGameComponent<any>(entity, binding.componentTag as any);
      if (comp && binding.fieldKey in comp) {
        val = comp[binding.fieldKey];
      }

      // 2. Fallback: resolve from Mission singleton if applicable
      if (val === undefined) {
        const missionEntity = scene.get(MISSION_ENTITY_ID);
        if (missionEntity) {
          const missionComp = getGameComponent<any>(missionEntity, binding.componentTag as any);
          if (missionComp && binding.fieldKey in missionComp) {
            val = missionComp[binding.fieldKey];
          }
        }
      }

      // 3. Fallback: search general scene entities for first matching component
      if (val === undefined) {
        for (const ent of scene) {
          const comp = getGameComponent<any>(ent, binding.componentTag as any);
          if (comp && binding.fieldKey in comp) {
            val = comp[binding.fieldKey];
            break;
          }
        }
      }

      // Convert value to a readable string format
      let valStr = "";
      if (val !== undefined && val !== null) {
        if (typeof val === "object") {
          if (Array.isArray(val)) {
            valStr = val.join(", ");
          } else {
            valStr = JSON.stringify(val);
          }
        } else {
          valStr = String(val);
        }
      }

      // Replace target placeholder in the template: {{key}} or {{ key }}
      const escapedKey = key.replace(/[-\/\\^$*+?.()|[\]{}]/g, '\\$&');
      const regex = new RegExp(`{{\\s*${escapedKey}\\s*}}`, "g");
      result = result.replace(regex, valStr);
    }

    return result;
  }

  update(scene: Scene, _dt: number, _ctx: SystemContext): void {
    const root = this.getRootElement();
    const activeEntityIds = new Set<EntityId>();

    for (const entity of scene) {
      const ui = getGameComponent<UIComponent>(entity, GameTag.UI);
      if (!ui) continue;

      activeEntityIds.add(entity.id);

      // Compute current interpolated HTML
      const rendered = this.renderTemplate(scene, entity, ui);
      this.renderedOutput.set(entity.id, rendered);

      // DOM updates (browser only)
      if (rendered !== ui.lastRendered) {
        ui.lastRendered = rendered;

        if (root) {
          let child = this.elementCache.get(entity.id);
          if (!child) {
            child = document.createElement("div");
            child.id = `ui-entity-${entity.id}`;
            child.className = "ui-element";
            child.style.position = "absolute";
            child.style.pointerEvents = "auto";
            root.appendChild(child);
            this.elementCache.set(entity.id, child);
          }
          child.innerHTML = rendered;
        }
      }
    }

    // Clean up stale DOM elements + cache entries for entities removed or UI detached
    for (const [id, element] of this.elementCache.entries()) {
      if (!activeEntityIds.has(id)) {
        if (root && element.parentNode === root) {
          root.removeChild(element);
        }
        this.elementCache.delete(id);
      }
    }

    // Clean up internal rendered output map for removed entities
    for (const id of this.renderedOutput.keys()) {
      if (!activeEntityIds.has(id)) {
        this.renderedOutput.delete(id);
      }
    }
  }

  /**
   * Helper to manually reset/clear elements (useful during tests or scene change).
   */
  clear(): void {
    const root = this.getRootElement();
    if (root) {
      for (const element of this.elementCache.values()) {
        if (element.parentNode === root) {
          root.removeChild(element);
        }
      }
    }
    this.elementCache.clear();
    this.renderedOutput.clear();
  }
}
