# Characters in AugEngine

How to get a rigged, animated character into an AugEngine scene, and what the engine
does with it. AugEngine *plays* characters (skinning, glTF clips, retargeting, IK,
ARKit-52 lip-sync morphs, personas with memory); it doesn't sculpt them. Pick a route
below to get the model.

> Public edition. The development monorepo also has an editor (AugEngine Studio) with an
> "Add character" wizard and an auto-rigger; neither is in this repository yet.

## Routes

| Route | You get | Cost | Notes |
|---|---|---|---|
| **Mixamo / AccuRIG** | Auto-rigged humanoid + large clip library | Free | Upload your mesh, download FBX (with skin), convert to `.glb` (e.g. Blender). Mixamo terms: no redistribution of raw files, no ML training. |
| **VRoid Studio** | Anime-style character, rigged, with face blendshapes | Free | Export VRM, convert to glTF (VRM is glTF-based; Blender VRM add-on → glb). |
| **MetaHuman** | Very realistic human | Free under $1M revenue (any engine since 5.6) | Export via Blender/Maya → glTF. Heavy for laptop GPUs — one hero, not a cast. |
| **Blender + Rigify / Maya** | Full control | Your time | Export glTF Binary (`.glb`). Apply scale first so 1 unit = 1 metre. |

Every route ends the same way: a skinned `.glb`, loaded with `loadGltfIntoRenderer` (see
[GETTING-STARTED.md](GETTING-STARTED.md)). A skinned model comes in as a rig (one entity
per bone), deforms, and plays its glTF clips.

## Retargeting

`retargetClip` (engine, `characters/retarget-clip.ts`) moves a clip between two
glTF skeletons:

- **Bone names** are matched through canonical humanoid ids
  (`characters/humanoid.ts`): Mixamo `mixamorig:LeftUpLeg`, VRM/VRoid
  `J_Bip_L_UpperLeg`, Unreal `thigh_l`, Rigify `DEF-thigh.L`, Quaternius
  `UpperLeg.L`, AugEngine `LeftUpperLeg`. The spine chain is assigned by
  structure (rigs disagree on Spine/Spine1/Chest/Torso naming).
- **World-space deltas**, mapped through each rig's facing (left/up/forward from
  the hips), so a rig facing −Z plays a +Z clip correctly.
- **Rest-pose correction for limbs only**: a T-pose clip on an A-pose rig points
  arms/legs the same way; spine, neck and head keep the target's own posture.
- **Hips travel** scaled by hip height; *in place* drops horizontal travel.

## IK

`applyIkComponents(scene, { heightAt })` (engine, `characters/ik-runtime.ts`)
runs after animation:

- `ik_two_bone` — exact two-bone solve (bone lengths kept, bend toward a pole or
  the current knee/elbow). `mode: "ground"` plants the tip at
  `heightAt(x, z) + groundOffset` (pass a terrain sampler; flat ground by
  default); `mode: "target"` reaches an entity or a fixed point. The chain must
  be connected (tip under mid under root): rigs that drive feet with separate
  IK control bones (Quaternius) are refused with that reason.
- `look_at` — turns a bone's axis toward an entity/point, clamped (default 70°).

## Faces and lip-sync

Name face morph targets after Apple's **ARKit-52** set (`jawOpen`,
`eyeBlinkLeft`, `mouthSmileLeft`…, list in `characters/humanoid.ts` `ARKIT_52`).
Lip-sync: live speech → in-browser viseme detection; scripted lines → bake offline
with NVIDIA Audio2Face (open-sourced Sept 2025; runs on an NVIDIA GPU). VRoid and
MetaHuman ship face shapes; generated (image-to-3D) meshes do not. The engine's
`buildFaceRig` (alpha) can add one: it finds the head, detects eyes/nose/mouth from the
texture and depth, and returns a face patch with all 52 ARKit morph targets
(`FACE_RIG_SHAPES`, audio2face channel order; `visemeFrameToFaceRig` maps visemes onto
them). Landmark detection is heuristic and can misplace features (in the 2026-11-27
review the mouth on a test mesh landed at eye height and blink moved the hair), so
inspect the result before shipping it. The editor in the development monorepo keeps it
off by default for that reason.

## Limits (honest list)

- **No LOD chain for skinned characters**: the rig mesh is one budget; switching
  skinned LODs at distance isn't implemented.
- **Retargeting** is humanoid ↔ humanoid; quadrupeds, tails, wings and props
  aren't mapped. Mirror-image rigs aren't detected.
- **Characters without face morphs get no lip-sync** until a face rig is added
  (VRoid/MetaHuman routes, or the alpha `buildFaceRig` with the caveat above).
