/**
 * The Wrenfield Observatory — the hand-authored case.
 *
 * Everything here is ground truth. The rule engine reads it; dialogue (LLM or
 * offline) only ever rephrases lines chosen from it. Nothing an LLM writes can
 * change who did it, how, when, or which clue exists where.
 *
 * Original setting and characters (no existing franchise, game or novel).
 */

export type RoomId = "parlour" | "dome" | "library" | "kitchen" | "conservatory";
export type SuspectId = "mara" | "desmond" | "ines" | "teodor";
export type ClueId =
  | "C_FLASK"
  | "C_LOG"
  | "C_LETTER"
  | "C_WILL"
  | "C_IOU"
  | "C_CABINET"
  | "C_TEA_NOTE"
  | "C_GLOVE";
export type TestimonyId = "T_TEODOR_SAW_DESMOND" | "T_TEODOR_ARGUED" | "T_INES_GARDEN" | "T_MARA_CARRIED";
/** Anything that can sit in the notebook and be presented to a suspect. */
export type EvidenceId = ClueId | TestimonyId;

export type MethodId = "poison_tea" | "struck" | "fall" | "smothered";
export type MotiveId = "inheritance" | "stolen_research" | "credit" | "grudge";

export interface Room {
  id: RoomId;
  name: string;
  blurb: string;
}

export interface RoomObject {
  id: string;
  room: RoomId;
  name: string;
  /** What examining it says. */
  text: string;
  /** Clue the object yields (once). */
  clue?: ClueId;
}

export interface Evidence {
  id: EvidenceId;
  name: string;
  /** Notebook text — what the detective knows once it is found. */
  text: string;
  /** Words a player might use when asking about it (offline intent matching). */
  keywords: string[];
  /**
   * Distinctive terms that would leak this evidence if a suspect said them
   * before the player has found it. The guardrail rejects such dialogue.
   */
  leakTerms: string[];
  /** Which accusation link it can support. */
  supports: Array<"suspect" | "method" | "motive">;
  /** Suspect the evidence points at (for scoring the "suspect" link). */
  pointsAt?: SuspectId;
}

export interface Answer {
  /** Gist the suspect conveys (offline text; the LLM rephrases it). */
  line: string;
  /** Testimony this answer puts in the notebook. */
  reveals?: TestimonyId;
}

export interface LiePolicy {
  /** Topic the lie covers. */
  topic: Topic;
  /** What they say while the lie holds. */
  lie: string;
  /** What they say once cracked. */
  truth: string;
  /**
   * Cracks when the player has presented evidence matching ANY of these sets
   * (all ids within a set must have been presented to this suspect).
   */
  crackWhen: EvidenceId[][];
  /** Line spoken at the moment it cracks. */
  crackLine: string;
}

export type Topic =
  | "greeting"
  | "alibi"
  | "victim"
  | "time"
  | "motive"
  | `suspect:${SuspectId}`
  | `clue:${EvidenceId}`
  | `room:${RoomId}`
  | "unknown";

export interface Suspect {
  id: SuspectId;
  name: string;
  role: string;
  /** Short persona sheet for the LLM. */
  persona: string;
  voice: string;
  /** Bust colours: [coat, accent]. */
  palette: [[number, number, number], [number, number, number]];
  answers: Partial<Record<Topic, Answer>>;
  /** Reaction when shown a piece of evidence they have nothing scripted for. */
  shrug: string;
  lies: LiePolicy[];
  /**
   * Terms this suspect must never say (facts they don't know, or truths they
   * are hiding). The guardrail rejects LLM text containing any of them unless
   * the term also appears in the gist chosen by the rules.
   */
  forbiddenTerms: string[];
}

export interface CaseTruth {
  killer: SuspectId;
  method: MethodId;
  motive: MotiveId;
  /** Shown after the accusation. */
  explanation: string[];
}

export const CASE_TITLE = "The Wrenfield Observatory";
export const CASE_INTRO =
  "Midnight, Wrenfield Observatory. Professor Aurel Vance was found dead at his telescope at ten past midnight. " +
  "The doctor says his heart stopped around 23:40. Four people were in the house. The local magistrate gives you one night " +
  "and one accusation. Search the rooms, question the household, and name the killer, the method and the motive.";

export const VICTIM = { name: "Professor Aurel Vance", role: "astronomer, owner of Wrenfield Observatory" } as const;

export const ROOMS: Room[] = [
  { id: "parlour", name: "Parlour", blurb: "The household waits here under the magistrate's eye. Question them here." },
  { id: "dome", name: "Observatory Dome", blurb: "Cold air, the great brass telescope, and the chair where Vance was found." },
  { id: "library", name: "Library", blurb: "Star charts, a dying fire, and Vance's writing desk." },
  { id: "kitchen", name: "Kitchen", blurb: "Copper pans, the kettle still warm, and Ines's herb cabinet." },
  { id: "conservatory", name: "Conservatory", blurb: "Glass walls onto the dark garden, orchids, and wet planters." },
];

export const OBJECTS: RoomObject[] = [
  { id: "dome.flask", room: "dome", name: "Tea flask", clue: "C_FLASK", text: "A steel tea flask beside the telescope. A bitter, green-tinged residue clings to the bottom." },
  { id: "dome.log", room: "dome", name: "Observation log", clue: "C_LOG", text: "Vance's log. The last entry, 23:05: 'Tea bitter tonight. Heart racing — must be the cold.'" },
  { id: "dome.desk", room: "dome", name: "Crumpled letter", clue: "C_LETTER", text: "A letter signed T. Lisk: 'You stole my comet. I will see you answer for it.'" },
  { id: "dome.telescope", room: "dome", name: "Telescope", text: "Pointed at the eastern sky. Nothing disturbed; no sign of a struggle anywhere in the dome." },
  { id: "library.drawer", room: "library", name: "Desk drawer", clue: "C_WILL", text: "A draft of a new will, to be signed tomorrow. It leaves the observatory and estate to Mara Quell. Desmond Pike's share is struck out." },
  { id: "library.coat", room: "library", name: "Coat on the chair", clue: "C_IOU", text: "Desmond's coat. In the pocket, IOU notes for forty thousand pounds, due on the first of the month, signed D. Pike." },
  { id: "library.fire", room: "library", name: "Fireplace", text: "Burned down to embers. Nobody has fed it for hours." },
  { id: "kitchen.cabinet", room: "kitchen", name: "Herb cabinet", clue: "C_CABINET", text: "Unlocked. A bottle labelled 'foxglove tincture' is nearly empty, with fresh drips down its side." },
  { id: "kitchen.note", room: "kitchen", name: "Note on the counter", clue: "C_TEA_NOTE", text: "In Ines's hand: 'Tea for the Professor, 21:40. Mara to take up.'" },
  { id: "kitchen.kettle", room: "kitchen", name: "Kettle", text: "Still faintly warm. Two cups' worth of water gone." },
  { id: "conservatory.planter", room: "conservatory", name: "Planter by the door", clue: "C_GLOVE", text: "Pushed into the soil: a single kid glove monogrammed 'D.P.', stained green and smelling bitter." },
  { id: "conservatory.orchids", room: "conservatory", name: "Orchids", text: "Vance's prize orchids. Through the glass you can see the whole garden path." },
];

export const EVIDENCE: Record<EvidenceId, Evidence> = {
  C_FLASK: { id: "C_FLASK", name: "Poisoned tea flask", text: "Vance's tea flask has a bitter green residue. The tea was poisoned.", keywords: ["flask", "tea", "residue", "drink"], leakTerms: ["residue"], supports: ["method"] },
  C_LOG: { id: "C_LOG", name: "Observation log", text: "Last entry 23:05: the tea tasted bitter and his heart was racing. He was alive and already poisoned at 23:05.", keywords: ["log", "entry", "23:05", "notebook"], leakTerms: ["23:05", "observation log"], supports: ["method"] },
  C_LETTER: { id: "C_LETTER", name: "Teodor's angry letter", text: "Teodor Lisk accused Vance of stealing his comet discovery.", keywords: ["letter", "comet", "stole", "stolen"], leakTerms: ["see you answer"], supports: ["motive"], pointsAt: "teodor" },
  C_WILL: { id: "C_WILL", name: "Draft new will", text: "Tomorrow Vance would have signed a will leaving everything to Mara and cutting Desmond out.", keywords: ["will", "inheritance", "estate", "inherit"], leakTerms: ["new will"], supports: ["motive"], pointsAt: "desmond" },
  C_IOU: { id: "C_IOU", name: "Desmond's IOUs", text: "Desmond owes forty thousand pounds, due on the first.", keywords: ["iou", "debt", "debts", "money", "owe", "gambling"], leakTerms: ["iou", "forty thousand"], supports: ["motive"], pointsAt: "desmond" },
  C_CABINET: { id: "C_CABINET", name: "Foxglove tincture", text: "Ines's herb cabinet was unlocked; the foxglove tincture is nearly empty, freshly used.", keywords: ["cabinet", "foxglove", "tincture", "bottle", "herb"], leakTerms: ["foxglove", "tincture"], supports: ["method"] },
  C_TEA_NOTE: { id: "C_TEA_NOTE", name: "Tea note", text: "Ines left the tea on the counter at 21:40 for Mara to carry up. Anyone could reach it.", keywords: ["note", "counter", "21:40"], leakTerms: ["note on the counter"], supports: ["method"] },
  C_GLOVE: { id: "C_GLOVE", name: "Stained glove", text: "A glove monogrammed D.P., stained green and smelling of the tincture, hidden in a planter.", keywords: ["glove", "monogram", "d.p.", "planter"], leakTerms: ["glove", "monogram"], supports: ["suspect"], pointsAt: "desmond" },
  T_TEODOR_SAW_DESMOND: { id: "T_TEODOR_SAW_DESMOND", name: "Teodor saw Desmond", text: "Teodor saw Desmond leave the kitchen around 21:55, wiping his hands.", keywords: ["saw desmond", "wiping"], leakTerms: ["wiping his hands"], supports: ["suspect"], pointsAt: "desmond" },
  T_TEODOR_ARGUED: { id: "T_TEODOR_ARGUED", name: "Teodor's argument", text: "Mara saw Teodor storm out of the dome at 22:30 after shouting at Vance.", keywords: ["argument", "argued", "shouting", "storm"], leakTerms: [], supports: [], pointsAt: "teodor" },
  T_INES_GARDEN: { id: "T_INES_GARDEN", name: "Ines in the garden", text: "Mara saw Ines in the garden through the conservatory glass at about 21:50.", keywords: ["garden"], leakTerms: [], supports: [] },
  T_MARA_CARRIED: { id: "T_MARA_CARRIED", name: "Mara carried the tea", text: "Mara carried the flask from the kitchen counter to the dome at 22:00.", keywords: ["carried", "took up"], leakTerms: [], supports: [] },
};

export const SUSPECTS: Record<SuspectId, Suspect> = {
  mara: {
    id: "mara",
    name: "Mara Quell",
    role: "research assistant",
    persona: "Twenty-nine, Vance's research assistant for four years. Earnest, precise, frightened. Found the body. Loved the work more than the man.",
    voice: "careful, exact times, apologetic, short sentences",
    palette: [[0.16, 0.34, 0.42], [0.85, 0.72, 0.36]],
    answers: {
      greeting: { line: "I found him, detective. I keep seeing his face. Ask me anything — I want this over." },
      alibi: { line: "At ten I took his tea up to the dome from the kitchen counter, then I worked on star plates in the library annexe until I went up at ten past midnight and found him.", reveals: "T_MARA_CARRIED" },
      victim: { line: "He was brilliant and he could be cruel. He took credit like other men take breath. But he kept me on when nobody else would." },
      time: { line: "I carried the flask up at ten exactly. At half past ten I heard shouting and saw Professor Lisk storm out of the dome. At ten past midnight I found him.", reveals: "T_TEODOR_ARGUED" },
      motive: { line: "Everyone had a quarrel with him. Professor Lisk most loudly." },
      "suspect:teodor": { line: "Professor Lisk shouted at him in the dome. I saw him storm out at half past ten.", reveals: "T_TEODOR_ARGUED" },
      "suspect:ines": { line: "Ines was in the garden — I saw her lantern through the conservatory glass at about ten to ten.", reveals: "T_INES_GARDEN" },
      "suspect:desmond": { line: "Desmond only comes when he needs money. He said he'd be in the library all night." },
      "room:kitchen": { line: "The flask was on the counter with Ines's note when I collected it at ten. Nobody was in the kitchen then." },
      "room:dome": { line: "He was slumped in the observing chair. The tea flask was by his hand." },
      "room:conservatory": { line: "Ines was out in the garden; I could see her lantern through the glass." },
      "clue:C_FLASK": { line: "I carried that flask. I didn't open it. I swear I didn't open it.", reveals: "T_MARA_CARRIED" },
      "clue:C_LOG": { line: "Five past eleven... so he was still alive when Professor Lisk had long gone." },
      "clue:C_LETTER": { line: "Professor Lisk always said the comet was his. Maybe it was." },
      "clue:C_TEA_NOTE": { line: "Yes, that's Ines's note. The flask sat there from twenty to ten until I took it at ten." },
      "clue:C_GLOVE": { line: "D.P.? That's Desmond's. He wears those gloves even indoors." },
    },
    shrug: "I don't know anything about that, detective.",
    lies: [
      {
        topic: "clue:C_WILL",
        lie: "A new will? I know nothing about his will.",
        truth: "He told me last week he was leaving the observatory to me. I begged him not to — I knew what Desmond would say.",
        crackWhen: [["C_WILL"]],
        crackLine: "...Yes. He told me last week. I didn't say so because I knew how it would look.",
      },
    ],
    forbiddenTerms: ["foxglove", "iou", "forty thousand", "kid glove", "wiping his hands"],
  },
  desmond: {
    id: "desmond",
    name: "Desmond Pike",
    role: "the professor's nephew and heir",
    persona: "Thirty-four, Vance's nephew and only family. Charming, sardonic, always short of money. Calm under pressure; jokes when cornered.",
    voice: "smooth, ironic, deflecting, never raises his voice",
    palette: [[0.32, 0.12, 0.14], [0.78, 0.66, 0.52]],
    answers: {
      greeting: { line: "Detective. Terrible business. Do sit — the magistrate seems to think we're all murderers." },
      victim: { line: "Uncle Aurel loved the sky more than anyone on the ground. Me included. Still, he was family." },
      time: { line: "Time? The evening dragged. I read in the library. I heard Lisk bellowing at some point." },
      motive: { line: "Why would anyone kill a man who spent his nights looking at stars? Ask Lisk. Ask the girl who gets the house." },
      "suspect:teodor": { line: "Lisk was screaming about his precious comet. If I were you I'd start with him." },
      "suspect:mara": { line: "Mara carried up the tea, didn't she? Devoted little Mara. Ask her about the will." },
      "suspect:ines": { line: "Ines and her potions. Half that kitchen could stop a heart." },
      "room:dome": { line: "I never go up to the dome. Too cold, too many stars." },
      "room:library": { line: "I spent the evening by the library fire with a book." },
      "clue:C_FLASK": { line: "Poisoned tea. How very Victorian. I don't touch tea, detective." },
      "clue:C_LOG": { line: "So he was alive at five past eleven. Where was Lisk at five past eleven?" },
      "clue:C_LETTER": { line: "There you are. 'I will see you answer for it.' Lisk wrote your case for you." },
      "clue:C_WILL": { line: "A draft. Unsigned. Uncle threatened new wills every winter. It meant nothing." },
      "clue:C_CABINET": { line: "Ines's cabinet, Ines's tincture. I'm not sure why you're showing me." },
      "clue:C_TEA_NOTE": { line: "So the tea sat on a counter for anyone. Including Mara. Especially Mara." },
    },
    shrug: "I can't help you there.",
    lies: [
      {
        topic: "alibi",
        lie: "I was in the library all evening. I never set foot in the kitchen.",
        truth: "I went down to the kitchen just before ten for the good brandy. I was in and out in five minutes. I never touched any tea.",
        crackWhen: [
          ["C_GLOVE", "T_TEODOR_SAW_DESMOND"],
          ["C_GLOVE", "C_CABINET"],
        ],
        crackLine: "...All right. I went down for the brandy, just before ten. That proves nothing.",
      },
      {
        topic: "clue:C_IOU",
        lie: "My finances are perfectly in order, thank you.",
        truth: "Fine — I owe forty thousand, due on the first. Uncle knew. It's why he was so dramatic about wills.",
        crackWhen: [["C_IOU"]],
        crackLine: "You went through my coat. ...Yes, I owe money. Plenty of people do.",
      },
      {
        topic: "clue:C_GLOVE",
        lie: "Lots of people have the initials D.P., detective.",
        truth: "It's my glove. I must have dropped it in the conservatory. I don't know how it got stained.",
        crackWhen: [["C_GLOVE", "T_TEODOR_SAW_DESMOND"], ["C_GLOVE", "C_CABINET"]],
        crackLine: "It's mine. I don't know how it got like that.",
      },
    ],
    forbiddenTerms: ["i poisoned", "i killed", "i did it", "foxglove tincture into", "i put the"],
  },
  ines: {
    id: "ines",
    name: "Ines Calloway",
    role: "housekeeper and herbalist",
    persona: "Fifty-two, housekeeper at Wrenfield for twenty years. Warm, blunt, proud of her garden and her remedies. Takes foxglove tincture for her own heart.",
    voice: "plain-spoken, motherly, a little defensive about her herbs",
    palette: [[0.24, 0.3, 0.18], [0.82, 0.8, 0.74]],
    answers: {
      greeting: { line: "Twenty years I kept this house. Ask what you need, love, but be quick — I've a kitchen to put right." },
      alibi: { line: "I made his tea at twenty to ten, left it on the counter with a note for Mara, and went out to cover the seedlings. I was in the garden till past ten.", reveals: "T_INES_GARDEN" },
      victim: { line: "Difficult man. Never once said thank you for a meal. But he didn't deserve this." },
      time: { line: "Tea made at twenty to ten. Garden till about ten past. Then bed." },
      motive: { line: "Money, usually. That's what it is in big houses. Money and pride." },
      "suspect:mara": { line: "Mara's a good girl. She took the tea up like I asked." },
      "suspect:desmond": { line: "Mr Desmond only visits when he's skint. He was hanging about the house all evening." },
      "suspect:teodor": { line: "The old professor shouted the house down. Didn't seem the poisoning sort, mind. More the shouting sort." },
      "room:kitchen": { line: "My kitchen. I was out of it from twenty to ten. Anyone could have walked in." },
      "room:conservatory": { line: "I came back through the conservatory a bit after ten. Didn't see anyone." },
      "clue:C_FLASK": { line: "That residue — that's not my tea. My tea's clean." },
      "clue:C_TEA_NOTE": { line: "That's my note. The tea sat there while I was in the garden." },
      "clue:C_GLOVE": { line: "Mr Desmond's glove. He leaves them everywhere. Never seen it green before." },
      "clue:C_LOG": { line: "Bitter? My tea is never bitter." },
    },
    shrug: "Can't help you with that, love.",
    lies: [
      {
        topic: "clue:C_CABINET",
        lie: "My cabinet is always locked. The key never leaves my belt.",
        truth: "I leave it unlocked when I'm cooking, to save fumbling. I left it open tonight. God forgive me.",
        crackWhen: [["C_CABINET"]],
        crackLine: "...It was unlocked. I leave it open when I'm cooking. Someone used my tincture.",
      },
    ],
    forbiddenTerms: ["new will", "iou", "forty thousand", "comet"],
  },
  teodor: {
    id: "teodor",
    name: "Teodor Lisk",
    role: "rival astronomer, house guest",
    persona: "Sixty-one, astronomer, Vance's former partner. Proud, prickly, bitter that Vance published 'his' comet. Honest to a fault once his pride is dealt with.",
    voice: "formal, pompous, sharp, uses astronomy metaphors",
    palette: [[0.2, 0.2, 0.26], [0.62, 0.62, 0.7]],
    answers: {
      greeting: { line: "I am a scientist, not a suspect. But I will indulge the magistrate." },
      alibi: { line: "I went to see Aurel in the dome at a quarter past ten and left at half past. Before that I walked the corridor to the library — and I saw young Pike come out of the kitchen, wiping his hands.", reveals: "T_TEODOR_SAW_DESMOND" },
      victim: { line: "Aurel was a gifted observer and a thief. Both things are true." },
      time: { line: "Five to ten I saw Pike by the kitchen. Quarter past ten I went up to the dome. Half past I left. He was alive and well and insufferable." , reveals: "T_TEODOR_SAW_DESMOND" },
      motive: { line: "Look for the man who gains from a death, not the man who lost a comet." },
      "suspect:desmond": { line: "Pike? I saw him come out of the kitchen at five to ten, wiping his hands on a handkerchief. He looked startled to see me.", reveals: "T_TEODOR_SAW_DESMOND" },
      "suspect:mara": { line: "The assistant did the real work. Aurel took the credit. Now there's a motive, if you want one." },
      "suspect:ines": { line: "The housekeeper was outside with a lantern. I saw it bobbing in the garden." },
      "room:kitchen": { line: "I passed the kitchen at five to ten. Pike came out of it, wiping his hands.", reveals: "T_TEODOR_SAW_DESMOND" },
      "room:dome": { line: "I left him at half past ten, at his eyepiece. Alive." },
      "clue:C_LOG": { line: "Five past eleven. I was in my room by then — and you see, he was alive long after I left." },
      "clue:C_CABINET": { line: "Foxglove. Digitalis. A heart poison. And Pike came out of that kitchen, wiping his hands.", reveals: "T_TEODOR_SAW_DESMOND" },
      "clue:C_GLOVE": { line: "That is Pike's glove. He wore a pair at dinner." },
      "clue:C_WILL": { line: "So the nephew was to be cut out. Interesting." },
    },
    shrug: "That is outside my field of observation.",
    lies: [
      {
        topic: "room:dome",
        lie: "We discussed the weather. Clear skies. Nothing more.",
        truth: "We argued. He published my comet under his own name and I told him I would ruin him. Then I left him alive, at half past ten.",
        crackWhen: [["C_LETTER"], ["T_TEODOR_ARGUED"]],
        crackLine: "...Very well. We argued. Bitterly. But I left him alive.",
      },
      {
        topic: "clue:C_LETTER",
        lie: "A private letter between colleagues. Irrelevant.",
        truth: "Yes, I wrote it. He stole my comet. I wanted him disgraced, not dead.",
        crackWhen: [["C_LETTER"]],
        crackLine: "I wrote it. I meant every word — about his reputation, not his life.",
      },
    ],
    forbiddenTerms: ["new will", "iou", "forty thousand", "unlocked", "planter"],
  },
};

export const TRUTH: CaseTruth = {
  killer: "desmond",
  method: "poison_tea",
  motive: "inheritance",
  explanation: [
    "Desmond Pike owed forty thousand pounds by the first of the month. His uncle meant to sign a new will tomorrow, leaving Wrenfield to Mara and cutting Desmond out.",
    "At 21:40 Ines left the professor's tea on the kitchen counter and went out to the garden, leaving her herb cabinet unlocked as she always did while cooking.",
    "Just before ten, Desmond slipped into the kitchen, emptied most of the foxglove tincture into the flask, and wiped his gloved hands. Teodor Lisk saw him come out.",
    "He hid the stained glove in a conservatory planter. At ten, Mara unknowingly carried the poisoned flask up to the dome.",
    "Teodor argued with Vance at a quarter past ten and left him alive. At five past eleven Vance wrote that his tea was bitter and his heart racing. By 23:40 he was dead.",
  ],
};

export const METHODS: Record<MethodId, string> = {
  poison_tea: "Poison in his tea",
  struck: "Struck from behind",
  fall: "A push from the observing platform",
  smothered: "Smothered in his chair",
};

export const MOTIVES: Record<MotiveId, string> = {
  inheritance: "To keep the inheritance before the new will",
  stolen_research: "Revenge for the stolen comet",
  credit: "Years of stolen credit for their work",
  grudge: "An old household grudge",
};

export function evidenceName(id: EvidenceId): string {
  return EVIDENCE[id].name;
}
