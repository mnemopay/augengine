/**
 * Dead Air — the second hand-authored case.
 *
 * Lagos, 1957. A small independent radio station goes silent at midnight in
 * the middle of its live show. Original setting and characters: Lagoon
 * Wireless, its staff and its songs are invented. No real people, stations,
 * brands or songs.
 *
 * Same contract as Wrenfield: everything here is ground truth; the rule
 * engine reads it and dialogue only rephrases the lines it picks.
 */
import type { CaseDef, CaseTruth, Evidence, Room, RoomObject, Suspect } from "./case.js";

export type DeadAirRoomId = "lobby" | "studio" | "bandroom" | "engineering" | "office";
export type DeadAirSuspectId = "ebun" | "kunle" | "adaeze" | "sammy";
export type DeadAirEvidenceId =
  | "C_CABLE"
  | "C_RUNSHEET"
  | "C_LETTER"
  | "C_BENCH"
  | "C_LEDGER"
  | "C_MEMO"
  | "C_TAG"
  | "C_CONTRACT"
  | "C_SONGSHEET"
  | "T_SAMMY_SAW_EBUN"
  | "T_ADAEZE_BALCONY"
  | "T_KUNLE_ROOF"
  | "T_FEMI_THIEF";

const TITLE = "Dead Air";

const INTRO =
  "Lagos, 1957. At one minute past midnight Lagoon Wireless went silent in the middle of its live show, Midnight Highlife. " +
  "Three minutes of dead air later, the station's owner and star voice, Femi Adeyinka-Cole, was found on the floor of Studio B, his hand burned. " +
  "Four people were in the building. The inspector gives you until dawn and one accusation. Search the station, question the staff in the lobby, " +
  "and name the killer, the method and the motive.";

const ROOMS: Array<Room & { id: DeadAirRoomId }> = [
  { id: "lobby", name: "Lobby", blurb: "Terrazzo floor, a ceiling fan turning slowly, and the staff on the bench under the inspector's eye. Question them here." },
  { id: "studio", name: "Studio B", blurb: "The on-air booth: a felt-lined desk, the microphone stand, and a record still turning. Femi died here." },
  { id: "bandroom", name: "Band Room", blurb: "Instrument cases, a battered upright piano, and a window that looks straight at the radio mast on the roof." },
  { id: "engineering", name: "Engineering Room", blurb: "Valves glowing in the transmitter rack, a soldering iron gone cold, and Kunle's repair bench." },
  { id: "office", name: "Station Office", blurb: "Two desks under a painting of the lagoon at dusk: Femi's, buried in letters, and Ebun's, neat as a pin." },
];

const OBJECTS: Array<RoomObject & { room: DeadAirRoomId }> = [
  { id: "studio.mic", room: "studio", name: "Microphone stand", clue: "C_CABLE", text: "The chrome stand is scorched at the grip. The examiner marked the lead as the source of the fatal shock. Its repair label is missing." },
  { id: "studio.runsheet", room: "studio", name: "Running order", clue: "C_RUNSHEET", text: "Tonight's running order in Femi's hand: '23:40 Lagoon Moon, long version, 8 min (F. to balcony). 00:00 midnight call, live.' The booth stood empty while the record played." },
  { id: "studio.jacket", room: "studio", name: "Jacket on the chair", clue: "C_LETTER", text: "Femi's linen jacket. In the pocket, an unsent letter to the station's auditors: 'Someone in my own office has been robbing this station for a year. On Monday I will give you the name and the proof.'" },
  { id: "studio.turntable", room: "studio", name: "Turntable", text: "The record has run out; the needle hisses in the run-off groove. Three minutes of silence went out over Lagos before the standby tape cut in." },
  { id: "engineering.bench", room: "engineering", name: "Repair bench", clue: "C_BENCH", text: "Kunle's bench. A hook marked 'FAULTY - DO NOT USE' is empty. His logbook, Tuesday: 'Studio B spare mic lead, failed electrical inspection. DANGEROUS. Tagged, hung on the faulty hook.'" },
  { id: "engineering.rack", room: "engineering", name: "Transmitter rack", text: "The valves are warm and steady. The transmitter itself never failed tonight; only the programme went silent." },
  { id: "engineering.ladder", room: "engineering", name: "Roof ladder", text: "The ladder up to the roof hatch and the mast. Fresh mud and rust on the rungs: someone climbed it tonight." },
  { id: "office.ledger", room: "office", name: "Advertising ledger", clue: "C_LEDGER", text: "Payments from a soap company and two cinemas are entered twice, once in full and once short. Over a year nearly two thousand pounds is missing. Every entry is in Ebun's neat hand." },
  { id: "office.memo", room: "office", name: "Memo on Femi's desk", clue: "C_MEMO", text: "Dated today, typed: 'K. Bankole to be dismissed at month end. Careless with the equipment. I will not have another fire.'" },
  { id: "office.basket", room: "office", name: "Ebun's wastepaper basket", clue: "C_TAG", text: "Under the carbon paper, a cardboard repair tag torn in two: 'FAULTY - DO NOT USE. K.B.'" },
  { id: "office.typewriter", room: "office", name: "Typewriter", text: "A letter about advertising rates, dated today, stops halfway through a sentence." },
  { id: "bandroom.case", room: "bandroom", name: "Music case", clue: "C_CONTRACT", text: "Adaeze's case. Inside, her contract with Lagoon Wireless: five more years, exclusive. Folded with it, a letter from a London record label offering her a deal 'if Mr Adeyinka-Cole will release you'." },
  { id: "bandroom.piano", room: "bandroom", name: "Sheet music on the piano", clue: "C_SONGSHEET", text: "'Lagoon Moon', words and music by Femi Adeyinka-Cole. Underneath, an older, creased copy in a different hand, signed 'S. Addo, Accra, 1954'." },
  { id: "bandroom.window", room: "bandroom", name: "Window", text: "It looks straight across at the radio mast on the roof. From here you would see anyone working up there with a torch." },
];

const EVIDENCE: Record<DeadAirEvidenceId, Evidence> = {
  C_CABLE: { id: "C_CABLE", name: "Rewired microphone lead", text: "The microphone stand's lead failed inspection. Femi was electrocuted when he took hold of the stand.", keywords: ["cable", "mic lead", "microphone", "mic stand", "plug", "electric", "current", "shock", "electrocuted", "wiring"], leakTerms: ["failed inspection", "repair label", "rewired"], supports: ["method"] },
  C_RUNSHEET: { id: "C_RUNSHEET", name: "Running order", text: "At 23:40 Femi put on an eight-minute record and went out to the balcony. The booth stood empty while it played.", keywords: ["running order", "run sheet", "runsheet", "schedule", "long record", "long version"], leakTerms: ["running order"], supports: ["method"] },
  C_LETTER: { id: "C_LETTER", name: "Femi's unsent letter", text: "Femi meant to tell the auditors on Monday who in his own office had been robbing the station.", keywords: ["letter", "auditor", "auditors", "monday"], leakTerms: ["auditor"], supports: ["motive"] },
  C_BENCH: { id: "C_BENCH", name: "Kunle's empty faulty hook", text: "On Tuesday Kunle tagged a spare Studio B microphone lead as dangerous and hung it on his faulty hook. The hook is empty.", keywords: ["bench", "hook", "logbook", "faulty", "repair"], leakTerms: ["faulty hook", "logbook"], supports: ["method"] },
  C_LEDGER: { id: "C_LEDGER", name: "Doctored ledger", text: "Nearly two thousand pounds of advertising money is missing, in entries written by Ebun Martins.", keywords: ["ledger", "books", "accounts", "advertising", "embezzle", "two thousand", "missing money"], leakTerms: ["two thousand", "ledger"], supports: ["motive"], pointsAt: "ebun" },
  C_MEMO: { id: "C_MEMO", name: "Dismissal memo", text: "Femi meant to sack Kunle at the end of the month for being careless with the equipment.", keywords: ["memo", "dismiss", "dismissed", "sack", "sacked", "fired"], leakTerms: ["dismiss", "memo"], supports: ["motive"], pointsAt: "kunle" },
  C_TAG: { id: "C_TAG", name: "Torn repair tag", text: "Kunle's 'FAULTY - DO NOT USE' tag from the missing lead, torn in two, in Ebun's wastepaper basket.", keywords: ["tag", "basket", "wastepaper", "torn"], leakTerms: ["torn tag", "wastepaper", "do not use"], supports: ["suspect"], pointsAt: "ebun" },
  C_CONTRACT: { id: "C_CONTRACT", name: "Adaeze's contract", text: "Femi held Adaeze to five more years and stood between her and a London record deal.", keywords: ["contract", "london", "record deal", "label", "five years"], leakTerms: ["london", "five more years", "record label"], supports: ["motive"], pointsAt: "adaeze" },
  C_SONGSHEET: { id: "C_SONGSHEET", name: "Two copies of 'Lagoon Moon'", text: "Femi published 'Lagoon Moon' under his own name, but an older copy is signed by Sammy Addo, Accra, 1954.", keywords: ["song", "lagoon moon", "sheet music", "composer", "wrote it"], leakTerms: ["1954", "older copy"], supports: ["motive"], pointsAt: "sammy" },
  T_SAMMY_SAW_EBUN: { id: "T_SAMMY_SAW_EBUN", name: "Sammy saw Ebun", text: "Sammy saw Ebun Martins come out of the engineering room at about a quarter to twelve, holding her handbag tight.", keywords: ["saw ebun", "handbag"], leakTerms: ["handbag"], supports: ["suspect"], pointsAt: "ebun" },
  T_ADAEZE_BALCONY: { id: "T_ADAEZE_BALCONY", name: "Adaeze followed Femi", text: "Sammy saw Adaeze follow Femi out to the balcony at twenty to twelve, and heard them shouting.", keywords: ["balcony", "shouting"], leakTerms: [], supports: [], pointsAt: "adaeze" },
  T_KUNLE_ROOF: { id: "T_KUNLE_ROOF", name: "Kunle on the roof", text: "Sammy watched Kunle's torch up by the mast from half past eleven until the station went silent.", keywords: ["torch", "mast", "roof"], leakTerms: [], supports: [] },
  T_FEMI_THIEF: { id: "T_FEMI_THIEF", name: "Femi feared a thief", text: "This week Femi told Adaeze that someone in his own office was robbing him.", keywords: ["thief", "robbing", "robbed"], leakTerms: [], supports: ["motive"] },
};

const SUSPECTS: Record<DeadAirSuspectId, Suspect> = {
  ebun: {
    id: "ebun",
    name: "Ebun Martins",
    role: "station secretary and bookkeeper",
    persona: "Forty-four, keeps the station's books and Femi's diary. Widowed young; pays school fees for two sons in England. Proper, devout, helpful, never raises her voice; tidies things when she is nervous.",
    voice: "polite, formal, churchy phrases, very precise about paperwork, calls the detective 'detective'",
    palette: [[0.36, 0.2, 0.3], [0.86, 0.7, 0.38]],
    answers: {
      greeting: { line: "Good evening, detective. I am Ebun Martins; I keep the books and the diary here. Whatever you need, I will find the paper for it." },
      victim: { line: "Mr Adeyinka-Cole built this station from one transmitter and a borrowed piano. He was demanding, but all of Lagos listened to him every night." },
      time: { line: "I was typing when the music stopped. Then the silence, three minutes of nothing on the monitor. I ran to Studio B and he was on the floor." },
      motive: { line: "A man with a voice like his makes enemies. Ask Kunle about his job. Ask that singer about her contract." },
      "suspect:kunle": { line: "Kunle is careless with his wires. Mr Adeyinka-Cole said so himself, only this morning. If a cable was bad, it was Kunle's cable." },
      "suspect:adaeze": { line: "Adaeze is young and wants to be famous overseas. She and Mr Adeyinka-Cole quarrelled about it all week." },
      "suspect:sammy": { line: "Mr Addo plays beautifully and drinks the same way. I did not see him after the band stopped." },
      "room:office": { line: "The office is my desk and Mr Adeyinka-Cole's. I keep mine tidy. He never kept his." },
      "room:studio": { line: "He was on the floor by the desk with the stand fallen across him. I did not touch anything. I know better than to touch anything electric." },
      "room:bandroom": { line: "The band room is the musicians' business. It smells of palm wine and valve oil." },
      "clue:C_CABLE": { line: "Electricity. God have mercy. I always said one of Kunle's wires would kill somebody." },
      "clue:C_RUNSHEET": { line: "He always took his cigarette during the long record. Everyone at the station knew that." },
      "clue:C_MEMO": { line: "I typed that memo myself this morning. Kunle was very angry when he heard." },
      "clue:C_CONTRACT": { line: "Five years. She signed it herself. He would never let a voice like hers go abroad." },
      "clue:C_SONGSHEET": { line: "I would not know about the music, detective. I only know the royalties came to the station." },
      "clue:C_BENCH": { line: "Kunle's bench. I would not know one wire from another." },
      "clue:T_SAMMY_SAW_EBUN": { line: "Mr Addo had been drinking. He saw what he wanted to see." },
    },
    shrug: "I am sorry, detective, I have no paper on that.",
    lies: [
      {
        topic: "alibi",
        lie: "I was at my typewriter in the office from eleven until the music stopped. I never left my desk.",
        truth: "I left my desk once, at a quarter to twelve, to fetch a typewriter ribbon from the engineering room. One minute. Then I went back and typed until the silence.",
        crackWhen: [["C_TAG", "T_SAMMY_SAW_EBUN"], ["C_TAG", "C_BENCH"]],
        crackLine: "...Very well. I went to the engineering room for a ribbon, only for a minute. That is all it was.",
      },
      {
        topic: "room:engineering",
        lie: "That is Kunle's room. I have no business in there. I have not set foot in it for weeks.",
        truth: "I was in there tonight, for a minute, at a quarter to twelve. Kunle keeps the spare typewriter ribbons.",
        crackWhen: [["C_TAG", "T_SAMMY_SAW_EBUN"], ["C_TAG", "C_BENCH"]],
        crackLine: "Yes, I was in that room tonight.",
      },
      {
        topic: "clue:C_TAG",
        lie: "A tag? Kunle leaves his rubbish everywhere. Anybody could have thrown it in my basket.",
        truth: "I found it on the floor by the engineering-room door and threw it away. I did not think. I did not read it.",
        crackWhen: [["C_TAG", "T_SAMMY_SAW_EBUN"], ["C_TAG", "C_BENCH"]],
        crackLine: "I threw it away. I did not read it, I swear I did not read it.",
      },
      {
        topic: "clue:C_LEDGER",
        lie: "The books balance to the penny, detective. I would stake my church on it.",
        truth: "There are irregularities. Small sums, borrowed against next month for my boys' school fees. I meant to put every penny back.",
        crackWhen: [["C_LEDGER"]],
        crackLine: "...The figures are mine. I meant to put it all back. Before God, I meant to.",
      },
      {
        topic: "clue:C_LETTER",
        lie: "Robbing the station? He never said a word to me about anything like that.",
        truth: "He asked me for the ledgers on Friday, and he would not look at me. I knew then that he knew something.",
        crackWhen: [["C_LETTER", "C_LEDGER"]],
        crackLine: "...He asked for the ledgers on Friday. He would not look me in the eye.",
      },
    ],
    forbiddenTerms: ["i rewired", "i swapped", "i took the lead", "i killed", "i did it", "failed inspection", "repair label", "earth wire", "live pin"],
  },
  kunle: {
    id: "kunle",
    name: "Kunle Bankole",
    role: "chief engineer",
    persona: "Forty-one, built half the station's equipment by hand. Gruff, proud of his work, frightened of losing his job; drinks a little on night shifts. There was a small studio fire last month that Femi never forgave.",
    voice: "blunt, technical, defensive, short sentences, the odd Lagos expression",
    palette: [[0.22, 0.26, 0.2], [0.72, 0.62, 0.42]],
    answers: {
      greeting: { line: "Kunle Bankole. Engineer. Ask your questions, I have a transmitter to watch." },
      alibi: { line: "At half past eleven the signal started crackling, so I climbed to the mast on the roof. I was up there with my torch until the station went silent. Then I came down running." },
      victim: { line: "Femi had the best voice in West Africa and the worst temper. He paid on time. Mostly." },
      time: { line: "Half past eleven, up the ladder. At midnight the carrier was fine but the programme was silence. That is when I knew something was wrong in the booth." },
      motive: { line: "Money. In this building it is always money. Ask the one who counts it." },
      "suspect:ebun": { line: "Madam Ebun counts every penny in this station. Very careful woman. Very careful." },
      "suspect:adaeze": { line: "The girl wants to sing overseas. Femi would not let her go. They were shouting about it all week." },
      "suspect:sammy": { line: "Sammy plays trumpet like an angel and complains like a goat. He says Femi stole his song." },
      "room:engineering": { line: "My room. Nobody touches my bench without asking. Nobody." },
      "room:studio": { line: "I wired Studio B myself. Every lead in that booth was sound when I left it." },
      "room:bandroom": { line: "From the band-room window you can see my mast. Ask the musicians, they saw my torch." },
      "clue:C_BENCH": { line: "My faulty hook, empty? I hung that dangerous lead there myself on Tuesday, with my tag on it. Somebody took it." },
      "clue:C_TAG": { line: "That is my tag. My handwriting. Who tore it? Where did you find it?" },
      "clue:C_RUNSHEET": { line: "Eight minutes of record. Long enough to change a lead, if you knew which lead to bring." },
      "clue:C_LEDGER": { line: "I don't read ledgers. I read meters." },
      "clue:T_KUNLE_ROOF": { line: "You see? Sammy saw my torch. I was on that roof the whole time." },
    },
    shrug: "I fix transmitters. I don't know about that.",
    lies: [
      {
        topic: "clue:C_CABLE",
        lie: "That lead was sound. I check every lead in Studio B myself. This was no fault of mine.",
        truth: "That is my dangerous spare, the one I tagged on Tuesday. Somebody took it off my faulty hook and put it on the stand.",
        crackWhen: [["C_BENCH"], ["C_CABLE", "T_KUNLE_ROOF"]],
        crackLine: "...God. That is my spare. The bad one. Somebody took it from my bench.",
      },
      {
        topic: "clue:C_MEMO",
        lie: "Dismissed? Nobody told me anything about that.",
        truth: "He told me this morning. One more month and out. I was angry, yes. But a man doesn't kill his own station.",
        crackWhen: [["C_MEMO"]],
        crackLine: "...Yes. He told me this morning. I was angry. That is not murder.",
      },
    ],
    forbiddenTerms: ["ledger", "two thousand", "london", "1954", "handbag"],
  },
  adaeze: {
    id: "adaeze",
    name: "Adaeze Nnadi",
    role: "singer with the station band",
    persona: "Twenty-four, the station's new star voice, discovered singing in a church choir. Ambitious, quick-tongued, proud; frightened that tonight will end her career before it starts.",
    voice: "quick, witty, emotional, flashes of temper that turn to tenderness",
    palette: [[0.42, 0.26, 0.12], [0.9, 0.78, 0.5]],
    answers: {
      greeting: { line: "Adaeze Nnadi. You've heard me on the wireless, even if you won't admit it. Ask what you want." },
      victim: { line: "He found me singing in a church and put me on the air. This week he was frightened. He told me someone in his own office was robbing him blind.", reveals: "T_FEMI_THIEF" },
      time: { line: "The band stopped at twenty to twelve for Femi's long record. At midnight we heard nothing, just the hiss. Then Ebun screamed." },
      motive: { line: "Somebody in that office was stealing from him. He told me so himself.", reveals: "T_FEMI_THIEF" },
      "suspect:ebun": { line: "Ebun Martins smiles at everybody and never laughs. She has typed every secret in this station." },
      "suspect:kunle": { line: "Kunle was up on the roof fiddling with the mast. He's rough, but he loved this station like his own child." },
      "suspect:sammy": { line: "Sammy taught me half of what I know. He was in the band room with his trumpet, sulking." },
      "room:bandroom": { line: "The band room is ours. Cases, the old piano, and Sammy's bottle that he thinks we don't know about." },
      "room:studio": { line: "Nobody goes in the booth while he's on air. He'd skin you." },
      "clue:C_SONGSHEET": { line: "Everyone in the band knows Lagoon Moon is Sammy's. Femi just had the louder name." },
      "clue:C_TAG": { line: "Kunle's tag, in Ebun's basket? Why would Ebun touch Kunle's things?" },
      "clue:C_LEDGER": { line: "So that's what he meant. It was the office all along." },
      "clue:C_RUNSHEET": { line: "He always smoked on the balcony during the long record. Always." },
      "clue:C_LETTER": { line: "Monday. He kept saying Monday, like it was a cliff." },
    },
    shrug: "Don't look at me like that. I don't know.",
    lies: [
      {
        topic: "alibi",
        lie: "I was in the band room the whole break, practising my scales.",
        truth: "I followed him to the balcony at twenty to twelve. We fought about my contract. He laughed and said he had bigger trouble, a thief to name on Monday. I left him there at a quarter to twelve, alive.",
        crackWhen: [["T_ADAEZE_BALCONY"], ["C_CONTRACT"]],
        crackLine: "...All right! I was on the balcony with him. We argued. He was alive when I left.",
      },
      {
        topic: "clue:C_CONTRACT",
        lie: "My contract is my business, and it's perfectly fine.",
        truth: "Five more years. London wanted me now. I begged him to release me and he refused. I hated him for one night. Only one night.",
        crackWhen: [["C_CONTRACT"]],
        crackLine: "Yes, he wouldn't release me. I was angry. Being angry isn't murder.",
      },
    ],
    forbiddenTerms: ["faulty hook", "failed inspection", "repair label", "repair bench", "two thousand"],
  },
  sammy: {
    id: "sammy",
    name: "Sammy Addo",
    role: "bandleader and trumpeter",
    persona: "Thirty-eight, a bandleader from Accra who came to Lagos for the radio money. Charming, wry, fond of a drink, honest when sober. Says Femi stole his best song.",
    voice: "warm, musical, wry jokes, calls everyone 'my friend'",
    palette: [[0.16, 0.22, 0.34], [0.84, 0.64, 0.3]],
    answers: {
      greeting: { line: "Sammy Addo, my friend. Trumpet, bandleader, and tonight, suspect. What a night." },
      alibi: { line: "We stopped at twenty to twelve for Femi's long record. I stayed by the band-room window with my bottle, and at about a quarter to twelve I saw Ebun Martins come out of the engineering room, holding her handbag tight like it had a secret in it.", reveals: "T_SAMMY_SAW_EBUN" },
      time: { line: "Twenty to twelve, the band stopped and Femi went out for his cigarette. A quarter to twelve, Madam Ebun came out of Kunle's room. Midnight, silence.", reveals: "T_SAMMY_SAW_EBUN" },
      victim: { line: "Femi could sell sand in the Sahara. He sold my song, too." },
      motive: { line: "Look for the person who was afraid of Monday. Femi kept saying, Monday, Monday." },
      "suspect:ebun": { line: "Madam Ebun? At about a quarter to twelve she came out of the engineering room in a hurry, holding her handbag tight. That woman never hurries.", reveals: "T_SAMMY_SAW_EBUN" },
      "suspect:adaeze": { line: "When we stopped at twenty to twelve, Adaeze went straight out to the balcony after Femi. I heard them shouting.", reveals: "T_ADAEZE_BALCONY" },
      "suspect:kunle": { line: "Kunle? From the band-room window I could see his torch up by the mast from half past eleven until the silence. He never came down.", reveals: "T_KUNLE_ROOF" },
      "room:engineering": { line: "I passed the engineering-room door at a quarter to twelve. Madam Ebun came out of it like a cat out of a kitchen.", reveals: "T_SAMMY_SAW_EBUN" },
      "room:bandroom": { line: "From the band-room window you see the whole roof. Kunle's torch was up there by the mast, bobbing about.", reveals: "T_KUNLE_ROOF" },
      "room:studio": { line: "I never went near the booth. Femi and I were not speaking this week." },
      "clue:C_CABLE": { line: "Electricity. A musician's nightmare. They'll say Kunle's wiring, but Kunle was on the roof." },
      "clue:C_TAG": { line: "Kunle's tag, in Madam Ebun's basket. And I saw her come out of his room. My friend, I think you are close." },
      "clue:C_RUNSHEET": { line: "Eight minutes of Lagoon Moon. My song, paying for his cigarette." },
      "clue:C_LETTER": { line: "Monday. So that is why he kept saying Monday." },
      "clue:C_CONTRACT": { line: "Five years for a voice like that. Femi kept birds in cages too." },
    },
    shrug: "That one I cannot play for you, my friend.",
    lies: [
      {
        topic: "clue:C_SONGSHEET",
        lie: "Lagoon Moon? Femi wrote it. Everybody knows that.",
        truth: "I wrote Lagoon Moon in Accra in 1954. Femi put his name on it and the money in his pocket. I wanted my name back, not his life.",
        crackWhen: [["C_SONGSHEET"]],
        crackLine: "...It is my song. My name was on it first. That's all.",
      },
    ],
    forbiddenTerms: ["ledger", "two thousand", "faulty hook", "dismiss"],
  },
};

const TRUTH: CaseTruth = {
  killer: "ebun",
  method: "electrocuted",
  motive: "hide_theft",
  solvedHeadline: "Case closed. Ebun Martins rigged the microphone stand to silence Femi before Monday.",
  explanation: [
    "For a year Ebun Martins had been skimming the station's advertising money: nearly two thousand pounds, hidden in a doctored ledger. On Friday Femi asked for the books. He had written to the auditors: on Monday he would name the thief.",
    "On Tuesday Kunle Bankole found a spare Studio B microphone lead with a dangerous electrical fault. He tagged it DO NOT USE and hung it on his faulty hook.",
    "At twenty to twelve Femi put on the eight-minute Lagoon Moon and went out to the balcony, as he always did. Adaeze followed him to fight about her contract. Kunle was up on the roof chasing a crackle at the mast.",
    "Ebun slipped into the engineering room, took the dangerous lead and tore off Kunle's tag. Sammy Addo saw her come out at a quarter to twelve. She swapped the lead onto the studio microphone stand and dropped the torn tag in her own wastepaper basket.",
    "At midnight Femi pulled the stand close for his live call. Three minutes of dead air went out over Lagos. Kunle's reputation for carelessness was meant to make it look like an accident.",
  ],
};

export const DEAD_AIR: CaseDef = {
  id: "deadair",
  title: TITLE,
  tagline: "Lagos, 1957. A radio station goes silent at midnight in the middle of its live show.",
  setting: "a small independent radio station in Lagos in 1957, on a hot, humid night; highlife music, ceiling fans, valve transmitters, the lagoon outside",
  intro: INTRO,
  victim: { name: "Femi Adeyinka-Cole", role: "owner and star presenter of Lagoon Wireless", portraitId: "femi", keywords: ["femi", "adeyinka", "victim", "boss", "owner", "presenter", "employer"] },
  authority: "the inspector",
  interviewRoom: "lobby",
  interviewHint: "The staff are waiting in the lobby. Go there to question them.",
  rooms: ROOMS,
  roomKeywords: [
    ["engineering", ["engineering", "transmitter", "roof", "mast", "ladder"]],
    ["studio", ["studio", "booth", "on air"]],
    ["bandroom", ["band room", "bandroom", " band", "piano", "window"]],
    ["office", ["office", "typewriter", " desk"]],
  ],
  objects: OBJECTS,
  evidence: EVIDENCE,
  suspects: SUSPECTS,
  suspectOrder: ["ebun", "kunle", "adaeze", "sammy"],
  truth: TRUTH,
  methods: {
    electrocuted: "A live current through the microphone stand",
    struck: "Struck down with a heavy object",
    poisoned: "Poison in his palm wine",
    strangled: "Strangled with a cable",
  },
  motives: {
    hide_theft: "To hide money stolen from the station",
    contract: "To escape a binding contract",
    stolen_song: "Revenge for a stolen song",
    job: "Rage at being dismissed",
  },
  unknownLine: "I'm not sure what you're asking, detective. Ask me about tonight.",
  assetPrefix: "deadair/",
};
