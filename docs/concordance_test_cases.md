# Concordance Ingest Test Cases

These are specific examples from the concordance that the ingest script must handle correctly.

## Test Case 1: Multiple Covernames with Brackets and "and"

**Input:**
```
KALIBR [CALIBER and CALIBRE] (cover name in Venona): David Greenglass. Venona New York KGB
1944, 602, 643, 714, 716, 729; Venona New York KGB 1945, 24; Venona Special Studies, 32, 79,
141, 154.
```

**Expected Output:**
- **Person entity:** "David Greenglass"
  - Aliases: ["David Greenglass"]
- **Covername entities:** "KALIBR", "CALIBER", "CALIBRE" (separate entities)
  - KALIBR aliases: ["KALIBR"]
  - CALIBER aliases: ["CALIBER"]
  - CALIBRE aliases: ["CALIBRE"]
- **Relationships:**
  - KALIBR --covername_of--> David Greenglass
  - CALIBER --covername_of--> David Greenglass
  - CALIBRE --covername_of--> David Greenglass

**Requirements:**
1. Split headword by brackets: `KALIBR [CALIBER and CALIBRE]` → `KALIBR`, `CALIBER`, `CALIBRE`
2. Split bracket contents by "and": `CALIBER and CALIBRE` → `CALIBER`, `CALIBRE`
3. Each covername becomes separate entity
4. All covernames link to same person

## Test Case 2: Cross-Reference Covername Alias

**Input:**
```
"Kalibr" (Russian original of a cover name in Vassiliev's notebooks): See "Caliber".
```

**Expected Output:**
- **Covername entity:** "Caliber" (target of crossref)
  - Aliases: ["Caliber", "Kalibr"] (Kalibr is alias of Caliber)
- **Relationship:** Kalibr --alias_of--> Caliber (or same entity with both aliases)

**Requirements:**
1. Detect cross-reference: "See X"
2. Extract target: "Caliber"
3. Extract source: "Kalibr" (from headword, remove quotes)
4. Create alias relationship: Kalibr → Caliber (same entity, different aliases)

## Test Case 3: Unidentified Covername (No Person)

**Input:**
```
"Kaiser" [Kayzer] (cover name in Vassiliev's notebooks): Unidentified Soviet intelligence contact, friend of
Harold Glasser. Described as American Army captain in Italy in 1944, then working in the
Treasury Depatment in Washington, and appointed to the staff of the Allied Control Commission
in Austria. Formerly active in the Washington CPUSA network. Vassiliev White Notebook #3,
52.
```

**Expected Output:**
- **Covername entity:** "Kaiser" (primary)
  - Aliases: ["Kaiser", "Kayzer"] (both variants)
  - Notes: "Unidentified Soviet intelligence contact..."
- **No person entity** (unidentified)
- **No relationships** (no person to link to)

**Requirements:**
1. Remove quotes from headword: `"Kaiser"` → `Kaiser`
2. Extract bracket variant: `[Kayzer]` → `Kayzer`
3. Both become aliases of same covername entity
4. No person referent (body says "Unidentified")
5. Store description in entity notes

## Test Case 4: Person Name with Comma Inversion

**Input:**
```
Kalinin, Mikhail Ivanovich: Bolshevik leader and official Soviet head of state, 1919–46. Vassiliev Yellow
Notebook #4, 66.
```

**Expected Output:**
- **Person entity:** "Mikhail Ivanovich Kalinin" (inverted)
  - Aliases: ["Mikhail Ivanovich Kalinin", "Kalinin, Mikhail Ivanovich"] (both forms)
- **No covername**

**Requirements:**
1. Detect comma-delimited person name: `Kalinin, Mikhail Ivanovich`
2. Invert to canonical: `Mikhail Ivanovich Kalinin`
3. Keep both forms as aliases (for matching flexibility)

## Test Case 5: Person with Question Mark (Uncertainty)

**Input:**
```
Kalinin, ?: Soviet sailor and Soviet internal security source. Cover name in Venona: ELKIN. Venona San
Francisco KGB, 88, 262; Venona Special Studies, 100.
```

**Expected Output:**
- **Person entity:** "Kalinin" (question mark removed)
  - Aliases: ["Kalinin"]
- **Covername entity:** "ELKIN"
  - Aliases: ["ELKIN"]
- **Relationship:** ELKIN --covername_of--> Kalinin

**Requirements:**
1. Remove question marks: `Kalinin, ?` → `Kalinin`
2. Extract person: "Kalinin" (no first name, just last name)
3. Extract covername from body: "Cover name in Venona: ELKIN" → "ELKIN"
4. Create relationship

## Test Case 6: Person with Full Name, No Covername

**Input:**
```
Kalinin, Tikhon Ivanovich: SGPC official. Venona San Francisco KGB, 77.
```

**Expected Output:**
- **Person entity:** "Tikhon Ivanovich Kalinin" (inverted)
  - Aliases: ["Tikhon Ivanovich Kalinin", "Kalinin, Tikhon Ivanovich"]
- **No covername**
- **Notes:** "SGPC official"

**Requirements:**
1. Invert comma-delimited name: `Kalinin, Tikhon Ivanovich` → `Tikhon Ivanovich Kalinin`
2. Keep both forms as aliases
3. Extract description: "SGPC official" → entity notes

## Implementation Checklist

- [ ] Split headword by brackets: extract `[X]` and split by "and"
- [ ] Handle quoted headwords: remove quotes but keep as alias
- [ ] Invert comma-delimited person names: `Last, First` → `First Last`
- [ ] Remove question marks from names: `Name, ?` → `Name`
- [ ] Detect cross-references: "See X" → create alias relationship
- [ ] Handle unidentified persons: no person entity, just covername
- [ ] Extract covernames from body: "Cover name in Venona: X"
- [ ] Create multiple covername entities when headword has brackets/and
- [ ] Link all covernames to same person when appropriate
