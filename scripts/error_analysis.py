import json
from jiwer import process_words

# Load predictions
with open("goal_predictions.json") as f:
    data = json.load(f)

# Known player/team/manager names from this match
ENTITIES = {
    "liverpool", "city", "manchester", "sterling", "aguero", "milner",
    "coutinho", "lallana", "firmino", "henderson", "clyne", "moreno",
    "sakho", "lovren", "mignolet", "lucas", "origi", "benteke", "ibe",
    "allen", "toure", "yaya", "fernandinho", "silva", "nasri", "navas",
    "otamendi", "mangala", "demichelis", "hart", "kompany", "sagna",
    "kolarov", "delph", "bony", "pellegrini", "klopp", "leicester",
    "sturridge", "can", "flanagan", "fernando", "iheanacho",
    "clichy", "zabaleta", "mane", "wijnaldum", "emre"
}

# Error counters
entity_sub = 0
entity_del = 0
entity_ins = 0
other_sub = 0
other_del = 0
other_ins = 0
total_ref_words = 0
entity_ref_words = 0

# Collect examples
entity_error_examples = []
other_error_examples = []

for seg in data:
    ref = seg["ref"]
    pred = seg["pred"]
    
    if not ref.strip() or not pred.strip():
        continue
    
    result = process_words(ref, pred)
    
    ref_words = ref.split()
    total_ref_words += len(ref_words)
    entity_ref_words += sum(1 for w in ref_words if w.lower() in ENTITIES)
    
    for chunk in result.alignments[0]:
        if chunk.type == "equal":
            continue
        
        # Get the words involved
        ref_slice = ref_words[chunk.ref_start_idx:chunk.ref_end_idx]
        pred_words = pred.split()
        hyp_slice = pred_words[chunk.hyp_start_idx:chunk.hyp_end_idx]
        
        is_entity = any(w.lower() in ENTITIES for w in ref_slice + hyp_slice)
        
        if chunk.type == "substitute":
            n = max(len(ref_slice), len(hyp_slice))
            if is_entity:
                entity_sub += n
                if len(entity_error_examples) < 20:
                    entity_error_examples.append(f"  SUB: '{' '.join(ref_slice)}' -> '{' '.join(hyp_slice)}'")
            else:
                other_sub += n
                if len(other_error_examples) < 20:
                    other_error_examples.append(f"  SUB: '{' '.join(ref_slice)}' -> '{' '.join(hyp_slice)}'")
        elif chunk.type == "delete":
            n = len(ref_slice)
            if is_entity:
                entity_del += n
                if len(entity_error_examples) < 20:
                    entity_error_examples.append(f"  DEL: '{' '.join(ref_slice)}'")
            else:
                other_del += n
                if len(other_error_examples) < 20:
                    other_error_examples.append(f"  DEL: '{' '.join(ref_slice)}'")
        elif chunk.type == "insert":
            n = len(hyp_slice)
            if is_entity:
                entity_ins += n
                if len(entity_error_examples) < 20:
                    entity_error_examples.append(f"  INS: '{' '.join(hyp_slice)}'")
            else:
                other_ins += n
                if len(other_error_examples) < 20:
                    other_error_examples.append(f"  INS: '{' '.join(hyp_slice)}'")

entity_total = entity_sub + entity_del + entity_ins
other_total = other_sub + other_del + other_ins
all_total = entity_total + other_total

print("=" * 60)
print("ERROR ANALYSIS — GOAL (Liverpool 3-0 Man City)")
print("=" * 60)
print(f"Total reference words: {total_ref_words}")
print(f"Entity words in reference: {entity_ref_words} ({100*entity_ref_words/total_ref_words:.1f}%)")
print(f"Total errors: {all_total}")
print()
print(f"ENTITY errors: {entity_total} ({100*entity_total/all_total:.1f}% of all errors)")
print(f"  Substitutions: {entity_sub}")
print(f"  Deletions:     {entity_del}")
print(f"  Insertions:    {entity_ins}")
print()
print(f"OTHER errors:  {other_total} ({100*other_total/all_total:.1f}% of all errors)")
print(f"  Substitutions: {other_sub}")
print(f"  Deletions:     {other_del}")
print(f"  Insertions:    {other_ins}")
print()
print(f"Entity WER contribution: {100*entity_total/total_ref_words:.2f}%")
print(f"Other WER contribution:  {100*other_total/total_ref_words:.2f}%")
print(f"Combined WER:            {100*all_total/total_ref_words:.2f}%")
print()
print("--- Entity error examples ---")
for ex in entity_error_examples:
    print(ex)
print()
print("--- Other error examples ---")
for ex in other_error_examples:
    print(ex)
