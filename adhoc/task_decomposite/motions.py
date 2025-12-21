from nltk.corpus import wordnet as wn
import nltk
import json

gestures = """Head nod

Head shake

Finger/Arm Point

Wave

Thumb-up

V-sign (Peace/Victory)

Beckon

Palms up

Shoulder shrug

Cross arms

Hold hands

Grasp arm

Push/Shove

Proxemics (Distance)

Flip/arrange hair

Tilt head up

Tap fingers

Scratch head

Touch nose

Cup hand over mouth

Clap

Clench fist

Interlace fingers

Wink

Finger to lips

OK sign (join thumb and index finger)

Lean in

Tap with a foot

Touch ears

Point to the chin with index finger

Hands on hips (Akimbos)

Clasp hands over knees

Lean back

Slouch

Erect posture

Leg bounce/shake

Tap feet

Expand/Restrict movements

Hunch over

Hands clasp behind back

Cup chin in hand

Walk speed (fast/slow)

Pace

Straighten clothes

Move feet under the table

Tug at hair

Wipe sweat from the forehead

Pound fist on table

Sit on the edge of the seat

Bow

Self-touch

Feet point outward

Feet point toward the exit

Shift weight to one leg

Touch knees or thighs

Handshake

Hug

Pat the shoulder

Place a hand on the back

High-five

Hug and back rub

Rub palms together

Stiff movements

Cover one ear with a hand

Spit

Spread arms wide"""
gestures = [r.lower() for r in gestures.split('\n') if len(r.strip()) > 0]


def save_json(data, path):
    with open(path, 'w') as f:
        json.dump(data, f, indent=2)


nltk.download('wordnet')
nltk.download('omw-1.4')  # for extended multilingual glosses

verbs = set()
for synset in wn.all_synsets('v'):
    for lemma in synset.lemmas():
        verbs.add(lemma.name())
        
verbs = list(verbs)
cnt = 0
gestures_in_verbs = 0
gestures_not_in_verbs = 0
vacant_examples = 0

total_verb_dict = []
for verb in verbs:
    synsets = wn.synsets(verb, pos='v')  # 'v' limits to verb senses
    total_verb_dict.extend([{
        'verb': verb,
        'definition': s.definition(),
        'example': s.examples()
    } for s in synsets])
    vacant_examples += sum(len(s.examples()) == 0 for s in synsets)
    cnt += len(synsets)
    if verb in gestures:
        gestures_in_verbs += 1

print(f"Gestures in verbs: {gestures_in_verbs}")
print(f"Gestures not in verbs: {len(gestures) - gestures_in_verbs}")

import os
os.makedirs('data', exist_ok=True)
save_json(total_verb_dict, 'data/verbnet.json')