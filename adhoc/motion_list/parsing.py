motions = """Head shaking: 1) Align head to center, 2) Rotate head horizontally to the left, 3) Rotate head horizontally to the right (2 to 3 repetitions).

Finger/Arm Pointing: 1) Lift arm toward the target, 2) Extend index finger while curling other fingers into a fist, 3) Lock elbow and wrist to maintain a straight line.

Waving: 1) Raise hand to shoulder or head height, 2) Turn palm outward toward the target, 3) Pivot wrist side-to-side (3 to 5 repetitions).

V-sign: 1) Raise hand to chest height, 2) Extend index and middle fingers in a 'V' shape while tucking thumb over other fingers, 3) Orient palm toward the target.

Beckoning: 1) Extend arm forward, 2) Turn palm upward, 3) Curl fingers toward the body and extend them back (2 to 4 repetitions).

Palms up: 1) Lift both hands to waist height, 2) Rotate forearms so palms face the sky, 3) Move hands slightly outward away from the torso.

Finger to lips: 1) Lift dominant hand toward the face, 2) Extend index finger vertically, 3) Place the side of the index finger against the center of the lips.

OK sign: 1) Raise hand to chest height, 2) Join the tips of the thumb and index finger to form a circle, 3) Extend the remaining three fingers upward and outward.

Shoulder shrug: 1) Keep arms at sides, 2) Lift both shoulders vertically toward the ears, 3) Briefly hold and drop shoulders back to neutral.

Crossing arms: 1) Lift both forearms to chest height, 2) Tuck the dominant hand under the opposite armpit, 3) Rest the non-dominant forearm over the dominant one.

Hands on hips (Akimbos): 1) Lift both hands to waist height, 2) Rotate palms to face downward/inward, 3) Press palms against the sides of the pelvis with elbows pointing outward.

Leaning in: 1) Keep lower body stationary, 2) Pivot the upper torso at the hips toward the front, 3) Shift the head forward beyond the center of gravity.

Leaning back: 1) Pivot the upper torso at the hips away from the front, 2) Extend the spine slightly, 3) Shift the center of gravity to the rear.

Slouching: 1) Drop shoulders forward and inward, 2) Curve the spine into a "C" shape, 3) Lower the head slightly.

Hunching over: 1) Bend the torso forward at a sharp angle, 2) Bring shoulders toward the knees, 3) Drop the head to face the ground.

Sitting on the edge of the seat: 1) Shift the pelvis forward, 2) Straighten the spine, 3) Place feet firmly on the ground with knees at a 90-degree angle.

Bowing: 1) Keep the spine straight, 2) Bend the torso forward from the hips (approx. 15-45 degrees), 3) Return slowly to a vertical position.

Flipping/arranging hair: 1) Lift hand to ear height, 2) Rotate wrist to sweep the back of the hand or fingers past the side of the head, 3) Lower the arm in a fluid arc.

Scratching head: 1) Raise hand to the top or side of the head, 2) Flex and extend fingers rapidly (3 to 5 repetitions), 3) Lower the hand.

Touching nose: 1) Raise dominant hand, 2) Briefly touch or rub the tip of the nose with the index finger or knuckle, 3) Return hand to neutral.

Cupping hand over mouth: 1) Raise hand to face, 2) Curve the palm and fingers into a "C" shape, 3) Place the hand over the mouth area.

Cupping chin in hand: 1) Raise dominant hand to face, 2) Rest the chin on the palm or the "V" between thumb and index finger, 3) Place the elbow on a surface or against the torso for support.

Straightening clothes: 1) Lower both hands to the chest or waist level, 2) Mimic a "grasp and pull" motion with fingers, 3) Sweep hands downward along the torso (2 repetitions).

Wiping sweat from forehead: 1) Raise the side of the index finger or the palm to the brow, 2) Move the hand horizontally across the forehead, 3) Shake the hand slightly at the end.

Touching ears: 1) Raise hand to head height, 2) Grasp the earlobe between thumb and index finger, 3) Gently pull or rub (2 repetitions).

Pointing to chin with index finger: 1) Raise dominant hand, 2) Extend index finger, 3) Rest the tip of the finger on the center of the chin (often combined with a slight head tilt).

Clapping: 1) Bring both hands to chest height with palms facing each other, 2) Rapidly strike the palms together, 3) Pull hands apart slightly (repeatedly).

Clenched fist: 1) Wrap four fingers tightly into the palm, 2) Lock the thumb over the index and middle fingers, 3) Tighten the forearm muscles.

Interlacing fingers: 1) Bring both hands together, 2) Slide the fingers of the left hand into the gaps between the fingers of the right hand, 3) Close the palms together.

Clasping hands over knees: 1) While seated, reach both hands to one knee, 2) Interlace fingers or place one hand over the other, 3) Lean the torso slightly forward.

Rubbing palms together: 1) Place both palms together at chest height, 2) Move hands back and forth in opposite directions (3 to 5 repetitions).

Handshake: 1) Extend dominant arm forward with thumb pointing up, 2) Close fingers around the other's "hand," 3) Move the arm up and down from the elbow (2 to 3 repetitions).

Hugging (Solo Robot): 1) Extend both arms wide to the sides, 2) Close arms in a circular motion toward the front, 3) Cross the forearms slightly as if overlapping another body.

Patting the shoulder: 1) Extend one arm forward and slightly up, 2) Keep the palm flat and facing down, 3) Move the hand up and down in short, rhythmic strokes.

High-five: 1) Raise dominant hand above head height, 2) Turn palm forward, 3) Push the hand forward in a short, forceful strike.

Spreading arms wide: 1) Lift both arms to shoulder height, 2) Extend them fully to the left and right, 3) Rotate palms forward (Expansive movement).

Tapping fingers: 1) Rest hand on a surface or leg, 2) Lift and lower each finger individually or in sequence (repeatedly).

Tapping with a foot: 1) Keep the heel on the ground, 2) Lift the front of the foot (toes), 3) Strike the floor repeatedly.

Leg bouncing/shaking: 1) While seated, lift the heel slightly off the ground, 2) Move the knee up and down rapidly using the calf muscle (continuous repetition).

Moving feet under the table: 1) Keep upper body still, 2) Shift feet back and forth or cross/uncross ankles.

Pushing/Shoving: 1) Bring both hands to chest height, 2) Turn palms forward with fingers up, 3) Forcefully extend elbows to thrust the hands forward.

Pounding fist on table: 1) Form a clenched fist, 2) Raise the arm 10-20cm above a surface, 3) Strike the bottom of the fist (pinky side) downward forcefully.

Spitting (Humanoid Mimicry): 1) Tilt head slightly forward, 2) Thrust the head forward quickly, 3) Return head to neutral (simulates the physical exertion of spitting)."""

motions = [motion.strip().split(": ") for motion in motions.split("\n") if len(motion.strip()) > 0]
motions = [
    {'cue' : motion[0], 'gestures': [gesture.strip().replace(", 2", "").replace(", 3", "") for gesture in motion[1].split(") ")[1:]]} for motion in motions
]

import json
with open('data/seed/motion_list.jsonl', 'w') as f:
    for motion in motions:
        f.write(json.dumps(motion, ensure_ascii=False) + '\n')