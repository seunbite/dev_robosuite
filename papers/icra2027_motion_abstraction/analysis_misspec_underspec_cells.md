# Cell-level misspec / underspec (GT vs Gemini 2.5 Pro)

Notation: `type(attr=gt→llm)` or `type(attr=gt)` when the GT slot is missing in the LLM (`∅` = absent).
**Bold** = spatial / meaning-critical (`location`, `facing`, `axis`, `path_shape`).

## Platform summary

| | misspec | underspec | both | GT comps | LLM comps |
|---|---:|---:|---:|---:|---:|
| all | 90% | 46% | 44% | 2.73 | 2.91 |
| Humanoid | 97% | 67% | 63% | 2.47 | 2.73 |
| Tiago | 100% | 50% | 50% | 2.30 | 2.87 |
| Single Arm | 90% | 17% | 17% | 2.37 | 3.00 |
| Stretch | 67% | 63% | 57% | 3.83 | 2.77 |
| Roomba | 97% | 33% | 33% | 2.67 | 3.20 |

90% includes weak diffs (e.g. speed). Spatial misspec (location/facing/axis/path_shape only): **79%** of cells.

## Per-cell detail (30 cues × 5 bodies)

### Nod yes (`nod_yes`)

| Body | Misspec | Underspec |
|---|---|---|
| Humanoid *[M]* | movement(degrees=(-22.0, 22.0)→(-18.0, 5.0); speed=1.5→1.6) | — |
| Tiago *[M]* | movement(degrees=(-22.0, 22.0)→(-20.0, 5.0)) | — |
| Single Arm *[M]* | pose(**facing=horizontal→vertical**); movement(**z=-25→-20**; **z=25→20**; speed=1.5→1.2; hold=positive→zero; speed=1.5→1.2; +1) | — |
| Stretch | — | — |
| Roomba *[M]* | path(**x=(-0.15, 0.15)→(0.12, -0.12)**; speed=2→1.8) | — |

### Shake head no (`shake_head_no`)

| Body | Misspec | Underspec |
|---|---|---|
| Humanoid *[M]* | movement(speed=2→2.2) | — |
| Tiago *[M]* | movement(hold=positive; speed=2→2.2) | — |
| Single Arm | — | — |
| Stretch | — | — |
| Roomba *[M]* | path(degrees=(-30.0, 30.0)→(35.0, 35.0); speed=2→2.8) | — |

### Draw circle repeat (`draw_circle_repeat`)

| Body | Misspec | Underspec |
|---|---|---|
| Humanoid *[M]* | pose(**z=60→80**); path(plane=xz→yz; radius=0.12→0.08) | — |
| Tiago *[M]* | pose(**z=60→70**); path(radius=0.12→0.08; speed=1.5→1.8) | — |
| Single Arm *[M]* | path(speed=2→1.5) | — |
| Stretch *[M]* | pose(**x=40**); path(radius=0.05→0.1; speed=1.4→1.8) | — |
| Roomba *[M]* | path(radius=0.4→0.25; speed=1.5→2) | — |

### Gaze shift avoid (`gaze_shift_avoid`)

| Body | Misspec | Underspec |
|---|---|---|
| Humanoid **[both]** | movement(hold=positive; speed=1→0.8) | movement |
| Tiago *[M]* | movement(hold=positive; speed=1→0.8); path(**mode=base→ee**; **shape=arc→line**; radius=0; speed=1→0.5; sweep=20) | — |
| Single Arm *[M]* | pose(**facing=horizontal→vertical**; **x=55**; **z=55→75**); movement(**y=10→-45**; speed=0.8→1.8; hold=positive→zero; speed=0.8→1.8) | — |
| Stretch *[U]* | — | path[hold] |
| Roomba *[M]* | path(degrees=120→75; hold=positive; speed=1→0.9) | — |

### Fist pump (`fist_pump`)

| Body | Misspec | Underspec |
|---|---|---|
| Humanoid **[both]** | pose(**arm=up→front**; **x=70**; **y=60**) | movement |
| Tiago *[M]* | pose(**arm=up→front**; **x=75**; **y=60**; **z=80→60**); movement(degrees=(0.0, 30.0)→(-20.0, 20.0); rep=single→plural) | — |
| Single Arm *[M]* | pose(**x=55**; **z=55→60**); movement(**z=35→25**; hold=positive→zero; rep=single→plural) | — |
| Stretch *[M]* | pose(**x=15**); movement(rep=single→plural) | — |
| Roomba *[M]* | path(**x=0.4→(0.15, -0.15)**; hold=positive; rep=single→plural; speed=4→3.8) | — |

### Tilt head curious (`tilt_head_curious`)

| Body | Misspec | Underspec |
|---|---|---|
| Humanoid **[both]** | movement(hold=positive; degrees=(0.0, 30.0)→(0.0, -25.0); speed=0.7→0.8) | movement |
| Tiago *[M]* | movement(hold=positive; degrees=(0.0, 30.0)→(0.0, 20.0); speed=0.7→0.6) | — |
| Single Arm *[M]* | pose(**facing=horizontal→vertical**; **z=70→75**); path(**axis=y→x**; distance=0.12→0; speed=0.7→0.5) | — |
| Stretch | — | — |
| Roomba *[M]* | path(**shape=arc→line**; degrees=15; radius=0) | — |

### Cover yawn sleepy (`cover_yawn_sleepy`)

| Body | Misspec | Underspec |
|---|---|---|
| Humanoid *[M]* | pose(**arm=up→in**; **facing=horizontal→vertical/palmin**; **head=up→center**; **x=70→60**; **y=50→75**; +1); movement(**axis=tilt→height**; hold=positive; degrees=(0.0, 15.0)→(0.18, 0.1795); joint=head→torso; speed=0.5→0.6); movement(**axis=height→tilt**; **height=low**; joint=torso→head; speed=0.6→0.5) | — |
| Tiago *[M]* | pose(**arm=in→front**; **x=70→65**; **y=75→70**; **z=60→55**); movement(**height=high**; hold=positive; speed=1.2→0.7); movement(**axis=height→tilt**; **height=mid**; joint=torso→head; speed=0.8→0.7); hold(t=1.5→1.2) | — |
| Single Arm **[both]** | path(**axis=z→x**; distance=-0.2→-0.05; speed=0.5→0.6) | path |
| Stretch | — | — |
| Roomba **[both]** | path(**shape=arc→line**; degrees=-10; radius=0.4; speed=0.2→0.5) | path[submotion] |

### Rub eye tired (`rub_eye_tired`)

| Body | Misspec | Underspec |
|---|---|---|
| Humanoid *[M]* | pose(**facing=vertical→vertical/palmback**; **head=center→down**; **x=75→60**; **y=50→60**); path(radius=0.03→0.02; speed=1.5→1.2) | — |
| Tiago *[M]* | pose(**arm=in→front**; **head=center→down**; **y=70→65**; **z=62→60**); path(radius=0.03→0.015; speed=0.8→1.2) | — |
| Single Arm *[M]* | pose(**dir=left→back**); path(**axis=z**; **shape=line→arc**; distance=(-0.05, 0.05); hold=zero→positive; speed=1→1.5) | — |
| Stretch | — | — |
| Roomba *[M]* | path(**y=(-0.05, 0.05)→-0.05**; rep=plural; speed=1.5→0.7); path(degrees=-20→(-60.0, 60.0); hold=positive; radius=0.4→0.08; speed=0.2→1.2) | — |

### Point you (`point_you`)

| Body | Misspec | Underspec |
|---|---|---|
| Humanoid **[both]** | pose(**z=50→70**) | hold[hold] |
| Tiago **[both]** | pose(**z=50→55**) | hold[hold] |
| Single Arm *[M]* | pose(**z=50→65**) | — |
| Stretch | — | — |
| Roomba *[M]* | path(**x=0→0.3**; speed=0.5→3.5) | — |

### Point self (`point_self`)

| Body | Misspec | Underspec |
|---|---|---|
| Humanoid **[both]** | pose(**arm=back→front**; **facing=horizontal→vertical**; **x=90→80**; **y=70→75**; **z=80→70**) | hold[hold] |
| Tiago **[both]** | pose(**z=60→45**) | hold[hold] |
| Single Arm *[M]* | pose(**z=40→55**) | — |
| Stretch | — | — |
| Roomba *[M]* | path(**shape=arc→line**; degrees=180; radius=0; speed=0.5→2.5) | — |

### Beckon come here (`beckon_come_here`)

| Body | Misspec | Underspec |
|---|---|---|
| Humanoid **[both]** | pose(**facing=vertical/palmin→horizontal/palmup**; **x=60**; **z=80→65**) | movement |
| Tiago **[both]** | pose(**facing=vertical→horizontal**; **x=70→75**) | movement |
| Single Arm *[M]* | movement(**x=-25→-35**; **x=25→35**; speed=2→2.8; hold=zero→positive; speed=2→2.8; +1) | — |
| Stretch *[U]* | — | path |
| Roomba **[both]** | path(**x=-0.15→(0.2, -0.15)**; speed=3→1.8) | path, path |

### Stop palm out (`stop_palm_out`)

| Body | Misspec | Underspec |
|---|---|---|
| Humanoid **[both]** | pose(**arm=up→front**; **z=60→80**) | hold[hold] |
| Tiago **[both]** | pose(**arm=up→front**; **z=60→65**; elbow_bended=false→true) | movement |
| Single Arm *[M]* | pose(**dir=front→up**; **z=60→65**); path(distance=0.1→0.08; hold=positive; rep=single; speed=1.5→2.5) | — |
| Stretch **[both]** | path(**mode=base→ee**; **x=0.18**; speed=2.4→1.8) | path |
| Roomba **[both]** | path(speed=2.5→3.5) | path |

### Salute (`salute`)

| Body | Misspec | Underspec |
|---|---|---|
| Humanoid **[both]** | pose(**arm=in→up**; **facing=horizontal/palmdown→horizontal/palmfront**; **x=40**; **y=50→70**; **z=80→90**) | path, movement |
| Tiago **[both]** | pose(**facing=vertical→horizontal**; **x=50→70**; **y=50→60**; **z=65→60**) | path |
| Single Arm *[M]* | pose(**facing=horizontal→vertical**; **x=55**; **y=50→40**; **z=55→50**); pose(**x=70**; **y=60→35**) | — |
| Stretch **[both]** | path(distance=0.25→0.3) | path |
| Roomba *[M]* | path(**shape=arc→line**; degrees=50; radius=0.4; speed=2.4→3) | — |

### Raise index one moment (`raise_index_one_moment`)

| Body | Misspec | Underspec |
|---|---|---|
| Humanoid **[both]** | pose(**facing=vertical/palmin→vertical**; **x=70**; **y=60**; **z=70→60**) | movement |
| Tiago **[both]** | pose(**z=70→45**) | movement[hold] |
| Single Arm *[M]* | pose(**x=55**; **y=50**; **z=55→70**) | — |
| Stretch **[both]** | path(**mode=base→ee**; **x=0.18**; speed=2.8→1.6) | pose[hold] |
| Roomba *[M]* | path(speed=2.8→2) | — |

### Highfive invite (`highfive_invite`)

| Body | Misspec | Underspec |
|---|---|---|
| Humanoid **[both]** | pose(**arm=up→front**); path(distance=0.1→0; rep=single; speed=3→0.5) | hold[hold] |
| Tiago **[both]** | pose(**arm=up→front**; **x=60→65**; **y=40**; **z=86→60**); path(distance=0.2→0; rep=single; speed=3→0.5) | hold[hold] |
| Single Arm **[both]** | pose(**dir=up→front**; **z=70→55**) | movement |
| Stretch **[both]** | path(**axis=x→z**; distance=0.12→0.15; speed=1.8→1.5); hold(t=1.3→1.5) | path, pose |
| Roomba **[both]** | path(**x=0.3→0.2**; speed=2→1.5) | path[submotion] |

### Shrug idk (`shrug_idk`)

| Body | Misspec | Underspec |
|---|---|---|
| Humanoid *[M]* | pose(**y=40**; **z=30→45**); movement(hold=positive; degrees=(20.0, -20.0)→(0.0, 25.0); speed=1→2.2) | — |
| Tiago *[M]* | pose(**y=30**; **z=50→60**); movement(hold=positive; degrees=(20.0, -20.0)→(0.0, -25.0)) | — |
| Single Arm *[M]* | pose(**facing=horizontal→vertical**; **z=50→45**); movement(**z=20→0**; **z=-20→20**; speed=1→2.2; hold=positive→zero; speed=1→2.2) | — |
| Stretch **[both]** | pose(**z=50→70**); movement(speed=2→2.5) | path, path, path, path, pose |
| Roomba **[both]** | path(degrees=28→35; radius=0→0.2; speed=1.4→1.8); path(**shape=line→arc**; **x=0.1**; speed=3.2→1.8); path(**x=-0.12→0**; speed=2.6→0.5) | path |

### Flex bicep (`flex_bicep`)

| Body | Misspec | Underspec |
|---|---|---|
| Humanoid **[both]** | pose(**x=50**) | hold[hold] |
| Tiago **[both]** | pose(**facing=horizontal→vertical**; **torso=mid→high**; **x=60**; **y=45**; **z=70→80**) | movement |
| Single Arm *[M]* | pose(**dir=left→up**; **x=70**; **z=65→40**); movement(**x=-30**; **x=30**; speed=1→2.5; hold=positive→zero; speed=1→2.5) | — |
| Stretch **[both]** | path(**mode=base→ee**; radius=0→0.12; speed=1.8→2; sweep=40→90); hold(t=1→1.4) | pose |
| Roomba *[M]* | path(**shape=arc→line**; degrees=360; radius=0.1; speed=4→1.2); path(degrees=(-5.0, 5.0)→25; rep=plural; speed=1.5→3.5) | — |

### Air quotes (`air_quotes`)

| Body | Misspec | Underspec |
|---|---|---|
| Humanoid *[M]* | pose(**facing=horizontal/palmdown→vertical/palmin**; **x=80**; **y=60→65**; **z=70→85**); movement(**axis=z→y**; speed=2→2.8) | — |
| Tiago *[M]* | pose(**facing=horizontal→vertical**; **x=75**; **y=60**); movement(**axis=z→x**; degrees=(-60.0, 60.0)→(-20.0, 20.0); speed=2→2.5) | — |
| Single Arm *[M]* | pose(**facing=horizontal→vertical**) | — |
| Stretch **[both]** | movement(**axis=z→x**; degrees=(-10.0, 10.0)→(-25.0, 25.0)) | path |
| Roomba *[M]* | path(**shape=line→arc**; **x=(-0.05, 0.05)**; speed=2.5→3.5) | — |

### Arms open welcome (`arms_open_welcome`)

| Body | Misspec | Underspec |
|---|---|---|
| Humanoid **[both]** | pose(**facing=horizontal/palmup→horizontal/palmfront**; **torso=mid→high**; **y=0**; **z=90→60**) | movement[hold] |
| Tiago *[M]* | movement(**axis=y→z**; hold=positive; degrees=(0.0, -10.0)→(0.0, 10.0); speed=2→0.8) | — |
| Single Arm *[M]* | — | — |
| Stretch **[both]** | path(**mode=base→ee**; radius=0.5→0.2; speed=1.8→1.2; sweep=100→-90); hold(t=1→1.5) | path, path[hold] |
| Roomba *[M]* | path(degrees=100→35; radius=0.55→0.4; speed=1.8→0.9); path(degrees=(-28.0, 28.0)→-35; radius=0→0.4; rep=plural; speed=2.4→0.9); path(**shape=arc→line**; degrees=100; radius=0.55; speed=1.8→0.5) | — |

### Hands up surrender (`hands_up_surrender`)

| Body | Misspec | Underspec |
|---|---|---|
| Humanoid *[M]* | pose(**y=20**; **z=85→95**) | — |
| Tiago **[both]** | pose(**z=90→80**) | path |
| Single Arm *[M]* | pose(**dir=up→front**; **z=90→45**) | — |
| Stretch **[both]** | pose(**arm=up→front**; **z=80→60**); path(**mode=base→ee**; radius=0→0.25; speed=2→1.8); hold(t=1→1.5) | path, hold, path[hold], hold, path |
| Roomba **[both]** | path(**x=-0.22→-0.15**; speed=1.1→2.5) | path, path[submotion], path, path, path |

### Stretch arms up (`stretch_arms_up`)

| Body | Misspec | Underspec |
|---|---|---|
| Humanoid **[both]** | pose(**facing=horizontal→vertical/palmin**; **z=100→90**) | hold[hold] |
| Tiago **[both]** | pose(**facing=horizontal→vertical**; **torso=mid→high**; **z=95→60**) | hold[hold] |
| Single Arm | — | — |
| Stretch **[both]** | pose(**z=30→70**); movement(speed=1.5→0.7); hold(t=1.2→1.5) | path[hold] |
| Roomba *[M]* | path(**x=0.5→0.3**; speed=3.5→0.6) | — |

### Raise fists victory (`raise_fists_victory`)

| Body | Misspec | Underspec |
|---|---|---|
| Humanoid **[both]** | pose(**facing=horizontal/palmback→vertical**; **torso=mid→high**; **x=65→40**; **y=40**; **z=65→90**; +2) | movement |
| Tiago *[M]* | pose(**arm=up→front**; **torso=mid→high**; **x=65**; **y=40**; **z=65→60**); path(**axis=z→x**; distance=0.2→0; rep=single; speed=3.5→0.5) | — |
| Single Arm *[M]* | pose(**dir=up→front**; **z=65→40**); path(**axis=z→x**; distance=0.2→0; rep=single; speed=3.5→0.5) | — |
| Stretch **[both]** | pose(**z=50→70**); hold(t=1→1.5) | path, path[hold], path |
| Roomba **[both]** | path(speed=4.5→4); path(**x=0.08→0**; speed=3.5→0.5) | path |

### Cross arms defensive (`cross_arms_defensive`)

| Body | Misspec | Underspec |
|---|---|---|
| Humanoid **[both]** | pose(**arm=in→out**; **facing=horizontal/palmdown→vertical**; **x=70**; **y=70**; **z=60→70**) | hold[hold] |
| Tiago **[both]** | pose(**facing=horizontal→vertical**; **x=65→60**; **y=100**; left_arm=mirror→in) | hold[hold] |
| Single Arm | — | — |
| Stretch **[both]** | pose(**arm=in→out**; **head=center→down**); path(**mode=base→ee**; **x=-0.15**; speed=1→1.2) | path |
| Roomba *[M]* | path(**shape=arc→line**; degrees=(-30.0, 30.0); radius=0; rep=plural; speed=1→1.8); path(**shape=line→arc**; **x=-0.15**; speed=1→2.5) | — |

### Palms together thank you (`palms_together_thank_you`)

| Body | Misspec | Underspec |
|---|---|---|
| Humanoid *[M]* | pose(**arm=up→in**; **facing=vertical→vertical/palmin**; **x=70**; **y=70→75**); movement(speed=0.5→0.7) | — |
| Tiago **[both]** | pose(**arm=up→in**; **z=65→55**) | hold[hold] |
| Single Arm **[both]** | pose(**dir=left→up**; **z=65→60**) | movement |
| Stretch **[both]** | pose(**arm=up→down**; **z=50→30**); path(distance=-0.12→0.15) | path, path, path, hold |
| Roomba **[both]** | path(**shape=arc→line**; degrees=-30; radius=0; speed=1.5→0.6); path(degrees=30→25; speed=1.5→0.8); path(**shape=arc→line**; degrees=30; radius=0; speed=1.5→0.5); path(degrees=-30→-25; speed=1.5→0.9) | path, path, path |

### Facepalm (`facepalm`)

| Body | Misspec | Underspec |
|---|---|---|
| Humanoid *[U]* | — | hold[hold] |
| Tiago *[M]* | pose(**arm=up→out**; **head=up→center**; **x=70**; **y=65**; **z=70→50**); movement(hold=positive; degrees=(0.0, 15.0)→(0.0, -25.0); speed=0.5→0.6) | — |
| Single Arm *[M]* | pose(**dir=left→back**; **facing=vertical→horizontal**); path(**axis=z→x**; distance=0.05→0; speed=0.4→0.5) | — |
| Stretch *[M]* | pose(**z=50→95**); movement(**height=high→mid**) | — |
| Roomba *[M]* | path(**shape=arc→line**; degrees=180; radius=0.3; speed=0.5→3.5); path(**x=0.05→0**; speed=0.3→0.5); path(**x=0→-0.15**; hold=positive) | — |

### Depict tree (`depict_tree`)

| Body | Misspec | Underspec |
|---|---|---|
| Humanoid **[both]** | pose(**arm=out→front**; **facing=horizontal/palmdown→horizontal/palmfront**; **y=0**) | hold[hold] |
| Tiago **[both]** | pose(**arm=up→still**; **x=65**; **y=0→70**; **z=60→50**) | hold[hold] |
| Single Arm **[both]** | pose(**facing=horizontal→vertical**) | movement |
| Stretch **[both]** | pose(**z=85→60**); hold(t=1.2→1.5) | movement |
| Roomba **[both]** | path(**x=0.35→0.2**; speed=1.5→0.7); path(degrees=35→(-30.0, 30.0); speed=1.2→1.8) | path |

### Depict big bowl (`depict_big_bowl`)

| Body | Misspec | Underspec |
|---|---|---|
| Humanoid **[both]** | pose(**arm=out→in**; **x=70**; **z=55→50**); path(speed=0.8→1.2; sweep=-140→-90) | hold[hold] |
| Tiago **[both]** | pose(**arm=out→front**; **facing=horizontal/palmup→horizontal**; **x=70→50**; **y=20→65**); path(radius=0.25→0.22; speed=0.8→1.2; sweep=-140→-90) | hold[hold] |
| Single Arm *[M]* | pose(**dir=right→up**; **x=20→70**; **z=55→65**); path(radius=0.25→0.2; speed=1.5→0.9; sweep=-180→-160) | — |
| Stretch **[both]** | pose(**torso=mid→high**; **x=20**; **z=55→80**); path(radius=0.18→0.25; speed=0.8→0.9); hold(t=1.5→1) | pose[submotion] |
| Roomba *[M]* | path(degrees=-180→90; radius=0.3→-0.35); path(**shape=line→arc**; **x=0**; hold=positive; speed=0.5→2) | — |

### Depict lion (`depict_lion`)

| Body | Misspec | Underspec |
|---|---|---|
| Humanoid **[both]** | pose(**facing=horizontal/palmdown→vertical**; **torso=mid→low**; **x=80**; **z=65→80**; use_feasible_bank=false); path(**axis=z→x**; distance=0.2→0.25; hold=zero; rep=single; speed=0.5→3.5); path(**axis=z→x**; distance=-0.3→0; hold=zero→positive; rep=single; speed=1.5→0.5) | movement |
| Tiago **[both]** | pose(**facing=horizontal→vertical**; **head=center→down**; **torso=mid→low**; **x=70**; **z=65→55**); path(distance=0.08→0.25; speed=2→3.5) | path, path, path[submotion] |
| Single Arm **[both]** | pose(**z=65→75**); path(**axis=z→x**; distance=0.1→0.3; rep=single; speed=0.4→3.8) | path |
| Stretch | — | — |
| Roomba | — | — |

### Depict small object (`depict_small_object`)

| Body | Misspec | Underspec |
|---|---|---|
| Humanoid **[both]** | pose(**arm=up→front**; **x=65**) | path, path |
| Tiago *[M]* | pose(**z=60→55**); path(**shape=arc→line**; plane=yz; radius=0.1; rep=single; speed=0.8→1.5; +1) | — |
| Single Arm *[M]* | path(**shape=arc→line**; plane=yz; radius=0.1; rep=single; speed=1.5→0.5; +1) | — |
| Stretch **[both]** | path(**mode=base→ee**; **y=-0.08**; speed=1.3→0.8); hold(t=0.8→1.2) | path, path[hold], path |
| Roomba **[both]** | path(**x=-0.08→0.12**; speed=1.3→0.7); path(**shape=line→arc**; **y=-0.08**; speed=1.3→2.8) | path[submotion], path |

### Depict flying bird (`depict_flying_bird`)

| Body | Misspec | Underspec |
|---|---|---|
| Humanoid *[M]* | pose(**y=0**; **z=65→60**); movement(speed=2.5→2) | — |
| Tiago *[M]* | pose(**torso=mid→high**; **y=0**); movement(degrees=(-25.0, 25.0)→(-35.0, 35.0); speed=2.5→2.8) | — |
| Single Arm *[M]* | pose(**y=100→20**) | — |
| Stretch **[both]** | movement(speed=4→2.8) | path[submotion] |
| Roomba *[M]* | path(degrees=360→-180; radius=0.8→0.2; rep=single; speed=2.5→2.2); path(**shape=line→arc**; **x=0.4**; speed=2.5→2.2); path(degrees=360→-180; radius=0.8→0.2; rep=single; speed=2.5→2.2) | — |
