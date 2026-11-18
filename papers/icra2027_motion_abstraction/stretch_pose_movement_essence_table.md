| cue | pose essence | movement essence | Stretch pose (Human) | Stretch pose (Gemini) | Stretch core movement (Human) | Stretch core movement (Gemini) |
|---|---|---|---|---|---|---|
| nod_yes | 고개 정면·토르소 mid | 고개 위아래 끄덕임 | head=center, torso=mid | 동일 (head=center, torso=mid) | head tilt [20,0]×2 | 동일 (head tilt [20,0]×2) |
| shake_head_no | 몸/팔 still·토르소 mid | 고개 좌우 흔듦 | dir=still, torso=mid | 동일 | head pan [-30,30]×2 | 동일 |
| gaze_shift_avoid | 정면 still에서 시작 | 시선을 옆으로 돌림(+회피) | dir=still, torso=mid | 동일 | head pan 45° & base yaw -35° + hold (머리+베이스 동시 회피) | head pan 45° + hold만 (베이스 path 없음) |
| tilt_head_curious | 정면 still | 한쪽으로 기울여 유지 | dir=still, torso=mid | 동일 | head tilt 25° + hold | 동일 |
| cover_yawn_sleepy | 손을 입 앞(front, z≈60) | 가리며 고개 숙이고 유지 | dir=front, grip=horiz, z=60, x=25 | 동일 | EE xz 호(sweep90) + head tilt -20 + hold | 동일 |
| rub_eye_tired | 손을 얼굴/눈쪽(in, z high) | 눈 주위 문지름 | dir=in, grip=vert, torso=high, z=90 | 동일 | EE 소원 arc(yz, r=0.03)×4 | 동일 |
| facepalm | 손을 얼굴 쪽(back/높음) | 덮고 고개·몸 숙임 | dir=back, grip=horiz, torso=high, z=50 | dir=back, grip=horiz, torso=high, z=95 (더 높음) | hold 후 head tilt -30 + torso height high | hold 후 head tilt -30 + torso height mid |
| point_you | 팔 앞·손가락/손 상대쪽 | 내밀어 지시 유지 | dir=front, grip=vert, z=60, x=30, elbow=F | 동일 | EE +x 0.12 후 hold | 동일 |
| point_self | 팔/손을 몸 쪽(back, elbow T) | 가슴 쪽으로 당겨 지시 | dir=back, grip=vert, elbow=T, x=20, z=55 | 동일 | EE -x 0.05 후 hold | 동일 |
| beckon_come_here | 팔 앞·손 내민 자세 | 몸 쪽으로 반복 끌어당김 | dir=front, grip=horiz, z=65, x=50 | 동일 | base 후진 + elbow y [-40,5]×3 (베이스+팔 동시) | elbow y [-40,5]×3만 (베이스 path 없음) |
| stop_palm_out | 손바닥 앞·높은 z | 앞으로 막아 유지 | dir=front, grip=horiz, torso=high, z=80, x=25 | 동일 | base 소각 yaw±12 + 전진 0.18 + hold (베이스로 차단감) | EE +x 0.15 + hold (팔만 내밈) |
| salute | 손을 이마/머리 쪽(in) | 올려 붙이고 유지 | dir=in, grip=horiz, z=60, x=10 | 동일 | base 호(deg45) + EE +z 0.25 + hold | EE +z 0.30 + hold (베이스 없음) |
| raise_index_one_moment | 검지/손 앞으로→위로 | 올려 들고 유지 | front+vert → 이어서 up pose | front+vert만 (up으로 안 바꿈) | base 전진 후 up pose + hold | EE +z 0.25 + hold (베이스·pose 전환 없음) |
| highfive_invite | 손바닥 앞→위로 든 초대 | 내밀어 유지 | front → up+torso high로 pose 전환 | front에서 시작, up 전환 없음 | base 전진 + EE +x + hold | EE +z 0.15 + hold (베이스 없음) |
| fist_pump | 주먹 위(up, vert, z high) | 위아래 펌핑 | dir=up, grip=vert, torso=high, z=85 | 유사 (x 슬롯 없음) | shoulder z [+20,-20]×1 + hold0.6 | shoulder z [+20,-20]×3 (더 반복) |
| shrug_idk | 팔 바깥·손바닥 옆 | 어깨/몸 으쓱 | out→나중에 in으로 풀림 | out, z=70 유지 | base 후진·선회 + torso high 으쓱 + 앞뒤 흔들림 | torso high→mid만 (베이스 path 전무) |
| flex_bicep | 팔 벌렸다 접어 이두 | 힘준 자세 유지 | out+elbow F → up pose로 전환 | out+elbow F 유지 | base yaw 40° 후 up hold | EE yz 호(sweep90) 후 hold (베이스 없음) |
| air_quotes | 손 앞·높게 (따옴표 손) | 손목으로 꺾기 반복 | dir=front, grip=vert, torso=high, z=85 | 동일 | base 전진 + wrist z ±10×2 | wrist x ±25×2만 (베이스 없음) |
| arms_open_welcome | 팔 앞·벌릴 준비 | 공간을 여는 개방 동작 | dir=front, grip=horiz, z=60 | 동일 | base 큰 호 100° + 소각±25 + 호 (베이스로 양팔 개방 대용) | EE xy 호 sweep-90 + hold (한 팔 스윕만) |
| hands_up_surrender | 손 위(up) | 든 채 유지/움츠림 | dir=up, grip=horiz, z=80 | dir=front에서 시작 (up 아님) | base 요란한 좌우 선회 + 후진 + hold | EE xz 호 sweep90 + hold (베이스 없음) |
| stretch_arms_up | 손 위·낮은 z에서 시작 | 리프트로 쭉 펴기 | dir=up, z=30 | dir=up, z=70 (이미 높음) | torso→high + base 소각±15 + hold | torso→high + hold (베이스 없음) |
| raise_fists_victory | 주먹 위 | 환호하며 올림 | dir=up, grip=vert, z=50 | dir=up, grip=vert, z=70 | base 720° + 큰 호 + 전진 (환호는 베이스) | torso→high + hold (암 펌핑/베이스 회전 없음) |
| cross_arms_defensive | 팔 몸쪽으로(in) | 닫힌 방어 유지 | dir=in, grip=vert, z=60 | dir=out, head=down (닫힘보다 외면) | base 소각±25 + 후진 + hold | EE +y 0.25 + hold (한 팔 가로막기) |
| palms_together_thank_you | 손 모아·토르소 low | 숙여 인사 | dir=up, grip=vert, torso=low, z=50 | dir=down, grip=vert, torso=low, z=30 | EE ↓↑ + base 미세 앞뒤 (배례) | EE +z + head tilt 25° hold (고개 숙임 강조) |
| draw_circle_repeat | 손 앞 | 원 반복 | dir=front, grip=vert, z=65, x=40 | dir=front, grip=vert, z=65 | EE yz 원 r=0.05×2 | EE yz 원 r=0.1×4 (더 큼·더 많이) |
| depict_tree | 팔/손 높게 가지 실루엣 | 미세 흔들며 유지 | dir=front, grip=horiz, torso=mid, z=85 | dir=front, z=60 (더 낮음) | wrist z ±12×2 + hold | EE yz 호 sweep-90 + hold (가지 펼침 path) |
| depict_big_bowl | 손 벌림→오므림 | 그릇 테두리 호 | out z70 → in z55로 pose 전환 | in, torso=high, z=80 단일 pose | EE yz 호 sweep-180 + hold | EE yz 호 sweep-180 + hold |
| depict_lion | 낮은 준비 자세(torso low) | 돌진+머리 흔들(할퀴기) | dir=front, grip=vert, torso=low, z=45 | 동일 | base 급전진 0.25 + head pan ±10×2 | 동일 |
| depict_small_object | 손 앞·고개 down (소물) | 작게 만지작 | dir=front, grip=vert, z=55, head=down | 동일 | base로 작은 사각 루프 + hold | EE +z 0.05 + hold (베이스 루프 없음) |
| depict_flying_bird | 팔 옆으로 펼침(out, high) | 날갯짓 | dir=out, grip=horiz, torso=high, z=80 | 동일 | shoulder z ±20×3 + base 대원 720° (날갯짓+비행) | shoulder z ±20×3만 (베이스 비행 경로 없음) |
