import matplotlib.pyplot as plt
from venn._venn import generate_colors, init_axes, generate_logics, draw_text, draw_ellipse
from venn._constants import SHAPE_COORDS, SHAPE_DIMS, SHAPE_ANGLES, PETAL_LABEL_COORDS

# 각 로봇이 가능한 행동 집합
humanoid = {
    "Head nodding", "Head shaking", "Pointing", "Waving",
    "Thumbs-up", "V-sign", "Beckoning", "Palms up",
    "Shoulder shrug", "Crossing arms", "Holding hands",
    "Grasping arm", "Pushing/Shoving", "Proxemics",
    "Tilting head up", "Leaning in", "Tapping with a foot",
    "Hands on hips", "Leaning back", "Slouching",
    "Erect posture", "Leg bouncing", "Hunching over",
    "Walking speed", "Pacing", "Wiping sweat",
    "Pounding fist", "Bowing", "Self-touching",
    "Shifting weight", "Touching knees", "Handshake",
    "Hugging", "Patting shoulder", "Hand on back",
    "High-five", "Hug and back rub", "Stiff movements",
    "Spreading arms wide"
}

robot_arm = {
    "Head nodding", "Head shaking", "Pointing", "Waving",
    "Beckoning", "Palms up", "Shoulder shrug",
    "Holding hands", "Grasping arm", "Pushing/Shoving",
    "Proxemics", "Leaning in", "Hands on hips",
    "Leaning back", "Slouching", "Erect posture",
    "Hunching over", "Walking speed", "Pacing",
    "Wiping sweat", "Pounding fist", "Bowing",
    "Self-touching", "Touching knees", "Handshake",
    "Patting shoulder", "Hand on back", "High-five",
    "Hug and back rub", "Stiff movements"
}

roomba = {
    "Pushing/Shoving",
    "Proxemics",
    "Leaning back",
    "Expansive/Restricted movements",
    "Walking speed",
    "Pacing",
    "Stiff movements"
}

robot_dog = {
    "Head nodding", "Head shaking", "Pointing", "Waving",
    "Palms up", "Shoulder shrug", "Holding hands",
    "Grasping arm", "Pushing/Shoving", "Proxemics",
    "Tilting head up", "Leaning in", "Tapping with a foot",
    "Leaning back", "Slouching", "Erect posture",
    "Leg bouncing", "Hunching over", "Walking speed",
    "Pacing", "Bowing", "Stiff movements",
    "Spreading arms wide"
}

sets = {
    # "Humanoid": humanoid,
    "Robot arm": robot_arm,
    "Roomba": roomba,
    "Robot dog": robot_dog
}

def generate_petal_labels_with_values(datasets):
    """Generate petal labels with actual values instead of just sizes"""
    datasets = list(datasets)
    n_sets = len(datasets)
    dataset_union = set.union(*datasets)
    petal_labels = {}
    for logic in generate_logics(n_sets):
        included_sets = [
            datasets[i] for i in range(n_sets) if logic[i] == "1"
        ]
        excluded_sets = [
            datasets[i] for i in range(n_sets) if logic[i] == "0"
        ]
        petal_set = (
            (dataset_union & set.intersection(*included_sets)) -
            set.union(set(), *excluded_sets)
        )
        # 실제 값들을 줄바꿈으로 구분하여 표시
        if len(petal_set) > 0:
            values_text = "\n".join(sorted(petal_set))
            petal_labels[logic] = values_text
        else:
            petal_labels[logic] = ""
    return petal_labels

# Venn 다이어그램 그리기
figsize = (16, 16)
fontsize = 15
ax = init_axes(None, figsize)
n_sets = len(sets)
colors = generate_colors(n_colors=n_sets, cmap="viridis", alpha=0.4)

# 도형 그리기
shape_params = zip(
    SHAPE_COORDS[n_sets], SHAPE_DIMS[n_sets], SHAPE_ANGLES[n_sets], colors
)
for coords, dims, angle, color in shape_params:
    draw_ellipse(ax, *coords, *dims, angle, color)

# 레이블 그리기
petal_labels = generate_petal_labels_with_values(sets.values())

for logic, petal_label in petal_labels.items():
    if logic in PETAL_LABEL_COORDS[n_sets]:
        x, y = PETAL_LABEL_COORDS[n_sets][logic]
        if petal_label:  # 빈 레이블은 표시하지 않음
            draw_text(ax, x, y, petal_label, fontsize=fontsize)

# 범례 추가
ax.legend(sets.keys(), loc="upper center", prop={"size": fontsize}, bbox_to_anchor=(0.5, 0.9), ncol=4, frameon=False)

plt.savefig("venn.png", bbox_inches="tight", dpi=300)
plt.close()
