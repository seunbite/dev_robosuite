"""
VLM Evaluation Experiment Script

This script evaluates Vision-Language Models (VLMs) on robot motion understanding tasks.

Pipeline:
1. Generate tiled images with meta_motion_generation.py (saves initial_pose_info.yml)
2. Manually annotate best_manual_index in the YAML
3. Run this script to evaluate using only the best variations

Evaluation types:
    - binary:      Yes/No — "Does this motion demonstrate X?"
    - MCQ:         Multiple choice (A/B/C/D) — "Which action best describes this motion?"
    - open_ended:  Free-form description — VLM describes what it sees and what action is being performed

Usage:
    python main_exp.py --vlm gemini --input_data IIWA --eval_type binary
    python main_exp.py --vlm gemini --input_data Humanoid --eval_type MCQ --num_samples 10
    python main_exp.py --vlm gemini --input_data GR1 --eval_type open_ended --log
"""

import os
import fire
import random
import yaml
import json
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from datetime import datetime
from google import genai
from google.genai import types
from PIL import Image
import base64
import io

# Configuration
CUES_PATH = "data/seed/cues.yml"
MOTIONS_BASE_PATH = "data/motions"
POSE_INFO_PATH = "data/motions/initial_pose_info.yml"


class VLMEvaluator:
    """Evaluator for VLM motion understanding."""
    
    def __init__(self, vlm_name: str, input_data: str, eval_type: str, 
                 random_if_no_index: bool = False, log: bool = False,
                 jsonl_log_path: Optional[str] = None):
        """
        Initialize evaluator.
        
        Args:
            vlm_name: VLM name ('gemini', etc.)
            input_data: Robot type ('IIWA', 'Panda', 'Humanoid', 'GR1')
            eval_type: Evaluation type ('binary', 'MCQ', or 'open_ended')
            random_if_no_index: If True, include cues without best_manual_index
                               and randomly pick from initial_indexes
            log: If True, print full prompts and VLM outputs
            jsonl_log_path: Path to JSONL log file (appends one line per sample)
        """
        self.vlm_name = vlm_name
        self.input_data = input_data
        self.eval_type = eval_type
        self.random_if_no_index = random_if_no_index
        self.log = log
        self.jsonl_log_path = jsonl_log_path
        
        # Load cues
        self.cues = self._load_cues()
        
        # Load pose info (for extracting best variations from tiled GIFs)
        self.pose_info = self._load_pose_info()
        
        # Initialize VLM
        self._init_vlm()
        
        # Get motion files (with best variation extraction)
        self.motion_files = self._get_motion_files()
        
        print(f"Initialized VLM Evaluator:")
        print(f"  VLM: {vlm_name}")
        print(f"  Robot: {input_data}")
        print(f"  Eval Type: {eval_type}")
        print(f"  random_if_no_index: {random_if_no_index}")
        print(f"  log: {log}")
        print(f"  Total cues: {len(self.cues)}")
        print(f"  Motion files: {len(self.motion_files)}")
        print(f"  Cues with best_manual_index: {self._count_annotated_cues()}")
    
    def _load_cues(self) -> List[str]:
        """Load all cues from cues.yml."""
        with open(CUES_PATH, 'r') as f:
            cues_data = yaml.safe_load(f)
        
        # Flatten all cues from all categories
        all_cues = []
        for category, cues in cues_data.items():
            if isinstance(cues, dict):
                all_cues.extend(cues.keys())
            elif isinstance(cues, list):
                all_cues.extend(cues)
        
        return list(set(all_cues))  # Remove duplicates
    
    def _load_pose_info(self) -> Dict:
        """Load initial pose info from YAML."""
        if not os.path.exists(POSE_INFO_PATH):
            print(f"Warning: Pose info file not found: {POSE_INFO_PATH}")
            return {}
        
        with open(POSE_INFO_PATH, 'r') as f:
            return yaml.safe_load(f) or {}
    
    def _get_robot_keys(self) -> List[str]:
        """Get pose_info YAML keys for the current robot type."""
        if self.input_data in ['Humanoid', 'GR1']:
            return ['GR1_right', 'GR1_left', 'GR1']
        return [self.input_data]
    
    def _get_robot_pose_info(self) -> Tuple[Dict, str]:
        """Get pose info dict and the matching robot key."""
        for key in self._get_robot_keys():
            if key in self.pose_info and self.pose_info[key]:
                return self.pose_info[key], key
        return {}, self._get_robot_keys()[0]
    
    def _count_annotated_cues(self) -> int:
        """Count cues that have best_manual_index annotated."""
        robot_pose_info, _ = self._get_robot_pose_info()
        return sum(1 for cue_data in robot_pose_info.values()
                   if isinstance(cue_data, dict) and cue_data.get('best_manual_index') is not None)
    
    def _init_vlm(self):
        """Initialize VLM API."""
        if self.vlm_name.lower().startswith('gemini'):
            # Get API key from environment
            api_key = os.getenv('GOOGLE_API_KEY') or os.getenv('GEMINI_API_KEY')
            if not api_key:
                raise ValueError("GOOGLE_API_KEY or GEMINI_API_KEY environment variable not set")
            
            # Initialize new google.genai client
            self.client = genai.Client(api_key=api_key)
            print(f"✓ Initialized Gemini model: {self.vlm_name}")
        else:
            raise ValueError(f"Unsupported VLM: {self.vlm_name}")
    
    def _append_jsonl_log(self, entry: Dict):
        """Append a single log entry to the JSONL log file."""
        if not self.jsonl_log_path:
            return
        os.makedirs(os.path.dirname(self.jsonl_log_path) or '.', exist_ok=True)
        with open(self.jsonl_log_path, 'a', encoding='utf-8') as f:
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')
    
    def _extract_best_variation_from_tiled(self, tiled_gif_path: str, variation_index: int,
                                             num_variations: int = 20, max_per_row: int = 10) -> Optional[str]:
        """
        Extract a specific variation from a tiled GIF (supports multi-row grid).
        
        Args:
            tiled_gif_path: Path to tiled GIF
            variation_index: 0-based index of the variation to extract
            num_variations: Total number of variations in the tiled GIF
            max_per_row: Max tiles per row (must match how the tiled GIF was created)
        
        Returns:
            Path to extracted single-variation GIF, or None if failed
        """
        import math
        try:
            gif = Image.open(tiled_gif_path)
            
            frames = []
            try:
                while True:
                    frames.append(gif.copy())
                    gif.seek(gif.tell() + 1)
            except EOFError:
                pass
            
            if not frames:
                return None
            
            total_width = frames[0].width
            total_height = frames[0].height
            
            # Compute grid layout (must match combine_variations_horizontally)
            cols = min(num_variations, max_per_row)
            rows = math.ceil(num_variations / max_per_row)
            tile_w = total_width // cols
            tile_h = total_height // rows
            
            # Locate the tile in the grid
            row = variation_index // max_per_row
            col = variation_index % max_per_row
            
            extracted_frames = []
            for frame in frames:
                left = col * tile_w
                top = row * tile_h
                cropped = frame.crop((left, top, left + tile_w, top + tile_h))
                extracted_frames.append(cropped)
            
            output_path = tiled_gif_path.replace('_tiled.gif', f'_v{variation_index}.gif')
            extracted_frames[0].save(
                output_path,
                save_all=True,
                append_images=extracted_frames[1:],
                duration=gif.info.get('duration', 100),
                loop=0,
                disposal=2,
                optimize=False
            )
            
            return output_path
        
        except Exception as e:
            print(f"Warning: Could not extract variation {variation_index} from {tiled_gif_path}: {e}")
            return None
    
    def _parse_cue_from_filename(self, filename_stem: str) -> Optional[str]:
        """
        Parse cue name from a tiled GIF filename.
        
        Supported formats:
            - Humanoid: YYYYMMDD_IDX_GR1_right_cuename_tiled
            - Manipulator: YYYYMMDD_pIDX_ROBOT_cuename_tiled
        """
        parts = filename_stem.split('_')
        try:
            if 'right' in parts or 'left' in parts:
                arm_idx = parts.index('right') if 'right' in parts else parts.index('left')
                cue_parts = parts[arm_idx + 1:-1]  # Between arm and 'tiled'
            else:
                # Find robot name position (skip date and index prefix)
                # e.g. ['20260212', 'p0', 'IIWA', 'beckoning', 'tiled']
                robot_idx = next(i for i, p in enumerate(parts) 
                                 if p in ['IIWA', 'Panda', 'XArm7', 'GR1'])
                cue_parts = parts[robot_idx + 1:-1]
            return '_'.join(cue_parts) if cue_parts else None
        except (ValueError, StopIteration):
            return None
    
    def _get_motion_files(self) -> List[Dict]:
        """
        Get motion GIF files for the robot.
        
        Uses best_manual_index (1-based) from initial_pose_info.yml to select the
        best variation from each tiled GIF. If random_if_no_index is True, cues
        without a best_manual_index are included with a random variation selected.
        """
        if self.input_data in ['Humanoid', 'GR1']:
            motion_dir = Path(MOTIONS_BASE_PATH) / "GR1"
        else:
            motion_dir = Path(MOTIONS_BASE_PATH) / self.input_data
        
        if not motion_dir.exists():
            print(f"Warning: Motion directory not found: {motion_dir}")
            return []
        
        robot_pose_info, robot_key = self._get_robot_pose_info()
        
        motion_files = []
        skipped_count = 0
        
        for gif_file in sorted(motion_dir.glob("*_tiled.gif")):
            cue_name = self._parse_cue_from_filename(gif_file.stem)
            if not cue_name:
                print(f"Warning: Could not parse cue from {gif_file.name}")
                skipped_count += 1
                continue
            
            # Look up in pose_info YAML
            cue_info = robot_pose_info.get(cue_name)
            if cue_info is None or not isinstance(cue_info, dict):
                print(f"Skipping {cue_name}: not in pose_info YAML ({robot_key})")
                skipped_count += 1
                continue
            
            initial_indexes = cue_info.get('initial_indexes', [])
            best_manual = cue_info.get('best_manual_index')  # 1-based
            num_variations = len(initial_indexes) if initial_indexes else 1
            
            if best_manual is not None:
                # best_manual_index is 1-based → convert to 0-based
                variation_idx = int(best_manual) - 1
                if variation_idx < 0 or variation_idx >= num_variations:
                    print(f"Warning: {cue_name} best_manual_index={best_manual} out of range (1-{num_variations})")
                    skipped_count += 1
                    continue
                pose_index = initial_indexes[variation_idx] if initial_indexes else None
            elif self.random_if_no_index:
                # No annotation — pick random variation
                variation_idx = random.randint(0, num_variations - 1)
                pose_index = initial_indexes[variation_idx] if initial_indexes else None
            else:
                skipped_count += 1
                continue
            
            # Extract the selected variation from the tiled GIF
            best_gif_path = self._extract_best_variation_from_tiled(
                str(gif_file), variation_idx, num_variations=num_variations
            )
            
            if best_gif_path:
                motion_files.append({
                    'file': best_gif_path,
                    'cue': cue_name,
                    'robot': self.input_data,
                    'variation_index': variation_idx,
                    'pose_index': pose_index,
                    'annotated': best_manual is not None,
                })
            else:
                print(f"Warning: Could not extract variation for {cue_name}")
                skipped_count += 1
        
        annotated = sum(1 for m in motion_files if m['annotated'])
        random_picked = sum(1 for m in motion_files if not m['annotated'])
        print(f"\nMotion files loaded: {len(motion_files)} total "
              f"({annotated} annotated, {random_picked} random)")
        if skipped_count > 0:
            print(f"Skipped: {skipped_count} cues")
        
        return motion_files
    
    def _count_gif_frames(self, gif_path: str) -> int:
        """Count the number of frames in a GIF file."""
        try:
            with Image.open(gif_path) as gif:
                n = 0
                while True:
                    n += 1
                    try:
                        gif.seek(gif.tell() + 1)
                    except EOFError:
                        break
                return n
        except Exception:
            return -1

    def _query_vlm_binary(self, gif_path: str, cue: str) -> bool:
        """
        Query VLM with binary question.
        
        Args:
            gif_path: Path to GIF file
            cue: Cue name to ask about
        
        Returns:
            True if VLM says yes, False otherwise
        """
        with open(gif_path, 'rb') as f:
            gif_bytes = f.read()
        
        prompt = f"""Look at this robot motion animation.

Does this motion demonstrate the action: "{cue.replace('_', ' ')}"?

Answer with only "Yes" or "No"."""
        
        if self.log:
            print(f"\n  [PROMPT]\n{prompt}")
        
        response = self.client.models.generate_content(
            model=self.vlm_name,
            contents=[
                types.Part.from_bytes(data=gif_bytes, mime_type='image/gif'),
                prompt
            ]
        )
        answer = response.text.strip()
        
        if self.log:
            print(f"  [OUTPUT]\n{answer}")
        
        return 'yes' in answer.lower()
    
    def _query_vlm_open_ended(self, gif_path: str, cue: str) -> str:
        """
        Query VLM with open-ended question (주관식).
        
        Instead of multiple-choice or binary, ask the VLM to freely describe
        what it sees and what action the robot is performing.
        
        Args:
            gif_path: Path to GIF file
            cue: Ground-truth cue name (for logging/comparison only)
        
        Returns:
            VLM's free-form description string
        """
        with open(gif_path, 'rb') as f:
            gif_bytes = f.read()
        
        prompt = """Look at this robot motion animation carefully.
This robot movement is not functional, rather expressive and communicative to deliever its mind state.

1. Describe in detail what action or gesture the robot is performing. Focus on the movement of its arms, hands, and body.
2. If this motion were a human gesture or action, what would it most likely represent? What intent or meaning does it convey?

Please provide a detailed and specific description for each point above."""
        
        if self.log:
            print(f"\n  [PROMPT]\n{prompt}")
        
        response = self.client.models.generate_content(
            model=self.vlm_name,
            contents=[
                types.Part.from_bytes(data=gif_bytes, mime_type='image/gif'),
                prompt
            ],
            config=types.GenerateContentConfig(
                max_output_tokens=4096,
            ),
        )
        answer = response.text.strip()
        
        if self.log:
            print(f"  [OUTPUT]\n{answer}")
        
        return answer
    
    def _query_vlm_mcq(self, gif_path: str, correct_cue: str, distractors: List[str]) -> Tuple[int, List[str]]:
        """
        Query VLM with multiple choice question.
        
        Args:
            gif_path: Path to GIF file
            correct_cue: Correct cue name
            distractors: List of 3 distractor cues
        
        Returns:
            (VLM choice index 0-3, options list)
        """
        options = distractors + [correct_cue]
        random.shuffle(options)
        
        with open(gif_path, 'rb') as f:
            gif_bytes = f.read()
        
        prompt = f"""Look at this robot motion animation.

Which of the following actions best describes this motion?

A) {options[0].replace('_', ' ')}
B) {options[1].replace('_', ' ')}
C) {options[2].replace('_', ' ')}
D) {options[3].replace('_', ' ')}

Answer with only the letter (A, B, C, or D)."""
        
        if self.log:
            print(f"\n  [PROMPT]\n{prompt}")
        
        response = self.client.models.generate_content(
            model=self.vlm_name,
            contents=[
                types.Part.from_bytes(data=gif_bytes, mime_type='image/gif'),
                prompt
            ]
        )
        answer = response.text.strip()
        
        if self.log:
            print(f"  [OUTPUT]\n{answer}")
        
        # Parse answer (extract first letter)
        for char in answer.upper():
            if char in 'ABCD':
                return ord(char) - ord('A'), options
        
        # Default to random if parsing fails
        return random.randint(0, 3), options
    
    def run_evaluation(self, num_samples: int = None) -> Dict:
        """
        Run evaluation on all motion files.
        
        Args:
            num_samples: Number of samples to evaluate (None = all)
        
        Returns:
            Dict with accuracy, results list, and metadata
        """
        if not self.motion_files:
            print("No motion files found!")
            return {"accuracy": 0.0, "results": [], "total": 0}
        
        samples = self.motion_files if num_samples is None else random.sample(
            self.motion_files, min(num_samples, len(self.motion_files))
        )
        
        correct = 0
        total = len(samples)
        results = []
        
        print(f"\nRunning evaluation on {total} samples...")
        print("=" * 60)
        
        for i, motion in enumerate(samples, 1):
            gif_path = motion['file']
            true_cue = motion['cue']
            annotated = motion.get('annotated', False)
            
            tag = "annotated" if annotated else "random"
            num_frames = self._count_gif_frames(gif_path)
            print(f"\n[{i}/{total}] {Path(gif_path).name} [{tag}] ({num_frames} frames)")
            print(f"  True cue: {true_cue}")
            
            try:
                if self.eval_type == 'binary':
                    prediction = self._query_vlm_binary(gif_path, true_cue)
                    is_correct = prediction is True
                    
                    print(f"  VLM answer: {'Yes' if prediction else 'No'}")
                    print(f"  Result: {'✓ Correct' if is_correct else '✗ Wrong'}")
                    
                    result_entry = {
                        "cue": true_cue, "correct": is_correct,
                        "prediction": "yes" if prediction else "no",
                        "annotated": annotated,
                    }
                    results.append(result_entry)
                    self._append_jsonl_log({
                        "timestamp": datetime.now().isoformat(),
                        "vlm": self.vlm_name,
                        "eval_type": self.eval_type,
                        "robot": self.input_data,
                        "cue": true_cue,
                        "gif": Path(gif_path).name,
                        "num_frames": num_frames,
                        "prediction": result_entry["prediction"],
                        "correct": is_correct,
                        "annotated": annotated,
                    })
                
                elif self.eval_type == 'MCQ':
                    distractors = random.sample(
                        [c for c in self.cues if c != true_cue],
                        min(3, len(self.cues) - 1)
                    )
                    
                    prediction_idx, options = self._query_vlm_mcq(gif_path, true_cue, distractors)
                    correct_idx = options.index(true_cue)
                    is_correct = prediction_idx == correct_idx
                    
                    print(f"  VLM choice: {chr(ord('A') + prediction_idx)} ({options[prediction_idx]})")
                    print(f"  Correct: {chr(ord('A') + correct_idx)} ({true_cue})")
                    print(f"  Result: {'✓ Correct' if is_correct else '✗ Wrong'}")
                    
                    result_entry = {
                        "cue": true_cue, "correct": is_correct,
                        "prediction": options[prediction_idx],
                        "options": options,
                        "annotated": annotated,
                    }
                    results.append(result_entry)
                    self._append_jsonl_log({
                        "timestamp": datetime.now().isoformat(),
                        "vlm": self.vlm_name,
                        "eval_type": self.eval_type,
                        "robot": self.input_data,
                        "cue": true_cue,
                        "gif": Path(gif_path).name,
                        "num_frames": num_frames,
                        "prediction": result_entry["prediction"],
                        "correct": is_correct,
                        "options": options,
                        "annotated": annotated,
                    })
                
                elif self.eval_type == 'open_ended':
                    description = self._query_vlm_open_ended(gif_path, true_cue)
                    
                    print(f"  VLM description:\n    {description[:200]}{'...' if len(description) > 200 else ''}")
                    
                    result_entry = {
                        "cue": true_cue, 
                        "description": description,
                        "annotated": annotated,
                    }
                    results.append(result_entry)
                    self._append_jsonl_log({
                        "timestamp": datetime.now().isoformat(),
                        "vlm": self.vlm_name,
                        "eval_type": self.eval_type,
                        "robot": self.input_data,
                        "cue": true_cue,
                        "gif": Path(gif_path).name,
                        "num_frames": num_frames,
                        "description": description,
                        "annotated": annotated,
                    })
                    # open_ended has no correct/wrong — skip accuracy tracking
                    is_correct = None
                
                else:
                    raise ValueError(f"Invalid eval_type: {self.eval_type}")
                
                if is_correct:
                    correct += 1
            
            except Exception as e:
                print(f"  Error: {e}")
                results.append({"cue": true_cue, "correct": False, "error": str(e)})
                self._append_jsonl_log({
                    "timestamp": datetime.now().isoformat(),
                    "vlm": self.vlm_name,
                    "eval_type": self.eval_type,
                    "robot": self.input_data,
                    "cue": true_cue,
                    "gif": Path(gif_path).name,
                    "num_frames": num_frames,
                    "error": str(e),
                })
                continue
        
        print("\n" + "=" * 60)
        print(f"EVALUATION COMPLETE")
        
        if self.eval_type == 'open_ended':
            print(f"  Total descriptions: {total}")
            print(f"  (open_ended mode — no accuracy metric)")
            print("=" * 60)
            return {"accuracy": None, "results": results, "total": total, "correct": None}
        else:
            accuracy = correct / total if total > 0 else 0.0
            print(f"  Correct: {correct}/{total}")
            print(f"  Accuracy: {accuracy:.2%}")
            print("=" * 60)
            return {"accuracy": accuracy, "results": results, "total": total, "correct": correct}


def main(
    vlm: str = 'gemini-2.5-flash',
    input_data: str = 'GR1',
    eval_type: str = 'open_ended', # binary, MCQ, open_ended
    num_samples: Optional[int] = None,
    random_if_no_index: bool = False,
    log: bool = False,
):
    """
    Run VLM evaluation on robot motions.
    
    Loads tiled GIFs from data/motions/{robot}/, selects the best variation
    per cue using best_manual_index (1-based) from initial_pose_info.yml,
    then queries the VLM for each motion.
    
    Args:
        vlm: VLM model name (e.g., 'gemini-2.5-flash', 'gemini-2.0-flash-exp')
        input_data: Robot type ('IIWA', 'Panda', 'XArm7', 'GR1', 'Humanoid')
        eval_type: Evaluation type ('binary', 'MCQ', or 'open_ended')
        num_samples: Number of samples to evaluate (None = all)
        random_if_no_index: If True, include cues without best_manual_index
                           and pick a random variation from initial_indexes
        log: If True, print full prompts and VLM outputs
    
    Examples:
        python main_exp.py --vlm gemini-2.5-flash --input_data GR1 --eval_type binary --log
        python main_exp.py --vlm gemini-2.5-flash --input_data IIWA --eval_type MCQ --num_samples 10
        python main_exp.py --vlm gemini-2.5-flash --input_data GR1 --eval_type open_ended --log
        python main_exp.py --input_data GR1 --random_if_no_index --log
    """
    valid_robots = ['IIWA', 'Panda', 'XArm7', 'GR1', 'Humanoid']
    if input_data not in valid_robots:
        raise ValueError(f"Invalid input_data: {input_data}. Choose from: {valid_robots}")
    
    if eval_type not in ['binary', 'MCQ', 'open_ended']:
        raise ValueError(f"Invalid eval_type: {eval_type}. Choose from: binary, MCQ, open_ended")
    
    # JSONL log file — appends one line per sample (survives interruptions)
    results_dir = "data/results"
    os.makedirs(results_dir, exist_ok=True)
    jsonl_log_path = os.path.join(results_dir, f"log_{vlm}_{input_data}_{eval_type}.jsonl")
    
    evaluator = VLMEvaluator(
        vlm_name=vlm,
        input_data=input_data,
        eval_type=eval_type,
        random_if_no_index=random_if_no_index,
        log=log,
        jsonl_log_path=jsonl_log_path,
    )
    
    eval_result = evaluator.run_evaluation(num_samples=num_samples)
    
    # Save summary results (JSON)
    output = {
        'vlm': vlm,
        'input_data': input_data,
        'eval_type': eval_type,
        'num_samples': eval_result['total'],
        'accuracy': eval_result['accuracy'],
        'correct': eval_result['correct'],
        'random_if_no_index': random_if_no_index,
        'results': eval_result['results'],
    }
    
    results_file = os.path.join(results_dir, f"results_{vlm}_{input_data}_{eval_type}.json")
    with open(results_file, 'w') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    
    print(f"\nResults saved to: {results_file}")
    print(f"JSONL log:        {jsonl_log_path}")
    
    if eval_type == 'open_ended':
        return eval_result['total']
    return eval_result['accuracy']


if __name__ == "__main__":
    fire.Fire(main)
