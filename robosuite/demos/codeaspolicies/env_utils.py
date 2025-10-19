"""Environment utilities for Code as Policies."""

import numpy as np
from shapely.geometry import *
from shapely.affinity import *

def get_obj_pos(obj_name):
    """Get object position in robot base frame."""
    obj_name = obj_name.replace('the', '').replace('_', ' ').strip()
    if obj_name in CORNER_POS:
        position = np.float32(np.array(CORNER_POS[obj_name]))
    else:
        pick_id = get_obj_id(obj_name)
        pose = pybullet.getBasePositionAndOrientation(pick_id)
        position = np.float32(pose[0])
    return position

def get_obj_names():
    """Get list of all object names in the environment."""
    return list(obj_name_to_id.keys())

def put_first_on_second(first, second):
    """Put first object on top of second object/position."""
    pick_pos = get_obj_pos(first) if isinstance(first, str) else first
    place_pos = get_obj_pos(second) if isinstance(second, str) else second
    env.step(action={'pick': pick_pos, 'place': place_pos})

def say(message):
    """Print robot's verbal response."""
    print(f"Robot says: {message}")

def get_corner_name(pos):
    """Get name of corner position."""
    corner_positions = get_corner_positions()
    corner_idx = np.argmin(np.linalg.norm(corner_positions - pos, axis=1))
    return ['top left corner', 'top right corner', 'bottom left corner', 'bottom right corner'][corner_idx]

def get_side_name(pos):
    """Get name of side position."""
    side_positions = get_side_positions()
    side_idx = np.argmin(np.linalg.norm(side_positions - pos, axis=1))
    return ['top side', 'right side', 'bottom side', 'left side'][side_idx]

def is_obj_visible(obj_name):
    """Check if object is visible in the scene."""
    return obj_name in get_obj_names()

def stack_objects_in_order(object_names):
    """Stack objects in specified order from bottom to top."""
    for i in range(len(object_names)-1):
        bottom_obj = object_names[i]
        top_obj = object_names[i+1]
        put_first_on_second(top_obj, bottom_obj)

def parse_obj_name(query, context):
    """Parse object name from natural language query."""
    # Get object positions
    block_names = [name for name in get_obj_names() if 'block' in name]
    block_positions = get_obj_positions_np(block_names)
    
    # Handle different query types
    if 'closest' in query:
        target_obj = query.split('closest to')[-1].strip()
        target_pos = get_obj_pos(target_obj)
        closest_idx = get_closest_idx(points=block_positions, point=target_pos)
        return block_names[closest_idx]
    
    elif 'left most' in query:
        left_idx = np.argmin(block_positions[:, 0])
        return block_names[left_idx]
    
    elif 'right most' in query:
        right_idx = np.argmax(block_positions[:, 0])
        return block_names[right_idx]
    
    elif 'blocks' in query:
        return block_names
    
    elif 'bowls' in query:
        return [name for name in get_obj_names() if 'bowl' in name]
    
    else:
        # Return exact match if exists
        for name in get_obj_names():
            if name in query:
                return name
    return None

def parse_position(query, context=""):
    """Parse position from natural language query."""
    if 'line' in query:
        # Parse line parameters
        if 'horizontal' in query:
            start_pos = denormalize_xy([0.25, 0.5])
            end_pos = denormalize_xy([0.75, 0.5])
        else:  # vertical
            start_pos = denormalize_xy([0.5, 0.25])
            end_pos = denormalize_xy([0.5, 0.75])
            
        line = make_line(start=start_pos, end=end_pos)
        n_points = int(query.split('with')[-1].split('points')[0].strip())
        return interpolate_pts_on_line(line=line, n=n_points)
        
    elif 'triangle' in query:
        size = float(query.split('size')[-1].split('cm')[0].strip()) / 100.0
        center = denormalize_xy([0.5, 0.5])
        polygon = make_triangle(size=size, center=center)
        return get_points_from_polygon(polygon)
        
    elif 'corner' in query:
        corners = get_corner_positions()
        if 'closest' in query:
            obj_name = query.split('closest to')[-1].strip()
            obj_pos = get_obj_pos(obj_name)
            return corners[get_closest_idx(corners, obj_pos)]
        elif 'top' in query:
            return corners[0] if 'left' in query else corners[1]
        else:  # bottom
            return corners[2] if 'left' in query else corners[3]
            
    elif 'side' in query:
        sides = get_side_positions()
        if 'closest' in query:
            obj_name = query.split('closest to')[-1].strip()
            obj_pos = get_obj_pos(obj_name)
            return sides[get_closest_idx(sides, obj_pos)]
        elif 'top' in query:
            return sides[0]
        elif 'right' in query:
            return sides[1]
        elif 'bottom' in query:
            return sides[2]
        else:  # left
            return sides[3]
            
    elif 'point' in query:
        if 'above' in query or 'top' in query:
            obj_name = parse_obj_name(query, context)
            return get_obj_pos(obj_name) + [0, 0.1]
        elif 'below' in query or 'bottom' in query:
            obj_name = parse_obj_name(query, context)
            return get_obj_pos(obj_name) + [0, -0.1]
        elif 'left' in query:
            obj_name = parse_obj_name(query, context)
            return get_obj_pos(obj_name) + [-0.1, 0]
        elif 'right' in query:
            obj_name = parse_obj_name(query, context)
            return get_obj_pos(obj_name) + [0.1, 0]
    
    return None

def parse_question(query, context):
    """Parse and answer questions about the environment state."""
    if 'is' in query:
        if 'right of' in query:
            obj1 = parse_obj_name(query.split('is')[1].split('right of')[0], context)
            obj2 = parse_obj_name(query.split('right of')[1], context)
            return get_obj_pos(obj1)[0] > get_obj_pos(obj2)[0]
            
        elif 'left of' in query:
            obj1 = parse_obj_name(query.split('is')[1].split('left of')[0], context)
            obj2 = parse_obj_name(query.split('left of')[1], context)
            return get_obj_pos(obj1)[0] < get_obj_pos(obj2)[0]
            
        elif 'above' in query or 'on top of' in query:
            obj1 = parse_obj_name(query.split('is')[1].split('above')[0].split('on top of')[0], context)
            obj2 = parse_obj_name(query.split('above')[-1].split('on top of')[-1], context)
            return get_obj_pos(obj1)[1] > get_obj_pos(obj2)[1]
            
    elif 'how many' in query:
        if 'yellow' in query:
            return len([name for name in get_obj_names() if 'yellow' in name])
        elif 'blocks' in query:
            return len([name for name in get_obj_names() if 'block' in name])
        elif 'bowls' in query:
            return len([name for name in get_obj_names() if 'bowl' in name])
            
    elif 'what' in query:
        if 'left of' in query:
            target = parse_obj_name(query.split('left of')[1], context)
            target_pos = get_obj_pos(target)
            return [name for name in get_obj_names() if get_obj_pos(name)[0] < target_pos[0]]
            
        elif 'right of' in query:
            target = parse_obj_name(query.split('right of')[1], context)
            target_pos = get_obj_pos(target)
            return [name for name in get_obj_names() if get_obj_pos(name)[0] > target_pos[0]]
    
    return None

def transform_shape_pts(query, shape_pts):
    """Transform shape points based on natural language query."""
    if 'bigger' in query or 'larger' in query:
        scale = float(query.split('by')[-1].strip())
        return scale_pts_around_centroid_np(shape_pts, scale_x=scale, scale_y=scale)
        
    elif 'smaller' in query:
        scale = float(query.split('by')[-1].strip()) if 'by' in query else 0.5
        return scale_pts_around_centroid_np(shape_pts, scale_x=scale, scale_y=scale)
        
    elif 'rotate' in query:
        angle = float(query.split('by')[-1].split('degrees')[0].strip())
        if 'clockwise' in query:
            angle = -angle
        return rotate_pts_around_centroid_np(shape_pts, angle=np.deg2rad(angle))
        
    elif 'move' in query or 'translate' in query:
        if 'toward' in query:
            target = parse_obj_name(query.split('toward')[-1], "")
            target_pos = get_obj_pos(target)
            mean_delta = np.mean(target_pos - shape_pts, axis=1)
            return translate_pts_np(shape_pts, mean_delta)
        elif 'right' in query:
            dist = float(query.split('by')[-1].split('cm')[0].strip()) / 100.0
            return translate_pts_np(shape_pts, [dist, 0])
        elif 'left' in query:
            dist = float(query.split('by')[-1].split('cm')[0].strip()) / 100.0
            return translate_pts_np(shape_pts, [-dist, 0])
        elif 'up' in query or 'top' in query:
            dist = float(query.split('by')[-1].split('cm')[0].strip()) / 100.0
            return translate_pts_np(shape_pts, [0, dist])
        elif 'down' in query or 'bottom' in query:
            dist = float(query.split('by')[-1].split('cm')[0].strip()) / 100.0
            return translate_pts_np(shape_pts, [0, -dist])
    
    return shape_pts

def get_obj_positions_np(obj_names):
    """Get numpy array of object positions."""
    return np.array([get_obj_pos(name) for name in obj_names])

def bbox_contains_pt(container_name, obj_name):
    """Check if object's bounding box contains a point."""
    container_bbox = get_bbox(container_name)
    obj_pos = get_obj_pos(obj_name)
    return (obj_pos[0] >= container_bbox[0] and 
            obj_pos[0] <= container_bbox[2] and
            obj_pos[1] >= container_bbox[1] and 
            obj_pos[1] <= container_bbox[3])