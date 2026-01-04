import threading
import sys
import numpy as np
from mujoco import viewer
import mujoco

glfw = None
try:
    import mujoco_viewer
    import glfw
    HAS_MUJOCO_VIEWER = True
except ImportError:
    HAS_MUJOCO_VIEWER = False

DEFAULT_FREE_CAM = {
    "lookat": [0, 0, 0.5],  # Look at origin
    "distance": 3,  # Increase distance to see more
    "azimuth": 180,
    "elevation": -30,  # Lower angle to see better
}


class MjviewerRenderer:
    def __init__(self, env, camera_id=None, cam_config=None):
        if cam_config is None:
            cam_config = DEFAULT_FREE_CAM
        self.env = env
        self.camera_id = camera_id
        self.viewer = None
        self.camera_config = cam_config
        self._viewer_initialized = False
        self._viewer_lock = threading.Lock()
        self._use_interactive = HAS_MUJOCO_VIEWER

    def render(self):
        pass

    def set_camera(self, camera_id):
        self.camera_id = camera_id

    def _init_viewer_interactive(self):
        """Initialize interactive viewer using mujoco_viewer"""
        try:
            # Check if we're on main thread (macOS requirement)
            is_main_thread = threading.current_thread() is threading.main_thread()
            
            if not is_main_thread and sys.platform == 'darwin':
                # On macOS, try to initialize on main thread
                # For now, we'll try anyway and catch the error
                pass
            
            self.viewer = mujoco_viewer.MujocoViewer(
                self.env.sim.model._model,
                self.env.sim.data._data,
                title="Robosuite Interactive Viewer",
                width=1200,
                height=800,
                hide_menus=False  # Show menus for interactive features
            )
            
            # Patch _add_marker_to_scene to handle MuJoCo 3.x compatibility (texid removed)
            original_add_marker = self.viewer._add_marker_to_scene
            def safe_add_marker(marker):
                """Wrapper that skips texid for MuJoCo 3.x compatibility"""
                if self.viewer.scn.ngeom >= self.viewer.scn.maxgeom:
                    print(f"DEBUG: Scene full, cannot add marker. ngeom={self.viewer.scn.ngeom}, maxgeom={self.viewer.scn.maxgeom}")
                    return
                g = self.viewer.scn.geoms[self.viewer.scn.ngeom]
                # Set default values (based on mujoco_custom_viewer.py)
                g.dataid = -1
                g.objtype = mujoco.mjtObj.mjOBJ_UNKNOWN
                g.objid = -1
                g.category = mujoco.mjtCatBit.mjCAT_DECOR
                # Skip texid (removed in MuJoCo 3.x)
                if hasattr(g, 'texuniform'):
                    g.texuniform = 0
                if hasattr(g, 'texrepeat'):
                    g.texrepeat[0] = 1
                    g.texrepeat[1] = 1
                g.emission = 0
                g.specular = 0.5
                g.shininess = 0.5
                g.reflectance = 0
                g.type = mujoco.mjtGeom.mjGEOM_BOX
                g.size[:] = np.ones(3) * 0.1
                g.mat[:] = np.eye(3)
                g.rgba[:] = np.ones(4)
                
                # Apply marker parameters
                for key, value in marker.items():
                    if key == 'texid':  # Skip texid
                        continue
                    if isinstance(value, (int, float, mujoco._enums.mjtGeom)):
                        setattr(g, key, value)
                    elif isinstance(value, (tuple, list, np.ndarray)):
                        # Check if attribute exists before setting
                        if hasattr(g, key):
                            attr = getattr(g, key)
                            attr[:] = np.asarray(value).reshape(attr.shape)
                        else:
                            print(f"WARNING: MjvGeom does not have '{key}' attribute! Marker may not be positioned correctly.")
                            # If pos doesn't exist, try to set it via mat matrix
                            if key == 'pos':
                                # Create transformation matrix with translation
                                pos = np.asarray(value)
                                g.mat[:3, 3] = pos  # Set translation part of matrix
                                print(f"Set position via mat matrix: {g.mat[:3, 3]}")
                    elif isinstance(value, str):
                        if key == "label":
                            if value is None:
                                g.label[0] = 0
                            else:
                                g.label = value
                    elif hasattr(g, key):
                        print(f"WARNING: Unknown type for {key}: {type(value)}")
                
                # Debug: print final marker state
                print(f"Marker added: type={g.type}, size={g.size}, rgba={g.rgba}")
                if hasattr(g, 'pos'):
                    print(f"  pos={g.pos}")
                else:
                    print(f"  mat translation={g.mat[:3, 3]}")
                
                self.viewer.scn.ngeom += 1
            
            # Replace _add_marker_to_scene with safe version
            self.viewer._add_marker_to_scene = safe_add_marker
            
            # Set up viewer attributes
            # mujoco_viewer uses vopt instead of opt
            if hasattr(self.viewer, 'vopt'):
                self.viewer.vopt.geomgroup[0] = 0
            elif hasattr(self.viewer, 'opt'):
                self.viewer.opt.geomgroup[0] = 0

            if self.camera_config is not None:
                self.viewer.cam.lookat = self.camera_config["lookat"]
                self.viewer.cam.distance = self.camera_config["distance"]
                self.viewer.cam.azimuth = self.camera_config["azimuth"]
                self.viewer.cam.elevation = self.camera_config["elevation"]

            if self.camera_id is not None:
                if self.camera_id >= 0:
                    self.viewer.cam.type = 2
                    self.viewer.cam.fixedcamid = self.camera_id
                else:
                    self.viewer.cam.type = 0
            
            return True
        except Exception as e:
            print(f"Warning: Failed to initialize interactive viewer: {e}")
            print("Falling back to passive viewer")
            return False

    def _init_viewer_passive(self):
        """Initialize passive viewer as fallback"""
        self.viewer = viewer.launch_passive(
            self.env.sim.model._model,
            self.env.sim.data._data,
            show_left_ui=False,
            show_right_ui=False,
        )
        self.viewer.opt.geomgroup[0] = 0

        if self.camera_config is not None:
            self.viewer.cam.lookat = self.camera_config["lookat"]
            self.viewer.cam.distance = self.camera_config["distance"]
            self.viewer.cam.azimuth = self.camera_config["azimuth"]
            self.viewer.cam.elevation = self.camera_config["elevation"]

        if self.camera_id is not None:
            if self.camera_id >= 0:
                self.viewer.cam.type = 2
                self.viewer.cam.fixedcamid = self.camera_id
            else:
                self.viewer.cam.type = 0

    def update(self):
        with self._viewer_lock:
            if self.viewer is None and not self._viewer_initialized:
                self._viewer_initialized = True
                
                if self._use_interactive:
                    success = self._init_viewer_interactive()
                    if not success:
                        self._use_interactive = False
                        self._init_viewer_passive()
                else:
                    self._init_viewer_passive()

        if self.viewer is not None:
            if self._use_interactive and hasattr(self.viewer, 'render'):
                # Interactive viewer: use _markers list (patched to handle texid)
                # Get controller and pending markers
                controller = getattr(self.env, '_ik_controller', None)
                has_markers = controller and hasattr(controller, '_pending_markers') and controller._pending_markers
                
                # Add markers to viewer's _markers list (will be added during render)
                if has_markers and hasattr(self.viewer, '_markers'):
                    # Clear previous target markers
                    self.viewer._markers[:] = [
                        m for m in self.viewer._markers 
                        if not (isinstance(m, dict) and m.get('label', '').startswith('target_cell_'))
                    ]
                    # Add new markers
                    for marker in controller._pending_markers:
                        marker_with_label = marker.copy()
                        marker_with_label['label'] = f'target_cell_{controller.target_cell_id}'
                        self.viewer._markers.append(marker_with_label)
                
                # Call render (markers will be added automatically via patched _add_marker_to_scene)
                self.viewer.render()
            elif hasattr(self.viewer, 'sync'):
                # Passive viewer uses sync()
                self.viewer.sync()

    def reset(self):
        pass

    def close(self):
        self.sim = None
        if self.viewer is not None:
            try:
                if hasattr(self.viewer, 'close'):
                    self.viewer.close()
            except Exception:
                pass
            self.viewer = None

    def add_keypress_callback(self, keypress_callback):
        self.keypress_callback = keypress_callback
