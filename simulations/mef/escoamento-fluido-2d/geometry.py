import numpy as np

def get_cylinder(params):
    cx, cy, r = params["cx"], params["cy"], params["r"]
    n_pts = 100
    theta = np.linspace(0, 2*np.pi, n_pts, endpoint=False)
    x_bound = cx + r * np.cos(theta)
    y_bound = cy + r * np.sin(theta)
    boundary_pts = np.column_stack([x_bound, y_bound])
    
    def mask_func(x, y):
        return (x - cx)**2 + (y - cy)**2 > (r * 1.01)**2

    return boundary_pts, mask_func

def get_rectangle(params):
    cx, cy = params["cx"], params["cy"]
    w, h = params["w"], params["h"]
    
    perim = 2 * (w + h)
    n_w = int(100 * (w / perim))
    n_h = int(100 * (h / perim))
    n_w = max(5, n_w)
    n_h = max(5, n_h)
    
    hw = w / 2
    hh = h / 2
    
    x_b = np.linspace(cx - hw, cx + hw, n_w, endpoint=False)
    y_b = np.full_like(x_b, cy - hh)
    
    y_r = np.linspace(cy - hh, cy + hh, n_h, endpoint=False)
    x_r = np.full_like(y_r, cx + hw)
    
    x_t = np.linspace(cx + hw, cx - hw, n_w, endpoint=False)
    y_t = np.full_like(x_t, cy + hh)
    
    y_l = np.linspace(cy + hh, cy - hh, n_h, endpoint=False)
    x_l = np.full_like(y_l, cx - hw)
    
    x_bound = np.concatenate([x_b, x_r, x_t, x_l])
    y_bound = np.concatenate([y_b, y_r, y_t, y_l])
    boundary_pts = np.column_stack([x_bound, y_bound])
    
    def mask_func(x, y):
        buffer = 1.01
        return (np.abs(x - cx) > hw * buffer) | (np.abs(y - cy) > hh * buffer)
        
    return boundary_pts, mask_func

def get_step(params):
    # Backward facing step
    # Step is at bottom left corner.
    # Dimensions: step_h, step_l
    step_h = params["step_h"]
    step_l = params["step_l"]
    
    # Boundary points for the step (L-shape)
    # 1. Top of step (0, step_h) -> (step_l, step_h)
    # 2. Back of step (step_l, step_h) -> (step_l, 0)
    
    n_pts = 50
    x_top = np.linspace(0, step_l, n_pts)
    y_top = np.full_like(x_top, step_h)
    
    y_back = np.linspace(step_h, 0, n_pts)
    x_back = np.full_like(y_back, step_l)
    
    # Concatenate
    x_bound = np.concatenate([x_top, x_back])
    y_bound = np.concatenate([y_top, y_back])
    boundary_pts = np.column_stack([x_bound, y_bound])
    
    def mask_func(x, y):
        # Remove if x < step_l AND y < step_h
        # Keep if x >= step_l OR y >= step_h
        # Add small buffer to avoid removing boundary nodes
        # We want to remove the block (0,0) to (step_l, step_h)
        # So keep if NOT (x < step_l-eps AND y < step_h-eps)
        eps = 1e-3
        return ~((x < step_l - eps) & (y < step_h - eps))
        
    return boundary_pts, mask_func



def get_none(params):
    # Empty (for Cavity or pure channel)
    return np.empty((0, 2)), lambda x, y: np.ones_like(x, dtype=bool)

def get_geometry(geo_type, params):
    if geo_type == "cylinder":
        return get_cylinder(params)
    elif geo_type == "rectangle":
        return get_rectangle(params)
    elif geo_type == "step":
        return get_step(params)
    else:
        return get_none(params)
