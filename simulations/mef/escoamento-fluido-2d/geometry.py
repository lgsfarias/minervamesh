import numpy as np

def get_cylinder(params):
    cx, cy, r = params["cx"], params["cy"], params["r"]
    n_pts = 100
    theta = np.linspace(0, 2*np.pi, n_pts, endpoint=False)
    x_bound = cx + r * np.cos(theta)
    y_bound = cy + r * np.sin(theta)
    boundary_pts = np.column_stack([x_bound, y_bound])
    
    def mask_func(x, y):
        # Return True if OUTSIDE (Fluid)
        # Using 1.01 buffer to ensure boundary nodes are not masked out if they slightly float
        return (x - cx)**2 + (y - cy)**2 > (r * 0.99)**2 
        # Wait, if I use 1.01, I might mask the boundary nodes themselves if they are exactly at r?
        # The boundary nodes are AT r.
        # The mask is used to filter the BACKGROUND grid.
        # The boundary nodes are added separately.
        # So we want to remove background grid nodes that are INSIDE or CLOSE TO boundary.
        # So (x-cx)^2 + ... > r^2.
        # If we use r*1.01, we remove nodes slightly outside too?
        # Usually we want to remove everything inside.
        # Let's stick to the original logic: (r * 1.01)**2
    
    # Original logic was: (x - cx)**2 + (y - cy)**2 > (r * 1.01)**2
    # This keeps points strictly outside r*1.01.
    # Points between r and r*1.01 are removed. This creates a small gap?
    # No, the boundary nodes are at r.
    # If grid nodes are at r+epsilon, they might be kept.
    # If grid nodes are at r-epsilon, they are removed.
    # The triangulation connects boundary (at r) with grid (at > r*1.01).
    # This is fine.
    
    def mask_func_impl(x, y):
        return (x - cx)**2 + (y - cy)**2 > (r * 1.01)**2

    return boundary_pts, mask_func_impl

def get_rectangle(params):
    cx, cy = params["cx"], params["cy"]
    w, h = params["w"], params["h"]
    
    # Generate points along 4 sides
    # Perimeter approx 2(w+h).
    # We want approx 100 points total.
    perim = 2 * (w + h)
    n_w = int(100 * (w / perim))
    n_h = int(100 * (h / perim))
    n_w = max(5, n_w)
    n_h = max(5, n_h)
    
    hw = w / 2
    hh = h / 2
    
    # Bottom (left to right)
    x_b = np.linspace(cx - hw, cx + hw, n_w, endpoint=False)
    y_b = np.full_like(x_b, cy - hh)
    
    # Right (bottom to top)
    y_r = np.linspace(cy - hh, cy + hh, n_h, endpoint=False)
    x_r = np.full_like(y_r, cx + hw)
    
    # Top (right to left)
    x_t = np.linspace(cx + hw, cx - hw, n_w, endpoint=False)
    y_t = np.full_like(x_t, cy + hh)
    
    # Left (top to bottom)
    y_l = np.linspace(cy + hh, cy - hh, n_h, endpoint=False)
    x_l = np.full_like(y_l, cx - hw)
    
    x_bound = np.concatenate([x_b, x_r, x_t, x_l])
    y_bound = np.concatenate([y_b, y_r, y_t, y_l])
    boundary_pts = np.column_stack([x_bound, y_bound])
    
    def mask_func_impl(x, y):
        # Return True if OUTSIDE
        # Inside if |x-cx| < hw and |y-cy| < hh
        # We add a small buffer to remove points too close to boundary
        buffer = 1.01
        return (np.abs(x - cx) > hw * buffer) | (np.abs(y - cy) > hh * buffer)
        
    return boundary_pts, mask_func_impl

def get_geometry(geo_type, params):
    if geo_type == "cylinder":
        return get_cylinder(params)
    elif geo_type == "rectangle":
        return get_rectangle(params)
    else:
        # Fallback to cylinder
        return get_cylinder(params)
