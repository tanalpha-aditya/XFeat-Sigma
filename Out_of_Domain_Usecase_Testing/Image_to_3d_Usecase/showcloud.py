import open3d as o3d
import sys

def show_point_cloud(ply_path):
    """Load and visualize a PLY point cloud file with keyboard zoom shortcuts."""
    pcd = o3d.io.read_point_cloud(ply_path)

    vis = o3d.visualization.VisualizerWithKeyCallback()  # Use VisualizerWithKeyCallback
    vis.create_window()
    vis.add_geometry(pcd)

    opt = vis.get_render_option()
    opt.background_color = [0.1, 0.1, 0.1]
    opt.point_size = 1.0

    view_control = vis.get_view_control()
    view_control.set_front([0, 0, -1])
    view_control.set_up([0, 1, 0])
    view_control.set_lookat([0, 0, 0])
    initial_zoom = 0.8  # Initial zoom level
    view_control.set_zoom(initial_zoom)

    zoom_step = 0.1  # Adjust this value to control zoom speed
    current_zoom_level = initial_zoom # Store zoom level here

    def zoom_in(vis):
        """Zoom in callback function."""
        nonlocal current_zoom_level  # Allow modification of outer variable
        current_zoom_level *= (1 - zoom_step) # Update stored zoom level
        view_control.set_zoom(current_zoom_level) # Set zoom using stored value
        return False  # Returning False continues the event loop

    def zoom_out(vis):
        """Zoom out callback function."""
        nonlocal current_zoom_level # Allow modification of outer variable
        current_zoom_level *= (1 + zoom_step) # Update stored zoom level
        view_control.set_zoom(current_zoom_level) # Set zoom using stored value
        return False

    # Register key callbacks
    vis.register_key_callback(ord('Z'), zoom_in)  # 'Z' key for zoom in
    vis.register_key_callback(ord('X'), zoom_out) # 'X' key for zoom out

    print("Press 'Z' to zoom in, 'X' to zoom out.")

    vis.run()
    vis.destroy_window()

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python showcloud.py path/to/pointcloud.ply")
        sys.exit(1)

    ply_path = sys.argv[1]
    show_point_cloud(ply_path)