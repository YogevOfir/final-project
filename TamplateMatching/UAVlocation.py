import math

def compute_uav_location(x0, y0, W, H, map_origin, scale, altitude, tilt_deg, yaw_deg):
    """
    Computes the UAV's ground location in the map coordinate system.

    Parameters:
      x0, y0: Top-left pixel coordinates of the best-match rectangle in the map.
      W, H: Width and Height (in pixels) of the UAV image (template).
      map_origin: Tuple (X0, Y0) representing the real-world coordinates corresponding to the top-left of the map.
      scale: Map scale in meters per pixel.
      altitude: Altitude of the UAV (meters).
      tilt_deg: Tilt (pitch) angle of the camera (in degrees) from nadir (0° means pointing straight down).
      yaw_deg: Yaw angle (heading) of the camera (in degrees), i.e., the direction in which the camera is tilted relative to map coordinates.
      
    Returns:
      (X_uav, Y_uav): Estimated UAV ground location in map coordinates (meters).
    """
    # Calculate the center of the matched rectangle (in pixel coordinates)
    x_center = x0 + W / 2.0
    y_center = y0 + H / 2.0

    # Convert the image center to map coordinates using the scale factor and map origin
    X_img = map_origin[0] + x_center * scale
    Y_img = map_origin[1] + y_center * scale

    # Convert tilt and yaw angles from degrees to radians
    tilt_rad = math.radians(tilt_deg)
    yaw_rad = math.radians(yaw_deg)

    # Compute the horizontal displacement due to the camera tilt
    # d = altitude * tan(tilt) gives the offset distance from the nadir point.
    d = altitude * math.tan(tilt_rad)

    # Calculate the offset components (assuming the tilt is in the direction of the yaw)
    offset_x = d * math.cos(yaw_rad)
    offset_y = d * math.sin(yaw_rad)

    # The UAV is located "behind" the image center by this offset
    X_uav = X_img - offset_x
    Y_uav = Y_img - offset_y

    return X_uav, Y_uav

# Example usage:
if __name__ == '__main__':
    # Inputs from template matching:
    x0, y0 = 100, 150          # Top-left corner of matched rectangle (pixels)
    W, H = 640, 480            # Size of UAV image (pixels)

    # Map parameters:
    map_origin = (5000.0, 10000.0)  # Real-world coordinates of the map's top-left pixel (meters)
    scale = 0.5                   # Map scale: 0.5 meters per pixel

    # UAV and camera parameters:
    altitude = 120.0             # UAV altitude in meters
    tilt_deg = 10.0              # Camera tilt angle (degrees from nadir)
    yaw_deg = 45.0               # Camera yaw angle (degrees relative to map coordinate system)

    # Compute UAV location:
    X_uav, Y_uav = compute_uav_location(x0, y0, W, H, map_origin, scale, altitude, tilt_deg, yaw_deg)
    print("Estimated UAV Location (meters): X =", X_uav, ", Y =", Y_uav)
