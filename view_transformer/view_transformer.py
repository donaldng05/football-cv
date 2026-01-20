import numpy as np
import cv2


class ViewTransformer:
    """
    Handles perspective transformation to map pixel coordinates from the video to real-world coordinates (meters).
    """

    def __init__(self):
        count_width = 68
        court_length = 23.32

        self.pixel_verticies = np.array(
            [
                [110, 1035],
                [265, 275],
                [910, 260],
                [1640, 915],
            ]
        )

        self.target_verticies = np.array(
            [
                [0, 0],
                [0, court_length],
                [count_width, court_length],
                [count_width, 0],
            ]
        )

        self.pixel_verticies = self.pixel_verticies.astype(np.float32)
        self.target_verticies = self.target_verticies.astype(np.float32)

        self.perspective_transformer = cv2.getPerspectiveTransform(
            self.pixel_verticies, self.target_verticies
        )

    def transform_point(self, point):
        """
        Transform a single 2D point from pixel coordinates to real-world coordinates.
        """
        p = int(point[0]), int(point[1])
        is_inside = cv2.pointPolygonTest(self.pixel_verticies, p, False) >= 0
        if not is_inside:
            return None

        reshaped_point = point.reshape(-1, 1, 2).astype(np.float32)
        transformed_point = cv2.perspectiveTransform(
            reshaped_point, self.perspective_transformer
        )
        return transformed_point.reshape(-1, 2)

    def add_transformed_position_to_tracks(self, tracks):
        """
        Apply perspective transformation to all tracked object positions.
        Updates the tracks dictionary in-place by adding 'position_transformed' to each track entry.
        """
        for object, object_tracks in tracks.items():
            for frame_num, track in enumerate(object_tracks):
                for track_id, track_info in track.items():
                    # Initialize position_transformed as None by default
                    tracks[object][frame_num][track_id]["position_transformed"] = None

                    if "position_adjusted" not in track_info:
                        continue

                    position = track_info["position_adjusted"]
                    position = np.array(position)
                    position_transformed = self.transform_point(position)

                    if position_transformed is not None:
                        position_transformed = position_transformed.squeeze().tolist()
                        tracks[object][frame_num][track_id][
                            "position_transformed"
                        ] = position_transformed
