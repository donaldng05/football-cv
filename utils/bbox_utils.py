def get_center_of_bbox(bbox):
    x1, y1, x2, y2 = bbox
    center_x = (x1 + x2) / 2
    center_y = (y1 + y2) / 2
    return int(center_x), int(center_y)


def get_bbox_width(bbox):
    return bbox[2] - bbox[0]


def measure_distance(p1, p2):
    return ((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2) ** 0.5


def measure_xy_distance(p1, p2):
    return p1[0] - p2[0], p1[1] - p2[1]


def get_foot_position(bbox):
    x1, y1, x2, y2 = bbox
    foot_x = (x1 + x2) / 2
    foot_y = y2
    return int(foot_x), int(foot_y)
