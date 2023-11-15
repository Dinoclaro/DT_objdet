from typing import Tuple


ASSETS_DIR = "/code/catkin_ws/src/DT_objdet/packages/nn_model"
IMAGE_SIZE = 416
SCORE = 0.6
AREA = 1000

def DT_TOKEN() -> str:
    dt_token = "dt1-3nT8KSoxVh4MnDRxovGLkXZDhPpgc4SzasJBTSxbRUfDguS-43dzqWFnWd8KBa1yev1g3UKnzVxZkkTbfeFCAD1kMCPQvvSVDYPfoXapvF29wVgdC7"

    return dt_token


def MODEL_NAME() -> str:

    return "yolov5n"


def NUMBER_FRAMES_SKIPPED() -> int:

    return 1


def filter_by_classes(pred_class: int) -> bool:
    """
    Remember the class IDs:

        | Object    | ID    |
        | ---       | ---   |
        | Duckie    | 0     |
        | Cone      | 1     |
        | Truck     | 2     |
        | Bus       | 3     |
        | Duckiebot | 4     | 

    Args:
        pred_class: the class of a prediction
    """
    # Only returning True for duckies
    if (pred_class == 0):
        return True


def filter_by_scores(score: float) -> bool:
    """
    Args:
        score: the confidence score of a prediction
    """
    if (score > SCORE):
        return True


def filter_by_bboxes(bbox: Tuple[int, int, int, int]) -> bool:
    """
    Args:
        bbox: is the bounding box of a prediction, in xyxy format
                This means the shape of bbox is (leftmost x pixel, topmost y, rightmost x, bottommost y)
    """

    if abs((bbox[2]-bbox[0])*(bbox[3]-bbox[1])) > AREA:
        return True