
from rotated.assigners.base import BaseTaskAlignedAssigner
from rotated.assigners.calculators import HorizontalIoUCalculator


class TaskAlignedAssigner(BaseTaskAlignedAssigner):
    """Task-aligned assigner specifically for horizontal bounding boxes.

    Expects horizontal boxes in format [cx, cy, w, h] where:
    - cx, cy: Center coordinates in absolute pixels
    - w, h: Width and height in absolute pixels

    Args:
        topk: Number of top candidates to select per GT object
        alpha: Exponent for classification score in alignment computation
        beta: Exponent for IoU score in alignment computation
        eps: Small value to prevent division by zero in assignment
        iou_eps: Small constant for numerical stability in IoU computation
    """

    def __init__(
        self,
        topk: int = 13,
        alpha: float = 1.0,
        beta: float = 6.0,
        eps: float = 1e-9,
        iou_eps: float = 1e-7,
    ):
        super().__init__(
            iou_calculator=HorizontalIoUCalculator(eps=iou_eps),
            box_format='horizontal',
            topk=topk,
            alpha=alpha,
            beta=beta,
            eps=eps,
        )
