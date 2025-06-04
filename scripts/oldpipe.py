from pathlib import Path
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
import shapely
from shapely import ops
from shapely.plotting import plot_line, plot_points, plot_polygon


class FishPointsOfInterestDetector():
    """Fish mask endpoint detector and classifier.

    1. Estimates endpoints.
    2. Classifies endpoints as either head/tail. Uses endpoints from:
        - previous step OR
        - specified parameter
    3. Uses separate heuristic to correct each.

    Included is a function that does all these steps in one.
    """

    def __init__(self) -> None:
        super().__init__()

        self.geo: FishGeometry = None

    def correct_tail_coord(self):  # TODO
        tail_poly = self.geo.get_tail_poly()

        tail_convex = tail_poly.convex_hull

        tail_convex_diff = shapely.difference(tail_convex, tail_poly).geoms

        distances = [
            shapely.distance(p, self.geo.get_tailpoint_extended())
            for p in tail_convex_diff
        ]

        tail_convex_diff = tail_convex_diff[distances.index(min(distances))]

        tail_point = shapely.geometry.Point(self.geo.get_tail_coord())
        tailpoint_in_poly = tail_convex_diff.boundary.contains(
            tail_point
        ) or tail_point.within(tail_convex_diff)
        print("Tail in poly? " + str(tailpoint_in_poly))

        if tailpoint_in_poly:
            _, tail_corrected = ops.nearest_points(
                self.geo.get_headpoint_extended(), tail_convex_diff.boundary
            )
        else:
            # we assume that the tail is non-convex
            # TODO: slice tail_poly with a line a little before tailpoint_line, trim the extra, and get the centroid
            tail_corrected = shapely.geometry.Point(self.geo.get_tail_coord())

        # plot_polygon(tail_convex_diff, add_points=False)
        # plot_points(tail_corrected)

        self.geo.set_tail_corrected(np.asarray([tail_corrected.x, tail_corrected.y]))

        return {"tail": self.geo.get_tail_corrected()}

    def correct_head_coord(self):
        # any point on head_poly_sliced is better or just as good as the initial estimated point
        head_poly_sliced = ops.split(
            self.geo.get_head_poly(), self.geo.get_headpoint_line()
        ).geoms

        # polygon closest to the extended headpoint is the one we want
        distances = [
            shapely.distance(p, self.geo.get_headpoint_extended())
            for p in head_poly_sliced
        ]
        head_poly_sliced = head_poly_sliced[distances.index(min(distances))]

        # we guess the tip of head_poly_sliced by finding its nearest point to another point far ahead
        try:
            _, head_corrected = ops.nearest_points(
                self.geo.get_headpoint_extended(), head_poly_sliced.convex_hull.boundary
            )
        except:
            # if a point can't be extracted, we default to the original estimation
            head_corrected = shapely.geometry.Point(self.geo.get_head_coord())
        self.geo.set_head_corrected([head_corrected.x, head_corrected.y])

        return {"head": self.geo.get_head_corrected()}

    def correct_endpoints(self):
        self.correct_head_coord()

        self.correct_tail_coord()

        return {
            "head": self.geo.get_head_corrected(),
            "tail": self.geo.get_tail_corrected(),
        }

    def classify_endpoints(self, endpoints=None) -> Dict[str, np.ndarray]:
        endpoints = (
            self.geo.get_estimated_endpoints() if endpoints == None else endpoints
        )
        self.geo.set_estimated_endpoints(endpoints)

        # get halves
        halves = self.geo.get_halves()

        # get the convex versions
        halves_convex = self.geo.get_halves_convex()

        # get the differences
        halves_difference = [
            shapely.difference(halves_convex[0], halves[0]),
            shapely.difference(halves_convex[1], halves[1]),
        ]

        # compare the areas and set head/tail polys
        half_areas = [halves_difference[0].area], [halves_difference[1].area]
        self.geo.set_tail_poly(halves[half_areas.index(max(half_areas))])
        self.geo.set_head_poly(halves[half_areas.index(min(half_areas))])

        # assign the classification to the endpoints
        head_coord = self.geo.get_head_coord(endpoints)
        tail_coord = self.geo.get_tail_coord(endpoints)

        # compute the confidence score
        confidence = float(
            (max(half_areas)[0] - min(half_areas)[0]) / max(half_areas)[0] / 2.0 + 0.5
        )
        confidence = int(confidence * 100) / 100

        return {"head": head_coord, "tail": tail_coord, "confidence": confidence}

    def estimate_endpoints(self) -> Tuple[np.ndarray, np.ndarray, float]:
        mask = self.geo.get_mask()

        left_coord, right_coord = PcaPointsOfInterestDetector().find_points_of_interest(
            mask
        )

        self.geo.set_estimated_endpoints([left_coord, right_coord])
        return self.geo.get_estimated_endpoints()

    def find_points_of_interest(
        self, mask: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        self.geo = FishGeometry(mask)

        self.estimate_endpoints()
        self.classify_endpoints()
        corrected = self.correct_endpoints()
        return corrected["head"], corrected["tail"]


class PcaPointsOfInterestDetector(PointsOfInterestDetector):
    def __init__(self) -> None:
        super().__init__()

    def find_points_of_interest(
        self, mask: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        # Find all the nonzero points.  These are the mask.
        y, x = mask.nonzero()
        x_min, x_max, y_min, y_max = [x.min(), x.max(), y.min(), y.max()]

        # Crop the mask using the nonzero
        mask_crop = mask[y_min:y_max, x_min:x_max]
        y, x = mask_crop.nonzero()

        # Necessary for using PCA
        x_mean = np.mean(x)
        y_mean = np.mean(y)

        x = x - x_mean
        y = y - y_mean
        x_min, x_max, y_min, y_max = [x.min(), x.max(), y.min(), y.max()]

        # PCA
        coords = np.vstack([x, y])
        cov = np.cov(coords)
        evals, evecs = np.linalg.eig(cov)

        # Choose the largest eigenvalue
        sort_indices = np.argsort(evals)[::-1]
        x_v1, y_v1 = evecs[:, sort_indices[0]]  # Eigenvector with largest eigenvalue

        height, width = mask_crop.shape

        # Scale the line until we reach the end.
        scale = height if height > width else width

        coord1 = np.array([-x_v1 * scale * 2, -y_v1 * scale * 2])
        coord2 = np.array([x_v1 * scale * 2, y_v1 * scale * 2])

        # Calculate the line
        coord1[0] -= x_min
        coord2[0] += x_min

        coord1[1] -= y_min
        coord2[1] += y_min

        m = y_v1 / x_v1
        b = coord1[1] - m * coord1[0]

        y, x = mask_crop.nonzero()
        y_target = m * x + b

        # Find the full line
        points_along_line = np.where(np.abs(y - y_target) < 1)
        x = x[points_along_line]
        y = y[points_along_line]

        coords = np.stack([x, y])
        left_coord = coords[:, np.argmin(x)]
        right_coord = coords[:, np.argmax(x)]

        y, x = mask.nonzero()
        x_min, x_max, y_min, y_max = [x.min(), x.max(), y.min(), y.max()]

        left_coord[0] += x_min
        right_coord[0] += x_min
        left_coord[1] += y_min
        right_coord[1] += y_min

        return left_coord, right_coord
    

class FishGeometry:
    def __init__(self, mask: np.ndarray):
        self.__mask = mask
        self.__reset()

    def __reset(self):
        self.__perimeter = None
        self.__estimated_endpoints = (
            None  # the first estimated endpoints (client sets this)
        )
        self.__ab = None  # the line connecting the first estimated endpoints
        self.__ab_perp = None  # the line perpendicular to ab
        self.__polygon = None  # a polygonal representation of the fish mask
        self.__halves = None  # two polygon halves sliced by ab_perp
        self.__halves_convex = None  # the respective convex hulls of the halves
        self.__tail_coord = None  # the estimated endpoint coordinate contained in tail_poly (client sets this)
        self.__head_coord = None  # the estimated endpoint coordinate contained in head_poly (clients sets this)
        self.__tail_poly = (
            None  # the polygon half containing tail_coord (client sets this)
        )
        self.__head_poly = (
            None  # the polygon half containing head_coord (client sets this)
        )
        # ^^ The client only needs to classify either the coords or the polygons. They don't have to do both.
        # The respective counterparts are inferred automatically.
        self.__headpoint_line = (
            None  # the line parallel to ab_perp with its centroid being head_coord
        )
        self.__tailpoint_line = (
            None  # the line parallel to ab_perp with its centroid being tail_coord
        )
        self.__headpoint_extended = (
            None  # a point further out from head_coord on an extended ab
        )
        self.__tailpoint_extended = (
            None  # a point further out form tail_coord on an extended ab
        )
        self.__head_corrected = None  # the corrected head point (client sets this)
        self.__tail_corrected = None  # the corrected tail point (client sets this)

    def set_estimated_endpoints(self, endpoints):
        if self.__estimated_endpoints != endpoints:
            self.__reset()
        left, right = endpoints
        self.__estimated_endpoints = [np.asarray(left), np.asarray(right)]

    def set_tail_poly(self, tail_poly):
        self.__tail_poly = tail_poly

    def set_head_poly(self, head_poly):
        self.__head_poly = head_poly

    def set_tail_coord(self, tail_coord):
        self.__tail_coord = np.asarray(tail_coord)

    def set_head_coord(self, head_coord):
        self.__head_coord = np.asarray(head_coord)

    def set_head_corrected(self, head_corrected):
        self.__head_corrected = np.asarray(head_corrected)

    def set_tail_corrected(self, tail_corrected):
        self.__tail_corrected = np.asarray(tail_corrected)

    def get_mask(self) -> np.ndarray:
        return self.__mask

    def get_estimated_endpoints(self) -> Tuple[np.ndarray, np.ndarray]:
        """The first estimated endpoints. The client must set this."""
        assert (
            self.__estimated_endpoints is not None
        ), "You need to set the endpoints first! Use set_estimated_endpoints()."
        return self.__estimated_endpoints

    def get_perimeter(self) -> List[np.ndarray]:
        """List of points representing the perimeter of the mask."""
        if self.__perimeter is not None:
            return self.__perimeter
        contours, _ = cv2.findContours(
            self.__mask.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
        )
        # print(len(contours)) TODO: select contour with the most points
        self.__perimeter = contours[0].reshape(-1, 2)
        return self.__perimeter

    def get_ab(self) -> shapely.geometry.linestring.LineString:
        """The line connecting the first estimated endpoints."""
        if self.__ab is not None:
            return self.__ab
        self.__ab = shapely.geometry.LineString(self.get_estimated_endpoints())
        return self.__ab

    def get_ab_perp(self) -> shapely.geometry.linestring.LineString:
        """The line perpendicular to ab."""
        if self.__ab_perp is not None:
            return self.__ab_perp
        ab = self.get_ab()
        perimeter = self.get_perimeter()
        vert_min = abs(perimeter[:, 1].min() - ab.centroid.y)
        vert_max = abs(perimeter[:, 1].max() - ab.centroid.y)
        vert_len = vert_min if vert_min > vert_max else vert_max
        ab_left = ab.parallel_offset(vert_len * 1.5, "left")
        ab_right = ab.parallel_offset(vert_len * 1.5, "right")
        self.__ab_perp = shapely.geometry.LineString(
            [ab_left.centroid, ab_right.centroid]
        )
        return self.__ab_perp

    def get_polygon(self) -> shapely.geometry.polygon.Polygon:
        """A polygonal representation of the fish mask."""
        if self.__polygon is not None:
            return self.__polygon
        self.__polygon = shapely.geometry.Polygon(self.get_perimeter())
        return self.__polygon

    def get_halves(
        self,
    ) -> Tuple[shapely.geometry.polygon.Polygon, shapely.geometry.polygon.Polygon]:
        """Two polygon halves sliced by ab_perp."""
        if self.__halves is not None:
            return self.__halves
        self.__halves = ops.split(self.get_polygon(), self.get_ab_perp()).geoms
        if len(self.__halves) > 2:
            self.__halves = sorted(self.__halves, key=lambda p: p.area, reverse=True)[
                :2
            ]
        return self.__halves

    def get_halves_convex(
        self,
    ) -> Tuple[shapely.geometry.polygon.Polygon, shapely.geometry.polygon.Polygon]:
        """The respective convex hulls of the halves."""
        if self.__halves_convex is not None:
            return self.__halves_convex
        halves = self.get_halves()
        self.__halves_convex = [halves[0].convex_hull, halves[1].convex_hull]
        return self.__halves_convex

    def get_head_poly(self) -> shapely.geometry.polygon.Polygon:
        """The polygon half containing head_coord. This or head_coord must be set by the client."""
        if self.__head_poly is not None:
            return self.__head_poly
        # if the head poly is not already set, try to infer it from the coord information
        head, tail = self.__find_polys_from_matching_endpoints()
        self.set_head_poly(head)
        self.set_tail_poly(tail)
        return self.__head_poly

    def get_tail_poly(self) -> shapely.geometry.polygon.Polygon:
        """The polygon half containing tail_coord. This or tail_coord must be set by the client."""
        if self.__tail_poly is not None:
            return self.__tail_poly
        # if the tail poly is not already set, try to infer it from the coord information
        head, tail = self.__find_polys_from_matching_endpoints()
        self.set_head_poly(head)
        self.set_tail_poly(tail)
        return self.__tail_poly

    def get_head_coord(self, endpoints=None) -> np.ndarray:
        """The estimated endpoint coordinate contained in head_poly. This or head_poly must be set by the client."""
        if self.__head_coord is not None:
            return self.__head_coord
        # if the head coord is not already set, try to infer it from the polygon information
        head, tail = self.__find_endpoints_from_matching_polys(endpoints)
        self.set_head_coord(head)
        self.set_tail_coord(tail)
        return self.__head_coord

    def get_tail_coord(self, endpoints=None) -> np.ndarray:
        """The estimated endpoint coordinate contained in tail_poly. This or tail_poly must be set by the client."""
        if self.__tail_coord is not None:
            return self.__tail_coord
        # if the tail coord is not already set, try to infer it from the polygon information
        head, tail = self.__find_endpoints_from_matching_polys(endpoints)
        self.set_head_coord(head)
        self.set_tail_coord(tail)
        return self.__tail_coord

    def get_headpoint_extended(self) -> shapely.geometry.point.Point:
        """A point further out from head_coord on an extended ab."""
        if self.__headpoint_extended is not None:
            return self.__headpoint_extended
        self.__find_outer_points()
        return self.__headpoint_extended

    def get_tailpoint_extended(self) -> shapely.geometry.point.Point:
        """A point further out from tail_coord on an extended ab."""
        if self.__tailpoint_extended is not None:
            return self.__tailpoint_extended
        self.__find_outer_points()
        return self.__tailpoint_extended

    def get_headpoint_line(self) -> shapely.geometry.linestring.LineString:
        """The line parallel to ab_perp with its centroid being head_coord."""
        if self.__headpoint_line is not None:
            return self.__headpoint_line
        self.__find_endpoint_lines()
        return self.__headpoint_line

    def get_tailpoint_line(self) -> shapely.geometry.linestring.LineString:
        """The line parallel to ab_perp with its centroid being tail_coord."""
        if self.__tailpoint_line is not None:
            return self.__tailpoint_line
        self.__find_endpoint_lines()
        return self.__tailpoint_line

    def get_head_corrected(self) -> np.ndarray:
        """The end result. Must be set by the client."""
        assert (
            self.__head_corrected is not None
        ), "You need to set head_corrected. Try set_head_corrected()."
        return self.__head_corrected

    def get_tail_corrected(self) -> np.ndarray:
        """The end result. Must be set by the client."""
        assert (
            self.__tail_corrected is not None
        ), "You need to set tail_corrected. Try set_tail_corrected()."
        return self.__tail_corrected

    def __find_endpoint_lines(self):
        head_coord = self.get_head_coord()
        tail_coord = self.get_tail_coord()
        ab = self.get_ab()
        ab_perp = self.get_ab_perp()
        # get the perpendicular line of ab from the headpoint
        half_len = shapely.distance(ab, shapely.Point(head_coord).centroid)
        endpoint_line1 = ab_perp.parallel_offset(half_len, "right")
        endpoint_line2 = ab_perp.parallel_offset(half_len, "left")
        # again choose the right line
        if shapely.distance(
            endpoint_line1.centroid, shapely.Point(head_coord)
        ) < shapely.distance(endpoint_line1.centroid, shapely.Point(tail_coord)):
            self.__headpoint_line = endpoint_line1
            self.__tailpoint_line = endpoint_line2
        else:
            self.__headpoint_line = endpoint_line2
            self.__tailpoint_line = endpoint_line1

    def __find_outer_points(self):
        ab_perp = self.get_ab_perp()
        perimeter = self.get_perimeter()
        # draw two lines parallel to ab_perp a little bit aways from the original endpoints
        hor_min = abs(perimeter[:, 0].min() - ab_perp.centroid.x)
        hor_max = abs(perimeter[:, 0].max() - ab_perp.centroid.x)
        hor_len = hor_min if hor_min > hor_max else hor_max
        nose_line1 = ab_perp.parallel_offset(hor_len * 2.0, "right")  # hor_len*2.0
        nose_line2 = ab_perp.parallel_offset(hor_len * 2.0, "left")  # hor_len*2.0
        # choose the line near the head and get the centroid
        if shapely.distance(
            nose_line1.centroid, shapely.Point(self.get_head_coord())
        ) < shapely.distance(nose_line1.centroid, shapely.Point(self.get_tail_coord())):
            self.__headpoint_extended = nose_line1.centroid
            self.__tailpoint_extended = nose_line2.centroid
        else:
            self.__headpoint_extended = nose_line2.centroid
            self.__tailpoint_extended = nose_line1.centroid

    def __find_endpoints_from_matching_polys(self, endpoints=None):
        assert (
            self.__head_poly is not None and self.__tail_poly is not None
        ), "You need to classify the polygons. Try set_head_poly() or set_tail_poly()."
        left_coord, right_coord = (
            self.__estimated_endpoints if endpoints == None else endpoints
        )
        if abs(shapely.distance(shapely.Point(left_coord), self.get_head_poly())) < abs(
            shapely.distance(shapely.Point(left_coord), self.get_tail_poly())
        ):
            head_coord = left_coord
            tail_coord = right_coord
        else:
            head_coord = right_coord
            tail_coord = left_coord
        return head_coord, tail_coord

    def __find_polys_from_matching_endpoints(self):
        assert (
            self.__head_coord is not None and self.__tail_coord is not None
        ), "You need to classify the endpoints. Try set_head_coord() or set_tail_coord()."
        halves = self.get_halves()
        if abs(shapely.distance(shapely.Point(self.__head_coord), halves[0])) < abs(
            shapely.distance(shapely.Point(self.__head_coord), halves[1])
        ):
            head_poly = halves[0]
            tail_poly = halves[1]
        else:
            head_poly = halves[1]
            tail_poly = halves[0]
        return head_poly, tail_poly