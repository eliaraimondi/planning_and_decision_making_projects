# from re import A
# from turtle import distance
# from networkx import edges, project
from sympy import plot
from pdm4ar.exercises_def.ex06.structures import *
from triangle import triangulate
import numpy as np
from typing import Union, Optional


class CollisionPrimitives_SeparateAxis:
    """
    Class for Implementing the Separate Axis Theorem


    A docstring with expected inputs and outputs is provided for each of the functions
    you are to implement. You do not need to adhere to the given variable names, but you should adhere to
    the datatypes of inputs and outputs.

    ## THEOREM
    Let A and B be two disjoint nonempty convex subsets of R^n. Then there exist a nonzero vector v anda  real number c s.t.
    <x,v> >= c and <y,v> <= c. For all x in A and y in B. i.e. the hyperplane <.,v> = c separates A and B.

    If both sets are closed, and at least one of them is compact, then the separation can be strict, that is,
    <x,v> > c_1 and <y,v> < c_2 for some c_1 > c_2


    In this exercise, we will be implementing the Separating Axis Theorem for 2d Primitives.

    """

    # Task 1
    @staticmethod
    def proj_polygon(p: Union[Polygon, Circle], ax: Segment) -> Segment:
        """
        Project the Polygon onto the axis, represented as a Segment.
        Inputs:
        Polygon p,
        a candidate axis ax to project onto

        Outputs:
        segment: a (shorter) segment with start and endpoints of where the polygon has been projected to.

        """
        # TODO: Implement function
        A = np.array([ax.p1.x, ax.p1.y])
        B = np.array([ax.p2.x, ax.p2.y])
        dir_ax = (B - A) / np.linalg.norm(B - A)

        # Find all the vertices of the polygon
        if isinstance(p, Polygon):
            vertices = [np.array([v.x, v.y]) for v in p.vertices]
        elif isinstance(p, Circle):
            center = np.array([p.center.x, p.center.y])
            vertices = [center + dir_ax * p.radius, center - dir_ax * p.radius]

        # Find the projection of the vertices on the axis
        projections = A + np.dot(vertices - A, dir_ax.T)[:, np.newaxis] * dir_ax

        # Find the projections closest to the start and end of the axis
        distances_fromA = {tuple(projection): np.linalg.norm(projection - A) for projection in projections}
        distances_fromB = {tuple(projection): np.linalg.norm(projection - B) for projection in projections}
        start = min(distances_fromA, key=distances_fromA.get)
        end = min(distances_fromB, key=distances_fromB.get)

        return Segment(Point(start[0], start[1]), Point(end[0], end[1]))

    # Task 2.a
    @staticmethod
    def overlap(s1: Segment, s2: Segment) -> bool:
        """
        Check if two segments overlap.
        Inputs:
        s1: a Segment
        s2: a Segment

        Outputs:
        bool: True if segments overlap. False o.w.
        """
        # TODO: Implement Function
        # Find the direction of the two segments
        A1 = np.array([s1.p1.x, s1.p1.y])
        B1 = np.array([s1.p2.x, s1.p2.y])
        A2 = np.array([s2.p1.x, s2.p1.y])
        B2 = np.array([s2.p2.x, s2.p2.y])

        if (
            np.abs(np.linalg.norm(A1 - A2)) < 0.001
            or np.abs(np.linalg.norm(B1 - B2)) < 0.001
            or np.abs(np.linalg.norm(A1 - B2)) < 0.001
            or np.abs(np.linalg.norm(A2 - B1)) < 0.001
        ):
            return True

        # Find in which direction respect to A1 are A2 and B2
        dir_A2_A1 = (A2 - A1) / np.linalg.norm(A2 - A1)
        dir_B2_A1 = (B2 - A1) / np.linalg.norm(B2 - A1)

        # If the direction is different, the segments overlap
        if np.dot(dir_A2_A1, dir_B2_A1) < 0:
            return True

        # Find in which direction respect to B1 are A2 and B2
        dir_A2_B1 = (A2 - B1) / np.linalg.norm(A2 - B1)
        dir_B2_B1 = (B2 - B1) / np.linalg.norm(B2 - B1)

        # If the direction is different, the segments overlap
        if np.dot(dir_A2_B1, dir_B2_B1) < 0:
            return True

        if np.dot(dir_A2_A1, dir_A2_B1) < 0:
            return True

        # If none of the above conditions are met, the segments do not overlap
        return False

    # Task 2.b
    @staticmethod
    def get_axes(p1: Polygon, p2: Polygon) -> list[Segment]:
        """
        Get all Candidate Separating Axes.
        Hint: These are 2D Polygons, recommend searching over axes that are orthogonal to the edges only.
        Rather than returning infinite Segments, return one axis per Edge1-Edge2 pairing.

        Inputs:
        p1, p2: Polygons to obtain separating Axes over.
        Outputs:
        list[Segment]: A list of segments of size N (worldlength) that represent each of the valid separating axes for the two polygons.
        """
        axes = []  # Populate with Segment types

        # TODO: Implement function
        # Find all the edges of the polygons
        edges_p1 = [
            np.array([p1.vertices[i + 1].x, p1.vertices[i + 1].y]) - np.array([p1.vertices[i].x, p1.vertices[i].y])
            for i in range(len(p1.vertices) - 1)
        ] + [np.array([p1.vertices[-1].x, p1.vertices[-1].y]) - np.array([p1.vertices[0].x, p1.vertices[0].y])]
        edges_p2 = [
            np.array([p2.vertices[i + 1].x, p2.vertices[i + 1].y]) - np.array([p2.vertices[i].x, p2.vertices[i].y])
            for i in range(len(p2.vertices) - 1)
        ] + [np.array([p2.vertices[-1].x, p2.vertices[-1].y]) - np.array([p2.vertices[0].x, p2.vertices[0].y])]
        edges = edges_p1 + edges_p2

        # Create a line for each edge and all axes
        for edge in edges:
            edge = edge / np.linalg.norm(edge)

            perpendicular_vector = np.array([edge[1], -edge[0]]) * 20

            axis = Segment(
                Point(-perpendicular_vector[0], -perpendicular_vector[1]),
                Point(perpendicular_vector[0], perpendicular_vector[1]),
            )
            axes.append(axis)

        return axes

    # Task 2.c
    @staticmethod
    def separating_axis_thm(
        p1: Polygon,
        p2: Union[Polygon, Circle],
    ) -> tuple[bool, Optional[Segment]]:
        """
        Get Candidate Separating Axes.
        Once obtained, loop over the Axes, project the polygons onto each axis and check overlap of the projected segments.
        If an axis with a non-overlapping projection is found, we can terminate early. Conclusion: The polygons do not collide.

        IMPORTANT
        This Method Evaluates task 2 and Task 3.
        Task 2 checks the separate axis theorem for two polygons.
        Task 3 checks the separate axis theorem for a circle and a polygon
        We have provided a skeleton on this method to distinguish the two test cases, feel free to use any helper methods above, but your output must come
        from  separating_axis_thm().

        Hint: You can use previously implemented methods, such as overlap() and get_axes()

        Inputs:
        p1, p2: Candidate Polygons
        Outputs:
        Output as a tuple
        bool: True if Polygons Collide. False o.w.
        Segment: An Optional argument that allows you to visualize the axis you are projecting onto.
        """
        axis = None
        if isinstance(p2, Polygon):  # Task 2c

            # TODO: Implement your solution for if polygon here. Exercise 2
            candidate_axes = CollisionPrimitives_SeparateAxis.get_axes(p1, p2)

            # For each norm, find if the projection of the polygons overlap
            for axis in candidate_axes:
                if not CollisionPrimitives_SeparateAxis.overlap(
                    CollisionPrimitives_SeparateAxis.proj_polygon(p1, axis),
                    CollisionPrimitives_SeparateAxis.proj_polygon(p2, axis),
                ):
                    return (False, axis)

            return (True, axis)

        elif isinstance(p2, Circle):  # Task 3b

            # TODO: Implement your solution for SAT for circles here. Exercise 3
            candidate_axes = CollisionPrimitives_SeparateAxis.get_axes_cp(p2, p1)

            # For each norm, find if the projection of the polygons overlap
            for ax in candidate_axes:
                if not CollisionPrimitives_SeparateAxis.overlap(
                    CollisionPrimitives_SeparateAxis.proj_polygon(p1, ax),
                    CollisionPrimitives_SeparateAxis.proj_polygon(p2, ax),
                ):
                    return (False, ax)

            return (True, axis)

        else:
            print("If we get here we have done a big mistake - TAs")
            return (bool, axis)

    # Task 3
    @staticmethod
    def get_axes_cp(circ: Circle, poly: Polygon) -> list[Segment]:
        """
        Get all Candidate Separating Axes.
        Hint: Notice that the circle is a polygon with infinite number of edges. Fortunately we do not need to check all axes normal to the edges.
        It's sufficient to check the axes normal to the polygon edges plus ONE axis formed by the circle center and the closest vertice of the polygon.

        Inputs:
        circ, poly: Cicle and Polygon to check, respectively.
        Ouputs:
        list[Segment]: A list of segments of size N (worldlength) that represent each of the valid separating axes for the two polygons.
        """
        axes = []

        # TODO
        # Find all the edges of the polygon
        edges_p = [
            np.array([poly.vertices[i + 1].x, poly.vertices[i + 1].y])
            - np.array([poly.vertices[i].x, poly.vertices[i].y])
            for i in range(len(poly.vertices) - 1)
        ] + [np.array([poly.vertices[-1].x, poly.vertices[-1].y]) - np.array([poly.vertices[0].x, poly.vertices[0].y])]

        # Find the edge of the circle
        closest_point = min(
            poly.vertices, key=lambda v: np.linalg.norm(np.array([v.x, v.y]) - np.array([circ.center.x, circ.center.y]))
        )
        edge_c = np.array([closest_point.x, closest_point.y]) - np.array([circ.center.x, circ.center.y])

        edges = edges_p + [edge_c]

        # Create a line for each edge and all axes
        for edge in edges:
            edge = edge / np.linalg.norm(edge)

            perpendicular_vector = np.array([edge[1], -edge[0]]) * 20

            axis = Segment(
                Point(-perpendicular_vector[0], -perpendicular_vector[1]),
                Point(perpendicular_vector[0], perpendicular_vector[1]),
            )
            axes.append(axis)

        return axes


class CollisionPrimitives:
    """
    Class of collision primitives
    """

    NUMBER_OF_SAMPLES = 10

    @staticmethod
    def circle_point_collision(c: Circle, p: Point) -> bool:
        """
        Given function.
        Checks if a circle and a point are in collision.

        Inputs:
        c: Circle primitive
        p: Point primitive

        Outputs:
        bool: True if in Collision, False otherwise
        """
        return (p.x - c.center.x) ** 2 + (p.y - c.center.y) ** 2 < c.radius**2

    @staticmethod
    def triangle_point_collision(t: Triangle, p: Point) -> bool:
        """
        Given function.
        Checks if a Triangle and a Point are in Collision

        Inputs:
        t: Triangle Primitive
        p: Point Primitive

        Outputs:
        bool: True if in Collision, False otherwise.
        """
        area_orig = np.abs((t.v2.x - t.v1.x) * (t.v3.y - t.v1.y) - (t.v3.x - t.v1.x) * (t.v2.y - t.v1.y))

        area1 = np.abs((t.v1.x - p.x) * (t.v2.y - p.y) - (t.v2.x - p.x) * (t.v1.y - p.y))
        area2 = np.abs((t.v2.x - p.x) * (t.v3.y - p.y) - (t.v3.x - p.x) * (t.v2.y - p.y))
        area3 = np.abs((t.v3.x - p.x) * (t.v1.y - p.y) - (t.v1.x - p.x) * (t.v3.y - p.y))

        if np.abs(area1 + area2 + area3 - area_orig) < 1e-3:
            return True

        return False

    @staticmethod
    def polygon_point_collision(poly: Polygon, p: Point) -> bool:
        """
        Given function.

        Input:
        poly: Polygon primitive
        p: Point primitive

        Outputs
        bool: True if in Collision, False otherwise.
        """
        triangulation_result = triangulate(dict(vertices=np.array([[v.x, v.y] for v in poly.vertices])))

        triangles = [
            Triangle(
                Point(triangle[0, 0], triangle[0, 1]),
                Point(triangle[1, 0], triangle[1, 1]),
                Point(triangle[2, 0], triangle[2, 1]),
            )
            for triangle in triangulation_result["vertices"][triangulation_result["triangles"]]
        ]

        for t in triangles:
            if CollisionPrimitives.triangle_point_collision(t, p):
                return True

        return False

    @staticmethod
    def circle_segment_collision(c: Circle, segment: Segment) -> bool:
        """
        Given function

        Input:
        c: Circle primitive
        segment: Segment primitive

        Outputs:
        bool: True if in collision, False otherwise.
        """
        inside_1 = CollisionPrimitives.circle_point_collision(c, segment.p1)
        inside_2 = CollisionPrimitives.circle_point_collision(c, segment.p2)

        if inside_1 or inside_2:
            return True

        dist_x = segment.p1.x - segment.p2.x
        dist_y = segment.p1.y - segment.p2.y
        segment_len = np.sqrt(dist_x**2 + dist_y**2)

        dot = (
            ((c.center.x - segment.p1.x) * (segment.p2.x - segment.p1.x))
            + ((c.center.y - segment.p1.y) * (segment.p2.y - segment.p1.y))
        ) / pow(segment_len, 2)

        closest_point = Point(
            segment.p1.x + (dot * (segment.p2.x - segment.p1.x)),
            segment.p1.y + (dot * (segment.p2.y - segment.p1.y)),
        )

        # Check whether point is on the segment segment or not
        segment_len_1 = np.sqrt((segment.p1.x - closest_point.x) ** 2 + (segment.p1.y - closest_point.y) ** 2)
        segment_len_2 = np.sqrt((segment.p2.x - closest_point.x) ** 2 + (segment.p2.y - closest_point.y) ** 2)

        if np.abs(segment_len_1 + segment_len_2 - segment_len) > 1e-3:
            return False

        closest_dist = np.sqrt((c.center.x - closest_point.x) ** 2 + (c.center.y - closest_point.y) ** 2)

        if closest_dist < c.radius:
            return True

        return False

    @staticmethod
    def sample_segment(segment: Segment) -> list[Point]:

        x_diff = (segment.p1.x - segment.p2.x) / CollisionPrimitives.NUMBER_OF_SAMPLES
        y_diff = (segment.p1.y - segment.p2.y) / CollisionPrimitives.NUMBER_OF_SAMPLES

        return [
            Point(x_diff * i + segment.p2.x, y_diff * i + segment.p2.y)
            for i in range(CollisionPrimitives.NUMBER_OF_SAMPLES)
        ]

    @staticmethod
    def triangle_segment_collision(t: Triangle, segment: Segment) -> bool:
        """
        Given function.

        Input:
        t: Triangle Primitive
        segment: Segment primitive

        Outputs:
        bool: True if in collision, False otherwise.
        """
        sampled_points = CollisionPrimitives.sample_segment(segment)

        for point in sampled_points:
            if CollisionPrimitives.triangle_point_collision(t, point):
                return True

        return False

    @staticmethod
    def polygon_segment_collision(p: Polygon, segment: Segment) -> bool:
        """
        Given function.

        Input:
        p: Polygon primitive
        segment: segment primitive

        Outputs:
        bool: True if in collision, False otherwise
        """
        sampled_points = CollisionPrimitives.sample_segment(segment)

        for point in sampled_points:
            if CollisionPrimitives.polygon_point_collision(p, point):
                return True

        return False

    @staticmethod
    def polygon_segment_collision_aabb(p: Polygon, segment: Segment) -> bool:
        """
        Given Function
        Casts a polygon into an AABB, then determines if the bounding box and a segment are in collision

        Inputs:
        p: Polygon primitive
        segment: Segment primitive

        Outputs:
        bool: True if in collision, False otherwise.
        """
        aabb = CollisionPrimitives._poly_to_aabb(p)
        sampled_points = CollisionPrimitives.sample_segment(segment)

        for point in sampled_points:

            if aabb.p_min.x > point.x or aabb.p_min.y > point.y:
                continue

            if aabb.p_max.x < point.x or aabb.p_max.y < point.y:
                continue

            if CollisionPrimitives.polygon_point_collision(p, point):
                return True

        return False

    @staticmethod
    def _poly_to_aabb(g: Polygon) -> AABB:
        """
        Given Function
        Casts a Polygon type into an AABB

        Inputs:
        g: Polygon

        Outputs:
        AABB: Bounding Box for the Polygon.
        """
        x_values = [v.x for v in g.vertices]
        y_values = [v.y for v in g.vertices]

        return AABB(Point(min(x_values), min(y_values)), Point(max(x_values), max(y_values)))

    @staticmethod
    def circle_circle_collision(c1: Circle, c2: Circle) -> bool:
        """
        Given function.

        Inputs:
        c1, c2: Circle primitives

        Outputs:
        bool: True if in collision, False otherwise.
        """
        return np.sqrt((c1.center.x - c2.center.x) ** 2 + (c1.center.y - c2.center.y) ** 2) < c1.radius + c2.radius

    @staticmethod
    def convert_triangle_to_polygon(triangle: Triangle) -> Polygon:
        """
        Converts a triangle to a polygon
        """
        return Polygon([triangle.v1, triangle.v2, triangle.v3])

    @staticmethod
    def circle_polygon_collision(c: Circle, p: Polygon) -> bool:
        """
        Given function.

        Inputs:
        c: Circle primitive
        p: Polygon primitive

        Outputs:
        bool: True if in collision, False otherwise.
        """
        vertices = p.vertices
        for i in range(len(vertices) - 1):
            segment = Segment(vertices[i], vertices[i + 1])
            if CollisionPrimitives.circle_segment_collision(c, segment):
                return True

        segment = Segment(vertices[-1], vertices[0])
        if CollisionPrimitives.circle_segment_collision(c, segment):
            return True

        return False

    @staticmethod
    def circle_triangle_collision(c: Circle, t: Triangle) -> bool:
        """
        Given function.

        Inputs:
        c: Circle primitive
        t: Triangle primitive

        Outputs:
        bool: True if in collision, False otherwise.
        """
        p = CollisionPrimitives.convert_triangle_to_polygon(t)
        return CollisionPrimitives.circle_polygon_collision(c, p)
