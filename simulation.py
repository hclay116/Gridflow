import pygame
import numpy as np
import random
import time
import argparse
import json
import os
import math
from collections import deque
import moderngl
from pyrr import Matrix44, Vector3, vector

# Initialize pygame
pygame.init()
pygame.display.gl_set_attribute(pygame.GL_CONTEXT_MAJOR_VERSION, 3)
pygame.display.gl_set_attribute(pygame.GL_CONTEXT_MINOR_VERSION, 3)
pygame.display.gl_set_attribute(pygame.GL_CONTEXT_PROFILE_MASK, pygame.GL_CONTEXT_PROFILE_CORE)

# Define constants
SCREEN_WIDTH = 1200 
SCREEN_HEIGHT = 800
ROAD_WIDTH = 12
INTERSECTION_SIZE = 20
SIDEWALK_WIDTH = 2
LANE_WIDTH = 5
VEHICLE_TYPES = ["car", "truck", "bus", "motorcycle"]

# Vehicle dimensions (in meters)
VEHICLE_DIMS = {
    "car": {"length": 5, "width": 2.5, "height": 1.5},
    "truck": {"length": 9, "width": 2.8, "height": 3.0},
    "bus": {"length": 12, "width": 2.8, "height": 3.2},
    "motorcycle": {"length": 2.5, "width": 1.0, "height": 1.5}
}

PEDESTRIAN_SPAWN_RATE = 0.005
MAX_SPEED = 15  # m/s (~ 54 km/h)

# Traffic light states
RED = 0
YELLOW = 1
GREEN = 2

# Traffic light colors (RGB float values)
LIGHT_COLORS = {
    RED: (1.0, 0.0, 0.0),
    YELLOW: (1.0, 1.0, 0.0),
    GREEN: (0.0, 1.0, 0.0)
}

# Set up pygame display with OpenGL support
pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT), pygame.OPENGL | pygame.DOUBLEBUF)
pygame.display.set_caption("3D Traffic Simulation")
clock = pygame.time.Clock()

# Initialize fonts
pygame.font.init()
small_font = pygame.font.SysFont("Arial", 14)
medium_font = pygame.font.SysFont("Arial", 18)
large_font = pygame.font.SysFont("Arial", 24)


# -------------------------- Shader and Camera Classes --------------------------
class Shader:
    """Represents a shader program in OpenGL."""
    def __init__(self, ctx, vertex_shader, fragment_shader):
        self.ctx = ctx
        self.program = ctx.program(
            vertex_shader=vertex_shader,
            fragment_shader=fragment_shader
        )
        self.vao = {}

    def create_vao(self, name, program, vbo, ibo=None):
        """Create a Vertex Array Object for a shader."""
        if ibo is not None:
            vao = self.ctx.vertex_array(program, [(vbo, '3f 3f', 'in_position', 'in_color')], ibo)
        else:
            vao = self.ctx.vertex_array(program, [(vbo, '3f 3f', 'in_position', 'in_color')])
        self.vao[name] = vao
        return vao


class Camera:
    """Represents a 3D camera for viewing the simulation."""
    def __init__(self, position=(0, 50, 150), target=(0, 0, 0), up=(0, 1, 0)):
        self.position = Vector3(position)
        self.target = Vector3(target)
        self.up = Vector3(up)
        self.fov = 45.0  # Field of view
        self.near = 0.1  # Near clipping plane
        self.far = 1000.0  # Far clipping plane
        self.movement_speed = 0.5
        self.rotation_speed = 0.5

        # Camera modes
        self.modes = ["overview", "follow_car", "first_person", "bird_eye"]
        self.current_mode = "overview"
        self.follow_target = None

        # Initial view angles
        self.yaw = -90.0
        self.pitch = -30.0
        self.update_vectors()

    def update_vectors(self):
        front = Vector3([
            math.cos(math.radians(self.yaw)) * math.cos(math.radians(self.pitch)),
            math.sin(math.radians(self.pitch)),
            math.sin(math.radians(self.yaw)) * math.cos(math.radians(self.pitch))
        ])
        self.front = front.normalized
        self.right = Vector3.cross(self.front, Vector3([0.0, 1.0, 0.0])).normalized
        self.up = Vector3.cross(self.right, self.front).normalized


    def set_mode(self, mode, target=None):
        """Set camera mode and optionally a target to follow."""
        if mode in self.modes:
            self.current_mode = mode
            self.follow_target = target
            if mode == "overview":
                self.position = Vector3([0, 50, 150])
                self.target = Vector3([0, 0, 0])
                self.yaw = -90.0
                self.pitch = -30.0
            elif mode == "bird_eye":
                self.position = Vector3([0, 150, 0])
                self.target = Vector3([0, 0, 0])
                self.yaw = -90.0
                self.pitch = -90.0

    def get_view_matrix(self):
        """Get the view matrix for the camera."""
        if self.current_mode == "follow_car" and self.follow_target:
            vehicle_pos = Vector3(self.follow_target.position)
            vehicle_dir = self.follow_target.direction_vector
            offset = vehicle_dir * -15.0
            offset.y += 5.0
            self.position = vehicle_pos + offset
            self.target = vehicle_pos
        elif self.current_mode == "first_person" and self.follow_target:
            vehicle_pos = Vector3(self.follow_target.position)
            vehicle_dir = self.follow_target.direction_vector
            offset = vehicle_dir * 2.0
            offset.y += 1.5
            self.position = vehicle_pos + offset
            self.target = vehicle_pos + vehicle_dir * 50.0
        return Matrix44.look_at(self.position, self.target if self.current_mode != "overview" else self.position + self.front, self.up)

    def get_projection_matrix(self, aspect_ratio):
        """Get the projection matrix for the camera."""
        return Matrix44.perspective_projection(self.fov, aspect_ratio, self.near, self.far)

    def process_input(self, keys_pressed, delta_time, mouse_rel=None):
        """Process keyboard and mouse input to move the camera."""
        if self.current_mode != "overview":
            return
        dt = delta_time
        speed = self.movement_speed * 50.0 * dt
        if keys_pressed[pygame.K_w]:
            self.position += self.front * speed
        if keys_pressed[pygame.K_s]:
            self.position -= self.front * speed
        if keys_pressed[pygame.K_a]:
            self.position -= self.right * speed
        if keys_pressed[pygame.K_d]:
            self.position += self.right * speed
        if keys_pressed[pygame.K_q]:
            self.position.y += speed
        if keys_pressed[pygame.K_e]:
            self.position.y -= speed
        if mouse_rel:
            self.yaw += mouse_rel[0] * self.rotation_speed
            self.pitch -= mouse_rel[1] * self.rotation_speed
            self.pitch = max(-89.0, min(89.0, self.pitch))
            self.update_vectors()


# -------------------------- Object Classes --------------------------
class Vehicle:
    """Represents a vehicle in the 3D simulation."""
    def __init__(self, vehicle_type, lane, direction, spawn_position, speed=None):
        self.vehicle_type = vehicle_type
        self.lane = lane  # 0 for rightmost, 1 for leftmost
        self.direction_name = direction  # "north", "south", "east", "west"
        self.position = Vector3(spawn_position)
        self.speed = speed if speed is not None else random.uniform(5, MAX_SPEED)
        self.waiting_time = 0
        self.stopped = False
        self.color = self._get_color()
        self.id = random.randint(10000, 99999)
        dims = VEHICLE_DIMS[vehicle_type]
        self.length = dims["length"]
        self.width = dims["width"]
        self.height = dims["height"]
        self.yaw = self._get_initial_yaw()
        self.direction_vector = self._calculate_direction_vector()
        self.bounding_box = self._calculate_bounding_box()
        self.vertices, self.indices = self._create_vehicle_mesh()

    def _get_color(self):
        if self.vehicle_type == "car":
            return random.choice([
                (0.8, 0.0, 0.0),
                (0.0, 0.0, 0.8),
                (0.0, 0.8, 0.0),
                (0.8, 0.8, 0.0),
                (0.8, 0.4, 0.0),
                (0.5, 0.5, 0.5),
                (1.0, 1.0, 1.0)
            ])
        elif self.vehicle_type == "truck":
            return random.choice([
                (0.6, 0.3, 0.0),
                (0.4, 0.4, 0.4),
                (0.0, 0.4, 0.6),
                (0.6, 0.6, 0.6)
            ])
        elif self.vehicle_type == "bus":
            return random.choice([
                (1.0, 0.8, 0.0),
                (1.0, 0.4, 0.0),
                (0.8, 0.8, 0.8)
            ])
        elif self.vehicle_type == "motorcycle":
            return random.choice([
                (0.0, 0.0, 0.0),
                (0.6, 0.0, 0.0),
                (0.0, 0.0, 0.6),
                (0.4, 0.4, 0.4)
            ])
        return (0.5, 0.5, 0.5)

    def _get_initial_yaw(self):
        if self.direction_name == "north":
            return 0
        elif self.direction_name == "south":
            return 180
        elif self.direction_name == "east":
            return 270
        elif self.direction_name == "west":
            return 90
        return 0

    def _calculate_direction_vector(self):
        radians = math.radians(self.yaw)
        return Vector3([math.sin(radians), 0.0, math.cos(radians)])

    def _calculate_bounding_box(self):
        half_length = self.length / 2
        half_width = self.width / 2
        local_corners = [
            Vector3([-half_length, 0, -half_width]),
            Vector3([half_length, 0, -half_width]),
            Vector3([half_length, 0, half_width]),
            Vector3([-half_length, 0, half_width])
        ]
        sin_yaw = math.sin(math.radians(self.yaw))
        cos_yaw = math.cos(math.radians(self.yaw))
        world_corners = []
        for corner in local_corners:
            rotated_x = corner.x * cos_yaw - corner.z * sin_yaw
            rotated_z = corner.x * sin_yaw + corner.z * cos_yaw
            world_corner = Vector3([rotated_x + self.position.x,
                                    corner.y + self.position.y,
                                    rotated_z + self.position.z])
            world_corners.append(world_corner)
        return world_corners

    def _create_vehicle_mesh(self):
        half_length = self.length / 2
        half_width = self.width / 2
        half_height = self.height / 2
        if self.vehicle_type == "car":
            vertices = [
                (-half_length, 0, -half_width, *self.color),
                (half_length, 0, -half_width, *self.color),
                (half_length, 0, half_width, *self.color),
                (-half_length, 0, half_width, *self.color),
                (-half_length * 0.8, half_height, -half_width * 0.8, *self.color),
                (half_length * 0.5, half_height, -half_width * 0.8, *self.color),
                (half_length * 0.5, half_height, half_width * 0.8, *self.color),
                (-half_length * 0.8, half_height, half_width * 0.8, *self.color),
            ]
        elif self.vehicle_type in ["truck", "bus"]:
            vertices = [
                (-half_length, 0, -half_width, *self.color),
                (half_length, 0, -half_width, *self.color),
                (half_length, 0, half_width, *self.color),
                (-half_length, 0, half_width, *self.color),
                (-half_length, half_height, -half_width, *self.color),
                (half_length, half_height, -half_width, *self.color),
                (half_length, half_height, half_width, *self.color),
                (-half_length, half_height, half_width, *self.color),
            ]
        else:  # motorcycle
            vertices = [
                (-half_length, 0, -half_width, *self.color),
                (half_length, 0, -half_width, *self.color),
                (half_length, 0, half_width, *self.color),
                (-half_length, 0, half_width, *self.color),
                (-half_length * 0.5, half_height, 0, *self.color),
                (half_length * 0.5, half_height, 0, *self.color),
            ]
        if self.vehicle_type == "motorcycle":
            indices = [
                0, 1, 2, 0, 2, 3,
                0, 4, 1, 1, 4, 5,
                1, 5, 2, 2, 5, 3,
                3, 5, 0, 0, 5, 4
            ]
        else:
            indices = [
                0, 1, 2, 0, 2, 3,
                4, 5, 6, 4, 6, 7,
                0, 4, 1, 1, 4, 5,
                1, 5, 2, 2, 5, 6,
                2, 6, 3, 3, 6, 7,
                3, 7, 0, 0, 7, 4
            ]
        return vertices, indices

    def update_position(self, delta_time, traffic_lights, other_vehicles):
        should_stop = self._check_traffic_light(traffic_lights)
        vehicle_ahead = self._check_vehicle_ahead(other_vehicles)
        if should_stop or vehicle_ahead:
            if not self.stopped:
                self.stopped = True
            self.waiting_time += delta_time
        else:
            self.stopped = False
            movement = self.direction_vector * self.speed * delta_time
            self.position += movement
        self.bounding_box = self._calculate_bounding_box()

    def _check_traffic_light(self, traffic_lights):
        if self.direction_name == "north":
            light = traffic_lights["south"]
            if (self.position.z > 0 and 
                self.position.z < INTERSECTION_SIZE/2 + 10 and 
                abs(self.position.x) < ROAD_WIDTH):
                return light.state in [RED, YELLOW]
        elif self.direction_name == "south":
            light = traffic_lights["north"]
            if (self.position.z < 0 and 
                self.position.z > -INTERSECTION_SIZE/2 - 10 and 
                abs(self.position.x) < ROAD_WIDTH):
                return light.state in [RED, YELLOW]
        elif self.direction_name == "east":
            light = traffic_lights["west"]
            if (self.position.x < 0 and 
                self.position.x > -INTERSECTION_SIZE/2 - 10 and 
                abs(self.position.z) < ROAD_WIDTH):
                return light.state in [RED, YELLOW]
        elif self.direction_name == "west":
            light = traffic_lights["east"]
            if (self.position.x > 0 and 
                self.position.x < INTERSECTION_SIZE/2 + 10 and 
                abs(self.position.z) < ROAD_WIDTH):
                return light.state in [RED, YELLOW]
        return False

    def _check_vehicle_ahead(self, other_vehicles):
        min_distance = 2.0 + self.speed * 0.5
        for vehicle in other_vehicles:
            if vehicle.id == self.id:
                continue
            if vehicle.lane != self.lane or vehicle.direction_name != self.direction_name:
                continue
            to_other = vehicle.position - self.position
            if to_other.dot(self.direction_vector) > 0:
                distance = to_other.length - self.length/2 - vehicle.length/2
                if distance < min_distance:
                    return True
        return False

    def is_out_of_bounds(self, bounds):
        x, _, z = self.position
        half_bound = bounds / 2
        return (abs(x) > half_bound or abs(z) > half_bound)

    def get_model_matrix(self):
        model = Matrix44.identity()
        model = model * Matrix44.from_translation(self.position)
        model = model * Matrix44.from_y_rotation(math.radians(self.yaw))
        return model


class Pedestrian:
    """Represents a pedestrian in the 3D simulation."""
    def __init__(self, direction, spawn_position):
        self.direction_name = direction
        self.position = Vector3(spawn_position)
        self.speed = random.uniform(1.0, 2.0)
        self.waiting_time = 0
        self.stopped = False
        self.color = (0.2, 0.2, 0.2)
        self.id = random.randint(10000, 99999)
        self.height = random.uniform(1.6, 1.9)
        self.width = 0.5
        self.yaw = self._get_initial_yaw()
        self.direction_vector = self._calculate_direction_vector()
        self.vertices, self.indices = self._create_pedestrian_mesh()

    def _get_initial_yaw(self):
        if self.direction_name == "north":
            return 0
        elif self.direction_name == "south":
            return 180
        elif self.direction_name == "east":
            return 270
        elif self.direction_name == "west":
            return 90
        return 0

    def _calculate_direction_vector(self):
        radians = math.radians(self.yaw)
        return Vector3([math.sin(radians), 0.0, math.cos(radians)])

    def _create_pedestrian_mesh(self):
        vertices = []
        indices = []
        segments = 8
        radius = self.width / 2
        body_height = self.height * 0.7
        for i in range(segments):
            angle = 2 * math.pi * i / segments
            x = radius * math.cos(angle)
            z = radius * math.sin(angle)
            vertices.append((x, 0, z, *self.color))
        for i in range(segments):
            angle = 2 * math.pi * i / segments
            x = radius * math.cos(angle)
            z = radius * math.sin(angle)
            vertices.append((x, body_height, z, *self.color))
        vertices.append((0, self.height, 0, *self.color))
        for i in range(segments):
            next_i = (i + 1) % segments
            indices.extend([i, next_i, segments + i])
            indices.extend([segments + i, next_i, segments + next_i])
        head_index = len(vertices) - 1
        for i in range(segments):
            next_i = (i + 1) % segments
            indices.extend([segments + i, segments + next_i, head_index])
        return vertices, indices

    def update_position(self, delta_time, traffic_lights):
        should_stop = self._check_traffic_light(traffic_lights)
        if should_stop:
            if not self.stopped:
                self.stopped = True
            self.waiting_time += delta_time
        else:
            self.stopped = False
            movement = self.direction_vector * self.speed * delta_time
            self.position += movement

    def _check_traffic_light(self, traffic_lights):
        if self.direction_name in ["north", "south"]:
            light_e = traffic_lights["east"]
            light_w = traffic_lights["west"]
            has_green = light_e.state == GREEN or light_w.state == GREEN
            if self.direction_name == "north":
                at_crosswalk = (self.position.z > INTERSECTION_SIZE/2 and 
                                self.position.z < INTERSECTION_SIZE/2 + SIDEWALK_WIDTH)
            else:
                at_crosswalk = (self.position.z < -INTERSECTION_SIZE/2 and 
                                self.position.z > -INTERSECTION_SIZE/2 - SIDEWALK_WIDTH)
            return has_green and at_crosswalk
        elif self.direction_name in ["east", "west"]:
            light_n = traffic_lights["north"]
            light_s = traffic_lights["south"]
            has_green = light_n.state == GREEN or light_s.state == GREEN
            if self.direction_name == "east":
                at_crosswalk = (self.position.x < -INTERSECTION_SIZE/2 and 
                                self.position.x > -INTERSECTION_SIZE/2 - SIDEWALK_WIDTH)
            else:
                at_crosswalk = (self.position.x > INTERSECTION_SIZE/2 and 
                                self.position.x < INTERSECTION_SIZE/2 + SIDEWALK_WIDTH)
            return has_green and at_crosswalk
        return False

    def is_out_of_bounds(self, bounds):
        x, _, z = self.position
        half_bound = bounds / 2
        return (abs(x) > half_bound or abs(z) > half_bound)

    def get_model_matrix(self):
        model = Matrix44.identity()
        model = model * Matrix44.from_translation(self.position)
        model = model * Matrix44.from_y_rotation(math.radians(self.yaw))
        return model


class TrafficLight:
    """Represents a traffic light in the 3D simulation."""
    def __init__(self, direction, position):
        self.direction_name = direction
        self.position = Vector3(position)
        self.state = RED
        self.timer = 0
        self.red_time = 30
        self.yellow_time = 5
        self.green_time = 30
        self.yaw = self._get_orientation()
        self.vertices, self.indices = self._create_traffic_light_mesh()
        self.light_colors = [
            (1.0, 0.0, 0.0),  # Red on
            (0.3, 0.3, 0.0),  # Yellow off
            (0.0, 0.3, 0.0)   # Green off
        ]

    def _get_orientation(self):
        if self.direction_name == "north":
            return 0
        elif self.direction_name == "south":
            return 180
        elif self.direction_name == "east":
            return 270
        elif self.direction_name == "west":
            return 90
        return 0

    def _create_traffic_light_mesh(self):
        pole_height = 5.0
        pole_radius = 0.2
        housing_width = 1.0
        housing_height = 3.0
        housing_depth = 0.5
        light_radius = 0.3
        vertices = []
        indices = []
        segments = 8
        pole_color = (0.2, 0.2, 0.2)
        for i in range(segments):
            angle = 2 * math.pi * i / segments
            x = pole_radius * math.cos(angle)
            z = pole_radius * math.sin(angle)
            vertices.append((x, 0, z, *pole_color))
        for i in range(segments):
            angle = 2 * math.pi * i / segments
            x = pole_radius * math.cos(angle)
            z = pole_radius * math.sin(angle)
            vertices.append((x, pole_height, z, *pole_color))
        base_index = 0
        for i in range(segments):
            next_i = (i + 1) % segments
            indices.extend([base_index + i, base_index + next_i, base_index + segments + i])
            indices.extend([base_index + segments + i, base_index + next_i, base_index + segments + next_i])
        housing_base = len(vertices)
        hw, hh, hd = housing_width/2, housing_height/2, housing_depth/2
        housing_bottom = pole_height - hh
        housing_top = pole_height + hh
        housing_vertices = [
            (-hw, housing_bottom, -hd, 0.1, 0.1, 0.1),
            (hw, housing_bottom, -hd, 0.1, 0.1, 0.1),
            (hw, housing_top, -hd, 0.1, 0.1, 0.1),
            (-hw, housing_top, -hd, 0.1, 0.1, 0.1),
            (-hw, housing_bottom, hd, 0.1, 0.1, 0.1),
            (hw, housing_bottom, hd, 0.1, 0.1, 0.1),
            (hw, housing_top, hd, 0.1, 0.1, 0.1),
            (-hw, housing_top, hd, 0.1, 0.1, 0.1),
        ]
        for vertex in housing_vertices:
            vertices.append(vertex)
        housing_indices = [
            housing_base, housing_base + 1, housing_base + 2,
            housing_base, housing_base + 2, housing_base + 3,
            housing_base + 4, housing_base + 6, housing_base + 5,
            housing_base + 4, housing_base + 7, housing_base + 6,
            housing_base, housing_base + 3, housing_base + 7,
            housing_base, housing_base + 7, housing_base + 4,
            housing_base + 1, housing_base + 5, housing_base + 6,
            housing_base + 1, housing_base + 6, housing_base + 2,
            housing_base + 3, housing_base + 2, housing_base + 6,
            housing_base + 3, housing_base + 6, housing_base + 7,
            housing_base, housing_base + 4, housing_base + 5,
            housing_base, housing_base + 5, housing_base + 1,
        ]
        indices.extend(housing_indices)
        bulb_base = len(vertices)
        bulb_spacing = housing_height / 4
        red_pos_y = housing_top - bulb_spacing
        green_pos_y = housing_bottom + bulb_spacing
        yellow_pos_y = (red_pos_y + green_pos_y) / 2
        for pos_y, color in [
            (red_pos_y, (1.0, 0.0, 0.0)),
            (yellow_pos_y, (1.0, 1.0, 0.0)),
            (green_pos_y, (0.0, 1.0, 0.0))
        ]:
            center_index = len(vertices)
            vertices.append((0, pos_y, -hd - 0.1, *color))
            for i in range(segments):
                angle = 2 * math.pi * i / segments
                x = light_radius * math.cos(angle)
                y = light_radius * math.sin(angle)
                vertices.append((x, pos_y + y, -hd - 0.1, *color))
            for i in range(segments):
                indices.extend([
                    center_index,
                    center_index + 1 + i,
                    center_index + 1 + ((i + 1) % segments)
                ])
        return vertices, indices

    def update(self, delta_time):
        self.timer += delta_time
        if self.state == RED and self.timer >= self.red_time:
            self.state = GREEN
            self.timer = 0
        elif self.state == GREEN and self.timer >= self.green_time:
            self.state = YELLOW
            self.timer = 0
        elif self.state == YELLOW and self.timer >= self.yellow_time:
            self.state = RED
            self.timer = 0

    def get_model_matrix(self):
        model = Matrix44.identity()
        model = model * Matrix44.from_translation(self.position)
        model = model * Matrix44.from_y_rotation(math.radians(self.yaw))
        return model


class Road:
    """Represents the road, intersection, sidewalks, and related geometry."""
    def __init__(self, road_width, intersection_size, sidewalk_width):
        self.road_width = road_width
        self.intersection_size = intersection_size
        self.sidewalk_width = sidewalk_width
        self.vertices, self.indices = self._create_road_mesh()

    def _create_road_mesh(self):
        vertices = []
        indices = []
        half_road = self.road_width / 2
        half_intersection = self.intersection_size / 2
        world_size = 200
        road_color = (0.2, 0.2, 0.2)
        crosswalk_color = (0.8, 0.8, 0.8)
        sidewalk_color = (0.6, 0.6, 0.6)
        grass_color = (0.1, 0.5, 0.1)
        lane_marking_color = (0.9, 0.9, 0.0)
        ground_index = len(vertices)
        ground_size = world_size / 2
        vertices.extend([
            (-ground_size, 0, -ground_size, *grass_color),
            (ground_size, 0, -ground_size, *grass_color),
            (ground_size, 0, ground_size, *grass_color),
            (-ground_size, 0, ground_size, *grass_color)
        ])
        indices.extend([
            ground_index, ground_index + 2, ground_index + 1,
            ground_index, ground_index + 3, ground_index + 2
        ])
        ns_road_index = len(vertices)
        vertices.extend([
            (-half_road, 0.01, -ground_size, *road_color),
            (half_road, 0.01, -ground_size, *road_color),
            (half_road, 0.01, -half_intersection, *road_color),
            (-half_road, 0.01, -half_intersection, *road_color),
            (-half_road, 0.01, half_intersection, *road_color),
            (half_road, 0.01, half_intersection, *road_color),
            (half_road, 0.01, ground_size, *road_color),
            (-half_road, 0.01, ground_size, *road_color)
        ])
        indices.extend([
            ns_road_index, ns_road_index + 1, ns_road_index + 2,
            ns_road_index, ns_road_index + 2, ns_road_index + 3,
            ns_road_index + 4, ns_road_index + 5, ns_road_index + 6,
            ns_road_index + 4, ns_road_index + 6, ns_road_index + 7
        ])
        ew_road_index = len(vertices)
        vertices.extend([
            (half_intersection, 0.01, -half_road, *road_color),
            (ground_size, 0.01, -half_road, *road_color),
            (ground_size, 0.01, half_road, *road_color),
            (half_intersection, 0.01, half_road, *road_color),
            (-half_intersection, 0.01, -half_road, *road_color),
            (-ground_size, 0.01, -half_road, *road_color),
            (-ground_size, 0.01, half_road, *road_color),
            (-half_intersection, 0.01, half_road, *road_color)
        ])
        indices.extend([
            ew_road_index, ew_road_index + 1, ew_road_index + 2,
            ew_road_index, ew_road_index + 2, ew_road_index + 3,
            ew_road_index + 4, ew_road_index + 5, ew_road_index + 6,
            ew_road_index + 4, ew_road_index + 6, ew_road_index + 7
        ])
        intersection_index = len(vertices)
        vertices.extend([
            (-half_road, 0.01, -half_road, *road_color),
            (half_road, 0.01, -half_road, *road_color),
            (half_road, 0.01, half_road, *road_color),
            (-half_road, 0.01, half_road, *road_color)
        ])
        indices.extend([
            intersection_index, intersection_index + 1, intersection_index + 2,
            intersection_index, intersection_index + 2, intersection_index + 3
        ])
        lane_width = 0.2
        lane_y = 0.02
        ns_lane_index = len(vertices)
        vertices.extend([
            (-lane_width/2, lane_y, -ground_size, *lane_marking_color),
            (lane_width/2, lane_y, -ground_size, *lane_marking_color),
            (lane_width/2, lane_y, -half_intersection - 2, *lane_marking_color),
            (-lane_width/2, lane_y, -half_intersection - 2, *lane_marking_color),
            (-lane_width/2, lane_y, half_intersection + 2, *lane_marking_color),
            (lane_width/2, lane_y, half_intersection + 2, *lane_marking_color),
            (lane_width/2, lane_y, ground_size, *lane_marking_color),
            (-lane_width/2, lane_y, ground_size, *lane_marking_color)
        ])
        indices.extend([
            ns_lane_index, ns_lane_index + 1, ns_lane_index + 2,
            ns_lane_index, ns_lane_index + 2, ns_lane_index + 3,
            ns_lane_index + 4, ns_lane_index + 5, ns_lane_index + 6,
            ns_lane_index + 4, ns_lane_index + 6, ns_lane_index + 7
        ])
        ew_lane_index = len(vertices)
        vertices.extend([
            (half_intersection + 2, lane_y, -lane_width/2, *lane_marking_color),
            (ground_size, lane_y, -lane_width/2, *lane_marking_color),
            (ground_size, lane_y, lane_width/2, *lane_marking_color),
            (half_intersection + 2, lane_y, lane_width/2, *lane_marking_color),
            (-half_intersection - 2, lane_y, -lane_width/2, *lane_marking_color),
            (-ground_size, lane_y, -lane_width/2, *lane_marking_color),
            (-ground_size, lane_y, lane_width/2, *lane_marking_color),
            (-half_intersection - 2, lane_y, lane_width/2, *lane_marking_color)
        ])
        indices.extend([
            ew_lane_index, ew_lane_index + 1, ew_lane_index + 2,
            ew_lane_index, ew_lane_index + 2, ew_lane_index + 3,
            ew_lane_index + 4, ew_lane_index + 5, ew_lane_index + 6,
            ew_lane_index + 4, ew_lane_index + 6, ew_lane_index + 7
        ])
        crosswalk_width = 2.0
        crosswalk_y = 0.02
        north_cw_index = len(vertices)
        vertices.extend([
            (-half_road, crosswalk_y, -half_intersection, *crosswalk_color),
            (half_road, crosswalk_y, -half_intersection, *crosswalk_color),
            (half_road, crosswalk_y, -half_intersection - crosswalk_width, *crosswalk_color),
            (-half_road, crosswalk_y, -half_intersection - crosswalk_width, *crosswalk_color)
        ])
        indices.extend([
            north_cw_index, north_cw_index + 1, north_cw_index + 2,
            north_cw_index, north_cw_index + 2, north_cw_index + 3
        ])
        south_cw_index = len(vertices)
        vertices.extend([
            (-half_road, crosswalk_y, half_intersection, *crosswalk_color),
            (half_road, crosswalk_y, half_intersection, *crosswalk_color),
            (half_road, crosswalk_y, half_intersection + crosswalk_width, *crosswalk_color),
            (-half_road, crosswalk_y, half_intersection + crosswalk_width, *crosswalk_color)
        ])
        indices.extend([
            south_cw_index, south_cw_index + 1, south_cw_index + 2,
            south_cw_index, south_cw_index + 2, south_cw_index + 3
        ])
        east_cw_index = len(vertices)
        vertices.extend([
            (half_intersection, crosswalk_y, -half_road, *crosswalk_color),
            (half_intersection + crosswalk_width, crosswalk_y, -half_road, *crosswalk_color),
            (half_intersection + crosswalk_width, crosswalk_y, half_road, *crosswalk_color),
            (half_intersection, crosswalk_y, half_road, *crosswalk_color)
        ])
        indices.extend([
            east_cw_index, east_cw_index + 1, east_cw_index + 2,
            east_cw_index, east_cw_index + 2, east_cw_index + 3
        ])
        west_cw_index = len(vertices)
        vertices.extend([
            (-half_intersection, crosswalk_y, -half_road, *crosswalk_color),
            (-half_intersection - crosswalk_width, crosswalk_y, -half_road, *crosswalk_color),
            (-half_intersection - crosswalk_width, crosswalk_y, half_road, *crosswalk_color),
            (-half_intersection, crosswalk_y, half_road, *crosswalk_color)
        ])
        indices.extend([
            west_cw_index, west_cw_index + 1, west_cw_index + 2,
            west_cw_index, west_cw_index + 2, west_cw_index + 3
        ])
        sw_width = self.sidewalk_width
        sw_y = 0.05
        north_sw_index = len(vertices)
        vertices.extend([
            (-half_road - sw_width, sw_y, -half_intersection - sw_width, *sidewalk_color),
            (half_road + sw_width, sw_y, -half_intersection - sw_width, *sidewalk_color),
            (half_road + sw_width, sw_y, -half_intersection - sw_width - 10, *sidewalk_color),
            (-half_road - sw_width, sw_y, -half_intersection - sw_width - 10, *sidewalk_color)
        ])
        indices.extend([
            north_sw_index, north_sw_index + 1, north_sw_index + 2,
            north_sw_index, north_sw_index + 2, north_sw_index + 3
        ])
        return vertices, indices

    def get_model_matrix(self):
        return Matrix44.identity()


# -------------------------- Simulation Class --------------------------
class Simulation:
    """Main simulation class that ties together the rendering, physics, and logic."""
    def __init__(self, spawn_rate=0.05, pedestrian_spawn_rate=0.005, bounds=200):
        self.ctx = moderngl.create_context()
        self.ctx.enable(moderngl.DEPTH_TEST)
        self.ctx.enable(moderngl.CULL_FACE)
        self.camera = Camera()
        self.road = Road(ROAD_WIDTH, INTERSECTION_SIZE, SIDEWALK_WIDTH)
        self.traffic_lights = {
            "north": TrafficLight("north", (0, 0, -INTERSECTION_SIZE/2)),
            "south": TrafficLight("south", (0, 0, INTERSECTION_SIZE/2)),
            "east": TrafficLight("east", (INTERSECTION_SIZE/2, 0, 0)),
            "west": TrafficLight("west", (-INTERSECTION_SIZE/2, 0, 0))
        }
        self.traffic_lights["north"].state = GREEN
        self.traffic_lights["south"].state = GREEN
        self.traffic_lights["east"].state = RED
        self.traffic_lights["west"].state = RED
        for tl in self.traffic_lights.values():
            tl.timer = 0
        self.vehicles = []
        self.pedestrians = []
        self.vehicle_spawn_rate = spawn_rate
        self.vehicle_spawn_timer = 0
        self.pedestrian_spawn_rate = pedestrian_spawn_rate
        self.pedestrian_spawn_timer = 0
        self.bounds = bounds
        self.stats = {
            "vehicles_spawned": 0,
            "vehicles_despawned": 0,
            "pedestrians_spawned": 0,
            "pedestrians_despawned": 0
        }
        self.show_stats = True
        self.follow_car = None
        self.create_shaders()
        self.create_meshes()

    def create_shaders(self):
        vertex_shader = """
            #version 330
            in vec3 in_position;
            in vec3 in_color;
            uniform mat4 model;
            uniform mat4 view;
            uniform mat4 projection;
            out vec3 color;
            void main() {
                color = in_color;
                gl_Position = projection * view * model * vec4(in_position, 1.0);
            }
        """
        fragment_shader = """
            #version 330
            in vec3 color;
            out vec4 fragColor;
            void main() {
                fragColor = vec4(color, 1.0);
            }
        """
        self.shader = Shader(self.ctx, vertex_shader, fragment_shader)

    def create_meshes(self):
        road_vertices = np.array(self.road.vertices, dtype='f4')
        road_indices = np.array(self.road.indices, dtype='i4')
        self.road_vbo = self.ctx.buffer(road_vertices)
        self.road_ibo = self.ctx.buffer(road_indices)
        self.shader.create_vao('road', self.shader.program, self.road_vbo, self.road_ibo)
        for direction, light in self.traffic_lights.items():
            vertices = np.array(light.vertices, dtype='f4')
            indices = np.array(light.indices, dtype='i4')
            vbo = self.ctx.buffer(vertices)
            ibo = self.ctx.buffer(indices)
            self.shader.create_vao(f'light_{direction}', self.shader.program, vbo, ibo)

    def update_meshes(self):
        # Remove previous dynamic VAOs
        for key in list(self.shader.vao.keys()):
            if key.startswith('vehicle_') or key.startswith('pedestrian_'):
                del self.shader.vao[key]
        for vehicle in self.vehicles:
            vertices = np.array(vehicle.vertices, dtype='f4')
            indices = np.array(vehicle.indices, dtype='i4')
            vbo = self.ctx.buffer(vertices)
            ibo = self.ctx.buffer(indices)
            self.shader.create_vao(f'vehicle_{vehicle.id}', self.shader.program, vbo, ibo)
        for pedestrian in self.pedestrians:
            vertices = np.array(pedestrian.vertices, dtype='f4')
            indices = np.array(pedestrian.indices, dtype='i4')
            vbo = self.ctx.buffer(vertices)
            ibo = self.ctx.buffer(indices)
            self.shader.create_vao(f'pedestrian_{pedestrian.id}', self.shader.program, vbo, ibo)

    def spawn_vehicle(self):
        direction = random.choice(["north", "south", "east", "west"])
        vehicle_type = random.choice(VEHICLE_TYPES)
        lane = random.randint(0, 1)
        half_road = ROAD_WIDTH / 2
        lane_offset = LANE_WIDTH / 2 + lane * LANE_WIDTH
        spawn_z = self.bounds / 2 * (-1 if direction == "north" else 1)
        spawn_x = self.bounds / 2 * (-1 if direction == "west" else 1)
        if direction in ["north", "south"]:
            lane_sign = -1 if direction == "north" else 1
            spawn_x = lane_sign * lane_offset
            spawn_position = (spawn_x, 0, spawn_z)
        else:
            lane_sign = 1 if direction == "east" else -1
            spawn_z = lane_sign * lane_offset
            spawn_position = (spawn_x, 0, spawn_z)
        vehicle = Vehicle(vehicle_type, lane, direction, spawn_position)
        self.vehicles.append(vehicle)
        self.stats["vehicles_spawned"] += 1

    def spawn_pedestrian(self):
        direction = random.choice(["north", "south", "east", "west"])
        half_intersection = INTERSECTION_SIZE / 2
        sidewalk_width = SIDEWALK_WIDTH
        road_width = ROAD_WIDTH
        if direction == "north":
            spawn_x = random.uniform(-road_width, road_width)
            spawn_z = -half_intersection - sidewalk_width - 5
            spawn_position = (spawn_x, 0, spawn_z)
        elif direction == "south":
            spawn_x = random.uniform(-road_width, road_width)
            spawn_z = half_intersection + sidewalk_width + 5
            spawn_position = (spawn_x, 0, spawn_z)
        elif direction == "east":
            spawn_z = random.uniform(-road_width, road_width)
            spawn_x = half_intersection + sidewalk_width + 5
            spawn_position = (spawn_x, 0, spawn_z)
        elif direction == "west":
            spawn_z = random.uniform(-road_width, road_width)
            spawn_x = -half_intersection - sidewalk_width - 5
            spawn_position = (spawn_x, 0, spawn_z)
        pedestrian = Pedestrian(direction, spawn_position)
        self.pedestrians.append(pedestrian)
        self.stats["pedestrians_spawned"] += 1

    def update(self, delta_time):
        # Update spawn timers and spawn new objects if needed
        self.vehicle_spawn_timer += delta_time
        self.pedestrian_spawn_timer += delta_time
        if self.vehicle_spawn_timer >= self.vehicle_spawn_rate:
            self.spawn_vehicle()
            self.vehicle_spawn_timer = 0
        if self.pedestrian_spawn_timer >= self.pedestrian_spawn_rate:
            self.spawn_pedestrian()
            self.pedestrian_spawn_timer = 0
        # Update traffic lights
        for tl in self.traffic_lights.values():
            tl.update(delta_time)
        # Update vehicles
        for vehicle in self.vehicles:
            vehicle.update_position(delta_time, self.traffic_lights, self.vehicles)
        # Update pedestrians
        for pedestrian in self.pedestrians:
            pedestrian.update_position(delta_time, self.traffic_lights)
        # Remove out-of-bound objects
        self.vehicles = [v for v in self.vehicles if not v.is_out_of_bounds(self.bounds)]
        self.pedestrians = [p for p in self.pedestrians if not p.is_out_of_bounds(self.bounds)]
        self.update_meshes()

    def render(self):
        aspect_ratio = SCREEN_WIDTH / SCREEN_HEIGHT
        view = self.camera.get_view_matrix()
        projection = self.camera.get_projection_matrix(aspect_ratio)
        self.shader.program["view"].write(view.astype("f4").tobytes())
        self.shader.program["projection"].write(projection.astype("f4").tobytes())
        # Render road
        self.shader.program["model"].write(Matrix44.identity().astype("f4").tobytes())
        self.shader.vao["road"].render()
        # Render traffic lights
        for direction, tl in self.traffic_lights.items():
            model = tl.get_model_matrix()
            self.shader.program["model"].write(model.astype("f4").tobytes())
            self.shader.vao[f"light_{direction}"].render()
        # Render vehicles
        for vehicle in self.vehicles:
            model = vehicle.get_model_matrix()
            self.shader.program["model"].write(model.astype("f4").tobytes())
            vao_key = f"vehicle_{vehicle.id}"
            if vao_key in self.shader.vao:
                self.shader.vao[vao_key].render()
        # Render pedestrians
        for pedestrian in self.pedestrians:
            model = pedestrian.get_model_matrix()
            self.shader.program["model"].write(model.astype("f4").tobytes())
            vao_key = f"pedestrian_{pedestrian.id}"
            if vao_key in self.shader.vao:
                self.shader.vao[vao_key].render()
        pygame.display.flip()

    def run(self):
        running = True
        last_time = time.time()
        while running:
            delta_time = time.time() - last_time
            last_time = time.time()
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                    running = False
            keys = pygame.key.get_pressed()
            mouse_rel = pygame.mouse.get_rel()
            self.camera.process_input(keys, delta_time, mouse_rel)
            self.update(delta_time)
            self.ctx.clear(0.1, 0.1, 0.1, 1.0)
            self.render()
            clock.tick(60)
        pygame.quit()


# -------------------------- Main Function --------------------------
def main():
    parser = argparse.ArgumentParser(description="3D Traffic Simulation")
    parser.add_argument("--config", type=str, help="Path to configuration file")
    args = parser.parse_args()
    simulation = Simulation()
    simulation.run()


if __name__ == "__main__":
    main()
